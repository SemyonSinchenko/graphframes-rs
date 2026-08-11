use std::sync::Arc;

use crate::expressions::common::{as_binary_like, downcast_int32};
use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanArray, Int32Array,
};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::HashMap;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::utils::format_state_name;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, EmitTo, Expr, GroupsAccumulator, Signature,
    Volatility,
};
use datafusion::scalar::ScalarValue;

#[derive(Debug)]
pub(crate) struct KCoreReduceAccumulator {
    // histogram of neighbors cores
    counts: HashMap<i32, u32>,
}

impl KCoreReduceAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct KCoreReduceGroupsAccumulator {
    // accumulator for all groups at once;
    // fast path.
    counts: Vec<HashMap<i32, u32>>,
}

impl KCoreReduceGroupsAccumulator {
    pub(crate) fn new() -> Self {
        Self { counts: Vec::new() }
    }
}

/// The main logic of choosing a new (uncapped) core.
///
/// Returns `uncapped_A = max{ l : #{neighbours with core >= l} >= l }`.
/// The `min(., current_core)` cap is applied later, in the Pregel vertex
/// update expression, because `current_core` is not visible to the
/// aggregate (see `skip_dest_state`).
///
/// Based on: Mandal, Aritra, and Mohammad Al Hasan. "A distributed k-core
/// decomposition algorithm on spark." 2017 IEEE International Conference
/// on Big Data (Big Data). IEEE, 2017.
///
/// Shared between Accumulator and GroupsAccumulator
fn uncapped_core(m: &HashMap<i32, u32>) -> i32 {
    if m.is_empty() {
        return 0i32;
    }

    let mut entries: Vec<(i32, u32)> = m.iter().map(|(&k, &v)| (k, v)).collect();
    // Descending by value so the running sum is the "# neighbours >= value".
    entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));

    let mut cum: u32 = 0; // running #{neighbours with value >= current}; bounded by degree < i32::MAX
    let mut best: i32 = 0; // uncapped_A >= 0 always (l=0 always satisfies ge(0)=total>=0)
    for (value, count) in entries {
        cum += count;
        // candidate = the largest l <= value with ge(l) >= l on this step
        let candidate = value.min(cum as i32);
        if candidate > best {
            best = candidate;
        }
    }

    best
}

fn se_map(m: &HashMap<i32, u32>) -> Vec<u8> {
    // We are assumming that a single node degree < i32::MAX
    let n = m.len() as u32;
    let mut buf = Vec::with_capacity(4 + 8usize * (n as usize));
    buf.extend_from_slice(&n.to_le_bytes());

    for (&k, &v) in m.iter() {
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

fn de_map_and_insert(buf: &[u8], map: &mut HashMap<i32, u32>) -> Result<()> {
    if buf.len() < 4 {
        return Err(DataFusionError::Execution(
            "k_core_reduce: corrupt state (length less 4)".to_string(),
        ));
    }
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if buf.len() != 4 + 8 * n {
        return Err(DataFusionError::Execution(
            "k_core_reduce: corrupt state (length mismatch)".to_string(),
        ));
    }

    // Each entry is 8 bytes: i32 key | u32 value (see `se_map`).
    for i in 0..n {
        let o = 4 + 8 * i;
        let k = i32::from_le_bytes(buf[o..o + 4].try_into().unwrap());
        let v = u32::from_le_bytes(buf[o + 4..o + 8].try_into().unwrap());
        *map.entry(k).or_insert(0u32) += v;
    }

    Ok(())
}

impl GroupsAccumulator for KCoreReduceGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let v = downcast_int32(&values[0], "k_core_reduce", "first")?;
        self.counts.resize(total_num_groups, HashMap::new());

        // Nulls are not expected by the Pregel message contract, but skip them
        // (and filtered-out rows) defensively: counting a null slot's raw bits
        // would silently corrupt the histogram.
        for i in 0..v.len() {
            if v.is_null(i) || opt_filter.is_some_and(|f| !f.value(i)) {
                continue;
            }
            let l = v.value(i);
            let cur = self.counts[group_indices[i]].entry(l).or_insert(0u32);
            *cur += 1u32;
        }

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let maps = emit_to.take_needed(&mut self.counts);
        let result: Int32Array = (0..maps.len()).map(|i| uncapped_core(&maps[i])).collect();

        Ok(Arc::new(result) as ArrayRef)
    }

    fn size(&self) -> usize {
        // size of self
        // + size of hashmap struct * num groups
        // + sum of sizes of each map (i32 + u32) * capacity of each
        let mut r = size_of::<Self>();
        r += self.counts.capacity() * size_of::<HashMap<i32, u32>>();
        for i in 0..self.counts.len() {
            r += self.counts[i].capacity() * (size_of::<i32>() + size_of::<u32>());
        }

        r
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let maps = emit_to.take_needed(&mut self.counts);
        let result = BinaryArray::from_iter_values((0..maps.len()).map(|i| se_map(&maps[i])));

        Ok(vec![Arc::new(result) as ArrayRef])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        let v = as_binary_like(&values[0], "k_core_reduce", "argument")?;

        self.counts.resize(total_num_groups, HashMap::new());

        // Null state rows appear when the skip-aggregation path filtered rows
        // out (`convert_to_state` emits nulls for them); they must be ignored,
        // otherwise the empty blob would be rejected as corrupt state.
        for i in 0..v.len() {
            if v.is_null(i) || opt_filter.is_some_and(|f| !f.value(i)) {
                continue;
            }
            de_map_and_insert(v.value(i), &mut self.counts[group_indices[i]])?;
        }

        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let v = downcast_int32(&values[0], "k_core_reduce", "first")?;

        // Each input row becomes its own single-entry histogram state.
        // Filtered-out (and null, not expected by contract) rows become null
        // states so the Final phase ignores them in `merge_batch`.
        let mut builder = BinaryBuilder::new();
        for i in 0..v.len() {
            if v.is_null(i) || opt_filter.is_some_and(|f| !f.value(i)) {
                builder.append_null();
            } else {
                let m = HashMap::from([(v.value(i), 1u32)]);
                builder.append_value(se_map(&m));
            }
        }

        Ok(vec![Arc::new(builder.finish()) as ArrayRef])
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }
}

impl Accumulator for KCoreReduceAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let labels = downcast_int32(&values[0], "k_core_reduce", "first")?;

        // No nulls expected: this is an internal aggregator over neighbour core
        // estimates, which are non-null by the Pregel message contract.
        for i in 0..labels.len() {
            let l = labels.value(i);

            let cur = self.counts.entry(l).or_insert(0u32);
            *cur += 1u32;
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let best = uncapped_core(&self.counts);
        Ok(ScalarValue::Int32(Some(best)))
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.counts.capacity() * (size_of::<i32>() + size_of::<u32>())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(se_map(&self.counts)))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let v = as_binary_like(&states[0], "k_core_reduce", "argument")?;

        // we are assumming no null-maps here;
        // null can be only an output of evaluate, not state
        for i in 0..v.len() {
            de_map_and_insert(v.value(i), &mut self.counts)?;
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct KCoreReduce {
    signature: Signature,
}

impl KCoreReduce {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Int32], Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for KCoreReduce {
    fn name(&self) -> &str {
        "k_core_reduce"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if (arg_types.len() != 1) || (arg_types[0] != DataType::Int32) {
            return Err(DataFusionError::Plan(format!(
                "k_core_reduce expects exactly one argument of type i32 but got {arg_types:?}"
            )));
        }

        Ok(DataType::Int32)
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(KCoreReduceAccumulator::new()))
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        _args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(KCoreReduceGroupsAccumulator::new()))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(vec![Arc::new(Field::new(
            format_state_name(args.name, "value"),
            DataType::Binary,
            true,
        ))])
    }
}

/// Builds an [`Expr`] that applies `k_core_reduce(a)`.
pub(crate) fn kcore_reduce(a: Expr) -> Expr {
    AggregateUDF::from(KCoreReduce::new()).call(vec![a])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{ArrayRef, BinaryArray, Int32Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionConfig;
    use datafusion::prelude::*;

    /// Feed a batch of i32 neighbour cores into a fresh accumulator.
    fn reduce(cores: &[i32]) -> KCoreReduceAccumulator {
        let mut acc = KCoreReduceAccumulator::new();
        acc.update_batch(&[Arc::new(Int32Array::from(cores.to_vec())) as ArrayRef])
            .unwrap();
        acc
    }

    /// Direct accumulator: uncapped_A over a neighbour multiset.
    #[test]
    fn test_accumulator_picks_uncapped_core() {
        // {3:3}: ge(3)=3 -> 3
        assert_eq!(
            reduce(&[3, 3, 3]).evaluate().unwrap(),
            ScalarValue::Int32(Some(3))
        );
        // {2:2, 1:1}: ge(2)=2 -> 2
        assert_eq!(
            reduce(&[2, 2, 1]).evaluate().unwrap(),
            ScalarValue::Int32(Some(2))
        );
        // {5:3}: only 3 neighbours, so uncapped capped at 3
        assert_eq!(
            reduce(&[5, 5, 5]).evaluate().unwrap(),
            ScalarValue::Int32(Some(3))
        );
        // {1:1}: single neighbour -> 1
        assert_eq!(
            reduce(&[1]).evaluate().unwrap(),
            ScalarValue::Int32(Some(1))
        );
        // {10:1, 1:1}: ge(1)=2, ge(2)=1 -> 1
        assert_eq!(
            reduce(&[10, 1]).evaluate().unwrap(),
            ScalarValue::Int32(Some(1))
        );
    }

    /// No neighbours -> uncapped_A = 0 (the empty multiset).
    #[test]
    fn test_accumulator_empty_returns_zero() {
        let mut acc = KCoreReduceAccumulator::new();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int32(Some(0)));
    }

    /// All neighbours core 0 -> no support -> uncapped 0.
    #[test]
    fn test_accumulator_all_zero_returns_zero() {
        assert_eq!(
            reduce(&[0, 0, 0, 0, 0]).evaluate().unwrap(),
            ScalarValue::Int32(Some(0))
        );
    }

    /// se_map/de_map round-trip preserves the histogram exactly.
    #[test]
    fn test_se_de_map_roundtrip_preserves_counts() {
        let mut original: HashMap<i32, u32> = HashMap::new();
        original.insert(1, 3);
        original.insert(2, 5);
        original.insert(7, 1);
        let bytes = se_map(&original);
        let mut restored: HashMap<i32, u32> = HashMap::new();
        de_map_and_insert(&bytes, &mut restored).unwrap();
        assert_eq!(restored.len(), original.len());
        for (k, v) in &original {
            assert_eq!(restored.get(k).copied().unwrap(), *v, "key {k}");
        }
    }

    /// Merge semantics: `de_map_and_insert` must ADD counts for shared keys,
    /// not overwrite. This is the property that makes partial->final merging
    /// correct across partitions.
    #[test]
    fn test_de_map_and_insert_accumulates_not_overwrites() {
        let mut m: HashMap<i32, u32> = HashMap::new();
        m.insert(3, 2);
        let mut partial: HashMap<i32, u32> = HashMap::new();
        partial.insert(3, 1); // shared key
        partial.insert(5, 4); // new key
        let bytes = se_map(&partial);
        de_map_and_insert(&bytes, &mut m).unwrap();
        assert_eq!(m.get(&3).copied().unwrap(), 3, "shared key must sum");
        assert_eq!(m.get(&5).copied().unwrap(), 4, "new key must appear");
    }

    /// A corrupt/truncated state blob must surface as a query error, not a panic.
    #[test]
    fn test_de_map_and_insert_malformed_errors() {
        let mut m: HashMap<i32, u32> = HashMap::new();
        // Header claims 1 entry but there is no payload -> length mismatch.
        assert!(de_map_and_insert(&1u32.to_le_bytes(), &mut m).is_err());
        // Empty buffer -> length < 4.
        assert!(de_map_and_insert(&[], &mut m).is_err());
    }

    /// Direct partial->final merge: serialize b's state and fold it into a.
    /// a={3:2} (uncapped 2), b={3:1} (uncapped 1); merged {3:3} -> uncapped 3.
    /// A no-op `merge_batch` yields 2; keeping only the local winner yields 1.
    #[test]
    fn test_accumulator_merge_unions_partial_states() {
        let mut a = reduce(&[3, 3]);
        let mut b = reduce(&[3]);

        let b_state = b.state().unwrap();
        let bytes = match &b_state[0] {
            ScalarValue::Binary(Some(x)) => x.clone(),
            other => panic!("expected Binary state, got {other:?}"),
        };
        a.merge_batch(&[Arc::new(BinaryArray::from(vec![Some(bytes.as_slice())])) as ArrayRef])
            .unwrap();

        assert_eq!(a.evaluate().unwrap(), ScalarValue::Int32(Some(3)));
    }

    /// GroupsAccumulator: multiple groups updated in a single batch evaluate
    /// to their own uncapped_A, in group-index order, and `EmitTo::All`
    /// resets the internal state.
    #[test]
    fn test_groups_accumulator_update_evaluate_multi_group() {
        let mut acc = KCoreReduceGroupsAccumulator::new();
        // g0: {3:3} -> 3; g1: {2:2, 1:1} -> 2; g2: {5:3} -> 3
        acc.update_batch(
            &[Arc::new(Int32Array::from(vec![3i32, 3, 3, 2, 2, 1, 5, 5, 5])) as ArrayRef],
            &[0, 0, 0, 1, 1, 1, 2, 2, 2],
            None,
            3,
        )
        .unwrap();

        let out = acc.evaluate(EmitTo::All).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.value(0), 3);
        assert_eq!(out.value(1), 2);
        assert_eq!(out.value(2), 3);

        // EmitTo::All released the state; a fresh batch starts from scratch.
        assert!(acc.counts.is_empty());
    }

    /// GroupsAccumulator: `state()`/`merge_batch()` round-trip must SUM the
    /// partial histograms for shared keys — the associativity property that
    /// makes two-phase (Partial -> Final) aggregation correct. This is the
    /// groups-level analogue of `test_accumulator_merge_unions_partial_states`.
    #[test]
    fn test_groups_accumulator_state_merge_roundtrip() {
        let mut a = KCoreReduceGroupsAccumulator::new();
        a.update_batch(
            &[Arc::new(Int32Array::from(vec![3i32, 3])) as ArrayRef],
            &[0, 0],
            None,
            1,
        )
        .unwrap();

        let mut b = KCoreReduceGroupsAccumulator::new();
        b.update_batch(
            &[Arc::new(Int32Array::from(vec![3i32, 5])) as ArrayRef],
            &[0, 0],
            None,
            1,
        )
        .unwrap();

        let states = b.state(EmitTo::All).unwrap();
        assert_eq!(states.len(), 1);
        // a={3:2}, b={3:1, 5:1}; merged {3:3, 5:1} -> ge(3)=3 -> uncapped 3
        a.merge_batch(&states, &[0], None, 1).unwrap();

        let out = a.evaluate(EmitTo::All).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), 3);
    }

    /// GroupsAccumulator: `EmitTo::First(n)` emits the first n groups and
    /// shifts the remaining groups' indices down by n.
    #[test]
    fn test_groups_accumulator_emit_to_first() {
        let mut acc = KCoreReduceGroupsAccumulator::new();
        acc.update_batch(
            &[Arc::new(Int32Array::from(vec![1i32, 10, 20, 20])) as ArrayRef],
            &[0, 1, 1, 1],
            None,
            2,
        )
        .unwrap();

        let out = acc.evaluate(EmitTo::First(1)).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        // g0: {1:1} -> 1
        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), 1);

        // g1 (now index 0): {10:1, 20:2} -> 3
        let out = acc.evaluate(EmitTo::All).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), 3);
    }

    /// GroupsAccumulator: `convert_to_state` turns each input row into its own
    /// single-entry histogram state, and merging those states reproduces the
    /// histogram of the original input. Filtered-out rows must become null
    /// states and `merge_batch` must skip them.
    #[test]
    fn test_groups_accumulator_convert_to_state() {
        let acc = KCoreReduceGroupsAccumulator::new();

        let states = acc
            .convert_to_state(
                &[Arc::new(Int32Array::from(vec![3i32, 3, 5])) as ArrayRef],
                None,
            )
            .unwrap();
        assert_eq!(states.len(), 1);
        let bin = states[0].as_any().downcast_ref::<BinaryArray>().unwrap();
        assert_eq!(bin.len(), 3);
        assert_eq!(bin.null_count(), 0);

        let mut merged = KCoreReduceGroupsAccumulator::new();
        merged.merge_batch(&states, &[0, 0, 0], None, 1).unwrap();
        let out = merged.evaluate(EmitTo::All).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        // {3:2, 5:1} -> ge(3)=3 -> uncapped 3
        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), 3);

        // Filtered-out row (index 1) must become a null state and be skipped.
        let filter = Arc::new(BooleanArray::from(vec![true, false, true]));
        let states = acc
            .convert_to_state(
                &[Arc::new(Int32Array::from(vec![3i32, 3, 5])) as ArrayRef],
                Some(filter.as_ref()),
            )
            .unwrap();
        let bin = states[0].as_any().downcast_ref::<BinaryArray>().unwrap();
        assert_eq!(bin.null_count(), 1);
        assert!(bin.is_null(1));

        let mut merged = KCoreReduceGroupsAccumulator::new();
        merged.merge_batch(&states, &[0, 0, 0], None, 1).unwrap();
        let out = merged.evaluate(EmitTo::All).unwrap();
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        // only rows 0 and 2 counted: {3:1, 5:1} -> ge(2)=2 -> uncapped 2
        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), 2);
    }

    /// GROUP BY: each group resolves to its own uncapped_A independently.
    #[tokio::test]
    async fn test_kcore_reduce_grouped() -> Result<()> {
        // g=0: cores [1,2,1] -> {1:2,2:1} -> uncapped 1
        // g=1: cores [10,20,20] -> {10:1,20:2} -> uncapped 3
        let df = dataframe!(
            "g" => vec![0i64, 0, 0, 1, 1, 1],
            "a" => vec![1i32, 2, 1, 10, 20, 20],
        )?;
        let out = df
            .aggregate(vec![col("g")], vec![kcore_reduce(col("a")).alias("k")])?
            .sort(vec![col("g").sort(true, true)])?
            .collect()
            .await?;
        let g = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let k = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.value(0), 0);
        assert_eq!(k.value(0), 1);
        assert_eq!(g.value(1), 1);
        assert_eq!(k.value(1), 3);
        Ok(())
    }

    /// Multi-partition: force the partial->final path via two `MemTable`
    /// partitions and `target_partitions(2)`. Vertex 0 receives core 3 twice in
    /// P1 and once in P2; only a correct `state()` + `merge_batch` (summing the
    /// shared key 3 -> 3) yields uncapped 3. This is the test that catches a
    /// broken `de_map` stride/offset or a wrong `state_fields`.
    #[tokio::test]
    async fn test_kcore_reduce_multi_partition_merge() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("a", DataType::Int32, false),
        ]));
        let mk = |cores: Vec<i32>| {
            let n = cores.len();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        std::iter::repeat(0i64).take(n),
                    )) as ArrayRef,
                    Arc::new(Int32Array::from(cores)) as ArrayRef,
                ],
            )
            .unwrap()
        };
        // P1: {3:2} -> uncapped 2 alone; P2: {3:1} -> uncapped 1 alone.
        let p1 = mk(vec![3, 3]);
        let p2 = mk(vec![3]);

        let table = MemTable::try_new(schema, vec![vec![p1], vec![p2]]).unwrap();
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        ctx.register_table("t", Arc::new(table)).unwrap();
        let out = ctx
            .table("t")
            .await
            .unwrap()
            .aggregate(vec![col("g")], vec![kcore_reduce(col("a")).alias("k")])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let k = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(k.len(), 1);
        // merged {3:3} -> uncapped 3
        assert_eq!(k.value(0), 3);
    }

    /// `return_type` must accept exactly (Int32) and reject the rest.
    #[test]
    fn test_kcore_reduce_return_type_validation() {
        let udf = KCoreReduce::new();
        assert_eq!(
            udf.return_type(&[DataType::Int32]).unwrap(),
            DataType::Int32
        );
        assert!(udf.return_type(&[DataType::Int64]).is_err());
        assert!(udf.return_type(&[]).is_err());
        assert!(
            udf.return_type(&[DataType::Int32, DataType::Int32])
                .is_err()
        );
    }
}
