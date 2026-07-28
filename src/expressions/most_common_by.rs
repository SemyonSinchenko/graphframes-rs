use std::sync::Arc;

use crate::expressions::common::{as_binary_like, downcast_int64};
use datafusion::arrow::array::{ArrayRef, Float64Array};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::HashMap;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::utils::format_state_name;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Expr, Signature, Volatility,
};
use datafusion::scalar::ScalarValue;

#[derive(Debug)]
pub(crate) struct MostCommonByAccumulator {
    sums: HashMap<i64, f64>,
}

impl MostCommonByAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            sums: HashMap::new(),
        }
    }
}

fn se_map(m: &HashMap<i64, f64>) -> Vec<u8> {
    // We are assumming that a single node degree < i32::MAX
    let n = m.len() as u32;
    let mut buf = Vec::with_capacity(4 + 16usize * (n as usize));
    buf.extend_from_slice(&n.to_le_bytes());

    for (&k, &v) in m.iter() {
        buf.extend_from_slice(&k.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

fn de_map_and_insert(buf: &[u8], map: &mut HashMap<i64, f64>) -> Result<()> {
    if buf.len() < 4 {
        return Err(DataFusionError::Execution(
            "most_common_by: corrupt state (length less 4)".to_string(),
        ));
    }
    let n = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    if buf.len() != 4 + 16 * n {
        return Err(DataFusionError::Execution(
            "most_common_by: corrupt state (length mismatch)".to_string(),
        ));
    }

    for i in 0..n {
        let o = 4 + 16 * i;
        let k = i64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        let v = f64::from_le_bytes(buf[o + 8..o + 16].try_into().unwrap());
        *map.entry(k).or_insert(0.0) += v;
    }

    Ok(())
}

impl Accumulator for MostCommonByAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let labels = downcast_int64(&values[0], "most_common_by", "first")?;
        let weights = &values[1].as_any().downcast_ref::<Float64Array>().ok_or(
            DataFusionError::Execution(
                "expected most_common_by secobd argument be f64 array".to_string(),
            ),
        )?;

        // we are assumming that there won't be nulls;
        // it is internal function that assumet to aggregate neighbors:
        // 1) nbr ID is not null by the DataFusion contract
        // 2) weights are not null by the contract of this aggregator
        for i in 0..labels.len() {
            let l = labels.value(i);
            let v = weights.value(i);

            let cur = self.sums.entry(l).or_insert(0.0);
            *cur += v;
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // That can be a possible case. No neighbors we should return null.
        if self.sums.is_empty() {
            return Ok(ScalarValue::Int64(None));
        }
        let mut max = -1;
        let mut max_value = f64::MIN;

        for (k, v) in self.sums.iter() {
            if v > &max_value {
                max_value = *v;
                max = *k;
            } else if v == &max_value {
                // tie-breaking: minimal key value:
                // LDBC's Label Propagation semantics
                if k < &max {
                    max_value = *v;
                    max = *k;
                }
            }
        }

        Ok(ScalarValue::Int64(Some(max)))
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.sums.capacity() * (size_of::<i64>() + size_of::<f64>())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(se_map(&self.sums)))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let v = as_binary_like(&states[0], "most_common_by", "argument")?;

        // we are assumming no null-maps here;
        // null can be only an output of evaluate, not state
        for i in 0..v.len() {
            de_map_and_insert(v.value(i), &mut self.sums)?;
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct MostCommonBy {
    signature: Signature,
}

impl MostCommonBy {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Int64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for MostCommonBy {
    fn name(&self) -> &str {
        "most_common_by"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if (arg_types.len() != 2)
            || !matches!(
                (&arg_types[0], &arg_types[1]),
                (DataType::Int64, DataType::Float64)
            )
        {
            return Err(DataFusionError::Plan(format!(
                "most_common_by expets exactly two arguments of types i64 and f64 but got {arg_types:?}"
            )));
        }

        Ok(DataType::Int64)
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(MostCommonByAccumulator::new()))
    }

    /// The intermediate state is a serialized `HashMap<i64, f64>` carried as an
    /// opaque `Binary` blob — a *different* type from the `Int64` final return
    /// value. The default `state_fields` mirrors `return_type`, so it would
    /// declare an `Int64` state column and the partial->final merge would never
    /// receive the serialized maps, silently breaking multi-partition
    /// aggregation. Override to declare the true `Binary` intermediate state.
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(vec![Arc::new(Field::new(
            format_state_name(args.name, "value"),
            DataType::Binary,
            true,
        ))])
    }
}

/// Builds an [`Expr`] that applies `most_common_by(a, b)`.
pub(crate) fn most_common_by(a: Expr, b: Expr) -> Expr {
    AggregateUDF::from(MostCommonBy::new()).call(vec![a, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{BinaryArray, Float64Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionConfig;
    use datafusion::prelude::*;

    /// Direct accumulator: weights summed per label, argmax picked.
    #[test]
    fn test_accumulator_update_picks_argmax() {
        let mut acc = MostCommonByAccumulator::new();
        // label 1 -> 5.0 (2+3), label 2 -> 10.0 -> winner 2
        acc.update_batch(&[
            Arc::new(Int64Array::from(vec![1i64, 2, 1])) as ArrayRef,
            Arc::new(Float64Array::from(vec![2.0f64, 10.0, 3.0])) as ArrayRef,
        ])
        .unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(Some(2)));
    }

    /// No inputs -> null result (vertex received no messages this iteration).
    #[test]
    fn test_accumulator_empty_returns_null() {
        let mut acc = MostCommonByAccumulator::new();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(None));
    }

    /// Tie on summed weight -> smallest label wins (LDBC LP semantics).
    #[test]
    fn test_accumulator_tie_picks_min_label() {
        let mut acc = MostCommonByAccumulator::new();
        acc.update_batch(&[
            Arc::new(Int64Array::from(vec![5i64, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![5.0f64, 5.0])) as ArrayRef,
        ])
        .unwrap();
        assert_eq!(acc.evaluate().unwrap(), ScalarValue::Int64(Some(2)));
    }

    /// se_map/de_map round-trip preserves the map exactly. This is the test
    /// that catches the LE/BE count mismatch (a swapped endianness yields a
    /// bogus `n` and the length check rejects the blob).
    #[test]
    fn test_se_de_map_roundtrip_preserves_sums() {
        let mut original: HashMap<i64, f64> = HashMap::new();
        original.insert(1, 3.0);
        original.insert(2, 5.5);
        original.insert(-7, 0.25);
        let bytes = se_map(&original);
        let mut restored: HashMap<i64, f64> = HashMap::new();
        de_map_and_insert(&bytes, &mut restored).unwrap();
        assert_eq!(restored.len(), original.len());
        for (k, v) in &original {
            assert!(
                (restored.get(k).copied().unwrap() - v).abs() < 1e-12,
                "key {k}"
            );
        }
    }

    /// Merge semantics: `de_map_and_insert` must ADD values for shared keys, not
    /// overwrite. This is the property that makes partial->final merging correct
    /// (keeping only the local winner would not be associative across partitions).
    #[test]
    fn test_de_map_and_insert_accumulates_not_overwrites() {
        let mut m: HashMap<i64, f64> = HashMap::new();
        m.insert(1, 3.0);
        let mut partial: HashMap<i64, f64> = HashMap::new();
        partial.insert(1, 2.0); // shared key
        partial.insert(5, 1.0); // new key
        let bytes = se_map(&partial);
        de_map_and_insert(&bytes, &mut m).unwrap();
        assert!(
            (m.get(&1).copied().unwrap() - 5.0).abs() < 1e-12,
            "shared key must sum"
        );
        assert!(
            (m.get(&5).copied().unwrap() - 1.0).abs() < 1e-12,
            "new key must appear"
        );
    }

    /// A corrupt/truncated state blob must surface as a query error, not a panic.
    #[test]
    fn test_de_map_and_insert_malformed_errors() {
        let mut m: HashMap<i64, f64> = HashMap::new();
        // Header claims 1 entry but there is no payload -> length mismatch.
        assert!(de_map_and_insert(&1u32.to_le_bytes(), &mut m).is_err());
        // Empty buffer -> length < 4.
        assert!(de_map_and_insert(&[], &mut m).is_err());
    }

    /// Direct partial->final merge: serialize b's state and fold it into a. The
    /// correct result is the argmax over the UNION; a no-op `merge_batch` yields
    /// null, and keeping only local winners yields a different key.
    #[test]
    fn test_accumulator_merge_unions_partial_states() {
        let mut a = MostCommonByAccumulator::new();
        let mut b = MostCommonByAccumulator::new();
        // a: {1->3, 2->2}  local winner 1
        a.update_batch(&[
            Arc::new(Int64Array::from(vec![1i64, 2])) as ArrayRef,
            Arc::new(Float64Array::from(vec![3.0f64, 2.0])) as ArrayRef,
        ])
        .unwrap();
        // b: {2->4}  local winner 2
        b.update_batch(&[
            Arc::new(Int64Array::from(vec![2i64])) as ArrayRef,
            Arc::new(Float64Array::from(vec![4.0f64])) as ArrayRef,
        ])
        .unwrap();

        // Serialize b's state and feed it back through merge_batch (final path).
        let b_state = b.state().unwrap();
        let bytes = match &b_state[0] {
            ScalarValue::Binary(Some(x)) => x.clone(),
            other => panic!("expected Binary state, got {other:?}"),
        };
        a.merge_batch(&[Arc::new(BinaryArray::from(vec![Some(bytes.as_slice())])) as ArrayRef])
            .unwrap();

        // merged: {1->3, 2->6} -> winner 2
        assert_eq!(a.evaluate().unwrap(), ScalarValue::Int64(Some(2)));
    }

    /// Single-group SQL aggregate with a clear winner.
    #[tokio::test]
    async fn test_most_common_by_clear_winner() -> Result<()> {
        // sums: 1 -> 6.0, 2 -> 3.0 -> winner 1
        let df = dataframe!(
            "g" => vec![0i64, 0, 0, 0],
            "a" => vec![1i64, 2, 1, 2],
            "b" => vec![5.0f64, 1.0, 1.0, 2.0],
        )?;
        let out = df
            .aggregate(
                vec![col("g")],
                vec![most_common_by(col("a"), col("b")).alias("m")],
            )?
            .collect()
            .await?;
        let m = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.value(0), 1);
        Ok(())
    }

    /// Tie at the SQL level -> minimal label.
    #[tokio::test]
    async fn test_most_common_by_tie_picks_min_label() -> Result<()> {
        let df = dataframe!(
            "a" => vec![5i64, 2],
            "b" => vec![5.0f64, 5.0],
        )?;
        let out = df
            .aggregate(vec![], vec![most_common_by(col("a"), col("b")).alias("m")])?
            .collect()
            .await?;
        let m = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(m.value(0), 2);
        Ok(())
    }

    /// GROUP BY: each group resolves to its own argmax independently.
    #[tokio::test]
    async fn test_most_common_by_grouped() -> Result<()> {
        let df = dataframe!(
            "g" => vec![0i64, 0, 0, 1, 1, 1],
            "a" => vec![1i64, 2, 1, 10, 20, 20],
            "b" => vec![1.0f64, 3.0, 1.0, 5.0, 2.0, 2.0],
        )?;
        let out = df
            .aggregate(
                vec![col("g")],
                vec![most_common_by(col("a"), col("b")).alias("m")],
            )?
            .sort(vec![col("g").sort(true, true)])?
            .collect()
            .await?;
        let g = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let m = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.value(0), 0);
        assert_eq!(g.value(1), 1);
        // g=0: 1->2.0, 2->3.0 -> 2
        assert_eq!(m.value(0), 2);
        // g=1: 10->5.0, 20->4.0 -> 10
        assert_eq!(m.value(1), 10);
        Ok(())
    }

    /// Multi-partition: force the partial->final path via two `MemTable`
    /// partitions and `target_partitions(2)`. Each partition favors a different
    /// label; only a correct merge (state() + merge_batch) yields the global
    /// argmax. A missing/wrong `state_fields` errors the query; a no-op
    /// `merge_batch` yields null; keeping only one partition yields key 1 not 2.
    #[tokio::test]
    async fn test_most_common_by_multi_partition_merge() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let mk = |labels: Vec<i64>, weights: Vec<f64>| {
            let n = labels.len();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        std::iter::repeat(0i64).take(n),
                    )) as ArrayRef,
                    Arc::new(Int64Array::from(labels)) as ArrayRef,
                    Arc::new(Float64Array::from(weights)) as ArrayRef,
                ],
            )
            .unwrap()
        };
        // P1: 100 edges label 1 -> {1: 100}; P2: 150 edges label 2 -> {2: 150}.
        let p1 = mk(vec![1i64; 100], vec![1.0f64; 100]);
        let p2 = mk(vec![2i64; 150], vec![1.0f64; 150]);

        let table = MemTable::try_new(schema, vec![vec![p1], vec![p2]]).unwrap();
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        ctx.register_table("t", Arc::new(table)).unwrap();
        let out = ctx
            .table("t")
            .await
            .unwrap()
            .aggregate(
                vec![col("g")],
                vec![most_common_by(col("a"), col("b")).alias("m")],
            )
            .unwrap()
            .collect()
            .await
            .unwrap();
        let m = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(m.len(), 1);
        // merged {1:100, 2:150} -> 2
        assert_eq!(m.value(0), 2);
    }

    /// `return_type` must accept exactly (Int64, Float64) and reject the rest.
    #[test]
    fn test_most_common_by_return_type_validation() {
        let udf = MostCommonBy::new();
        assert_eq!(
            udf.return_type(&[DataType::Int64, DataType::Float64])
                .unwrap(),
            DataType::Int64
        );
        assert!(udf.return_type(&[DataType::Int64]).is_err());
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Int64])
                .is_err()
        );
        assert!(
            udf.return_type(&[DataType::Float64, DataType::Float64])
                .is_err()
        );
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Float64, DataType::Int64])
                .is_err()
        );
    }
}
