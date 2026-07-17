use datafusion::arrow::array::{Array, ArrayRef, Int64Array, ListArray};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use std::sync::Arc;

/// Per-vertex update rule of the distributed k-core decomposition.
///
/// Given a vertex's neighbours' current core estimates and the vertex's own
/// current core, returns the largest value `l` such that at least `l`
/// neighbours have a core estimate `>= l`. The result is bounded above by
/// `current_core`, so core values are monotonically non-increasing. An empty
/// neighbour list yields `0`.
///
/// This is the hot-path implementation used by the UDF below: the histogram
/// is written into a caller-provided buffer (reused across rows) and the
/// neighbour values are pulled from a lazy iterator, so evaluating a whole
/// batch performs no per-row heap allocation. `num_neighbors` is only an
/// upper bound on the real neighbour count (e.g. the list slot count, nulls
/// included); a looser bound merely widens the top of the histogram with zero
/// buckets and cannot change the result.
///
/// Based on: Mandal, Aritra, and Mohammad Al Hasan. "A distributed k-core
/// decomposition algorithm on spark." 2017 IEEE International Conference on
/// Big Data (Big Data). IEEE, 2017.
fn kcore_merge_into(
    counts: &mut Vec<i64>,
    num_neighbors: usize,
    current_core: i64,
    neighbors: impl Iterator<Item = i64>,
) -> i64 {
    // The new core can never exceed the number of neighbours (nor
    // current_core), so the counts table only needs to be that large. This
    // bounds the allocation independently of `current_core` (seeded from the
    // degree), defending against a pathological value such as `i64::MAX`.
    let cap = (current_core.max(0) as usize).min(num_neighbors);
    counts.clear();
    counts.resize(cap + 1, 0);
    for el in neighbors {
        // Anything above `cap` lands in the top bucket; negatives clamp to 0.
        let bucket = (el.max(0) as usize).min(cap);
        counts[bucket] += 1;
    }
    let mut current_weight = 0i64;
    for i in (1..=cap).rev() {
        current_weight += counts[i];
        if (i as i64) <= current_weight {
            return i as i64;
        }
    }
    0
}

fn list_int64_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Int64, true)))
}

/// DataFusion scalar UDF implementing the k-core update rule
/// ([`kcore_merge_into`]) over `(List<Int64>, Int64) -> Int64`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct KCoreMerge {
    signature: Signature,
}

impl KCoreMerge {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![list_int64_type(), DataType::Int64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for KCoreMerge {
    fn name(&self) -> &str {
        "kcore_merge"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::List(f), DataType::Int64] if f.data_type() == &DataType::Int64 => {
                Ok(DataType::Int64)
            }
            _ => Err(DataFusionError::Plan(format!(
                "kcore_merge expects (List<Int64>, Int64), got: {arg_types:?}"
            ))),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "kcore_merge expects exactly two arguments, got: {}",
                arrays.len()
            )));
        }
        let list = downcast_list(&arrays[0], "first")?;
        let cores = downcast_int64(&arrays[1], "second")?;

        let values = list.values();
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "kcore_merge list elements must be Int64, got: {:?}",
                    list.value_type()
                ))
            })?;

        let offsets = list.value_offsets();
        let len = arrays
            .iter()
            .map(|arr| arr.len())
            .max()
            .unwrap_or(0)
            .max(args.number_rows);

        // Reuse a single histogram buffer across rows so the hot path performs
        // no per-row heap allocation; the neighbour values are read lazily
        // straight off the Arrow array (skipping nulls) instead of being
        // collected into a fresh `Vec` per row.
        let mut counts: Vec<i64> = Vec::new();
        let result: Int64Array = (0..len)
            .map(|i| {
                let core_idx = i % cores.len();
                if cores.is_null(core_idx) {
                    return Some(0i64);
                }
                let current_core = cores.value(core_idx);
                let row = i % list.len();
                if list.is_null(row) {
                    // No message this iteration: keep the current core.
                    return Some(current_core);
                }
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                let neighbors = (start..end)
                    .filter(|&j| !values.is_null(j))
                    .map(|j| values.value(j));
                Some(kcore_merge_into(
                    &mut counts,
                    end - start,
                    current_core,
                    neighbors,
                ))
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

fn downcast_list<'a>(array: &'a ArrayRef, label: &str) -> Result<&'a ListArray> {
    array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "kcore_merge {label} argument must be a list of Int64, got: {:?}",
            array.data_type()
        ))
    })
}

fn downcast_int64<'a>(array: &'a ArrayRef, label: &str) -> Result<&'a Int64Array> {
    array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "kcore_merge {label} argument must be Int64, got: {:?}",
            array.data_type()
        ))
    })
}

/// Builds an [`Expr`] that applies `kcore_merge(neighbors, current_core)`.
pub(crate) fn kcore_merge_expr(neighbors: Expr, current_core: Expr) -> Expr {
    ScalarUDF::from(KCoreMerge::new()).call(vec![neighbors, current_core])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    /// Thin slice-based wrapper around [`kcore_merge_into`] for the pure-function
    /// tests below: hides the buffer plumbing behind the natural `(&[i64], i64)`
    /// signature.
    fn kcore_merge(neighbor_cores: &[i64], current_core: i64) -> i64 {
        let mut counts = Vec::new();
        kcore_merge_into(
            &mut counts,
            neighbor_cores.len(),
            current_core,
            neighbor_cores.iter().copied(),
        )
    }

    #[test]
    fn test_merge_empty_returns_zero() {
        assert_eq!(kcore_merge(&[], 5), 0);
        assert_eq!(kcore_merge(&[], 0), 0);
    }

    #[test]
    fn test_merge_two_connected() {
        // Single neighbour with core 1, current core 1 -> 1.
        assert_eq!(kcore_merge(&[1], 1), 1);
    }

    #[test]
    fn test_merge_triangle() {
        // Two neighbours both with core 2, current core 2 -> 2.
        assert_eq!(kcore_merge(&[2, 2], 2), 2);
    }

    #[test]
    fn test_merge_star_center() {
        // Center with degree 3: three leaves with core 1.
        // largest l with >= l neighbours >= l: l=1 (3 neighbours >= 1).
        assert_eq!(kcore_merge(&[1, 1, 1], 3), 1);
    }

    #[test]
    fn test_merge_capped_values() {
        // Neighbour estimates above the current core are capped into the
        // top bucket: [5, 5, 5], current 3 -> 3 neighbours >= 3 -> l=3.
        assert_eq!(kcore_merge(&[5, 5, 5], 3), 3);
    }

    #[test]
    fn test_merge_decreases_core() {
        // current core 4, neighbours [2, 2, 1]:
        // i=4: weight 0; i=3: weight 0; i=2: weight 2 -> 2<=2 -> 2.
        assert_eq!(kcore_merge(&[2, 2, 1], 4), 2);
    }

    #[test]
    fn test_merge_ignores_negatives() {
        // Negative estimates are clamped to bucket 0 and cannot satisfy the
        // condition for i >= 1.
        assert_eq!(kcore_merge(&[-1, -1, 3], 3), 1);
    }

    #[test]
    fn test_merge_pathological_current_core_does_not_oom() {
        // A corrupted / huge `current_core` must not allocate a buffer of that
        // size. The result is bounded by the neighbour count, so a tiny
        // allocation suffices and the answer is still correct.
        assert_eq!(kcore_merge(&[3, 3, 3], i64::MAX), 3);
        assert_eq!(kcore_merge(&[i64::MAX, i64::MAX], i64::MAX), 2);
        assert_eq!(kcore_merge(&[], i64::MAX), 0);
    }

    #[tokio::test]
    async fn test_merge_udf_dataframe() -> Result<()> {
        use datafusion::arrow::array::{Int64Builder, ListBuilder, RecordBatch};
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        // Build a List<Int64> column: row 0 = [2,2], row 1 = [1,1,1].
        let mut list_builder = ListBuilder::new(Int64Builder::new());
        list_builder.values().append_value(2);
        list_builder.values().append_value(2);
        list_builder.append(true);
        list_builder.values().append_value(1);
        list_builder.values().append_value(1);
        list_builder.values().append_value(1);
        list_builder.append(true);
        let list_arr = list_builder.finish();

        let core_arr = Int64Array::from(vec![2, 3]);

        let schema = Arc::new(Schema::new(vec![
            Field::new("neighbors", list_int64_type(), false),
            Field::new("core", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(list_arr) as ArrayRef,
                Arc::new(core_arr) as ArrayRef,
            ],
        )?;

        let ctx = SessionContext::new();
        let df = ctx.read_batch(batch)?;
        let result = df
            .select(vec![
                kcore_merge_expr(col("neighbors"), col("core")).alias("k"),
            ])?
            .collect()
            .await?;

        let arr = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(arr.value(0), 2);
        assert_eq!(arr.value(1), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_merge_udf_return_type_validation() {
        let udf = KCoreMerge::new();
        assert_eq!(
            udf.return_type(&[list_int64_type(), DataType::Int64])
                .unwrap(),
            DataType::Int64
        );
        // Wrong arity / wrong type must error.
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Int64])
                .is_err()
        );
        assert!(udf.return_type(&[list_int64_type()]).is_err());
    }
}
