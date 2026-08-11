//! Per-vertex k-core update as a scalar UDF over the collected neighbour-core
//! list (`array_agg` result).

use crate::expressions::common::downcast_int32;
use datafusion::arrow::array::{Array, ArrayRef, Int32Array, ListArray};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use std::sync::Arc;

/// Counts are `u32`:
/// each bucket holds at most `num_neighbors` entries, bounded by the "degree <
/// i32::MAX" assumption shared with the rest of the codebase.
fn kcore_merge_into(
    counts: &mut Vec<u32>,
    num_neighbors: usize,
    neighbors: impl Iterator<Item = i32>,
) -> i32 {
    let cap = num_neighbors;
    counts.clear();
    counts.resize(cap + 1, 0);
    for el in neighbors {
        let bucket = (el.max(0) as usize).min(cap);
        counts[bucket] += 1;
    }
    let mut current_weight = 0u32;
    for i in (1..=cap).rev() {
        current_weight += counts[i];
        if (i as u32) <= current_weight {
            return i as i32;
        }
    }
    0
}

fn list_int32_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Int32, true)))
}

/// Scalar UDF `kcore_merge(List<Int32>) -> Int32`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct KCoreMerge {
    signature: Signature,
}

impl KCoreMerge {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::exact(vec![list_int32_type()], Volatility::Immutable),
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
            [DataType::List(f)] if f.data_type() == &DataType::Int32 => Ok(DataType::Int32),
            _ => Err(DataFusionError::Plan(format!(
                "kcore_merge expects (List<Int32>), got: {arg_types:?}"
            ))),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 1 {
            return Err(DataFusionError::Plan(format!(
                "kcore_merge expects exactly one argument, got: {}",
                arrays.len()
            )));
        }
        let list = arrays[0]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "kcore_merge argument must be List, got: {:?}",
                    arrays[0].data_type()
                ))
            })?;
        let values = downcast_int32(list.values(), "kcore_merge", "list elements")?;
        let offsets = list.offsets();
        let len = args.number_rows.max(list.len());

        let mut counts: Vec<u32> = Vec::new();
        let result: Int32Array = (0..len)
            .map(|i| {
                let row = i % list.len();
                if list.is_null(row) {
                    // No neighbours: nothing can support l >= 1.
                    return Some(0i32);
                }
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                let neighbors = (start..end)
                    .filter(|&j| !values.is_null(j))
                    .map(|j| values.value(j));
                Some(kcore_merge_into(&mut counts, end - start, neighbors))
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Builds an [`Expr`] that applies `kcore_merge(neighbors)`.
pub(crate) fn kcore_merge_expr(neighbors: Expr) -> Expr {
    ScalarUDF::from(KCoreMerge::new()).call(vec![neighbors])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    /// Thin slice wrapper over the counting-array reducer.
    fn reduce(neighbors: &[i32]) -> i32 {
        let mut counts = Vec::new();
        kcore_merge_into(&mut counts, neighbors.len(), neighbors.iter().copied())
    }

    /// Uncapped_A over a neighbour multiset (mirrors the old accumulator tests).
    #[test]
    fn test_kcore_merge_picks_uncapped_core() {
        // {3:3}: ge(3)=3 -> 3
        assert_eq!(reduce(&[3, 3, 3]), 3);
        // {2:2, 1:1}: ge(2)=2 -> 2
        assert_eq!(reduce(&[2, 2, 1]), 2);
        // {5:3}: only 3 neighbours, so uncapped capped at 3
        assert_eq!(reduce(&[5, 5, 5]), 3);
        // {1:1}: single neighbour -> 1
        assert_eq!(reduce(&[1]), 1);
        // {10:1, 1:1}: ge(1)=2, ge(2)=1 -> 1
        assert_eq!(reduce(&[10, 1]), 1);
        // neighbours' cores above the degree clamp to the top bucket
        assert_eq!(reduce(&[1000, 1000, 1000, 1000]), 4);
    }

    /// No neighbours -> uncapped_A = 0; all-zero neighbours -> 0.
    #[test]
    fn test_kcore_merge_empty_and_zero() {
        assert_eq!(reduce(&[]), 0);
        assert_eq!(reduce(&[0, 0, 0, 0, 0]), 0);
    }

    /// Negative neighbour cores clamp to bucket 0 and cannot support l >= 1.
    #[test]
    fn test_kcore_merge_negative_clamps() {
        assert_eq!(reduce(&[-5, -5]), 0);
        assert_eq!(reduce(&[-5, 3]), 1);
    }

    /// DataFusion accepts a scalar-over-aggregate expression directly in
    /// `DataFrame::aggregate` and splits it into the array_agg aggregate plus
    /// a kcore_merge projection on top — so k-core spills only the reduced
    /// O(|V|) result. Regression test for the `k_core.rs` wiring.
    #[tokio::test]
    async fn test_nested_scalar_over_aggregate() -> Result<()> {
        use datafusion::arrow::array::{Int32Array as I32A, Int64Array as I64A};
        use datafusion::functions_aggregate::array_agg::array_agg;
        let df = dataframe!(
            "g" => vec![0i64, 0, 1, 1, 1],
            "a" => vec![3i32, 3, 10, 20, 20],
        )?;

        let out = df
            .aggregate(
                vec![col("g")],
                vec![kcore_merge_expr(array_agg(col("a"))).alias("k")],
            )?
            .collect()
            .await?;
        let mut pairs: Vec<(i64, i32)> = Vec::new();
        for b in &out {
            let g = b.column(0).as_any().downcast_ref::<I64A>().unwrap();
            let k = b.column(1).as_any().downcast_ref::<I32A>().unwrap();
            for r in 0..g.len() {
                pairs.push((g.value(r), k.value(r)));
            }
        }
        pairs.sort_unstable();
        // g=0: {3:2} -> uncapped 2; g=1: {10:1,20:2} -> uncapped 3
        assert_eq!(pairs, vec![(0, 2), (1, 3)]);
        Ok(())
    }
}
