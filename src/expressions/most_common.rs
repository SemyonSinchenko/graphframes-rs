//! Mode-with-minimal-label scalar UDF for classical label propagation.
//!
//! LDBC CDLP semantics: the label with the largest total weight among a
//! vertex's neighbours wins; ties break toward the smallest label. With the
//! unit weights used by classical_lp this is exactly the mode.

use crate::expressions::common::downcast_int64;
use datafusion::arrow::array::{Array, ArrayRef, Int64Array, ListArray};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use std::sync::Arc;

/// Most frequent label, ties broken by the smallest label. Returns `None` for
/// an empty multiset;
fn mode_min_label(scratch: &mut Vec<i64>) -> Option<i64> {
    if scratch.is_empty() {
        return None;
    }
    scratch.sort_unstable();
    let mut best: Option<i64> = None;
    let mut best_count: i64 = 0;
    let mut i = 0;
    while i < scratch.len() {
        let label = scratch[i];
        let mut count: i64 = 0;
        while i < scratch.len() && scratch[i] == label {
            count += 1;
            i += 1;
        }
        if count > best_count || (count == best_count && best.map_or(true, |b| label < b)) {
            best = Some(label);
            best_count = count;
        }
    }
    best
}

fn list_int64_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Int64, true)))
}

/// Scalar UDF `most_common(List<Int64>) -> Int64` (nullable).
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct MostCommon {
    signature: Signature,
}

impl MostCommon {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::exact(vec![list_int64_type()], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for MostCommon {
    fn name(&self) -> &str {
        "most_common"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::List(f)] if f.data_type() == &DataType::Int64 => Ok(DataType::Int64),
            _ => Err(DataFusionError::Plan(format!(
                "most_common expects (List<Int64>), got: {arg_types:?}"
            ))),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 1 {
            return Err(DataFusionError::Plan(format!(
                "most_common expects exactly one argument, got: {}",
                arrays.len()
            )));
        }
        let list = arrays[0]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                DataFusionError::Plan(format!(
                    "most_common argument must be List, got: {:?}",
                    arrays[0].data_type()
                ))
            })?;
        let values = downcast_int64(list.values(), "most_common", "list elements")?;
        let offsets = list.offsets();
        let len = args.number_rows.max(list.len());

        // Reuse the sort scratch across rows: no per-row allocation.
        // One scratch buffer reused across all rows: no per-row allocation.
        let mut scratch: Vec<i64> = Vec::new();
        let result: Int64Array = (0..len)
            .map(|i| {
                let row = i % list.len();
                if list.is_null(row) {
                    return None;
                }
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                scratch.clear();
                for j in start..end {
                    if !values.is_null(j) {
                        scratch.push(values.value(j));
                    }
                }
                mode_min_label(&mut scratch)
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Builds an [`Expr`] that applies `most_common(labels)`.
pub(crate) fn most_common_expr(labels: Expr) -> Expr {
    ScalarUDF::from(MostCommon::new()).call(vec![labels])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    fn mode(labels: &[i64]) -> Option<i64> {
        let mut scratch = labels.to_vec();
        mode_min_label(&mut scratch)
    }

    /// LDBC semantics: most frequent label wins.
    #[test]
    fn test_mode_picks_most_frequent() {
        assert_eq!(mode(&[5, 5, 5, 3, 3]), Some(5));
        assert_eq!(mode(&[7]), Some(7));
        assert_eq!(mode(&[1, 2, 3, 4, 4, 4, 2, 2]), Some(2));
    }

    /// Ties break toward the smallest label.
    #[test]
    fn test_mode_tie_picks_min_label() {
        assert_eq!(mode(&[5, 5, 2, 2]), Some(2));
        assert_eq!(mode(&[10, 10, 1, 1]), Some(1));
        assert_eq!(mode(&[3, 3, 3, 7, 7, 7]), Some(3));
    }

    /// Empty multiset -> None (SQL NULL), matching the old accumulator.
    #[test]
    fn test_mode_empty_returns_none() {
        assert_eq!(mode(&[]), None);
    }

    /// Nested scalar-over-aggregate wiring (the `classical_lp.rs` shape):
    /// most_common(array_agg(label)) per group.
    #[tokio::test]
    async fn test_nested_most_common_aggregate() -> Result<()> {
        use datafusion::arrow::array::Int64Array as I64A;
        use datafusion::functions_aggregate::array_agg::array_agg;
        let df = dataframe!(
            "g" => vec![0i64, 0, 0, 1, 1, 1],
            "a" => vec![5i64, 5, 3, 10, 20, 20],
        )?;

        let out = df
            .aggregate(
                vec![col("g")],
                vec![most_common_expr(array_agg(col("a"))).alias("c")],
            )?
            .collect()
            .await?;
        let mut pairs: Vec<(i64, i64)> = Vec::new();
        for b in &out {
            let g = b.column(0).as_any().downcast_ref::<I64A>().unwrap();
            let c = b.column(1).as_any().downcast_ref::<I64A>().unwrap();
            for r in 0..g.len() {
                pairs.push((g.value(r), c.value(r)));
            }
        }
        pairs.sort_unstable();
        // g=0: {5:2, 3:1} -> 5; g=1: {10:1, 20:2} -> 20
        assert_eq!(pairs, vec![(0, 5), (1, 20)]);
        Ok(())
    }
}
