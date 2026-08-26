//! Scalar UDFs over `f32` vectors: `l2_norm`, `l2_distance`,
//! `cosine_distance`.
//!
//! This module holds the DataFusion wrappers only: SIMD-logic free; the
//! kernels live in [`crate::ml::linalg`]. Contract:
//! vector rows are non-null; the two-argument UDFs require both rows
//! to have the same length.

use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::plan_err;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use crate::expressions::common::as_f32_list_like;
use crate::ml::{cosine_distance, l2_distance, l2_norm};

/// Both arguments must be same-sized `f32` vectors.
fn validate_vector_args(arg_types: &[DataType], arity: usize, fname: &str) -> Result<()> {
    if arg_types.len() != arity {
        return plan_err!(
            "{fname} expects {arity} argument(s), got {}",
            arg_types.len()
        );
    }
    for (i, t) in arg_types.iter().enumerate() {
        let ok = match t {
            DataType::FixedSizeList(f, _) => f.data_type() == &DataType::Float32,
            DataType::List(f) => f.data_type() == &DataType::Float32,
            _ => false,
        };
        if !ok {
            return plan_err!(
                "{fname} argument {i} must be FixedSizeList<Float32> or List<Float32>, got {t:?}"
            );
        }
    }
    Ok(())
}

/// Scalar UDF `l2_norm(v) -> Float64`: true L2 norm of a vector.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct L2NormUDF {
    signature: Signature,
}

impl L2NormUDF {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl Default for L2NormUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for L2NormUDF {
    fn name(&self) -> &str {
        "l2_norm"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        validate_vector_args(arg_types, 1, "l2_norm")?;
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let v = as_f32_list_like(&arrays[0], "l2_norm", "first")?;
        let result: Float64Array = (0..v.len())
            .map(|i| {
                let row = v.value(i);
                Some(l2_norm(row, row.len()) as f64)
            })
            .collect();
        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Scalar UDF `l2_distance(v1, v2) -> Float64`: true L2 distance between two
/// vectors.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct L2DistanceUDF {
    signature: Signature,
}

impl L2DistanceUDF {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for L2DistanceUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for L2DistanceUDF {
    fn name(&self) -> &str {
        "l2_distance"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        validate_vector_args(arg_types, 2, "l2_distance")?;
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let v1 = as_f32_list_like(&arrays[0], "l2_distance", "first")?;
        let v2 = as_f32_list_like(&arrays[1], "l2_distance", "second")?;
        let len = v1.len().max(v2.len());
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let a = v1.value(i % v1.len());
            let b = v2.value(i % v2.len());
            if a.len() != b.len() {
                return plan_err!(
                    "l2_distance vectors must have the same length, got {} and {}",
                    a.len(),
                    b.len()
                );
            }
            values.push(Some(l2_distance(a, b, a.len()).sqrt() as f64));
        }
        let result: Float64Array = values.into_iter().collect();
        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Scalar UDF `cosine_distance(v1, v2) -> Float64`: cosine distance between
/// two vectors (`1 - cosine similarity`; zero-norm rows score `0.0`);
///
/// scikit-learn semantics
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct CosineDistanceUDF {
    signature: Signature,
}

impl CosineDistanceUDF {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for CosineDistanceUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarUDFImpl for CosineDistanceUDF {
    fn name(&self) -> &str {
        "cosine_distance"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        validate_vector_args(arg_types, 2, "cosine_distance")?;
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let v1 = as_f32_list_like(&arrays[0], "cosine_distance", "first")?;
        let v2 = as_f32_list_like(&arrays[1], "cosine_distance", "second")?;
        let len = v1.len().max(v2.len());
        let mut values = Vec::with_capacity(len);
        for i in 0..len {
            let a = v1.value(i % v1.len());
            let b = v2.value(i % v2.len());
            if a.len() != b.len() {
                return plan_err!(
                    "cosine_distance vectors must have the same length, got {} and {}",
                    a.len(),
                    b.len()
                );
            }
            values.push(Some(cosine_distance(
                a,
                b,
                a.len(),
                l2_norm(a, a.len()),
                l2_norm(b, b.len()),
            ) as f64));
        }
        let result: Float64Array = values.into_iter().collect();
        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Builds an [`Expr`] computing the L2 norm of `v`.
pub(crate) fn l2_norm_expr(v: Expr) -> Expr {
    ScalarUDF::from(L2NormUDF::new()).call(vec![v])
}

/// Builds an [`Expr`] computing the true L2 distance between `v1` and `v2`.
pub(crate) fn l2_distance_expr(v1: Expr, v2: Expr) -> Expr {
    ScalarUDF::from(L2DistanceUDF::new()).call(vec![v1, v2])
}

/// Builds an [`Expr`] computing the cosine distance between `v1` and `v2`.
pub(crate) fn cosine_distance_expr(v1: Expr, v2: Expr) -> Expr {
    ScalarUDF::from(CosineDistanceUDF::new()).call(vec![v1, v2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
    use datafusion::arrow::datatypes::{Field, Schema};
    use datafusion::common::Result;
    use datafusion::prelude::{DataFrame, SessionContext, col, lit};

    // ---------------- UDFs ----------------

    fn two_col_table(rows: &[[f32; 2]], rows2: &[[f32; 2]]) -> Result<DataFrame> {
        let mk = |rows: &[[f32; 2]], name: &str| -> Result<(Arc<Schema>, ArrayRef)> {
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            let fsl = FixedSizeListArray::try_new(
                Arc::new(Field::new("el", DataType::Float32, false)),
                2,
                Arc::new(Float32Array::from(flat)),
                None,
            )?;
            let schema = Schema::new(vec![Field::new(
                name,
                DataType::FixedSizeList(Arc::new(Field::new("el", DataType::Float32, false)), 2),
                false,
            )]);
            Ok((Arc::new(schema), Arc::new(fsl) as ArrayRef))
        };
        let (s1, a1) = mk(rows, "v1")?;
        let (s2, a2) = mk(rows2, "v2")?;
        let schema = Schema::new(vec![s1.field(0).clone(), s2.field(0).clone()]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![a1, a2])?;
        let ctx = SessionContext::new();
        Ok(ctx.read_batch(batch)?)
    }

    #[tokio::test]
    async fn udf_l2_norm_returns_true_norm() -> Result<()> {
        let df = two_col_table(&[[3.0, 4.0], [0.0, 0.0]], &[[0.0, 0.0], [0.0, 0.0]])?;
        let out = df
            .clone()
            .select(vec![l2_norm_expr(col("v1")).alias("n")])?
            .collect()
            .await?;
        let n = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((n.value(0) - 5.0).abs() < 1e-5, "{}", n.value(0));
        assert_eq!(n.value(1), 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn udf_l2_distance_returns_true_distance() -> Result<()> {
        let df = two_col_table(&[[0.0, 0.0], [1.0, 1.0]], &[[3.0, 4.0], [1.0, 1.0]])?;
        let out = df
            .clone()
            .select(vec![l2_distance_expr(col("v1"), col("v2")).alias("d")])?
            .collect()
            .await?;
        let d = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((d.value(0) - 5.0).abs() < 1e-5, "{}", d.value(0)); // 3-4-5
        assert!((d.value(1) - 0.0).abs() < 1e-6, "{}", d.value(1));
        Ok(())
    }

    #[tokio::test]
    async fn udf_cosine_distance_canonical_values() -> Result<()> {
        let df = two_col_table(
            &[[1.0, 0.0], [1.0, 2.0], [0.0, 0.0]],
            &[[1.0, 0.0], [2.0, 4.0], [3.0, 4.0]],
        )?;
        let out = df
            .clone()
            .select(vec![cosine_distance_expr(col("v1"), col("v2")).alias("d")])?
            .collect()
            .await?;
        let d = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!((d.value(0) - 0.0).abs() < 1e-6, "{}", d.value(0)); // parallel
        assert!((d.value(1) - 0.0).abs() < 1e-6, "{}", d.value(1)); // collinear, scaled
        assert!((d.value(2) - 0.0).abs() < 1e-6, "{}", d.value(2)); // zero vector policy
        Ok(())
    }

    #[tokio::test]
    async fn udf_l2_distance_broadcasts_literal_vector() -> Result<()> {
        // One column plus a literal vector: the literal broadcasts per row.
        let df = two_col_table(&[[0.0, 0.0], [6.0, 8.0]], &[[0.0, 0.0], [0.0, 0.0]])?;
        let lit_vec =
            datafusion::scalar::ScalarValue::FixedSizeList(Arc::new(FixedSizeListArray::try_new(
                Arc::new(Field::new("el", DataType::Float32, false)),
                2,
                Arc::new(Float32Array::from(vec![0.0f32, 0.0])),
                None,
            )?));
        let out = df
            .clone()
            .select(vec![l2_distance_expr(col("v1"), lit(lit_vec)).alias("d")])?
            .collect()
            .await?;
        let d = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(d.len(), 2);
        assert!((d.value(0) - 0.0).abs() < 1e-6, "{}", d.value(0));
        assert!((d.value(1) - 10.0).abs() < 1e-5, "{}", d.value(1)); // 6-8-10
        Ok(())
    }

    #[tokio::test]
    async fn udf_l2_distance_rejects_mismatched_lengths() -> Result<()> {
        // v1: one row of length 3 (List); v2: one row of length 2 (List).
        use datafusion::arrow::array::ListArray;
        use datafusion::arrow::datatypes::Float32Type;
        let v1 = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(vec![
            Some(1.0f32),
            Some(2.0),
            Some(3.0),
        ])]);
        let v2 = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(vec![
            Some(1.0f32),
            Some(2.0),
        ])]);
        let v1_field = Field::new("v1", v1.data_type().clone(), false);
        let v2_field = Field::new("v2", v2.data_type().clone(), false);
        let schema = Schema::new(vec![v1_field, v2_field]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(v1) as ArrayRef, Arc::new(v2) as ArrayRef],
        )?;
        let ctx = SessionContext::new();
        let result = ctx
            .read_batch(batch)?
            .select(vec![l2_distance_expr(col("v1"), col("v2")).alias("d")])?
            .collect()
            .await;
        assert!(
            result.is_err(),
            "mismatched vector lengths must surface as an error"
        );
        Ok(())
    }

    #[test]
    fn builders_reference_udf_names() {
        assert!(format!("{}", l2_norm_expr(col("v"))).contains("l2_norm"));
        assert!(format!("{}", l2_distance_expr(col("a"), col("b"))).contains("l2_distance"));
        assert!(
            format!("{}", cosine_distance_expr(col("a"), col("b"))).contains("cosine_distance")
        );
    }
}
