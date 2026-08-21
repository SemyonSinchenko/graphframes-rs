//! Final cluster assignment (`k_means_assign`) and nearest-center cost
//! (`k_means_cost`) scalar UDFs for K-Means.

use datafusion::arrow::array::{ArrayRef, Float64Array, Int32Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::expressions::common::as_f32_list_like;
use crate::ml::{DistanceMetric, nearest_center, nearest_centers};

/// Shared construction of the 1-argument signature:
/// `features` is either `FixedSizeList<Float32>(d)` or `List<Float32>`.
fn features_signature(d: usize) -> Signature {
    let field = || {
        Arc::new(datafusion::arrow::datatypes::Field::new(
            "el",
            DataType::Float32,
            false,
        ))
    };
    Signature::uniform(
        1,
        vec![
            DataType::FixedSizeList(field(), d as i32),
            DataType::List(field()),
        ],
        Volatility::Immutable,
    )
}

/// `Eq`/`Hash` consistent over the float centers: comparison and hashing are
/// both bit-based (`f32::to_bits`), so equal instances always hash equal.
/// `signature` is a pure function of `d` and therefore not compared.
macro_rules! impl_kmeans_udf_eq_hash {
    ($t:ty) => {
        impl PartialEq for $t {
            fn eq(&self, other: &Self) -> bool {
                self.k == other.k
                    && self.d == other.d
                    && self.metric == other.metric
                    && self.centers.len() == other.centers.len()
                    && self
                        .centers
                        .iter()
                        .zip(&other.centers)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
            }
        }

        impl Eq for $t {}

        impl Hash for $t {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.k.hash(state);
                self.d.hash(state);
                self.metric.hash(state);
                self.centers.len().hash(state);
                for c in &self.centers {
                    c.to_bits().hash(state);
                }
            }
        }
    };
}

/// Scalar UDF `k_means_assign(features) -> Int32`: index of the nearest
/// center per row (ties break to the first center).
#[derive(Debug)]
pub(crate) struct KMeansAssign {
    signature: Signature,
    k: usize,
    d: usize,
    centers: Vec<f32>,
    metric: DistanceMetric,
}

impl KMeansAssign {
    pub(crate) fn new(k: usize, d: usize, centers: Vec<f32>, metric: DistanceMetric) -> Self {
        Self {
            signature: features_signature(d),
            k,
            d,
            centers,
            metric,
        }
    }
}

impl_kmeans_udf_eq_hash!(KMeansAssign);

impl ScalarUDFImpl for KMeansAssign {
    fn name(&self) -> &str {
        "k_means_assign"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let v = as_f32_list_like(&arrays[0], "k_means_assign", "first")?;
        let result: Int32Array = (0..v.len())
            .map(|i| {
                Some(nearest_center(v.value(i), &self.centers, self.k, self.d, self.metric) as i32)
            })
            .collect();
        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Scalar UDF `k_means_cost(features) -> Float64`: distance to the nearest
/// center per row (squared L2 for the L2 metric — the D^2 cost used by the
/// k-means|| sampling; cosine distance for the Cosine metric).
#[derive(Debug)]
pub(crate) struct KMeansCost {
    signature: Signature,
    k: usize,
    d: usize,
    centers: Vec<f32>,
    metric: DistanceMetric,
}

impl KMeansCost {
    pub(crate) fn new(k: usize, d: usize, centers: Vec<f32>, metric: DistanceMetric) -> Self {
        Self {
            signature: features_signature(d),
            k,
            d,
            centers,
            metric,
        }
    }
}

impl_kmeans_udf_eq_hash!(KMeansCost);

impl ScalarUDFImpl for KMeansCost {
    fn name(&self) -> &str {
        "k_means_cost"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let v = as_f32_list_like(&arrays[0], "k_means_cost", "first")?;
        let result: Float64Array = (0..v.len())
            .map(|i| {
                let (_, dist) =
                    nearest_centers(v.value(i), &self.centers, self.k, self.d, self.metric);
                Some(dist as f64)
            })
            .collect();
        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

/// Builds an [`Expr`] assigning each row to its nearest center.
pub(crate) fn kmeans_assign_expr(
    features: Expr,
    k: usize,
    d: usize,
    centers: Vec<f32>,
    metric: DistanceMetric,
) -> Expr {
    ScalarUDF::from(KMeansAssign::new(k, d, centers, metric)).call(vec![features])
}

/// Builds an [`Expr`] computing the distance from each row to its nearest
/// center.
pub(crate) fn kmeans_cost_expr(
    features: Expr,
    k: usize,
    d: usize,
    centers: Vec<f32>,
    metric: DistanceMetric,
) -> Expr {
    ScalarUDF::from(KMeansCost::new(k, d, centers, metric)).call(vec![features])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Array, FixedSizeListArray, Float32Array, RecordBatch};
    use datafusion::arrow::datatypes::{Field, Schema};
    use datafusion::common::Result;
    use datafusion::prelude::*;

    fn centers_2x2() -> Vec<f32> {
        vec![10.0, 10.0, 0.0, 0.0]
    }

    fn features_table(rows: &[[f32; 2]]) -> Result<DataFrame> {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let schema = Schema::new(vec![Field::new(
            "feat",
            DataType::FixedSizeList(Arc::new(Field::new("el", DataType::Float32, false)), 2),
            false,
        )]);
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("el", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(flat)),
            None,
        )?;
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(fsl) as ArrayRef])?;
        let ctx = SessionContext::new();
        Ok(ctx.read_batch(batch)?)
    }

    #[tokio::test]
    async fn assign_picks_nearest_center_per_row() -> Result<()> {
        let df = features_table(&[[11.0, 11.0], [1.0, 1.0], [4.0, 4.0]])?;
        let out = df
            .clone()
            .select(vec![
                kmeans_assign_expr(col("feat"), 2, 2, centers_2x2(), DistanceMetric::L2)
                    .alias("cluster"),
            ])?
            .collect()
            .await?;
        let clusters = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(clusters.len(), 3);
        assert_eq!(clusters.value(0), 0);
        assert_eq!(clusters.value(1), 1);
        // (4,4): squared L2 to (10,10) = 72, to (0,0) = 32 -> cluster 1.
        assert_eq!(clusters.value(2), 1);
        Ok(())
    }

    #[tokio::test]
    async fn cost_returns_squared_l2_to_nearest_center() -> Result<()> {
        let df = features_table(&[[11.0, 11.0], [1.0, 1.0]])?;
        let out = df
            .clone()
            .select(vec![
                kmeans_cost_expr(col("feat"), 2, 2, centers_2x2(), DistanceMetric::L2)
                    .alias("cost"),
            ])?
            .collect()
            .await?;
        let costs = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(costs.len(), 2);
        assert!((costs.value(0) - 2.0).abs() < 1e-6, "{}", costs.value(0));
        assert!((costs.value(1) - 2.0).abs() < 1e-6, "{}", costs.value(1));
        Ok(())
    }

    #[test]
    fn builders_reference_the_udf_names() {
        let a = kmeans_assign_expr(col("feat"), 2, 2, centers_2x2(), DistanceMetric::L2);
        assert!(format!("{a}").contains("k_means_assign"));
        let c = kmeans_cost_expr(col("feat"), 2, 2, centers_2x2(), DistanceMetric::L2);
        assert!(format!("{c}").contains("k_means_cost"));
    }

    #[test]
    fn eq_and_hash_agree_on_signed_zero() {
        let a = KMeansAssign::new(1, 2, vec![0.0f32, 0.0], DistanceMetric::L2);
        let b = KMeansAssign::new(1, 2, vec![-0.0f32, 0.0], DistanceMetric::L2);
        if a == b {
            let ha = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                a.hash(&mut h);
                h.finish()
            };
            let hb = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                b.hash(&mut h);
                h.finish()
            };
            assert_eq!(ha, hb, "equal objects must hash equal");
        }
    }

    #[test]
    fn eq_respects_centers_length() {
        let a = KMeansCost::new(1, 2, vec![1.0f32], DistanceMetric::L2);
        let b = KMeansCost::new(1, 2, vec![1.0f32, 2.0], DistanceMetric::L2);
        assert_ne!(a, b);
    }
}
