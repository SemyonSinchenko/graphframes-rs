use datafusion::error::Result;
use datafusion::logical_expr::function::AccumulatorArgs;
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Expr, Signature, Volatility,
};

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, FixedSizeListArray, Float64Array};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::scalar::ScalarValue;

use crate::expressions::common::{as_f32_list_like, as_f64_list_like};
use crate::ml::{DistanceMetric, nearest_centers};

#[derive(Debug)]
pub(crate) struct KMeansStepAccumulator {
    k: usize,
    d: usize,
    centers: Vec<f32>,
    state: Vec<f64>,
    metric: DistanceMetric,
}

impl KMeansStepAccumulator {
    pub(crate) fn new(k: usize, d: usize, centers: Vec<f32>, metric: DistanceMetric) -> Self {
        Self {
            k: k,
            d: d,
            centers: centers,
            state: vec![0.0; k * d + k],
            metric: metric,
        }
    }

    pub(crate) fn default(k: usize, d: usize, centers: Vec<f32>) -> Self {
        Self {
            k: k,
            d: d,
            centers: centers,
            state: vec![0.0; k * d + k],
            metric: DistanceMetric::L2,
        }
    }

    fn to_scalar(&self) -> ScalarValue {
        ScalarValue::FixedSizeList(Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("el", DataType::Float64, false)),
                (self.k * self.d + self.k) as i32,
                Arc::new(Float64Array::from(self.state.clone())),
                None,
            )
            .expect("valid fixed size list"),
        ))
    }
}

impl Accumulator for KMeansStepAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        debug_assert_eq!(values[0].null_count(), 0usize);
        let v = as_f32_list_like(&values[0], "k_means_step", "first")?;

        // no nulls are assumed in feature (embeddings)
        for i in 0..v.len() {
            let vv = v.value(i);
            let (cluster, _) = nearest_centers(vv, &self.centers, self.k, self.d, self.metric);
            self.state[self.k * self.d + cluster] += 1.0f64;

            for t in 0..self.d {
                self.state[cluster * self.d + t] += vv[t] as f64;
            }
        }

        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.to_scalar()])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let s = as_f64_list_like(&states[0], "k_means_step", "state")?;

        let n = self.k * self.d + self.k;
        // non-nulls semantic by contract
        for i in 0..s.len() {
            let ss = s.value(i);
            for t in 0..n {
                self.state[t] += ss[t];
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        // 2 usize values + 1 enum + 2x vec struct + k*d of f32 + (k*d + k) of f64
        size_of::<usize>() * 2
            + size_of::<DistanceMetric>()
            + (self.k * self.d) * size_of::<f32>()
            + (self.k * self.d + self.k) * size_of::<f64>()
            + size_of::<Vec<f64>>()
            + size_of::<Vec<f32>>()
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(self.to_scalar())
    }
}

#[derive(Debug)]
pub(crate) struct KMeansStep {
    signature: Signature,
    k: usize,
    d: usize,
    centers: Vec<f32>,
    metric: DistanceMetric,
}

impl KMeansStep {
    pub(crate) fn new(k: usize, d: usize, centers: Vec<f32>, metric: DistanceMetric) -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![
                    DataType::FixedSizeList(
                        Arc::new(Field::new("el", DataType::Float32, false)),
                        d as i32,
                    ),
                    DataType::List(Arc::new(Field::new("el", DataType::Float32, false))),
                ],
                Volatility::Immutable,
            ),
            k: k,
            d: d,
            centers: centers,
            metric: metric,
        }
    }

    pub(crate) fn default(k: usize, d: usize, centers: Vec<f32>) -> Self {
        Self::new(k, d, centers, DistanceMetric::L2)
    }
}

impl PartialEq for KMeansStep {
    fn eq(&self, other: &Self) -> bool {
        // note: signature cannot differ because it is not exposed anyhow
        // to the caller
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

impl Eq for KMeansStep {}

impl Hash for KMeansStep {
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

impl AggregateUDFImpl for KMeansStep {
    fn name(&self) -> &str {
        "k_means_step"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::FixedSizeList(
            Arc::new(Field::new("el", DataType::Float64, false)),
            (self.k * self.d + self.k) as i32,
        ))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(KMeansStepAccumulator::new(
            self.k,
            self.d,
            self.centers.clone(),
            self.metric,
        )))
    }
}

/// Builds an [`Expr`] applying `k_means_step` to `features`.
///
/// `k`, `d`, `centers` and `metric` are captured in the UDF instance at
/// construction time. The caller must build a fresh expression
/// per iteration.
pub(crate) fn kmeans_step_expr(
    features: Expr,
    k: usize,
    d: usize,
    centers: Vec<f32>,
    metric: DistanceMetric,
) -> Expr {
    AggregateUDF::from(KMeansStep::new(k, d, centers, metric)).call(vec![features])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, RecordBatch,
    };
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::Result;
    use datafusion::datasource::MemTable;
    use datafusion::prelude::*;
    use datafusion::scalar::ScalarValue;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    fn centers_2x2() -> Vec<f32> {
        vec![10.0, 10.0, 0.0, 0.0]
    }

    /// `FixedSizeList<Float32>` column (d=2) from flat rows.
    fn feat_batch(rows: &[[f32; 2]]) -> ArrayRef {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("el", DataType::Float32, false)),
                2,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .unwrap(),
        )
    }

    /// Reads `[k*d sums, k counts]` out of the accumulator state regardless of
    /// the current state representation (List today, FixedSizeList per the
    /// declared return type once the representation is aligned).
    fn state_vec(acc: &mut KMeansStepAccumulator) -> Vec<f64> {
        let st = acc.state().unwrap();
        match &st[0] {
            ScalarValue::FixedSizeList(arr) => {
                let binding = arr.value(0);
                let v = binding.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..v.len()).map(|i| v.value(i)).collect()
            }
            ScalarValue::List(la) => {
                let binding = la.value(0);
                let v = binding.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..v.len()).map(|i| v.value(i)).collect()
            }
            other => panic!("unexpected state: {other:?}"),
        }
    }

    /// State as an [`ArrayRef`] suitable for `merge_batch`.
    fn state_array(acc: &mut KMeansStepAccumulator) -> ArrayRef {
        let st = acc.state().unwrap();
        match &st[0] {
            ScalarValue::FixedSizeList(arr) => arr.clone() as ArrayRef,
            ScalarValue::List(la) => la.clone() as ArrayRef,
            other => panic!("unexpected state: {other:?}"),
        }
    }

    #[test]
    fn sums_land_in_assigned_center_block() {
        // rows (11,11) -> c0, (1,1) -> c1 against c0=(10,10), c1=(0,0)
        let mut acc = KMeansStepAccumulator::new(2, 2, centers_2x2(), DistanceMetric::L2);
        acc.update_batch(&vec![feat_batch(&[[11.0, 11.0], [1.0, 1.0]])])
            .unwrap();
        assert_eq!(
            state_vec(&mut acc),
            vec![11.0, 11.0, 1.0, 1.0, 1.0, 1.0],
            "per-cluster sums + counts"
        );
    }

    #[test]
    fn merge_batch_adds_partials() {
        let mut a = KMeansStepAccumulator::new(2, 2, centers_2x2(), DistanceMetric::L2);
        a.update_batch(&vec![feat_batch(&[[11.0, 11.0]])]).unwrap();
        let mut b = KMeansStepAccumulator::new(2, 2, centers_2x2(), DistanceMetric::L2);
        b.update_batch(&vec![feat_batch(&[[1.0, 1.0]])]).unwrap();
        a.merge_batch(&[state_array(&mut b)]).unwrap();
        assert_eq!(
            state_vec(&mut a),
            vec![11.0, 11.0, 1.0, 1.0, 1.0, 1.0],
            "merged partials equal the single-batch result"
        );
    }

    #[test]
    fn state_shape_matches_declared_return_type() {
        // return_type declares FixedSizeList(Float64, k*d+k); state must match.
        let mut acc = KMeansStepAccumulator::new(2, 2, centers_2x2(), DistanceMetric::L2);
        acc.update_batch(&vec![feat_batch(&[[1.0, 0.0]])]).unwrap();
        let st = acc.state().unwrap();
        match &st[0] {
            ScalarValue::FixedSizeList(arr) => {
                assert_eq!(arr.len(), 1);
                assert_eq!(arr.value_length() as usize, 6);
            }
            other => panic!("state must be ScalarValue::FixedSizeList, got {other:?}"),
        }
    }

    fn hash_of(x: &KMeansStep) -> u64 {
        let mut h = DefaultHasher::new();
        x.hash(&mut h);
        h.finish()
    }

    #[test]
    fn eq_respects_centers_length() {
        let a = KMeansStep::new(1, 2, vec![1.0f32], DistanceMetric::L2);
        let b = KMeansStep::new(1, 2, vec![1.0f32, 2.0], DistanceMetric::L2);
        assert_ne!(a, b, "prefix-equal centers of different length must differ");
    }

    #[test]
    fn eq_and_hash_agree_on_signed_zero() {
        // Numeric equality says +0.0 == -0.0; the bit-level hash differs.
        // Whatever side equality lands on, the Eq/Hash contract must hold.
        let a = KMeansStep::new(1, 2, vec![0.0f32, 0.0], DistanceMetric::L2);
        let b = KMeansStep::new(1, 2, vec![-0.0f32, 0.0], DistanceMetric::L2);
        if a == b {
            assert_eq!(hash_of(&a), hash_of(&b), "equal objects must hash equal");
        }
    }

    #[test]
    fn builder_produces_kmeans_step_expr() {
        let expr = kmeans_step_expr(col("feat"), 2, 2, centers_2x2(), DistanceMetric::L2);
        let s = format!("{expr}");
        assert!(s.contains("k_means_step"), "expr display: {s}");
    }

    async fn register_features(
        ctx: &SessionContext,
        rows: &[[f32; 2]],
        partitions: usize,
    ) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "feat",
            DataType::FixedSizeList(Arc::new(Field::new("el", DataType::Float32, false)), 2),
            false,
        )]));
        let chunks = rows.chunks(rows.len().div_ceil(partitions));
        let batches: Vec<Vec<RecordBatch>> = chunks
            .map(|chunk| {
                let flat: Vec<f32> = chunk.iter().flatten().copied().collect();
                let fsl = FixedSizeListArray::try_new(
                    Arc::new(Field::new("el", DataType::Float32, false)),
                    2,
                    Arc::new(Float32Array::from(flat)),
                    None,
                )
                .unwrap();
                vec![RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl) as ArrayRef]).unwrap()]
            })
            .collect();
        let table = MemTable::try_new(schema, batches)?;
        ctx.register_table("t", Arc::new(table))?;
        Ok(())
    }

    async fn run_step(ctx: &SessionContext, centers: Vec<f32>) -> Result<Vec<f64>> {
        let udf = AggregateUDF::from(KMeansStep::new(2, 2, centers, DistanceMetric::L2));
        let out = ctx
            .table("t")
            .await?
            .aggregate(vec![], vec![udf.call(vec![col("feat")]).alias("r")])?
            .collect()
            .await?;
        assert_eq!(out[0].num_rows(), 1);
        let arr = out[0].column(0);
        let fsl = arr
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("result must be a FixedSizeList");
        let binding = fsl.value(0);
        let v = binding.as_any().downcast_ref::<Float64Array>().unwrap();
        Ok((0..v.len()).map(|i| v.value(i)).collect())
    }

    #[tokio::test]
    async fn e2e_aggregate_single_partition() -> Result<()> {
        let ctx = SessionContext::new();
        register_features(&ctx, &[[11.0, 11.0], [1.0, 1.0]], 1).await?;
        let out = run_step(&ctx, centers_2x2()).await?;
        assert_eq!(out, vec![11.0, 11.0, 1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[tokio::test]
    async fn e2e_aggregate_two_partitions() -> Result<()> {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        register_features(&ctx, &[[11.0, 11.0], [1.0, 1.0]], 2).await?;
        let out = run_step(&ctx, centers_2x2()).await?;
        assert_eq!(out, vec![11.0, 11.0, 1.0, 1.0, 1.0, 1.0]);
        Ok(())
    }
}
