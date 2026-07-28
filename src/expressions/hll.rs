//! HyperLogLog sketch utilities for graph algorithms.
//!
//! A thin DataFusion wrapper over the Apache Datasketches HLL implementation,
//! exposing sketch construction, union, and cardinality estimation as
//! scalar/aggregate UDFs so they can be composed directly inside Pregel-style
//! expressions.
//!
//! The scope is deliberately narrow: only `Int64` inputs (every graph id in
//! this crate is a 64-bit long) and only the operations needed to build
//! neighbourhood-sketch algorithms:
//! - `hll_long` builds a singleton sketch `{x}` from one id (scalar UDF);
//! - `hll_long_union` unions two sketches (scalar UDF);
//! - `hll_long_aggregate` unions a group of sketches into one per group
//!   (aggregate UDF);
//! - `hll_long_estimate` returns the estimated cardinality of a sketch
//!   (scalar UDF).
//!
//! Sketches are carried as `Binary` columns, but the UDFs accept `BinaryView`
//! as well, because parquet spills — the foundation of the out-of-core Pregel
//! engine — round-trip `Binary` as `BinaryView`. Note that `hll_long_aggregate`
//! is pinned to `lg_k = 12` (`DEFAULT_LG_K`); keep it paired with sketches
//! built at the same `lg_k`, or it will silently downsample.
//!
//! These are the building blocks for HyperANF / HyperBALL and approximate
//! closeness centrality (see `algorithm::centrality::hyperanf`).
use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, BinaryArray, Float64Array},
        datatypes::DataType,
    },
    error::{DataFusionError, Result},
    logical_expr::{
        Accumulator, AggregateUDF, AggregateUDFImpl, ColumnarValue, Expr, ScalarFunctionArgs,
        ScalarUDF, ScalarUDFImpl, Signature, Volatility, function::AccumulatorArgs,
    },
    scalar::ScalarValue,
};
use datasketches::hll::{HllSketch, HllType};

use crate::expressions::common::as_binary_like;
use crate::expressions::common::downcast_int64;

const DEFAULT_HLL_TYPE: HllType = HllType::Hll8;
const DEFAULT_LG_K: u8 = 12;

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct HllLong {
    signature: Signature,
    lg_k: u8,
}

impl HllLong {
    pub(crate) fn new(lg_k: u8) -> Self {
        if (lg_k < 4) || (lg_k > 21) {
            panic!("lg_k should be in [4, 21]!");
        }
        Self {
            signature: Signature::uniform(1, vec![DataType::Int64], Volatility::Immutable),
            lg_k: lg_k,
        }
    }
}

impl ScalarUDFImpl for HllLong {
    fn name(&self) -> &str {
        "hll_long"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> datafusion::error::Result<DataType> {
        if (arg_types.len() != 1) || (arg_types[0] != DataType::Int64) {
            return Err(DataFusionError::Plan(format!(
                "hll_long expects exactly one Int64 argument, got: {arg_types:?}"
            )));
        }

        Ok(DataType::Binary)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 1 {
            return Err(DataFusionError::Plan(format!(
                "expected exactly one argument, got: {}",
                arrays.len(),
            )));
        }

        let v = downcast_int64(&arrays[0], "hll_long", "first")?;
        let len = v.len().max(args.number_rows);
        let result: BinaryArray = if v.null_count() == 0 {
            (0..len)
                .map(|i| {
                    let mut hll8 = HllSketch::new(self.lg_k, DEFAULT_HLL_TYPE);
                    hll8.update(v.value(i % v.len()));
                    Some(hll8.serialize())
                })
                .collect()
        } else {
            (0..len)
                .map(|i| {
                    if v.is_null(i % v.len()) {
                        None
                    } else {
                        let mut hll8 = HllSketch::new(self.lg_k, DEFAULT_HLL_TYPE);
                        hll8.update(v.value(i % v.len()));
                        Some(hll8.serialize())
                    }
                })
                .collect()
        };

        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

pub(crate) fn hll_long(col: Expr, lg_k: u8) -> Expr {
    ScalarUDF::from(HllLong::new(lg_k)).call(vec![col])
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct HllEstimate {
    signature: Signature,
}

impl HllEstimate {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![DataType::Binary, DataType::BinaryView],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for HllEstimate {
    fn name(&self) -> &str {
        "hll_estimate"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if (arg_types.len() != 1)
            || !matches!(arg_types[0], DataType::Binary | DataType::BinaryView)
        {
            return Err(DataFusionError::Plan(format!(
                "hll_estimate expects exactly one Binary/BinaryView argument, got: {arg_types:?}"
            )));
        }

        Ok(DataType::Float64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 1 {
            return Err(DataFusionError::Plan(format!(
                "expected exactly one argument, got: {}",
                arrays.len(),
            )));
        }

        let v = as_binary_like(&arrays[0], "hll_estimate", "first")?;

        let len = v.len().max(args.number_rows);
        let estimates: Vec<Option<f64>> = (0..len)
            .map(|i| {
                let idx = i % v.len();
                if v.is_null(idx) {
                    return Ok(None);
                }
                let sketch = HllSketch::deserialize(v.value(idx))
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                Ok(Some(sketch.estimate()))
            })
            .collect::<Result<_>>()?;

        let result = Float64Array::from(estimates);
        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

pub(crate) fn hll_long_estimate(col: Expr) -> Expr {
    ScalarUDF::from(HllEstimate::new()).call(vec![col])
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct HllUnion {
    signature: Signature,
}

impl HllUnion {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::uniform(
                2,
                vec![DataType::Binary, DataType::BinaryView],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for HllUnion {
    fn name(&self) -> &str {
        "hll_union"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if (arg_types.len() != 2)
            || !arg_types
                .iter()
                .all(|dt| matches!(dt, DataType::Binary | DataType::BinaryView))
        {
            return Err(DataFusionError::Plan(format!(
                "hll_union expects exactly two Binary/BinaryView arguments, got: {arg_types:?}"
            )));
        }

        Ok(DataType::Binary)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "expected exactly two arguments, got: {}",
                arrays.len(),
            )));
        }

        let left = as_binary_like(&arrays[0], "hll_union", "first")?;

        let right = as_binary_like(&arrays[1], "hll_union", "second")?;

        let len = (if left.len() > right.len() {
            left.len()
        } else {
            right.len()
        })
        .max(args.number_rows);

        let unioned: BinaryArray = match (left.null_count(), right.null_count()) {
            (0, 0) => (0..len)
                .map(|i| {
                    let sketch_left = HllSketch::deserialize(left.value(i % left.len()))
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                    let sketch_right = HllSketch::deserialize(right.value(i % right.len()))
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

                    let mut union = datasketches::hll::HllUnion::new(
                        sketch_left.lg_config_k().max(sketch_right.lg_config_k()),
                    );
                    union.update(&sketch_left);
                    union.update(&sketch_right);

                    Ok(Some(union.to_sketch(DEFAULT_HLL_TYPE).serialize()))
                })
                .collect::<Result<_>>()?,
            _ => (0..len)
                .map(|i| {
                    if i >= left.len() || left.is_null(i) || i >= right.len() || right.is_null(i) {
                        Ok(None)
                    } else {
                        let sketch_left = HllSketch::deserialize(left.value(i))
                            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                        let sketch_right = HllSketch::deserialize(right.value(i))
                            .map_err(|e| DataFusionError::Execution(e.to_string()))?;

                        let mut union = datasketches::hll::HllUnion::new(
                            sketch_left.lg_config_k().max(sketch_right.lg_config_k()),
                        );
                        union.update(&sketch_left);
                        union.update(&sketch_right);

                        Ok(Some(union.to_sketch(DEFAULT_HLL_TYPE).serialize()))
                    }
                })
                .collect::<Result<_>>()?,
        };

        let result = BinaryArray::from(unioned);

        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

pub(crate) fn hll_long_union(a: Expr, b: Expr) -> Expr {
    ScalarUDF::from(HllUnion::new()).call(vec![a, b])
}

#[derive(Debug)]
pub(crate) struct HllAccumulator {
    state: datasketches::hll::HllUnion,
}

impl HllAccumulator {
    pub(crate) fn new(lg_k: u8) -> Self {
        Self {
            state: datasketches::hll::HllUnion::new(lg_k),
        }
    }
}

impl Accumulator for HllAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> std::result::Result<(), DataFusionError> {
        let v = as_binary_like(&values[0], "hll_long_aggregate", "argument")?;

        if v.null_count() == 0 {
            (0..v.len())
                .map(|i| {
                    let sketch = HllSketch::deserialize(v.value(i))
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                    self.state.update(&sketch);
                    Ok(())
                })
                .collect::<Result<()>>()?
        } else {
            (0..v.len())
                .map(|i| {
                    if !v.is_null(i) {
                        let sketch = HllSketch::deserialize(v.value(i))
                            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                        self.state.update(&sketch);
                        Ok(())
                    } else {
                        Ok(())
                    }
                })
                .collect::<Result<()>>()?
        };

        Ok(())
    }

    fn evaluate(&mut self) -> std::result::Result<ScalarValue, DataFusionError> {
        Ok(ScalarValue::Binary(Some(
            self.state.to_sketch(DEFAULT_HLL_TYPE).serialize(),
        )))
    }

    fn size(&self) -> usize {
        // The gadget's register array is ~2^lg_max_k
        // bytes for Hll8, plus the union/sketch struct overhead.
        std::mem::size_of::<Self>() + (1usize << self.state.lg_max_k())
    }

    fn state(&mut self) -> std::result::Result<Vec<ScalarValue>, DataFusionError> {
        Ok(vec![ScalarValue::Binary(Some(
            self.state.to_sketch(DEFAULT_HLL_TYPE).serialize(),
        ))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> std::result::Result<(), DataFusionError> {
        // merge is the same as update because internally we are
        // serializing state as sketch.
        self.update_batch(states)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct HllAggregate {
    signature: Signature,
    lg_k: u8,
}

impl HllAggregate {
    pub(crate) fn new(lg_k: u8) -> Self {
        if (lg_k < 4) || (lg_k > 21) {
            panic!("lg_k should be in [4, 21]!");
        }
        Self {
            signature: Signature::uniform(
                1,
                vec![DataType::Binary, DataType::BinaryView],
                Volatility::Immutable,
            ),
            lg_k,
        }
    }
}

impl AggregateUDFImpl for HllAggregate {
    fn name(&self) -> &str {
        "hll_long_aggregate"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if (arg_types.len() != 1)
            || !matches!(arg_types[0], DataType::Binary | DataType::BinaryView)
        {
            return Err(DataFusionError::Plan(format!(
                "hll_aggregate expects exactly one Binary/BinaryView argument, got: {arg_types:?}"
            )));
        }

        Ok(DataType::Binary)
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(HllAccumulator::new(self.lg_k)))
    }
}

pub(crate) fn hll_long_aggregate(col: Expr) -> Expr {
    AggregateUDF::from(HllAggregate::new(DEFAULT_LG_K)).call(vec![col])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    /// A single distinct value hashes into a sketch that estimates exactly 1.0
    /// (HLL is exact in List mode for tiny cardinalities).
    #[tokio::test]
    async fn test_hll_long_singleton_estimates_one() -> Result<()> {
        let df = dataframe!("id" => vec![7i64, 42i64, -1i64, i64::MAX])?;
        let out = df
            .select(vec![hll_long(col("id"), 12).alias("sketch")])?
            .collect()
            .await?;

        let sketches = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        assert_eq!(sketches.len(), 4);
        for i in 0..sketches.len() {
            assert!(!sketches.is_null(i), "row {i} unexpectedly null");
            let sketch = HllSketch::deserialize(sketches.value(i))
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            assert!(
                (sketch.estimate() - 1.0).abs() < 1e-9,
                "row {i}: expected estimate 1.0, got {}",
                sketch.estimate()
            );
        }
        Ok(())
    }

    /// The same value must produce a byte-identical sketch (deterministic
    /// Murmur hash); distinct values must produce distinct sketches.
    #[tokio::test]
    async fn test_hll_long_is_deterministic_and_distinguishes_values() -> Result<()> {
        let df = dataframe!("id" => vec![1i64, 1, 2, 2, 3])?;
        let out = df
            .select(vec![hll_long(col("id"), 12).alias("sketch")])?
            .collect()
            .await?;

        let sketches = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        let bytes: Vec<Vec<u8>> = (0..sketches.len())
            .map(|i| sketches.value(i).to_vec())
            .collect();

        assert_eq!(bytes[0], bytes[1], "same id must serialize identically");
        assert_eq!(bytes[2], bytes[3], "same id must serialize identically");
        assert_ne!(
            bytes[0], bytes[2],
            "distinct ids must produce distinct sketches"
        );
        assert_ne!(
            bytes[0], bytes[4],
            "distinct ids must produce distinct sketches"
        );
        Ok(())
    }

    /// Null inputs must propagate to null outputs (no sketch built).
    #[tokio::test]
    async fn test_hll_long_null_propagates() -> Result<()> {
        let df = dataframe!("id" => vec![Some(1i64), None, Some(3i64)])?;
        let out = df
            .select(vec![hll_long(col("id"), 12).alias("sketch")])?
            .collect()
            .await?;

        let sketches = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();

        assert_eq!(sketches.len(), 3);
        assert!(!sketches.is_null(0));
        assert!(sketches.is_null(1), "null input must yield null output");
        assert!(!sketches.is_null(2));
        Ok(())
    }

    /// `return_type` must accept exactly one Int64 and reject everything else.
    #[tokio::test]
    async fn test_hll_long_return_type_validation() {
        let udf = HllLong::new(12);
        assert_eq!(
            udf.return_type(&[DataType::Int64]).unwrap(),
            DataType::Binary
        );
        // Wrong arity.
        assert!(udf.return_type(&[]).is_err());
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Int64])
                .is_err()
        );
        // Wrong type.
        assert!(udf.return_type(&[DataType::Utf8]).is_err());
        assert!(udf.return_type(&[DataType::Binary]).is_err());
    }

    /// `lg_k` outside the supported `[4, 21]` range must fail fast.
    #[test]
    #[should_panic(expected = "lg_k should be in [4, 21]!")]
    fn test_hll_long_rejects_lg_k_below_min() {
        let _ = HllLong::new(3);
    }

    #[test]
    #[should_panic(expected = "lg_k should be in [4, 21]!")]
    fn test_hll_long_rejects_lg_k_above_max() {
        let _ = HllLong::new(22);
    }

    /// Boundary values of `lg_k` are accepted (no panic).
    #[test]
    fn test_hll_long_accepts_boundary_lg_k() {
        let _ = HllLong::new(4);
        let _ = HllLong::new(21);
    }

    /// `hll_long_estimate(hll_long(id))` round-trips: a singleton sketch
    /// estimates exactly 1.0 through the full UDF pipeline.
    #[tokio::test]
    async fn test_hll_long_estimate_singleton() -> Result<()> {
        let df = dataframe!("id" => vec![7i64, 42i64, -1i64])?;
        let out = df
            .select(vec![
                hll_long_estimate(hll_long(col("id"), 12)).alias("est"),
            ])?
            .collect()
            .await?;

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(est.len(), 3);
        for i in 0..est.len() {
            assert!(!est.is_null(i), "row {i} unexpectedly null");
            assert!(
                (est.value(i) - 1.0).abs() < 1e-9,
                "row {i}: expected 1.0, got {}",
                est.value(i)
            );
        }
        Ok(())
    }

    /// Null sketches (from a null id) must propagate to a null estimate, not
    /// a panic and not a spurious 0.0.
    #[tokio::test]
    async fn test_hll_long_estimate_null_propagates() -> Result<()> {
        let df = dataframe!("id" => vec![Some(1i64), None, Some(3i64)])?;
        let out = df
            .select(vec![
                hll_long_estimate(hll_long(col("id"), 12)).alias("est"),
            ])?
            .collect()
            .await?;

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(est.len(), 3);
        assert!(!est.is_null(0));
        assert!(est.is_null(1), "null sketch must yield null estimate");
        assert!(!est.is_null(2));
        Ok(())
    }

    /// `return_type` must accept exactly one Binary and reject everything else.
    #[tokio::test]
    async fn test_hll_long_estimate_return_type_validation() {
        let udf = HllEstimate::new();
        assert_eq!(
            udf.return_type(&[DataType::Binary]).unwrap(),
            DataType::Float64
        );
        assert!(udf.return_type(&[]).is_err());
        assert!(
            udf.return_type(&[DataType::Binary, DataType::Binary])
                .is_err()
        );
        assert!(udf.return_type(&[DataType::Int64]).is_err());
        assert!(udf.return_type(&[DataType::Utf8]).is_err());
    }

    /// A corrupt/truncated sketch byte image must surface as a query error,
    /// not abort execution with a panic.
    #[tokio::test]
    async fn test_hll_long_estimate_malformed_bytes_errors() {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::{Field, Schema};

        // A single zero byte is not a valid HLL preamble: deserialization runs
        // out of bytes and must error instead of panicking.
        let sketch_col: BinaryArray = BinaryArray::from(vec![Some(&[0u8][..])]);
        let schema = Arc::new(Schema::new(vec![Field::new(
            "sketch",
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(sketch_col) as ArrayRef]).unwrap();

        let ctx = SessionContext::new();
        let df = ctx.read_batch(batch).unwrap();
        let result = df
            .select(vec![hll_long_estimate(col("sketch")).alias("est")])
            .unwrap()
            .collect()
            .await;
        assert!(
            result.is_err(),
            "malformed sketch bytes must produce a query error, not succeed or panic"
        );
    }

    /// A literal/scalar argument must be broadcast across `number_rows`
    /// (the broadcast-len fix): `hll_long(lit(7), 12)` over a 3-row frame must
    /// yield 3 sketches of {7}, not 1 row.
    #[tokio::test]
    async fn test_hll_long_broadcasts_scalar_literal() -> Result<()> {
        let df = dataframe!("id" => vec![1i64, 2, 3])?;
        let out = df
            .select(vec![hll_long(lit(7i64), 12).alias("s")])?
            .collect()
            .await?;

        let sketches = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(
            sketches.len(),
            3,
            "scalar literal must broadcast to all rows"
        );
        // Every broadcast row must carry the same sketch of the literal value 7.
        let first = sketches.value(0).to_vec();
        for i in 0..sketches.len() {
            assert_eq!(
                sketches.value(i),
                first,
                "row {i} should equal the broadcast value"
            );
            assert_eq!(estimate_of(sketches.value(i)), 1.0);
        }
        Ok(())
    }

    /// Same broadcast guarantee for `hll_long_estimate`: a literal sketch
    /// argument over a 3-row frame must yield 3 estimates, not 1.
    #[tokio::test]
    async fn test_hll_long_estimate_broadcasts_scalar_literal() -> Result<()> {
        use datafusion::common::ScalarValue;

        // Fresh sketch of 3 distinct values (List mode => estimate exactly 3.0).
        let sketch_bytes = build_hll8(&[10, 20, 30]);
        let df = dataframe!("id" => vec![1i64, 2, 3])?;
        let out = df
            .select(vec![
                hll_long_estimate(lit(ScalarValue::Binary(Some(sketch_bytes)))).alias("e"),
            ])?
            .collect()
            .await?;

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(
            est.len(),
            3,
            "scalar literal sketch must broadcast to all rows"
        );
        for i in 0..est.len() {
            assert!(
                (est.value(i) - 3.0).abs() < 1e-3,
                "row {i}: got {}",
                est.value(i)
            );
        }
        Ok(())
    }

    /// Build an Hll8 sketch (lg_k=12) over the given ids and serialize it,
    /// for tests that need real multi-element sketches to feed into the union UDF.
    fn build_hll8(vals: &[i64]) -> Vec<u8> {
        let mut s = HllSketch::new(12, HllType::Hll8);
        for v in vals {
            s.update(*v);
        }
        s.serialize()
    }

    /// Estimate a serialized sketch, for test assertions.
    fn estimate_of(bytes: &[u8]) -> f64 {
        HllSketch::deserialize(bytes)
            .expect("test sketches must deserialize")
            .estimate()
    }

    /// Build a Binary column from per-row optional byte blobs. Copies the bytes
    /// through a `BinaryBuilder` so there are no dangling borrows from temporary
    /// `Vec`s (only the `&[u8]` slice `From` impls exist for `BinaryArray`).
    fn binary_col(rows: &[Option<Vec<u8>>]) -> BinaryArray {
        use datafusion::arrow::array::BinaryBuilder;
        let mut b = BinaryBuilder::new();
        for row in rows {
            match row {
                Some(v) => b.append_value(v),
                None => b.append_null(),
            }
        }
        b.finish()
    }

    /// Full Expr pipeline: `estimate(union(sketch(a), sketch(b)))` over
    /// singleton sketches. Distinct ids union to 2; equal ids dedup to 1.
    #[tokio::test]
    async fn test_hll_long_union_pipeline_singletons() -> Result<()> {
        // Row 0: {1}∪{4} = 2 ; Row 1: {2}∪{2} = 1 (dedup) ; Row 2: {3}∪{1} = 2.
        let df = dataframe!(
            "a" => vec![1i64, 2, 3],
            "b" => vec![4i64, 2, 1],
        )?;
        let out = df
            .select(vec![
                hll_long_estimate(hll_long_union(
                    hll_long(col("a"), 12),
                    hll_long(col("b"), 12),
                ))
                .alias("est"),
            ])?
            .collect()
            .await?;

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(est.len(), 3);
        // Union estimates carry tiny HLL estimator noise (HLL RE ~1.6% at
        // lg_k=12), so a 1e-3 tolerance is both stable and tight enough to
        // catch e.g. an ignored operand (would give 1.0 instead of 2.0).
        assert!(
            (est.value(0) - 2.0).abs() < 1e-3,
            "row 0: got {}",
            est.value(0)
        );
        assert!(
            (est.value(1) - 1.0).abs() < 1e-3,
            "row 1 (dedup): got {}",
            est.value(1)
        );
        assert!(
            (est.value(2) - 2.0).abs() < 1e-3,
            "row 2: got {}",
            est.value(2)
        );
        Ok(())
    }

    /// True set union over multi-element sketches built directly (not via the
    /// singleton-only `hll_long` path). HLL is exact at these tiny
    /// cardinalities, so estimates equal the true union size.
    #[tokio::test]
    async fn test_hll_long_union_multi_element_sets() {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::{Field, Schema};

        let left = binary_col(&[
            Some(build_hll8(&[1, 2, 3])),
            Some(build_hll8(&[1, 2, 3, 4, 5, 6])),
        ]);
        let right = binary_col(&[
            Some(build_hll8(&[3, 4, 5])),       // {1,2,3} ∪ {3,4,5} = 5
            Some(build_hll8(&[4, 5, 6, 7, 8])), // {1..6} ∪ {4..8} = 8
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Binary, true),
            Field::new("b", DataType::Binary, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(left) as ArrayRef, Arc::new(right) as ArrayRef],
        )
        .unwrap();

        let ctx = SessionContext::new();
        let out = ctx
            .read_batch(batch)
            .unwrap()
            .select(vec![
                hll_long_estimate(hll_long_union(col("a"), col("b"))).alias("est"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(est.len(), 2);
        assert!(
            (est.value(0) - 5.0).abs() < 1e-3,
            "row 0: got {}",
            est.value(0)
        );
        assert!(
            (est.value(1) - 8.0).abs() < 1e-3,
            "row 1: got {}",
            est.value(1)
        );
    }

    /// Union is idempotent: union(s, s) must estimate the same cardinality as s.
    #[tokio::test]
    async fn test_hll_long_union_idempotent() {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::{Field, Schema};

        let sketch = build_hll8(&[1, 2, 3, 4, 5, 6, 7]);
        let before = estimate_of(&sketch);
        let arr = binary_col(&[Some(sketch.clone())]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Binary, true),
            Field::new("b", DataType::Binary, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arr.clone()) as ArrayRef, Arc::new(arr) as ArrayRef],
        )
        .unwrap();

        let ctx = SessionContext::new();
        let out = ctx
            .read_batch(batch)
            .unwrap()
            .select(vec![
                hll_long_estimate(hll_long_union(col("a"), col("b"))).alias("est"),
            ])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(est.len(), 1);
        assert!(
            (est.value(0) - before).abs() < 1e-3,
            "union(s,s) != s: got {}",
            est.value(0)
        );
    }

    /// A null in either operand must propagate to a null result (not a panic,
    /// not a spurious 0.0).
    #[tokio::test]
    async fn test_hll_long_union_null_propagates() {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::{Field, Schema};

        let left = binary_col(&[Some(build_hll8(&[1])), None, Some(build_hll8(&[3]))]);
        let right = binary_col(&[
            Some(build_hll8(&[4])),
            Some(build_hll8(&[5])),
            Some(build_hll8(&[6])),
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Binary, true),
            Field::new("b", DataType::Binary, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(left) as ArrayRef, Arc::new(right) as ArrayRef],
        )
        .unwrap();

        let ctx = SessionContext::new();
        let out = ctx
            .read_batch(batch)
            .unwrap()
            .select(vec![hll_long_union(col("a"), col("b")).alias("u")])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let u = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        assert_eq!(u.len(), 3);
        assert!(!u.is_null(0));
        assert!(u.is_null(1), "null in left operand must yield null");
        assert!(!u.is_null(2));
    }

    /// `return_type` must accept exactly two Binary arguments and reject
    /// everything else.
    #[tokio::test]
    async fn test_hll_long_union_return_type_validation() {
        let udf = HllUnion::new();
        assert_eq!(
            udf.return_type(&[DataType::Binary, DataType::Binary])
                .unwrap(),
            DataType::Binary
        );
        assert!(udf.return_type(&[DataType::Binary]).is_err());
        assert!(
            udf.return_type(&[DataType::Binary, DataType::Binary, DataType::Binary])
                .is_err()
        );
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Int64])
                .is_err()
        );
        assert!(
            udf.return_type(&[DataType::Binary, DataType::Utf8])
                .is_err()
        );
    }

    /// A corrupt byte image in either operand must surface as a query error,
    /// not a panic.
    #[tokio::test]
    async fn test_hll_long_union_malformed_bytes_errors() {
        use datafusion::arrow::array::RecordBatch;
        use datafusion::arrow::datatypes::{Field, Schema};

        let left = binary_col(&[Some(build_hll8(&[1]))]);
        let right = binary_col(&[Some(vec![0u8])]); // invalid preamble
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Binary, true),
            Field::new("b", DataType::Binary, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(left) as ArrayRef, Arc::new(right) as ArrayRef],
        )
        .unwrap();

        let ctx = SessionContext::new();
        let result = ctx
            .read_batch(batch)
            .unwrap()
            .select(vec![hll_long_union(col("a"), col("b")).alias("u")])
            .unwrap()
            .collect()
            .await;
        assert!(
            result.is_err(),
            "malformed sketch bytes must produce a query error, not succeed or panic"
        );
    }

    /// Direct test of the partial->final merge path: serialize one
    /// accumulator's state and merge it into another. This is the path a
    /// default (no-op) `merge_batch` would silently corrupt under
    /// `target_partitions > 1`.
    #[test]
    fn test_hll_accumulator_merge_unions_partial_states() {
        let mut a = HllAccumulator::new(12);
        let mut b = HllAccumulator::new(12);

        a.update_batch(&[Arc::new(binary_col(&[Some(build_hll8(&[1, 2, 3]))])) as ArrayRef])
            .unwrap();
        b.update_batch(&[Arc::new(binary_col(&[Some(build_hll8(&[4, 5, 6]))])) as ArrayRef])
            .unwrap();

        // Serialize b's state and feed it back through merge_batch (final path).
        let b_state = b.state().unwrap();
        let bytes = match &b_state[0] {
            ScalarValue::Binary(Some(b)) => b.clone(),
            other => panic!("expected Binary state, got {other:?}"),
        };
        a.merge_batch(&[Arc::new(binary_col(&[Some(bytes)])) as ArrayRef])
            .unwrap();

        let merged = a.evaluate().unwrap();
        let est = match merged {
            ScalarValue::Binary(Some(b)) => estimate_of(&b),
            other => panic!("expected Binary result, got {other:?}"),
        };
        // {1,2,3} ∪ {4,5,6} = 6
        assert!(
            (est - 6.0).abs() < 1e-3,
            "merge of {{1,2,3}} + {{4,5,6}} should estimate ~6, got {est}"
        );
    }

    /// Full pipeline: each row -> singleton sketch, aggregate -> union.
    /// One group with ids {1..5} => estimate ~5.
    #[tokio::test]
    async fn test_hll_long_aggregate_basic() -> Result<()> {
        let df = dataframe!("g" => vec![0i64, 0, 0, 0, 0], "id" => vec![1i64, 2, 3, 4, 5])?;
        let out = df
            .aggregate(
                vec![col("g")],
                vec![hll_long_aggregate(hll_long(col("id"), 12)).alias("s")],
            )?
            .select(vec![hll_long_estimate(col("s")).alias("est")])?
            .collect()
            .await?;

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(est.len(), 1);
        assert!((est.value(0) - 5.0).abs() < 1e-3, "got {}", est.value(0));
        Ok(())
    }

    /// GROUP BY: each group unions its own sketches independently.
    #[tokio::test]
    async fn test_hll_long_aggregate_grouped() -> Result<()> {
        use datafusion::arrow::array::Int64Array;

        let df = dataframe!(
            "g" => vec![0i64, 0, 0, 1, 1, 1, 1],
            "id" => vec![1i64, 2, 3, 10, 20, 30, 40],
        )?;
        let out = df
            .aggregate(
                vec![col("g")],
                vec![hll_long_aggregate(hll_long(col("id"), 12)).alias("s")],
            )?
            .select(vec![col("g"), hll_long_estimate(col("s")).alias("est")])?
            .sort(vec![col("g").sort(true, true)])?
            .collect()
            .await?;

        let g = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let est = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.value(0), 0);
        assert_eq!(g.value(1), 1);
        assert!(
            (est.value(0) - 3.0).abs() < 1e-3,
            "group 0: got {}",
            est.value(0)
        );
        assert!(
            (est.value(1) - 4.0).abs() < 1e-3,
            "group 1: got {}",
            est.value(1)
        );
        Ok(())
    }

    /// Null sketch inputs are skipped (not an error, not added to the union).
    #[tokio::test]
    async fn test_hll_long_aggregate_skips_nulls() {
        use datafusion::arrow::array::{Int64Array, RecordBatch};
        use datafusion::arrow::datatypes::{Field, Schema};

        // sketches {1},{2},null,{4},{5} -> union {1,2,4,5} = 4
        let sketch_col = binary_col(&[
            Some(build_hll8(&[1])),
            Some(build_hll8(&[2])),
            None,
            Some(build_hll8(&[4])),
            Some(build_hll8(&[5])),
        ]);
        let g_col = Int64Array::from(vec![0i64, 0, 0, 0, 0]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("s", DataType::Binary, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(g_col) as ArrayRef,
                Arc::new(sketch_col) as ArrayRef,
            ],
        )
        .unwrap();

        let ctx = SessionContext::new();
        let out = ctx
            .read_batch(batch)
            .unwrap()
            .aggregate(
                vec![col("g")],
                vec![hll_long_aggregate(col("s")).alias("m")],
            )
            .unwrap()
            .select(vec![hll_long_estimate(col("m")).alias("est")])
            .unwrap()
            .collect()
            .await
            .unwrap();
        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(est.len(), 1);
        assert!(
            (est.value(0) - 4.0).abs() < 1e-3,
            "nulls skipped, union of 4 distinct = 4, got {}",
            est.value(0)
        );
    }

    /// Multi-partition: force the partial->final path (one partition's partial
    /// state must be merged into the other via `merge_batch`). A no-op
    /// `merge_batch` would yield ~100 (one partition) or 0 instead of ~200.
    #[tokio::test]
    async fn test_hll_long_aggregate_multi_partition_merge() {
        use datafusion::arrow::array::{Int64Array, RecordBatch};
        use datafusion::arrow::datatypes::{Field, Schema};
        use datafusion::datasource::MemTable;
        use datafusion::prelude::SessionConfig;

        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("id", DataType::Int64, false),
        ]));
        let mk = |ids: Vec<i64>| {
            let n = ids.len();
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        std::iter::repeat(0i64).take(n),
                    )) as ArrayRef,
                    Arc::new(Int64Array::from(ids)) as ArrayRef,
                ],
            )
            .unwrap()
        };
        let batch1 = mk((0..100).collect());
        let batch2 = mk((100..200).collect());

        // Two explicit partitions -> partial aggregate on each, final merge.
        let table = MemTable::try_new(schema, vec![vec![batch1], vec![batch2]]).unwrap();
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        ctx.register_table("t", Arc::new(table)).unwrap();
        let out = ctx
            .table("t")
            .await
            .unwrap()
            .aggregate(
                vec![col("g")],
                vec![hll_long_aggregate(hll_long(col("id"), 12)).alias("s")],
            )
            .unwrap()
            .select(vec![hll_long_estimate(col("s")).alias("est")])
            .unwrap()
            .collect()
            .await
            .unwrap();

        let est = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(est.len(), 1);
        assert!(
            (est.value(0) - 200.0).abs() / 200.0 < 0.05,
            "multi-partition union of {{0..199}} should be ~200, got {}",
            est.value(0)
        );
    }

    /// `return_type` must accept exactly one Binary and reject everything else.
    #[tokio::test]
    async fn test_hll_long_aggregate_return_type_validation() {
        let udf = HllAggregate::new(12);
        assert_eq!(
            udf.return_type(&[DataType::Binary]).unwrap(),
            DataType::Binary
        );
        assert!(udf.return_type(&[]).is_err());
        assert!(
            udf.return_type(&[DataType::Binary, DataType::Binary])
                .is_err()
        );
        assert!(udf.return_type(&[DataType::Int64]).is_err());
    }

    /// `lg_k` outside the supported `[4, 21]` range must fail fast at
    /// construction, not later when the accumulator is built.
    #[test]
    #[should_panic(expected = "lg_k should be in [4, 21]!")]
    fn test_hll_aggregate_rejects_bad_lg_k() {
        let _ = HllAggregate::new(3);
    }
}
