//! Scalar UDFs and vector aggregates over `f32` vectors: `l2_norm`,
//! `l2_distance`, `cosine_distance`, the `vec_sum` aggregate and the
//! FastRP `fastrp_init` random-projection init.
//!
//! This module holds the DataFusion wrappers only: SIMD-logic free; the
//! kernels live in [`crate::ml::linalg`]. Contract:
//! vector rows are non-null; the two-argument UDFs require both rows
//! to have the same length.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, ListArray,
};
use datafusion::arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::plan_err;
use datafusion::error::Result;
use datafusion::logical_expr::function::AccumulatorArgs;
use datafusion::logical_expr::groups_accumulator::{EmitTo, GroupsAccumulator};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, ColumnarValue, Expr, ScalarFunctionArgs,
    ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::scalar::ScalarValue;

use crate::expressions::common::{as_f32_list_like, downcast_int64};
use crate::ml::{
    cosine_distance, fastrp_init_fill, l2_distance, l2_norm, vec_add, vec_scale, vec_weighted_sum,
};

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

#[allow(dead_code)]
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

#[allow(dead_code)]
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

/// Builds the `el: Float32` child field shared by all vector columns here.
pub(crate) fn f32_child_field() -> Arc<Field> {
    Arc::new(Field::new("el", DataType::Float32, false))
}

/// Wraps a flat buffer into a non-null `FixedSizeList<Float32, d>` array.
fn fsl_array(d: usize, flat: Vec<f32>) -> Arc<FixedSizeListArray> {
    debug_assert_eq!(
        flat.len() % d,
        0,
        "flat vector buffer length must be a multiple of d"
    );
    Arc::new(
        FixedSizeListArray::try_new(
            f32_child_field(),
            d as i32,
            Arc::new(Float32Array::from(flat)),
            None,
        )
        .expect("valid fixed size list"),
    )
}

// ---------------- vec_sum: aggregate UDAF ----------------
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct VectorSum {
    signature: Signature,
    d: usize,
}

impl VectorSum {
    pub(crate) fn new(d: usize) -> Self {
        Self {
            // Nested-type signatures would pin the child field name and
            // nullability, but a parquet round-trip renames/loosens it
            // (`el` -> `element`, nullable). Accept any single argument and
            // validate the vector shape in `return_type` instead.
            signature: Signature::any(1, Volatility::Immutable),
            d,
        }
    }
}

impl AggregateUDFImpl for VectorSum {
    fn name(&self) -> &str {
        "vec_sum"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        validate_vector_args(arg_types, 1, "vec_sum")?;
        Ok(DataType::FixedSizeList(f32_child_field(), self.d as i32))
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VecSumAccumulator {
            inner: VecSumGroupsAccumulator::new(self.d),
        }))
    }

    fn groups_accumulator_supported(&self, _args: AccumulatorArgs) -> bool {
        true
    }

    fn create_groups_accumulator(
        &self,
        _args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(VecSumGroupsAccumulator::new(self.d)))
    }
}

/// Grouped running sums for [`VectorSum`]: one dense `Vec<f32>` holding
/// `total_num_groups * d` floats; group `g` occupies `sums[g*d .. (g+1)*d]`.
///
/// The buffer is zero-initialized (the sum identity), so a group is never
/// null. New groups arriving with a larger `total_num_groups` extend the
/// buffer; `EmitTo::First(n)` drains the first `n * d` floats so the
/// remaining group indices shift down, exactly as the trait contract
/// requires.
#[derive(Debug)]
struct VecSumGroupsAccumulator {
    d: usize,
    sums: Vec<f32>,
}

impl VecSumGroupsAccumulator {
    fn new(d: usize) -> Self {
        Self {
            d,
            sums: Vec::new(),
        }
    }

    /// Grow the dense buffer to `total_num_groups` zero vectors.
    fn resize(&mut self, total_num_groups: usize) {
        let needed = total_num_groups.saturating_mul(self.d);
        if self.sums.len() < needed {
            self.sums.resize(needed, 0.0f32);
        }
    }

    /// Add the rows selected by `group_indices` (optionally filtered) into
    /// the running group sums.
    fn accumulate(
        &mut self,
        v: &crate::expressions::common::F32ListLike,
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        use crate::expressions::common::F32ListLike;

        self.resize(total_num_groups);
        match v {
            // Fast path: FixedSizeList rows are constant-sized, one length
            // check is enough for the whole array.
            F32ListLike::Fixed(_) => {
                if v.value_length() != self.d {
                    return plan_err!(
                        "vec_sum: vector size {} does not match the declared dimension {}",
                        v.value_length(),
                        self.d
                    );
                }
            }
            // List rows are checked per row below (parquet round-trips make
            // this the common on-disk representation).
            F32ListLike::View(_) => {}
        }

        for (row, &g) in group_indices.iter().enumerate() {
            if let Some(filter) = opt_filter
                && (filter.is_null(row) || !filter.value(row))
            {
                continue;
            }
            let vals = v.value(row);
            if vals.len() != self.d {
                return plan_err!(
                    "vec_sum: vector length {} does not match the declared dimension {}",
                    vals.len(),
                    self.d
                );
            }
            let base = g * self.d;
            vec_add(&mut self.sums[base..base + self.d], vals);
        }
        Ok(())
    }

    /// Detach the emitted groups' flat sums, honoring the `EmitTo::First(n)`
    /// "shift down" contract. Groups beyond the accumulated tail (defensively,
    /// if the hash table allocated more groups than this accumulator ever saw
    /// rows for) keep the zero identity.
    fn take_state(&mut self, emit_to: EmitTo) -> Vec<f32> {
        match emit_to {
            EmitTo::All => std::mem::take(&mut self.sums),
            EmitTo::First(n) => {
                let n_floats = n.saturating_mul(self.d);
                let mut taken = Vec::with_capacity(n_floats);
                if n_floats <= self.sums.len() {
                    taken.extend(self.sums.drain(0..n_floats));
                } else {
                    taken.extend(self.sums.drain(0..));
                    taken.resize(n_floats, 0.0);
                }
                taken
            }
        }
    }
}

impl GroupsAccumulator for VecSumGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert_eq!(values.len(), 1, "vec_sum takes exactly one argument");
        // Non-null contract: without a filter the whole batch must be
        // null-free; with a FILTER clause the excluded rows are allowed to
        // be null (they never reach the accumulation loop).
        if opt_filter.is_none() {
            debug_assert_eq!(
                values[0].null_count(),
                0,
                "vec_sum input must be non-null by contract"
            );
        }
        let v = as_f32_list_like(&values[0], "vec_sum", "first")?;
        self.accumulate(&v, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let flat = self.take_state(emit_to);
        Ok(fsl_array(self.d, flat))
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        // SUM is self-combinable: the partial state is the (partial) sum.
        let flat = self.take_state(emit_to);
        Ok(vec![fsl_array(self.d, flat) as ArrayRef])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> Result<()> {
        // State may come back from a parquet spill as List<Float32> — accept
        // both representations like every vector consumer in this crate.
        let v = as_f32_list_like(&values[0], "vec_sum", "state")?;
        self.accumulate(&v, group_indices, None, total_num_groups)
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        // Partial-aggregation bypass: for SUM the input rows *are* valid
        // state rows; filtered-out rows become the zero-vector identity.
        let v = as_f32_list_like(&values[0], "vec_sum", "first")?;
        let mut flat = Vec::with_capacity(v.len() * self.d);
        for row in 0..v.len() {
            let keep = opt_filter.map_or(true, |f| !f.is_null(row) && f.value(row));
            if keep {
                let vals = v.value(row);
                if vals.len() != self.d {
                    return plan_err!(
                        "vec_sum: vector length {} does not match the declared dimension {}",
                        vals.len(),
                        self.d
                    );
                }
                flat.extend_from_slice(vals);
            } else {
                flat.extend(std::iter::repeat_n(0.0f32, self.d));
            }
        }
        Ok(vec![fsl_array(self.d, flat) as ArrayRef])
    }

    fn size(&self) -> usize {
        // capacity-based so DataFusion's spill accounting sees the real cost
        self.sums.capacity() * size_of::<f32>() + 2 * size_of::<usize>()
    }
}

/// No-GROUP-BY fallback: a single implicit group over the grouped
/// accumulator.
#[derive(Debug)]
struct VecSumAccumulator {
    inner: VecSumGroupsAccumulator,
}

impl Accumulator for VecSumAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // every row belongs to the single implicit group 0
        let indices = vec![0usize; values[0].len()];
        self.inner.update_batch(values, &indices, None, 1)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let indices = vec![0usize; states[0].len()];
        self.inner.merge_batch(states, &indices, 1)
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let arr = self.inner.evaluate(EmitTo::All)?;
        ScalarValue::try_from_array(&arr, 0)
    }
}

/// Builds an [`Expr`] summing same-sized `f32` vectors per group.
pub(crate) fn vec_sum_expr(v: Expr, d: usize) -> Expr {
    AggregateUDF::from(VectorSum::new(d)).call(vec![v])
}

// ---------------- vec_scale: scalar UDF ----------------

/// Scalar UDF `vec_scale(v, s) -> v * s`: element-wise product of an `f32`
/// vector with a scalar (Float32 or Float64; the factor may vary per row).
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct VecScale {
    signature: Signature,
}

impl VecScale {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl Default for VecScale {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_vec_scale_args(arg_types: &[DataType]) -> Result<()> {
    if arg_types.len() != 2 {
        return plan_err!("vec_scale expects 2 arguments, got {}", arg_types.len());
    }
    let is_vector = match &arg_types[0] {
        DataType::FixedSizeList(f, _) => f.data_type() == &DataType::Float32,
        DataType::List(f) => f.data_type() == &DataType::Float32,
        _ => false,
    };
    if !is_vector {
        return plan_err!(
            "vec_scale argument 0 must be FixedSizeList<Float32> or List<Float32>, got {:?}",
            arg_types[0]
        );
    }
    match &arg_types[1] {
        DataType::Float32 | DataType::Float64 => Ok(()),
        other => plan_err!("vec_scale argument 1 must be Float32 or Float64, got {other:?}"),
    }
}

impl ScalarUDFImpl for VecScale {
    fn name(&self) -> &str {
        "vec_scale"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        validate_vec_scale_args(arg_types)?;
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let v = as_f32_list_like(&arrays[0], "vec_scale", "first")?;

        // per-row factor (a broadcast literal becomes a constant array)
        let factors: Vec<f32> = match arrays[1].data_type() {
            DataType::Float32 => arrays[1]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
                .collect(),
            DataType::Float64 => arrays[1]
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .iter()
                .map(|x| *x as f32)
                .collect(),
            other => {
                return plan_err!(
                    "vec_scale scalar factor must be Float32 or Float64, got {other:?}"
                );
            }
        };

        match &v {
            crate::expressions::common::F32ListLike::Fixed(a) => {
                let size = a.value_length() as usize;
                let mut flat: Vec<f32> = Vec::with_capacity(a.len() * size);
                let mut validity: Vec<bool> = Vec::with_capacity(a.len());
                let mut buf = vec![0.0f32; size];
                for row in 0..a.len() {
                    if a.is_null(row) {
                        validity.push(false);
                        flat.extend(std::iter::repeat_n(0.0f32, size));
                    } else {
                        validity.push(true);
                        let s = factors[row % factors.len()];
                        buf.copy_from_slice(v.value(row));
                        vec_scale(&mut buf, s);
                        flat.extend_from_slice(&buf);
                    }
                }
                let out = FixedSizeListArray::try_new(
                    child_field(&arrays[0])?,
                    size as i32,
                    Arc::new(Float32Array::from(flat)),
                    Some(NullBuffer::from(validity)),
                )?;
                Ok(ColumnarValue::Array(Arc::new(out)))
            }
            crate::expressions::common::F32ListLike::View(a) => {
                let mut flat: Vec<f32> = Vec::with_capacity(a.values().len());
                let mut buf: Vec<f32> = Vec::new();
                for row in 0..a.len() {
                    let len = a.value(row).len();
                    if a.is_null(row) {
                        // masked by the validity buffer; keep offsets valid
                        flat.extend(std::iter::repeat_n(0.0f32, len));
                    } else {
                        let s = factors[row % factors.len()];
                        buf.clear();
                        buf.extend_from_slice(v.value(row));
                        vec_scale(&mut buf, s);
                        flat.extend_from_slice(&buf);
                    }
                }
                let out = ListArray::new(
                    child_field(&arrays[0])?,
                    a.offsets().clone(),
                    Arc::new(Float32Array::from(flat)),
                    a.nulls().cloned(),
                );
                Ok(ColumnarValue::Array(Arc::new(out)))
            }
        }
    }
}

/// Child field of a vector column (`List(Float32, f)` / `FixedSizeList(f, d)`).
fn child_field(array: &ArrayRef) -> Result<Arc<Field>> {
    match array.data_type() {
        DataType::List(f) => Ok(Arc::clone(f)),
        DataType::FixedSizeList(f, _) => Ok(Arc::clone(f)),
        other => plan_err!("expected a vector column, got {other:?}"),
    }
}

/// Scalar UDF `vec_weighted_sum(w0, v0, w1, v1, ...) -> FixedSizeList<Float32, d>`:
/// the linear combination `Σ w_i · v_i` computed in one fused pass (SIMD
/// kernel in [`crate::ml::linalg`]).
///
/// The number of (scalar, vector) term pairs is fixed at plan time. A NULL
/// vector row contributes zero — the "never reached" FastRP state — so the
/// combination never propagates NULLs. Vectors must be same-sized `f32`
/// (length `d`, captured at plan time); scalars are Float32 or Float64 and
/// may vary per row.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct VecWeightedSum {
    signature: Signature,
    d: usize,
    terms: usize,
}

impl VecWeightedSum {
    pub(crate) fn new(d: usize, terms: usize) -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            d,
            terms,
        }
    }
}

fn validate_vec_weighted_sum_args(arg_types: &[DataType], terms: usize) -> Result<()> {
    if arg_types.len() != 2 * terms {
        return plan_err!(
            "vec_weighted_sum expects {} arguments ({} (weight, vector) pairs), got {}",
            2 * terms,
            terms,
            arg_types.len()
        );
    }
    for i in 0..terms {
        let (w, v) = (&arg_types[2 * i], &arg_types[2 * i + 1]);
        match w {
            DataType::Float32 | DataType::Float64 => {}
            other => {
                return plan_err!(
                    "vec_weighted_sum weight must be Float32 or Float64, got {other:?}"
                );
            }
        }
        let is_vector = match v {
            DataType::FixedSizeList(f, _) => f.data_type() == &DataType::Float32,
            DataType::List(f) => f.data_type() == &DataType::Float32,
            _ => false,
        };
        if !is_vector {
            return plan_err!(
                "vec_weighted_sum vector must be FixedSizeList<Float32> or List<Float32>, got {v:?}"
            );
        }
    }
    Ok(())
}

impl ScalarUDFImpl for VecWeightedSum {
    fn name(&self) -> &str {
        "vec_weighted_sum"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        validate_vec_weighted_sum_args(arg_types, self.terms)?;
        Ok(DataType::FixedSizeList(f32_child_field(), self.d as i32))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        // user-defined signature: accept the validated types as-is
        validate_vec_weighted_sum_args(arg_types, self.terms)?;
        Ok(arg_types.to_vec())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let n = arrays[0].len();

        // per-row factors (a broadcast literal becomes a constant array)
        let factors: Vec<Vec<f32>> = (0..self.terms)
            .map(|i| match arrays[2 * i].data_type() {
                DataType::Float32 => arrays[2 * i]
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
                    .collect(),
                DataType::Float64 => arrays[2 * i]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .map(|x| *x as f32)
                    .collect(),
                other => unreachable!("validated at plan time: {other:?}"),
            })
            .collect();

        let vecs: Vec<crate::expressions::common::F32ListLike> = (0..self.terms)
            .map(|i| as_f32_list_like(&arrays[2 * i + 1], "vec_weighted_sum", "vector"))
            .collect::<Result<_>>()?;

        let mut flat = vec![0.0f32; n * self.d];
        let mut buf = vec![0.0f32; self.d];
        let mut terms: Vec<(f32, &[f32])> = Vec::with_capacity(self.terms);
        for row in 0..n {
            terms.clear();
            for ti in 0..self.terms {
                let v = &vecs[ti];
                if !v.is_null(row) {
                    let vals = v.value(row);
                    if vals.len() != self.d {
                        return plan_err!(
                            "vec_weighted_sum: vector length {} does not match the declared dimension {}",
                            vals.len(),
                            self.d
                        );
                    }
                    terms.push((factors[ti][row % factors[ti].len()], vals));
                }
            }
            vec_weighted_sum(&mut buf, &terms);
            flat[row * self.d..(row + 1) * self.d].copy_from_slice(&buf);
        }

        Ok(ColumnarValue::Array(fsl_array(self.d, flat)))
    }
}

/// Builds an [`Expr`] computing the linear combination `Σ w_i · v_i` over
/// `(weight, vector)` term pairs (vectors may be NULL: a NULL term is zero).
pub(crate) fn vec_weighted_sum_expr(terms: &[(Expr, Expr)], d: usize) -> Expr {
    let args: Vec<Expr> = terms
        .iter()
        .flat_map(|(w, v)| vec![w.clone(), v.clone()])
        .collect();
    ScalarUDF::from(VecWeightedSum::new(d, terms.len())).call(args)
}

/// Builds an [`Expr`] scaling an `f32` vector by a scalar (per row).
pub(crate) fn vec_scale_expr(v: Expr, s: Expr) -> Expr {
    ScalarUDF::from(VecScale::new()).call(vec![v, s])
}

// ---------------- fastrp_init: scalar UDF ----------------

/// Scalar UDF `fastrp_init(id) -> FixedSizeList<Float32, d>`: the
/// deterministic sparse random projection init of FastRP.
///
/// `d` and `seed` are captured at plan time; the vector for a row is a pure
/// function of `(id, seed)` (see [`crate::ml::fastrp_init_fill`]), so init is
/// reproducible across re-scans and runs.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct FastRPInit {
    signature: Signature,
    d: usize,
    seed: u64,
}

impl FastRPInit {
    pub(crate) fn new(d: usize, seed: u64) -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Int64], Volatility::Immutable),
            d,
            seed,
        }
    }
}

impl ScalarUDFImpl for FastRPInit {
    fn name(&self) -> &str {
        "fastrp_init"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.len() != 1 || arg_types[0] != DataType::Int64 {
            return plan_err!("fastrp_init expects a single Int64 argument, got {arg_types:?}");
        }
        Ok(DataType::FixedSizeList(f32_child_field(), self.d as i32))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let ids = downcast_int64(&arrays[0], "fastrp_init", "first")?;

        let mut flat = Vec::with_capacity(ids.len() * self.d);
        let mut buf = vec![0.0f32; self.d];
        for i in 0..ids.len() {
            fastrp_init_fill(ids.value(i), self.seed, self.d, &mut buf);
            flat.extend_from_slice(&buf);
        }
        Ok(ColumnarValue::Array(fsl_array(self.d, flat)))
    }
}

/// Builds an [`Expr`] initializing the FastRP random projection vectors.
pub(crate) fn fastrp_init_expr(id: Expr, d: usize, seed: u64) -> Expr {
    ScalarUDF::from(FastRPInit::new(d, seed)).call(vec![id])
}

/// Zero-vector [`ScalarValue`] matching the concrete vector type of a column
/// (`List<Float32>` or `FixedSizeList<Float32, d>`, including the very same
/// child field).
///
/// A parquet round-trip turns `FixedSizeList` into `List`, so the literal
/// must adapt to stay type-compatible with `coalesce` / `when`.
pub(crate) fn vec_zero_scalar(vector_type: &DataType, d: usize) -> Result<ScalarValue> {
    let zeros = Arc::new(Float32Array::from(vec![0.0f32; d]));
    match vector_type {
        DataType::List(f) => {
            let arr = ListArray::new(
                Arc::clone(f),
                OffsetBuffer::new(ScalarBuffer::from(vec![0i32, d as i32])),
                zeros,
                None,
            );
            Ok(ScalarValue::List(Arc::new(arr)))
        }
        DataType::FixedSizeList(f, size) if *size as usize == d => {
            let arr = FixedSizeListArray::try_new(Arc::clone(f), *size, zeros, None)?;
            Ok(ScalarValue::FixedSizeList(Arc::new(arr)))
        }
        other => plan_err!("expected a Float32 vector column of length {d}, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use datafusion::arrow::array::{
        Array, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    };
    use datafusion::arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
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

    // ---------------- vec_sum / fastrp_init ----------------

    use crate::ml::fastrp_init_fill;
    use datafusion::logical_expr::ExprFunctionExt;

    /// `(g: Int64, v: FixedSizeList<Float32, 3>)` table from `(group, vec)` rows.
    fn group_table(rows: &[(i64, &[f32; 3])]) -> Result<DataFrame> {
        let flat: Vec<f32> = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        let groups = Int64Array::from(rows.iter().map(|(g, _)| *g).collect::<Vec<_>>());
        let schema = Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("v", DataType::FixedSizeList(f32_child_field(), 3), false),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(groups), fsl_array(3, flat)])?;
        Ok(SessionContext::new().read_batch(batch)?)
    }

    /// Serial per-group reference sums.
    fn group_sums_ref(rows: &[(i64, &[f32; 3])]) -> Vec<(i64, [f32; 3])> {
        let mut order: Vec<i64> = Vec::new();
        let mut sums: HashMap<i64, [f32; 3]> = HashMap::new();
        for (g, v) in rows {
            if !sums.contains_key(g) {
                order.push(*g);
                sums.insert(*g, [0.0; 3]);
            }
            let s = sums.get_mut(g).unwrap();
            for (t, x) in v.iter().enumerate() {
                s[t] += x;
            }
        }
        order.into_iter().map(|g| (g, sums[&g])).collect()
    }

    /// Downcast column `idx` into per-row `f32` vectors (any representation).
    async fn rows_as_f32(df: DataFrame, idx: usize) -> Result<Vec<Vec<f32>>> {
        let batches = df.collect().await?;
        let mut out = Vec::new();
        for batch in &batches {
            let col = batch.column(idx);
            let v = crate::expressions::common::as_f32_list_like(col, "test", "first")?;
            for i in 0..v.len() {
                out.push(v.value(i).to_vec());
            }
        }
        Ok(out)
    }

    #[tokio::test]
    async fn udaf_vec_sum_matches_serial_reference() -> Result<()> {
        let rows: Vec<(i64, &[f32; 3])> = vec![
            (2, &[1.0, 1.0, 1.0]),
            (1, &[10.0, 0.0, -4.0]),
            (2, &[0.5, 0.5, 0.5]),
            (1, &[1.0, 2.0, 3.0]),
            (2, &[8.25, -0.5, 0.125]),
            (0, &[7.0, 7.0, 7.0]),
        ];
        let df = group_table(&rows)?;
        let out = df.aggregate(vec![col("g")], vec![vec_sum_expr(col("v"), 3)])?;
        // group order is not guaranteed: sort by key for comparison
        let mut got: Vec<(i64, [f32; 3])> = out
            .collect()
            .await?
            .iter()
            .flat_map(|b| {
                let g = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
                let v = crate::expressions::common::as_f32_list_like(b.column(1), "test", "first")
                    .unwrap();
                (0..b.num_rows()).map(move |i| {
                    let mut a = [0.0f32; 3];
                    a.copy_from_slice(v.value(i));
                    (g.value(i), a)
                })
            })
            .collect();
        got.sort_by_key(|(g, _)| *g);

        let mut expected = group_sums_ref(&rows);
        expected.sort_by_key(|(g, _)| *g);
        assert_eq!(got, expected);
        Ok(())
    }

    #[tokio::test]
    async fn udaf_vec_sum_accepts_list_input() -> Result<()> {
        // Simulate a parquet round-trip: List<Float32> instead of FixedSizeList.
        let rows: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.5, 0.5, 0.5],
            vec![-1.0, 0.0, 1.0],
        ];
        let values = Arc::new(Float32Array::from(
            rows.iter().flatten().copied().collect::<Vec<f32>>(),
        ));
        let lists = ListArray::new(
            f32_child_field(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 3, 6, 9])),
            values,
            None,
        );
        let groups = Int64Array::from(vec![1i64, 1, 2]);
        let schema = Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("v", DataType::List(f32_child_field()), false),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(groups), Arc::new(lists)])?;
        let df = SessionContext::new().read_batch(batch)?;

        let out = df
            .aggregate(vec![col("g")], vec![vec_sum_expr(col("v"), 3)])?
            .sort(vec![col("g").sort(true, true)])?;
        let got = rows_as_f32(out, 1).await?;
        assert_eq!(got, vec![vec![1.5, 2.5, 3.5], vec![-1.0, 0.0, 1.0]]);
        Ok(())
    }

    #[tokio::test]
    async fn udaf_vec_sum_rejects_mismatched_list_lengths() -> Result<()> {
        let values = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]));
        let lists = ListArray::new(
            f32_child_field(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 3, 4])),
            values,
            None,
        );
        let groups = Int64Array::from(vec![1i64, 1]);
        let schema = Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("v", DataType::List(f32_child_field()), false),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(groups), Arc::new(lists)])?;
        let df = SessionContext::new().read_batch(batch)?;
        let result = df
            .aggregate(vec![col("g")], vec![vec_sum_expr(col("v"), 3)])?
            .collect()
            .await;
        assert!(
            result.is_err(),
            "unequal vector lengths must surface as an error"
        );
        Ok(())
    }

    #[tokio::test]
    async fn udaf_vec_sum_respects_aggregate_filter() -> Result<()> {
        // The FastRP pattern: NULL sources are excluded by an aggregate
        // FILTER (WHERE v IS NOT NULL) and must not poison the sums.
        // The input mimics a parquet round-trip: `List<Float32>` with a
        // nullable child field (not the canonical `el`).
        let flat: Vec<f32> = vec![1.0, 2.0, 3.0, 10.0, 10.0, 10.0];
        let lists = ListArray::new(
            Arc::new(Field::new("element", DataType::Float32, true)),
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 3, 3, 6])),
            Arc::new(Float32Array::from(flat)),
            // the middle row is a genuine NULL vector (never-reached source)
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let groups = Int64Array::from(vec![1i64, 1, 1]);
        let schema = Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new(
                "v",
                DataType::List(Arc::new(Field::new("element", DataType::Float32, true))),
                true,
            ),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(schema), vec![Arc::new(groups), Arc::new(lists)])?;
        let df = SessionContext::new().read_batch(batch)?;

        let agg = vec_sum_expr(col("v"), 3)
            .filter(col("v").is_not_null())
            .build()?;
        let out = df.aggregate(vec![col("g")], vec![agg])?;
        let got = rows_as_f32(out, 1).await?;
        assert_eq!(got, vec![vec![11.0, 12.0, 13.0]]);
        Ok(())
    }

    #[tokio::test]
    async fn accumulator_single_group_fallback() -> Result<()> {
        // No GROUP BY: DataFusion uses the plain Accumulator path.
        let rows: Vec<&[f32; 3]> = vec![&[1.0, 1.0, 1.0], &[2.0, 0.5, -3.0], &[4.0, 4.0, 4.0]];
        let df = group_table(&rows.iter().map(|v| (0i64, *v)).collect::<Vec<_>>())?;
        let out = df.aggregate(vec![], vec![vec_sum_expr(col("v"), 3)])?;
        let got = rows_as_f32(out, 0).await?;
        assert_eq!(got, vec![vec![7.0, 5.5, 2.0]]);
        Ok(())
    }

    // --------- direct GroupsAccumulator trait-level tests ---------

    fn fsl_batch(rows: &[[f32; 3]]) -> ArrayRef {
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        fsl_array(3, flat)
    }

    #[test]
    fn groups_accumulator_updates_and_emits_in_group_order() -> Result<()> {
        let mut acc = VecSumGroupsAccumulator::new(3);
        acc.update_batch(
            &[fsl_batch(&[
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [10.0, 0.0, 0.0],
            ])],
            &[0, 1, 0],
            None,
            2,
        )?;
        acc.update_batch(
            &[fsl_batch(&[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])],
            &[1, 0],
            None,
            2,
        )?;

        let out = acc.evaluate(EmitTo::All)?;
        let v = crate::expressions::common::as_f32_list_like(&out, "test", "first")?;
        assert_eq!(v.len(), 2);
        assert_eq!(v.value(0), &[12.0, 2.0, 2.0]);
        assert_eq!(v.value(1), &[2.5, 2.5, 2.5]);
        Ok(())
    }

    #[test]
    fn groups_accumulator_emit_first_shifts_indices() -> Result<()> {
        let mut acc = VecSumGroupsAccumulator::new(3);
        acc.update_batch(
            &[fsl_batch(&[
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ])],
            &[0, 1, 2, 3],
            None,
            4,
        )?;

        // Emit groups 0 and 1; indices 2, 3 shift down to 0, 1.
        let emitted = acc.evaluate(EmitTo::First(2))?;
        let v = crate::expressions::common::as_f32_list_like(&emitted, "test", "first")?;
        assert_eq!(v.value(0), &[1.0, 1.0, 1.0]);
        assert_eq!(v.value(1), &[2.0, 2.0, 2.0]);

        acc.update_batch(
            &[fsl_batch(&[[100.0, 100.0, 100.0], [10.0, 10.0, 10.0]])],
            &[0, 1], // old groups 2, 3
            None,
            2,
        )?;
        let out = acc.evaluate(EmitTo::All)?;
        let v = crate::expressions::common::as_f32_list_like(&out, "test", "first")?;
        assert_eq!(v.value(0), &[103.0, 103.0, 103.0]);
        assert_eq!(v.value(1), &[14.0, 14.0, 14.0]);
        Ok(())
    }

    #[test]
    fn groups_accumulator_merge_accepts_parquet_list_state() -> Result<()> {
        // Phase 1: partial sums.
        let mut partial = VecSumGroupsAccumulator::new(3);
        partial.update_batch(
            &[fsl_batch(&[[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])],
            &[0, 1],
            None,
            2,
        )?;
        let state = partial.state(EmitTo::All)?;

        // Simulate a parquet spill: FixedSizeList state comes back as List.
        let state_list =
            datafusion::arrow::compute::cast(&state[0], &DataType::List(f32_child_field()))?;

        // Phase 2: merge (as List!) into a fresh accumulator.
        let mut final_acc = VecSumGroupsAccumulator::new(3);
        final_acc.merge_batch(&[state_list], &[0, 1], 2)?;
        final_acc.update_batch(
            &[fsl_batch(&[[1.0, 1.0, 1.0], [7.0, 7.0, 7.0]])],
            &[0, 1],
            None,
            2,
        )?;

        let out = final_acc.evaluate(EmitTo::All)?;
        let v = crate::expressions::common::as_f32_list_like(&out, "test", "first")?;
        assert_eq!(v.value(0), &[2.0, 3.0, 4.0]);
        assert_eq!(v.value(1), &[7.5, 7.5, 7.5]);
        Ok(())
    }

    #[test]
    fn groups_accumulator_respects_opt_filter() -> Result<()> {
        let mut acc = VecSumGroupsAccumulator::new(3);
        let filter = BooleanArray::from(vec![true, false, true]);
        acc.update_batch(
            &[fsl_batch(&[
                [1.0, 0.0, 0.0],
                [1000.0, 1000.0, 1000.0],
                [0.0, 2.0, 0.0],
            ])],
            &[0, 0, 0],
            Some(&filter),
            1,
        )?;
        let out = acc.evaluate(EmitTo::All)?;
        let v = crate::expressions::common::as_f32_list_like(&out, "test", "first")?;
        assert_eq!(v.value(0), &[1.0, 2.0, 0.0]);
        Ok(())
    }

    #[test]
    fn groups_accumulator_convert_to_state_zeroes_filtered_rows() -> Result<()> {
        let acc = VecSumGroupsAccumulator::new(3);
        let filter = BooleanArray::from(vec![true, false, true]);
        let state = acc.convert_to_state(
            &[fsl_batch(&[
                [1.0, 1.0, 1.0],
                [9.0, 9.0, 9.0],
                [3.0, 3.0, 3.0],
            ])],
            Some(&filter),
        )?;
        let v = crate::expressions::common::as_f32_list_like(&state[0], "test", "first")?;
        assert_eq!(v.value(0), &[1.0, 1.0, 1.0]);
        assert_eq!(
            v.value(1),
            &[0.0, 0.0, 0.0],
            "filtered row -> zero identity"
        );
        assert_eq!(v.value(2), &[3.0, 3.0, 3.0]);
        Ok(())
    }

    // --------- fastrp_init ---------

    #[tokio::test]
    async fn udf_fastrp_init_is_deterministic_and_matches_kernel() -> Result<()> {
        let ids = Int64Array::from(vec![1i64, 2, 3, -7]);
        let schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids)])?;
        let df = SessionContext::new().read_batch(batch)?;

        let got = rows_as_f32(
            df.clone()
                .select(vec![fastrp_init_expr(col("id"), 4, 42)])?,
            0,
        )
        .await?;

        let mut expected = vec![0.0f32; 4];
        for (i, id) in [1i64, 2, 3, -7].iter().enumerate() {
            fastrp_init_fill(*id, 42, 4, &mut expected);
            assert_eq!(got[i], expected, "fastrp_init({id}) must match the kernel");
        }

        // Same input, different seed: different init.
        let other = rows_as_f32(df.select(vec![fastrp_init_expr(col("id"), 4, 43)])?, 0).await?;
        assert_ne!(got, other);
        Ok(())
    }

    #[test]
    fn vec_zero_scalar_matches_column_type() -> Result<()> {
        let list_type = DataType::List(f32_child_field());
        let list_zero = vec_zero_scalar(&list_type, 3)?;
        assert_eq!(list_zero.data_type(), list_type);

        let fsl_type = DataType::FixedSizeList(f32_child_field(), 3);
        let fsl_zero = vec_zero_scalar(&fsl_type, 3)?;
        assert_eq!(fsl_zero.data_type(), fsl_type);

        // Each literal holds a single row of d zeros.
        match &list_zero {
            ScalarValue::List(arr) => {
                assert_eq!(arr.len(), 1);
                let row = arr.value(0);
                let f32s = row.as_any().downcast_ref::<Float32Array>().unwrap();
                assert_eq!(f32s.values(), &[0.0f32; 3]);
            }
            other => panic!("expected List scalar, got {other:?}"),
        }
        assert!(matches!(fsl_zero, ScalarValue::FixedSizeList(_)));
        Ok(())
    }

    #[test]
    fn builders_reference_vector_udf_names() {
        assert!(format!("{}", vec_sum_expr(col("v"), 4)).contains("vec_sum"));
        assert!(format!("{}", fastrp_init_expr(col("id"), 4, 0)).contains("fastrp_init"));
        assert!(format!("{}", vec_scale_expr(col("v"), lit(2.0f32))).contains("vec_scale"));
        assert!(
            format!(
                "{}",
                vec_weighted_sum_expr(&[(lit(1.0f64), col("a")), (lit(2.0f64), col("b"))], 4)
            )
            .contains("vec_weighted_sum")
        );
    }

    // ---------------- vec_weighted_sum ----------------

    #[tokio::test]
    async fn udf_vec_weighted_sum_matches_serial_reference() -> Result<()> {
        // rows: [v0, v1, NULL]; weights: [1.5, -2.0, 0.5]
        let terms: Vec<(Expr, Expr)> = vec![
            (lit(1.5f64), col("h1")),
            (lit(-2.0f64), col("h2")),
            (lit(0.5f64), col("h3")),
        ];
        let d = 3;
        let mk = |vals: Vec<f32>| {
            FixedSizeListArray::try_new(
                f32_child_field(),
                d as i32,
                Arc::new(Float32Array::from(vals)),
                None,
            )
            .unwrap()
        };
        // h1 = [(1,2,3), (0,0,0), NULL]
        let h1 = FixedSizeListArray::try_new(
            f32_child_field(),
            d as i32,
            Arc::new(Float32Array::from(vec![
                Some(1.0f32),
                Some(2.0),
                Some(3.0),
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(9.0),
                Some(9.0),
                Some(9.0),
            ])),
            Some(NullBuffer::from(vec![true, true, false])),
        )?;
        // h2 = [(0.5,1,-1); 3 rows]
        let h2 = mk(vec![0.5, 1.0, -1.0, 0.5, 1.0, -1.0, 0.5, 1.0, -1.0]);
        // h3 = [(2,2,2); 3 rows]
        let h3 = mk(vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);

        let schema = Schema::new(vec![
            Field::new(
                "h1",
                DataType::FixedSizeList(f32_child_field(), d as i32),
                true,
            ),
            Field::new(
                "h2",
                DataType::FixedSizeList(f32_child_field(), d as i32),
                false,
            ),
            Field::new(
                "h3",
                DataType::FixedSizeList(f32_child_field(), d as i32),
                false,
            ),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(h1), Arc::new(h2), Arc::new(h3)],
        )?;
        let df = SessionContext::new().read_batch(batch)?;

        let out = df.select(vec![vec_weighted_sum_expr(&terms, d).alias("c")])?;
        let got = rows_as_f32(out, 0).await?;

        // row 0: 1.5*(1,2,3) - 2*(0.5,1,-1) + 0.5*(2,2,2) = (1.5, 2, 7.5)
        assert!(
            (got[0][0] - 1.5).abs() < 1e-6
                && (got[0][1] - 2.0).abs() < 1e-6
                && (got[0][2] - 7.5).abs() < 1e-4,
            "{:?}",
            got[0]
        );
        // row 1: 1.5*(0,0,0) - 2*(0.5,1,-1) + 0.5*(2,2,2) = (0,-1,3)
        assert!(
            (got[1][0].abs()) < 1e-6
                && (got[1][1] + 1.0).abs() < 1e-6
                && (got[1][2] - 3.0).abs() < 1e-5,
            "{:?}",
            got[1]
        );
        // row 2: h1 is NULL -> contributes zero, i.e. identical to row 1's
        // explicit zero vector: -2*(0.5,1,-1) + 0.5*(2,2,2) = (0,-1,3)
        assert_eq!(
            got[2], got[1],
            "NULL vector must behave like the zero vector"
        );
        Ok(())
    }

    #[test]
    fn udf_vec_weighted_sum_validates_args() -> Result<()> {
        // one (weight, vector) pair -> exactly 2 arguments
        let udf = VecWeightedSum::new(2, 1);
        assert!(
            udf.return_type(&[DataType::Float64, DataType::List(f32_child_field())])
                .is_ok()
        );
        // wrong arity
        assert!(udf.return_type(&[DataType::Float64]).is_err());
        // non-float scalar
        assert!(
            udf.return_type(&[DataType::Int64, DataType::List(f32_child_field())])
                .is_err()
        );
        // non-vector argument
        assert!(
            udf.return_type(&[DataType::Float64, DataType::Float32])
                .is_err()
        );
        Ok(())
    }

    // ---------------- vec_scale ----------------

    /// `(v: F32 vector of len `d`)` single-column table.
    fn vec_table(rows: Vec<Vec<Option<f32>>>, d: usize) -> Result<DataFrame> {
        let flat: Vec<Option<f32>> = rows.iter().flatten().copied().collect();
        let fsl = FixedSizeListArray::try_new(
            f32_child_field(),
            d as i32,
            Arc::new(Float32Array::from(flat)),
            None,
        )?;
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(f32_child_field(), d as i32),
            true,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(fsl)])?;
        Ok(SessionContext::new().read_batch(batch)?)
    }

    #[tokio::test]
    async fn udf_vec_scale_multiplies_by_scalar_literal() -> Result<()> {
        let df = vec_table(vec![vec![Some(1.0), Some(-2.0), Some(0.5)]], 3)?;
        let out = df
            .clone()
            .select(vec![vec_scale_expr(col("v"), lit(2.0f32)).alias("s")])?;
        let got = rows_as_f32(out, 0).await?;
        assert_eq!(got, vec![vec![2.0, -4.0, 1.0]]);

        // Float64 factor is accepted too.
        let got = rows_as_f32(
            df.select(vec![vec_scale_expr(col("v"), lit(0.5f64)).alias("s")])?,
            0,
        )
        .await?;
        assert_eq!(got, vec![vec![0.5, -1.0, 0.25]]);
        Ok(())
    }

    #[tokio::test]
    async fn udf_vec_scale_passes_nulls_through() -> Result<()> {
        // Mimics the FastRP message path: NULL states stay NULL after scaling.
        let flat: Vec<Option<f32>> = vec![Some(1.0), Some(2.0), None, None];
        let nullable_child = Arc::new(Field::new("el", DataType::Float32, true));
        let fsl = FixedSizeListArray::try_new(
            Arc::clone(&nullable_child),
            2,
            Arc::new(Float32Array::from(flat)),
            Some(NullBuffer::from(vec![true, false])),
        )?;
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(nullable_child, 2),
            true,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(fsl)])?;
        let df = SessionContext::new().read_batch(batch)?;

        let out = df.select(vec![vec_scale_expr(col("v"), lit(3.0f32)).alias("s")])?;
        let batches = out.collect().await?;
        let v =
            crate::expressions::common::as_f32_list_like(batches[0].column(0), "test", "first")?;
        assert!(!v.is_null(0));
        assert_eq!(v.value(0), &[3.0, 6.0]);
        assert!(v.is_null(1), "null vector must stay null");
        Ok(())
    }

    #[tokio::test]
    async fn udf_vec_scale_with_l2_norm_gives_unit_norm() -> Result<()> {
        // The norm_output composition: v / ||v|| has unit L2 norm.
        let df = vec_table(vec![vec![Some(3.0), Some(4.0)]], 2)?;
        let out = df.select(vec![
            vec_scale_expr(col("v"), lit(1.0f64) / l2_norm_expr(col("v"))).alias("s"),
        ])?;
        let got = rows_as_f32(out, 0).await?;
        assert!(
            (got[0][0] - 0.6).abs() < 1e-6 && (got[0][1] - 0.8).abs() < 1e-6,
            "expected (0.6, 0.8), got {:?}",
            got[0]
        );
        let norm = (got[0][0] * got[0][0] + got[0][1] * got[0][1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[tokio::test]
    async fn udf_vec_scale_preserves_list_representation() -> Result<()> {
        let values = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]));
        let lists = ListArray::new(
            f32_child_field(),
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 2, 4])),
            values,
            None,
        );
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::List(f32_child_field()),
            false,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(lists)])?;
        let df = SessionContext::new().read_batch(batch)?;

        let out = df.select(vec![vec_scale_expr(col("v"), lit(10.0f32)).alias("s")])?;
        let batches = out.collect().await?;
        assert_eq!(
            batches[0].column(0).data_type(),
            &DataType::List(f32_child_field()),
            "List in -> List out"
        );
        let v =
            crate::expressions::common::as_f32_list_like(batches[0].column(0), "test", "first")?;
        assert_eq!(v.value(0), &[10.0, 20.0]);
        assert_eq!(v.value(1), &[30.0, 40.0]);
        Ok(())
    }
}
