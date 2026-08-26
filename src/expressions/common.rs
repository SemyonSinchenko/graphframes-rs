use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, FixedSizeListArray, Float32Array, Float64Array,
    Int32Array, Int64Array, ListArray,
};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result};

/// Helper for other UDFs: vertexId and edgeSrc / edgeDst are all i64
pub(crate) fn downcast_int64<'a>(
    array: &'a ArrayRef,
    fname: &str,
    label: &str,
) -> Result<&'a Int64Array> {
    array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "{fname} {label} argument must be Int64, got: {:?}",
            array.data_type()
        ))
    })
}

/// Helper for other UDFs: with assumption degree < i32::MAX
/// this one is a common thing.
pub(crate) fn downcast_int32<'a>(
    array: &'a ArrayRef,
    fname: &str,
    label: &str,
) -> Result<&'a Int32Array> {
    array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "{fname} {label} argument must be Int32, got: {:?}",
            array.data_type()
        ))
    })
}

/// Read-only accessor over either a [`BinaryArray`] or a [`BinaryViewArray`].
///
/// Parquet spills round-trip `Binary` sketch columns as `BinaryView` (DataFusion
/// reads parquet `Binary` as `BinaryView`), so every sketch-consuming UDF must
/// accept both representations.
pub(crate) enum BinaryLike<'a> {
    Fixed(&'a BinaryArray),
    View(&'a BinaryViewArray),
}

impl<'a> BinaryLike<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            BinaryLike::Fixed(a) => a.len(),
            BinaryLike::View(a) => a.len(),
        }
    }

    pub(crate) fn null_count(&self) -> usize {
        match self {
            BinaryLike::Fixed(a) => a.null_count(),
            BinaryLike::View(a) => a.null_count(),
        }
    }

    pub(crate) fn is_null(&self, i: usize) -> bool {
        match self {
            BinaryLike::Fixed(a) => a.is_null(i),
            BinaryLike::View(a) => a.is_null(i),
        }
    }

    pub(crate) fn value(&self, i: usize) -> &[u8] {
        match self {
            BinaryLike::Fixed(a) => a.value(i),
            BinaryLike::View(a) => a.value(i),
        }
    }
}

/// Read-only accessor over either a [`FixedSizeListArray`] or a [`ListArray`]
/// whose elements are `Float32`.
///
/// Fixed-width float vectors are typically materialized as `FixedSizeList`,
/// while a parquet round-trip surfaces them as `List`, so every
/// vector-consuming UDF must accept both representations.
pub(crate) enum F32ListLike<'a> {
    Fixed(&'a FixedSizeListArray),
    View(&'a ListArray),
}

impl<'a> F32ListLike<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            F32ListLike::Fixed(a) => a.len(),
            F32ListLike::View(a) => a.len(),
        }
    }

    pub(crate) fn null_count(&self) -> usize {
        match self {
            F32ListLike::Fixed(a) => a.null_count(),
            F32ListLike::View(a) => a.null_count(),
        }
    }

    pub(crate) fn is_null(&self, i: usize) -> bool {
        match self {
            F32ListLike::Fixed(a) => a.is_null(i),
            F32ListLike::View(a) => a.is_null(i),
        }
    }

    /// Zero-copy `f32` slice for row `i`.
    pub(crate) fn value(&self, i: usize) -> &'a [f32] {
        match self {
            F32ListLike::Fixed(a) => {
                let child = a
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("F32ListLike child validated as Float32");
                let size = a.value_length() as usize;
                &child.values()[i * size..(i + 1) * size]
            }
            F32ListLike::View(a) => {
                let child = a
                    .values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("F32ListLike child validated as Float32");
                let offsets = a.value_offsets();
                &child.values()[offsets[i] as usize..offsets[i + 1] as usize]
            }
        }
    }
}

/// Read-only accessor over either a [`FixedSizeListArray`] or a [`ListArray`]
/// whose elements are `Float64`.
///
/// Fixed-width float vectors are typically materialized as `FixedSizeList`,
/// while a parquet round-trip surfaces them as `List`, so every
/// vector-consuming UDF must accept both representations.
pub(crate) enum F64ListLike<'a> {
    Fixed(&'a FixedSizeListArray),
    View(&'a ListArray),
}

impl<'a> F64ListLike<'a> {
    pub(crate) fn len(&self) -> usize {
        match self {
            F64ListLike::Fixed(a) => a.len(),
            F64ListLike::View(a) => a.len(),
        }
    }

    pub(crate) fn null_count(&self) -> usize {
        match self {
            F64ListLike::Fixed(a) => a.null_count(),
            F64ListLike::View(a) => a.null_count(),
        }
    }

    pub(crate) fn is_null(&self, i: usize) -> bool {
        match self {
            F64ListLike::Fixed(a) => a.is_null(i),
            F64ListLike::View(a) => a.is_null(i),
        }
    }

    /// Zero-copy `f64` slice for row `i`.
    pub(crate) fn value(&self, i: usize) -> &'a [f64] {
        match self {
            F64ListLike::Fixed(a) => {
                let child = a
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("F64ListLike child validated as Float64");
                let size = a.value_length() as usize;
                &child.values()[i * size..(i + 1) * size]
            }
            F64ListLike::View(a) => {
                let child = a
                    .values()
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("F64ListLike child validated as Float64");
                let offsets = a.value_offsets();
                &child.values()[offsets[i] as usize..offsets[i + 1] as usize]
            }
        }
    }
}

/// Helper for binary vectors
pub(crate) fn as_binary_like<'a>(
    array: &'a ArrayRef,
    fname: &str,
    label: &str,
) -> Result<BinaryLike<'a>> {
    if let Some(a) = array.as_any().downcast_ref::<BinaryArray>() {
        return Ok(BinaryLike::Fixed(a));
    }
    if let Some(a) = array.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok(BinaryLike::View(a));
    }
    Err(DataFusionError::Plan(format!(
        "{fname} {label} argument must be Binary or BinaryView, got: {:?}",
        array.data_type()
    )))
}

/// Helper for f32 vectors
pub(crate) fn as_f32_list_like<'a>(
    array: &'a ArrayRef,
    fname: &str,
    label: &str,
) -> Result<F32ListLike<'a>> {
    if let Some(a) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        if a.values().data_type() == &DataType::Float32 {
            return Ok(F32ListLike::Fixed(a));
        }
    }
    if let Some(a) = array.as_any().downcast_ref::<ListArray>() {
        if a.values().data_type() == &DataType::Float32 {
            return Ok(F32ListLike::View(a));
        }
    }
    Err(DataFusionError::Plan(format!(
        "{fname} {label} argument must be FixedSizeList(Float32) or List(Float32), got: {:?}",
        array.data_type()
    )))
}

/// Helper for f64 vectors
pub(crate) fn as_f64_list_like<'a>(
    array: &'a ArrayRef,
    fname: &str,
    label: &str,
) -> Result<F64ListLike<'a>> {
    if let Some(a) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        if a.values().data_type() == &DataType::Float64 {
            return Ok(F64ListLike::Fixed(a));
        }
    }
    if let Some(a) = array.as_any().downcast_ref::<ListArray>() {
        if a.values().data_type() == &DataType::Float64 {
            return Ok(F64ListLike::View(a));
        }
    }
    Err(DataFusionError::Plan(format!(
        "{fname} {label} argument must be FixedSizeList(Float64) or List(Float64), got: {:?}",
        array.data_type()
    )))
}
