use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Int32Array, Int64Array,
};
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
