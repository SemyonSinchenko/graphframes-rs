use datafusion::arrow::array::{ArrayRef, Int64Array};
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
