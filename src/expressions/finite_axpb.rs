use datafusion::arrow::array::{Array, ArrayRef, Int64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, Expr, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use std::sync::Arc;

/// GF(2^64) affine hash: `(a ⊗ x) ⊕ b`.
///
/// - `⊗` is carry-less (polynomial) multiplication over GF(2^64),
///   reduced modulo the irreducible polynomial
///   `x^64 + x^4 + x^3 + x + 1` (reduction constant `0x1b`).
/// - `⊕` is XOR (the field's addition).
///
pub(crate) fn axpb(a: i64, x: i64, b: i64) -> i64 {
    let mut r: u64 = 0;
    let irrpoly: u64 = 0x1b;
    let mut current_a: u64 = a as u64;
    let mut current_x: u64 = x as u64;
    while current_x != 0 {
        if (current_x & 1) != 0 {
            r ^= current_a;
        }
        current_x >>= 1;
        if (current_a & (1u64 << 63)) != 0 {
            current_a = (current_a << 1) ^ irrpoly;
        } else {
            current_a <<= 1;
        }
    }
    (r ^ (b as u64)) as i64
}

/// DataFusion scalar UDF implementing [`axpb`] over three `Int64` columns.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct FiniteAxPlusB {
    signature: Signature,
}

impl FiniteAxPlusB {
    pub(crate) fn new() -> Self {
        Self {
            signature: Signature::uniform(3, vec![DataType::Int64], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for FiniteAxPlusB {
    fn name(&self) -> &str {
        "finite_axpb"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.len() != 3 || !arg_types.iter().all(|dt| dt == &DataType::Int64) {
            return Err(DataFusionError::Plan(format!(
                "finite_axpb expects exactly three Int64 arguments, got: {arg_types:?}"
            )));
        }
        Ok(DataType::Int64)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        if arrays.len() != 3 {
            return Err(DataFusionError::Plan(format!(
                "finite_axpb expects exactly three arguments, got: {}",
                arrays.len()
            )));
        }
        let a = downcast_int64(&arrays[0], "first")?;
        let x = downcast_int64(&arrays[1], "second")?;
        let b = downcast_int64(&arrays[2], "third")?;

        let len = arrays
            .iter()
            .map(|arr| arr.len())
            .max()
            .unwrap_or(0)
            .max(args.number_rows);

        let result: Int64Array = match (a.null_count(), x.null_count(), b.null_count()) {
            (0, 0, 0) => (0..len)
                .map(|i| {
                    Some(axpb(
                        a.value(i % a.len()),
                        x.value(i % x.len()),
                        b.value(i % b.len()),
                    ))
                })
                .collect(),
            _ => (0..len)
                .map(|i| {
                    if i >= a.len()
                        || a.is_null(i)
                        || i >= x.len()
                        || x.is_null(i)
                        || i >= b.len()
                        || b.is_null(i)
                    {
                        None
                    } else {
                        Some(axpb(a.value(i), x.value(i), b.value(i)))
                    }
                })
                .collect(),
        };

        Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
    }
}

fn downcast_int64<'a>(array: &'a ArrayRef, label: &str) -> Result<&'a Int64Array> {
    array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        DataFusionError::Plan(format!(
            "finite_axpb {label} argument must be Int64, got: {:?}",
            array.data_type()
        ))
    })
}

/// Builds an [`Expr`] that applies `finite_axpb(a, x, b)` to three
/// `Int64` expressions.
///
/// Prefer this helper over manual `ScalarUDF` construction inside the
/// algorithm code (e.g. `finite_axpb(lit(r_a), col("dst"), lit(r_b))`).
pub(crate) fn finite_axpb(a: Expr, x: Expr, b: Expr) -> Expr {
    ScalarUDF::from(FiniteAxPlusB::new()).call(vec![a, x, b])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[test]
    fn test_axpb_zero_multiplier_is_b() {
        // 0 ⊗ x = 0 in GF(2^64), so the result is just `b`.
        assert_eq!(axpb(0, 12345, -1), -1);
        assert_eq!(axpb(0, 0, 42), 42);
        assert_eq!(axpb(0, i64::MIN, i64::MAX), i64::MAX);
    }

    #[test]
    fn test_axpb_zero_input_is_b() {
        // The loop body never runs when `x == 0`, so `r == 0` and the
        // result is `b`.
        assert_eq!(axpb(7, 0, 99), 99);
        assert_eq!(axpb(i64::MIN, 0, -3), -3);
    }

    #[test]
    fn test_axpb_identity_multiplier() {
        // `a == 1` copies `x` into `r` bit-for-bit (the shift of `current_a`
        // only matters once it exceeds `1 << 63`, which the single `x` bit
        // under `1 << 63` never triggers). Result: `x ^ b`.
        for (x, b) in [(0i64, 0i64), (1, 0), (255, 0), (12345, -1), (1, 0xdeadbeef)] {
            assert_eq!(axpb(1, x, b), x ^ b, "axpb(1, {x}, {b})");
        }
    }

    #[test]
    fn test_axpb_commutativity_of_addition() {
        // GF(2^64) addition is XOR and is commutative, so `axpb(a, x, b)`
        // should equal `axpb(a, x, 0) ^ b` for any `b`.
        let h = axpb(0x1234_5678_9abc_def0, 987654321, 0);
        for b in [0i64, 1, -1, 0x0f0f_0f0f_0f0f_0f0f] {
            assert_eq!(axpb(0x1234_5678_9abc_def0, 987654321, b), h ^ b);
        }
    }

    #[test]
    fn test_axpb_reduction_path_bit63() {
        // Force `current_a` through the `0x1b` reduction branch by making
        // the loop run while `current_a` already has bit 63 set. With
        // `a == i64::MIN` (bit 63 set) and `x == 1` the loop runs once:
        //   r ^= a                                  -> r = a
        //   current_x >>= 1                         -> 0 (loop ends)
        //   bit 63 of a set, so current_a = (a<<1)^0x1b  (unused afterwards)
        //   final = r ^ b
        let a = i64::MIN; // 0x8000_0000_0000_0000
        assert_eq!(axpb(a, 1, 0), a);

        // Two iterations with `a` having bit 63 set, to actually exercise
        // the `0x1b` reduction inside the multiply accumulation. Hand-traced:
        //   iter1 (x-bit=1): r ^= a                       -> r = 0x8000_0000_0000_0000
        //                    current_a = (a<<1) ^ 0x1b    -> 0x1b   (top bit shifted out, then reduced)
        //   iter2 (x-bit=1): r ^= current_a               -> r = 0x8000_0000_0000_001b
        //   x exhausted; result = r ^ b
        assert_eq!(axpb(a, 0b11, 0), a ^ 0x1b);
    }

    #[test]
    fn test_axpb_scalar_roundtrip_compose() {
        // The back-pass composes two affine maps:
        //   accA' = axpb(oldAccA, poppedA, 0)
        //   accB' = axpb(oldAccA, poppedB, accB)
        // Sanity-check that composing the identity affine map (a=1, b=0)
        // with an arbitrary map leaves it unchanged.
        let acc_a = 1i64;
        let acc_b = 0i64;
        let popped_a = 0xdeadbeef;
        let popped_b = 42;
        let new_acc_a = axpb(acc_a, popped_a, 0);
        let new_acc_b = axpb(acc_a, popped_b, acc_b);
        assert_eq!(new_acc_a, popped_a);
        assert_eq!(new_acc_b, popped_b ^ acc_b);
    }

    #[tokio::test]
    async fn test_axpb_udf_dataframe() -> Result<()> {
        let df = dataframe!(
            "a" => vec![0i64, 1, 7, i64::MIN],
            "x" => vec![12345i64, 99, 256, 1],
            "b" => vec![0i64, -1, 42, 0],
        )?;

        let result = df
            .select(vec![finite_axpb(col("a"), col("x"), col("b")).alias("h")])?
            .collect()
            .await?;

        let arr = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let a = [0i64, 1, 7, i64::MIN];
        let x = [12345i64, 99, 256, 1];
        let b = [0i64, -1, 42, 0];
        for (i, (&av, (&xv, &bv))) in a.iter().zip(x.iter().zip(b.iter())).enumerate() {
            assert_eq!(arr.value(i), axpb(av, xv, bv), "row {i}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_axpb_udf_return_type_validation() {
        let udf = FiniteAxPlusB::new();
        assert_eq!(
            udf.return_type(&[DataType::Int64, DataType::Int64, DataType::Int64])
                .unwrap(),
            DataType::Int64
        );
        // Wrong arity / wrong type must error.
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Int64])
                .is_err()
        );
        assert!(
            udf.return_type(&[DataType::Int64, DataType::Utf8, DataType::Int64])
                .is_err()
        );
    }
}
