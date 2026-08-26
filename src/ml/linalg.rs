//! SIMD linear-algebra kernels over `f32` vectors.
//!
//! All manual SIMD code for the ml module lives here; this module is
//! DataFusion-free on purpose — the scalar-UDF wrappers live in
//! [`crate::expressions::linalg`].
//!
//! Contract: vectors are non-null, same-sized slices; for the two-argument
//! kernels both slices have the same length.

use std::ops::Add;

use wide::f32x8;

/// Squared L2 distance `||x - c||^2`.
///
/// Kept squared (no `sqrt`) because the only hot-loop caller, the K-Means
/// nearest-center search, compares distances and `sqrt` is monotone.
pub(crate) fn l2_distance(x: &[f32], c: &[f32], d: usize) -> f32 {
    let mut acc = f32x8::splat(0.0);
    let mut t = 0;
    while t + 8 <= d {
        let xv = f32x8::from(&x[t..t + 8]);
        let cv = f32x8::from(&c[t..t + 8]);
        acc = (xv - cv).mul_add(xv - cv, acc);
        t += 8;
    }

    let mut dist = acc.reduce_add();

    while t < d {
        let xi = x[t];
        let ci = c[t];
        dist += (xi - ci) * (xi - ci);
        t += 1;
    }

    dist
}

/// True L2 norm `sqrt(||x||^2)`.
pub(crate) fn l2_norm(x: &[f32], d: usize) -> f32 {
    let mut acc = f32x8::splat(0.0);
    let mut t = 0;
    while t + 8 <= d {
        let xv = f32x8::from(&x[t..t + 8]);
        acc = xv.mul_add(xv, acc);
        t += 8;
    }

    let mut l2 = acc.reduce_add();

    while t < d {
        let xi = x[t];
        l2 += xi * xi;
        t += 1;
    }

    l2.sqrt()
}

/// Cosine distance `1 - (x . c) / (|x| |c|)`.
///
/// Zero-norm vectors score `0.0`: scikit-learn semantics
pub(crate) fn cosine_distance(x: &[f32], c: &[f32], d: usize, x_norm: f32, c_norm: f32) -> f32 {
    let denom = x_norm * c_norm;

    if denom == 0f32 {
        return 0f32;
    }

    let mut mult = f32x8::splat(0.0);
    let mut t = 0;

    while t + 8 <= d {
        let xv = f32x8::from(&x[t..t + 8]);
        let cv = f32x8::from(&c[t..t + 8]);

        mult = mult.add(xv * cv);
        t += 8;
    }

    let mut numerator = mult.reduce_add();

    while t < d {
        let xi = x[t];
        let ci = c[t];
        numerator += xi * ci;
        t += 1;
    }

    1f32 - numerator / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive scalar squared-L2 reference.
    fn l2_ref(x: &[f32], c: &[f32]) -> f32 {
        x.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum()
    }

    /// True L2 norm reference: `sqrt(sum of squares)`.
    fn norm_ref(x: &[f32]) -> f32 {
        x.iter().map(|a| a * a).sum::<f32>().sqrt()
    }

    /// Naive scalar cosine-distance reference: `1 - dot / (|x| |c|)`.
    fn cosine_ref(x: &[f32], c: &[f32]) -> f32 {
        let dot: f32 = x.iter().zip(c).map(|(a, b)| a * b).sum();
        let norm_x = norm_ref(x);
        let norm_c = norm_ref(c);
        if norm_x == 0.0 || norm_c == 0.0 {
            0.0
        } else {
            1.0 - dot / (norm_x * norm_c)
        }
    }

    fn assert_close(actual: f32, expected: f32, what: &str) {
        let scale = 1.0 + actual.abs() + expected.abs();
        assert!(
            (actual - expected).abs() <= 1e-3 * scale,
            "{what}: expected {expected}, got {actual}"
        );
    }

    /// Deterministic pseudo-random vector pair with mixed signs and magnitudes.
    fn vec_pair(d: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) % 2001) as f32 / 100.0 - 10.0
        };
        let x = (0..d).map(|_| next()).collect();
        let c = (0..d).map(|_| next()).collect();
        (x, c)
    }

    // ---------------- kernels ----------------

    #[test]
    fn test_l2_distance_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, c) = vec_pair(d, 42 + d as u32);
            assert_close(
                l2_distance(&x, &c, d),
                l2_ref(&x, &c),
                &format!("l2_distance, d={d}"),
            );
        }
    }

    #[test]
    fn test_l2_distance_identical_vectors_is_zero() {
        let x = vec![1.0f32, -2.0, 3.5, 0.0];
        assert_eq!(l2_distance(&x, &x, x.len()), 0.0);
    }

    #[test]
    fn test_l2_norm_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, _) = vec_pair(d, 42 + d as u32);
            assert_close(l2_norm(&x, d), norm_ref(&x), &format!("l2_norm, d={d}"));
        }
        // Zero vector.
        let zero = vec![0.0f32; 17];
        assert_eq!(l2_norm(&zero, 17), 0.0);
        // 3-4-5 triangle.
        let t = vec![3.0f32, 4.0];
        assert_close(l2_norm(&t, 2), 5.0, "3-4-5");
    }

    #[test]
    fn test_cosine_distance_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, c) = vec_pair(d, 7 + d as u32);
            assert_close(
                cosine_distance(&x, &c, d, norm_ref(&x), norm_ref(&c)),
                cosine_ref(&x, &c),
                &format!("cosine_distance, d={d}"),
            );
        }
    }

    #[test]
    fn test_cosine_distance_scaled_parallel_vectors_is_zero() {
        // Collinear vectors must have cosine distance 0 regardless of scale.
        let x = vec![1.0f32, 2.0, 3.0];
        let c = vec![2.0f32, 4.0, 6.0];
        assert_close(
            cosine_distance(&x, &c, 3, norm_ref(&x), norm_ref(&c)),
            0.0,
            "scaled parallel",
        );
    }

    #[test]
    fn test_cosine_distance_unit_vector_canonical_values() {
        let e1 = vec![1.0f32, 0.0];
        let e2 = vec![0.0f32, 1.0];
        assert_close(cosine_distance(&e1, &e1, 2, 1.0, 1.0), 0.0, "parallel");
        assert_close(cosine_distance(&e1, &e2, 2, 1.0, 1.0), 1.0, "orthogonal");
        assert_close(
            cosine_distance(&e1, &vec![-1.0, 0.0], 2, 1.0, 1.0),
            2.0,
            "opposite",
        );
    }

    #[test]
    fn test_cosine_distance_zero_vector_policy() {
        // Pinned decision: a zero vector scores as "identical to everything".
        let zero = vec![0.0f32, 0.0, 0.0];
        let v = vec![1.0f32, 2.0, 3.0];
        let nv = norm_ref(&v);
        assert_eq!(cosine_distance(&zero, &v, 3, 0.0, nv), 0.0);
        assert_eq!(cosine_distance(&v, &zero, 3, nv, 0.0), 0.0);
        assert_eq!(cosine_distance(&zero, &zero, 3, 0.0, 0.0), 0.0);
    }
}
