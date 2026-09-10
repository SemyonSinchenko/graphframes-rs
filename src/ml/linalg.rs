//! SIMD linear-algebra kernels and pure `f32` vector helpers.
//!
//! All manual SIMD code for the ml module lives here; this module is
//! DataFusion-free on purpose — the scalar-UDF / UDAF wrappers live in
//! [`crate::expressions::linalg`].
//!
//! Contract: vectors are non-null, same-sized slices; for the two-argument
//! kernels both slices have the same length.

use std::ops::Add;

use wide::f32x8;

/// Accumulate `v` into `acc` in place: `acc[i] += v[i]` for every `i`.
///
/// Hot path of the `vec_sum` group accumulator: both slices have the same
/// length (the equal-size, non-null vector contract of this module).
pub(crate) fn vec_add(acc: &mut [f32], v: &[f32]) {
    debug_assert_eq!(
        acc.len(),
        v.len(),
        "vec_add slices must have the same length"
    );
    let d = acc.len().min(v.len());

    let mut t = 0;
    while t + 8 <= d {
        let av = f32x8::from(&acc[t..t + 8]);
        let vv = f32x8::from(&v[t..t + 8]);
        acc[t..t + 8].copy_from_slice((av + vv).as_array_ref());
        t += 8;
    }

    while t < d {
        acc[t] += v[t];
        t += 1;
    }
}

/// Scale `v` in place: `v[i] *= s` for every `i`.
///
/// Used by the `vec_scale` scalar UDF (per-step FastRP normalization and the
/// final output normalization).
pub(crate) fn vec_scale(v: &mut [f32], s: f32) {
    let d = v.len();
    let sv = f32x8::splat(s);

    let mut t = 0;
    while t + 8 <= d {
        let vv = f32x8::from(&v[t..t + 8]);
        v[t..t + 8].copy_from_slice((vv * sv).as_array_ref());
        t += 8;
    }

    while t < d {
        v[t] *= s;
        t += 1;
    }
}

/// Weighted sum of `terms` into `out`: `out[i] = Σ (w * v[i])`.
///
/// Fused form of "scale every vector, then add them up": one pass, one
/// output buffer. Hot path of the `vec_weighted_sum` scalar UDF (the final
/// linear combination of FastRP iterates). All vectors have the same length
/// (the equal-size, non-null contract of this module).
pub(crate) fn vec_weighted_sum(out: &mut [f32], terms: &[(f32, &[f32])]) {
    debug_assert!(
        terms.iter().all(|(_, v)| v.len() == out.len()),
        "vec_weighted_sum terms must have the same length as out"
    );
    let d = out.len();

    let mut t = 0;
    while t + 8 <= d {
        let mut acc = f32x8::splat(0.0);
        for (w, v) in terms {
            acc = f32x8::from(&v[t..t + 8]).mul_add(f32x8::splat(*w), acc);
        }
        out[t..t + 8].copy_from_slice(acc.as_array_ref());
        t += 8;
    }

    while t < d {
        let mut acc = 0.0f32;
        for (w, v) in terms {
            acc = v[t].mul_add(*w, acc);
        }
        out[t] = acc;
        t += 1;
    }
}

/// Deterministic sparse random projection vector of a single vertex — the
/// `U^(0)` init of FastRP (Chen & King, CIKM 2019).
///
/// Entries are drawn from `{−1, 0, +1}`: `+1` and `−1` each with probability
/// `1 / (2·√q)`, zero otherwise, where `q` is the largest power of two not
/// exceeding `d` (the paper's `2^d*`). The result is a pure function of
/// `(id, seed)`, so re-scans and re-runs reproduce the same init.
pub(crate) fn fastrp_init_fill(id: i64, seed: u64, d: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), d, "fastrp_init_fill output must have length d");
    let q = if d.is_power_of_two() {
        d
    } else {
        d.next_power_of_two() >> 1
    };
    let inv = 1.0f32 / (q as f32).sqrt(); // per-sign probability

    let mut state = splitmix64(seed ^ id.cast_unsigned().wrapping_mul(GOLDEN));
    for slot in out.iter_mut().take(d) {
        let u = splitmix_unit(&mut state);
        *slot = if u < 0.5 * inv {
            1.0
        } else if u < inv {
            -1.0
        } else {
            0.0
        };
    }
}

/// `2^64 / φ`, the SplitMix64 golden-gamma constant.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 next-state mixer (used as a tiny dependency-free RNG).
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `f32` in `[0, 1)` from a SplitMix64 stream (53 random bits).
fn splitmix_unit(state: &mut u64) -> f32 {
    let z = splitmix64(*state);
    *state = z;
    (z >> 11) as f32 / (1u64 << 53) as f32
}

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

    // ---------------- vec_add / fastrp_init ----------------

    /// Naive scalar `acc += v` reference.
    fn vec_add_ref(acc: &mut [f32], v: &[f32]) {
        for (a, b) in acc.iter_mut().zip(v) {
            *a += *b;
        }
    }

    #[test]
    fn test_vec_add_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, c) = vec_pair(d, 11 + d as u32);
            let mut acc = x.clone();
            let mut acc_ref = x.clone();
            vec_add(&mut acc, &c);
            vec_add_ref(&mut acc_ref, &c);
            assert_eq!(acc, acc_ref, "vec_add, d={d}");
        }
    }

    #[test]
    fn test_vec_add_repeated_accumulation() {
        let d = 20;
        let (_, c) = vec_pair(d, 5);
        let mut acc = vec![0.0f32; d];
        for _ in 0..7 {
            vec_add(&mut acc, &c);
        }
        let mut expected = vec![0.0f32; d];
        for _ in 0..7 {
            vec_add_ref(&mut expected, &c);
        }
        assert_eq!(acc, expected);
    }

    /// Naive scalar `v *= s` reference.
    fn vec_scale_ref(v: &mut [f32], s: f32) {
        for x in v.iter_mut() {
            *x *= s;
        }
    }

    #[test]
    fn test_vec_scale_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, _) = vec_pair(d, 3 + d as u32);
            let mut v = x.clone();
            let mut v_ref = x.clone();
            let s = 0.375f32; // exact in f32
            vec_scale(&mut v, s);
            vec_scale_ref(&mut v_ref, s);
            assert_eq!(v, v_ref, "vec_scale, d={d}");
        }
        // identity and zero scale
        let mut v = vec![1.0f32, -2.0, 3.5];
        vec_scale(&mut v, 1.0);
        assert_eq!(v, vec![1.0, -2.0, 3.5]);
        vec_scale(&mut v, 0.0);
        assert_eq!(v, vec![0.0; 3]);
    }

    #[test]
    fn test_vec_weighted_sum_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 16, 17, 64, 100] {
            let (a, _) = vec_pair(d, 21);
            let (b, _) = vec_pair(d, 22);
            let (c, _) = vec_pair(d, 23);
            let terms = vec![
                (0.5f32, a.as_slice()),
                (-1.25, b.as_slice()),
                (2.0, c.as_slice()),
            ];

            let mut out = vec![0.0f32; d];
            vec_weighted_sum(&mut out, &terms);

            let mut expected = vec![0.0f32; d];
            for (w, v) in &terms {
                for i in 0..d {
                    expected[i] = v[i].mul_add(*w, expected[i]);
                }
            }
            // wide's mul_add may be emulated (not hardware-fused), so the
            // last bits can differ from scalar f32::mul_add
            for i in 0..d {
                assert_close(
                    out[i],
                    expected[i],
                    &format!("vec_weighted_sum, d={d}, i={i}"),
                );
            }
        }
        // no terms at all -> zero vector
        let mut out = vec![9.0f32; 5];
        vec_weighted_sum(&mut out, &[]);
        assert_eq!(out, vec![0.0; 5]);
    }

    #[test]
    fn test_fastrp_init_is_deterministic_and_in_domain() {
        for d in [1usize, 2, 63, 64, 65, 256] {
            let mut a = vec![0.0f32; d];
            let mut b = vec![0.0f32; d];
            fastrp_init_fill(1234, 42, d, &mut a);
            fastrp_init_fill(1234, 42, d, &mut b);
            assert_eq!(a, b, "same (id, seed) must reproduce the init, d={d}");
            assert!(
                a.iter().all(|x| *x == -1.0 || *x == 0.0 || *x == 1.0),
                "entries must be sparse ternary, d={d}"
            );
        }
    }

    #[test]
    fn test_fastrp_init_varies_with_id_and_seed() {
        let d = 256;
        let mut base = vec![0.0f32; d];
        fastrp_init_fill(7, 42, d, &mut base);

        let mut other = vec![0.0f32; d];
        fastrp_init_fill(8, 42, d, &mut other);
        assert_ne!(base, other, "different ids must init differently");

        fastrp_init_fill(7, 43, d, &mut other);
        assert_ne!(base, other, "different seeds must init differently");
    }

    #[test]
    fn test_fastrp_init_density_is_sparse() {
        // density 1/sqrt(q) per sign with q = largest power of two <= d;
        // for d=1024 -> q=1024 -> ~3/4 of the entries are zero. Both extremes
        // are astronomically unlikely ((3/4)^1024 and (1/4)^1024).
        let d = 1024;
        let mut v = vec![0.0f32; d];
        fastrp_init_fill(3, 42, d, &mut v);
        let non_zero = v.iter().filter(|x| **x != 0.0).count();
        assert!(non_zero > 0, "expected some ±1 entries");
        assert!(non_zero < d, "expected some zero entries");
    }
}
