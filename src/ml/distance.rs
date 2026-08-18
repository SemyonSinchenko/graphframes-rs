use std::ops::Add;

use wide::f32x8;

#[derive(Debug, Clone, Copy)]
pub(crate) enum DistanceMetric {
    L2,
    Cosine,
}

fn l2(x: &[f32], c: &[f32], d: usize) -> f32 {
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

fn l2norm(x: &[f32], d: usize) -> f32 {
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

fn cosine(x: &[f32], c: &[f32], d: usize, x_norm: f32, c_norm: f32) -> f32 {
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

pub(crate) fn nearest_centers(
    feat: &[f32],
    centers: &[f32],
    k: usize,
    d: usize,
    metric: DistanceMetric,
) -> (usize, f32) {
    let mut best_cluster = 0usize;
    let mut best_dist = f32::MAX;

    match metric {
        DistanceMetric::L2 => {
            for t in 0..k {
                let dist = l2(feat, &centers[(t * d)..(t * d + d)], d);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = t;
                }
            }

            (best_cluster, best_dist)
        }
        DistanceMetric::Cosine => {
            let mut c_norms = vec![0.0f32; k];
            for i in 0..k {
                c_norms[i] = l2norm(&centers[(i * d)..(i * d + d)], d);
            }
            let x_norm = l2norm(feat, d);

            for t in 0..k {
                let dist = cosine(feat, &centers[(t * d)..(t * d + d)], d, x_norm, c_norms[t]);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = t;
                }
            }

            (best_cluster, best_dist)
        }
    }
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

    #[test]
    fn test_l2_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, c) = vec_pair(d, 42 + d as u32);
            assert_close(l2(&x, &c, d), l2_ref(&x, &c), &format!("l2, d={d}"));
        }
    }

    #[test]
    fn test_l2_identical_vectors_is_zero() {
        let x = vec![1.0f32, -2.0, 3.5, 0.0];
        assert_eq!(l2(&x, &x, x.len()), 0.0);
    }

    #[test]
    fn test_l2norm_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, _) = vec_pair(d, 42 + d as u32);
            assert_close(l2norm(&x, d), norm_ref(&x), &format!("l2norm, d={d}"));
        }
        // Zero vector.
        let zero = vec![0.0f32; 17];
        assert_eq!(l2norm(&zero, 17), 0.0);
        // 3-4-5 triangle.
        let t = vec![3.0f32, 4.0];
        assert_close(l2norm(&t, 2), 5.0, "3-4-5");
    }

    #[test]
    fn test_cosine_matches_scalar_reference() {
        for d in [1usize, 3, 7, 8, 9, 15, 16, 17, 64, 100] {
            let (x, c) = vec_pair(d, 7 + d as u32);
            assert_close(
                cosine(&x, &c, d, norm_ref(&x), norm_ref(&c)),
                cosine_ref(&x, &c),
                &format!("cosine, d={d}"),
            );
        }
    }

    #[test]
    fn test_cosine_scaled_parallel_vectors_is_zero() {
        // Collinear vectors must have cosine distance 0 regardless of scale.
        let x = vec![1.0f32, 2.0, 3.0];
        let c = vec![2.0f32, 4.0, 6.0];
        assert_close(
            cosine(&x, &c, 3, norm_ref(&x), norm_ref(&c)),
            0.0,
            "scaled parallel",
        );
    }

    #[test]
    fn test_cosine_unit_vector_canonical_values() {
        let e1 = vec![1.0f32, 0.0];
        let e2 = vec![0.0f32, 1.0];
        assert_close(cosine(&e1, &e1, 2, 1.0, 1.0), 0.0, "parallel");
        assert_close(cosine(&e1, &e2, 2, 1.0, 1.0), 1.0, "orthogonal");
        assert_close(cosine(&e1, &vec![-1.0, 0.0], 2, 1.0, 1.0), 2.0, "opposite");
    }

    #[test]
    fn test_cosine_zero_vector_policy() {
        // Pinned decision: a zero vector scores as "identical to everything".
        let zero = vec![0.0f32, 0.0, 0.0];
        let v = vec![1.0f32, 2.0, 3.0];
        let nv = norm_ref(&v);
        assert_eq!(cosine(&zero, &v, 3, 0.0, nv), 0.0);
        assert_eq!(cosine(&v, &zero, 3, nv, 0.0), 0.0);
        assert_eq!(cosine(&zero, &zero, 3, 0.0, 0.0), 0.0);
    }

    #[test]
    fn test_nearest_centers_l2_picks_argmin() {
        let feat = vec![0.0f32, 0.0];
        let centers = vec![1.0f32, 1.0, 10.0, 0.0, 5.0, 5.0];
        let (best, dist) = nearest_centers(&feat, &centers, 3, 2, DistanceMetric::L2);
        assert_eq!(best, 0);
        assert_close(dist, 2.0, "squared L2 to center 0");
    }

    #[test]
    fn test_nearest_centers_metrics_disagree_on_purpose() {
        // Unit vectors keep cosine exact: c0 is orthogonal (cos dist 1.0), c1
        // is parallel (cos dist 0.0) but far in L2. L2 must pick 0, cosine 1.
        let feat = vec![1.0f32, 0.0];
        let centers = vec![0.0f32, 1.0, 1000.0, 0.0];
        let (l2_best, _) = nearest_centers(&feat, &centers, 2, 2, DistanceMetric::L2);
        let (cos_best, _) = nearest_centers(&feat, &centers, 2, 2, DistanceMetric::Cosine);
        assert_eq!(l2_best, 0);
        assert_eq!(cos_best, 1);
    }

    #[test]
    fn test_nearest_centers_tie_breaks_to_first() {
        let feat = vec![0.0f32, 0.0];
        let centers = vec![1.0f32, 0.0, 1.0, 0.0];
        let (best, dist) = nearest_centers(&feat, &centers, 2, 2, DistanceMetric::L2);
        assert_eq!(best, 0);
        assert_close(dist, 1.0, "squared L2 to the first center");
    }

    #[test]
    fn test_nearest_centers_single_center() {
        let feat = vec![3.0f32, 4.0];
        let centers = vec![0.0f32, 0.0];
        let (best, dist) = nearest_centers(&feat, &centers, 1, 2, DistanceMetric::L2);
        assert_eq!(best, 0);
        assert_close(dist, 25.0, "3-4-5 triangle, squared");
    }
}
