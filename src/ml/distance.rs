use crate::ml::linalg::{cosine_distance, l2_distance, l2_norm};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum DistanceMetric {
    L2,
    Cosine,
}

/// Nearest center to `feat` among `k` flat `centers` (`k * d` values).
///
/// Returns the center index (ties break to the first center) and the distance
/// under the metric: **squared** L2 for [`DistanceMetric::L2`] (the D^2 cost
/// the k-means|| sampling consumes) and cosine distance for
/// [`DistanceMetric::Cosine`].
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
                let dist = l2_distance(feat, &centers[(t * d)..(t * d + d)], d);
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
                c_norms[i] = l2_norm(&centers[(i * d)..(i * d + d)], d);
            }
            let x_norm = l2_norm(feat, d);

            for t in 0..k {
                let dist =
                    cosine_distance(feat, &centers[(t * d)..(t * d + d)], d, x_norm, c_norms[t]);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = t;
                }
            }

            (best_cluster, best_dist)
        }
    }
}

/// Index of the center nearest to `feat` (ties break to the first center).
///
/// Thin wrapper over [`nearest_centers`] for the assignment scalar UDF, which
/// does not need the distance value.
pub(crate) fn nearest_center(
    feat: &[f32],
    centers: &[f32],
    k: usize,
    d: usize,
    metric: DistanceMetric,
) -> usize {
    nearest_centers(feat, centers, k, d, metric).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, what: &str) {
        let scale = 1.0 + actual.abs() + expected.abs();
        assert!(
            (actual - expected).abs() <= 1e-3 * scale,
            "{what}: expected {expected}, got {actual}"
        );
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

    #[test]
    fn test_nearest_center_returns_index_only() {
        let feat = vec![0.0f32, 0.0];
        let centers = vec![5.0f32, 5.0, 1.0, 0.0];
        assert_eq!(nearest_center(&feat, &centers, 2, 2, DistanceMetric::L2), 1);
        // Tie: both centers at squared distance 1 -> first one wins.
        let tie_centers = vec![1.0f32, 0.0, -1.0, 0.0];
        assert_eq!(
            nearest_center(&feat, &tie_centers, 2, 2, DistanceMetric::L2),
            0
        );
    }
}
