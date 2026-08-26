mod distance;
mod kmeans;
mod linalg;

pub use distance::DistanceMetric;
pub(crate) use distance::{nearest_center, nearest_centers};
pub use kmeans::{KMeansBuilder, KMeansResult, KMeansRun};
pub(crate) use linalg::{cosine_distance, l2_distance, l2_norm};
