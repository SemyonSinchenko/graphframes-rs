use datafusion::error::Result;
use datafusion::logical_expr::Accumulator;

use datafusion::arrow::array::ArrayRef;

use crate::expressions::common::as_f32_list_like;
use crate::ml::{DistanceMetric, nearest_centers};


#[derive(Debug)]
pub(crate) struct KMeansStepAccumulator {
    k: usize,
    d: usize,
    state: Vec<f64>,
    metric: DistanceMetric,
}

impl KMeansStepAccumulator {
    pub(crate) fn new(k: usize, d: usize) -> Self {
        Self {
            k: k,
            d: d,
            state: vec![0.0; k * d + k]
        }
    }
}

impl Accumulator for KMeansStepAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let v = as_f32_list_like(&values[0], "k_means_step", "first")?;
        let c = as_f32_list_like(&values[1], "k_means_step", "second")?;

        // centers are the same for all the rows
        let centers = c.value(0);

        // no nulls are assumed in feature (embeddings)
        for i in 0..v.len() {
            let vv = v.value(i);
            let (cluster, dist) = nearest_centers(vv, centers, self.k, self.d, self.metric);
            self.state[self.k * self.d + cluster] += dist as f64;

            for t in 0..self.d {
                self.state[t] += vv[t] as f64;
            }
        }

        Ok(())
    }
}
