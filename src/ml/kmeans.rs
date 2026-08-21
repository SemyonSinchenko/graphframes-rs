//! Out-of-core K-Means driver.
//!
//! The run itself writes nothing: every iteration is one streaming scan of
//! the features (no caching, no checkpoints) and the result — iteration
//! count, per-k total metric and centers — is returned to the caller.
//!
//! Initialization follows
//! [Spark MLlib's k-means initializationf](https://github.com/apache/spark/blob/dc46b1479450e8e656bce703d831e3aada95ea5c/mllib/src/main/scala/org/apache/spark/mllib/clustering/KMeans.scala#L387-L459):
//! one hash-selected first center, `init_steps` rounds of D^2-weighted
//! candidate sampling (via the engine's per-row `random()`, since
//! `DataFrame::sample` does not exist), then a weighted local k-means++ over
//! the candidates. Only the candidate sampling is non-deterministic across
//! runs; everything seeded in the driver is derived from `seed`.
//!
//! Bahmani et al., Scalable K-Means++, VLDB 2012
//!
//!
//! Contract: the feature column holds non-null, same-sized `f32` vectors
//! (`FixedSizeList<Float32>` or `List<Float32>`).

use datafusion::arrow::array::{Array, FixedSizeListArray, Float64Array, Int32Array, Int64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Result, plan_err};
use datafusion::functions::math::random;
use datafusion::functions_aggregate::count::count;
use datafusion::functions_aggregate::sum::sum;
use datafusion::logical_expr::Expr;
use datafusion::prelude::*;
use rand::rngs::StdRng;

/// The engine's per-row `random()` as an [`Expr`] (it is exposed as a bare
/// `ScalarUDF`, not an expression function).
fn random_expr() -> Expr {
    random().call(vec![])
}
use rand::{Rng, SeedableRng};

use crate::VERTEX_ID;
use crate::expressions::{
    as_f32_list_like, finite_axpb, kmeans_assign_expr, kmeans_cost_expr, kmeans_step_expr,
    l2_norm_expr,
};
use crate::ml::{DistanceMetric, nearest_centers};

/// `power(base, exp)` as an [`Expr`] (exposed as a bare `ScalarUDF`).
fn power_expr(base: Expr, exp: Expr) -> Expr {
    datafusion::functions::math::power().call(vec![base, exp])
}

/// Result of a single K (effective K: when the data has fewer distinct points
/// than requested, K is clamped down, mirroring Spark returning fewer
/// distinct candidate centers).
#[derive(Debug, Clone)]
pub(crate) struct KMeansRun {
    /// Effective number of centers.
    pub k: usize,
    /// Sum over all rows of the distance to the assigned center
    /// (squared L2 — i.e. inertia — for the L2 metric, cosine distance sum
    /// for the Cosine metric).
    pub total_metric: f64,
    /// Flat `k * d` center coordinates.
    pub centers: Vec<f32>,
}

/// Result of a [`KMeansBuilder`] run.
#[derive(Debug, Clone)]
pub(crate) struct KMeansResult {
    /// Lloyd iterations executed (shared by all K values; the loop stops when
    /// every K converged or `max_iter` was reached).
    pub num_iterations: usize,
    /// Feature dimension.
    pub d: usize,
    /// One run per requested K, in request order.
    pub runs: Vec<KMeansRun>,
}

struct RunState {
    k_requested: usize,
    k_eff: usize,
    centers: Vec<f32>,
    delta: f64,
    /// `sum_c ||S_c||^2 / n_c` over the last Lloyd step's cluster sums;
    l2_term: f64,
}

/// Builder for an out-of-core K-Means run over a feature column.
pub(crate) struct KMeansBuilder<'a> {
    features: &'a DataFrame,
    feature_col: String,
    ks: Vec<usize>,
    metric: DistanceMetric,
    max_iter: usize,
    tol: f64,
    init_steps: usize,
    seed: u64,
}

impl<'a> KMeansBuilder<'a> {
    pub(crate) fn new(features: &'a DataFrame, feature_col: impl Into<String>) -> Self {
        Self {
            features,
            feature_col: feature_col.into(),
            ks: Vec::new(),
            metric: DistanceMetric::L2,
            max_iter: 20,
            tol: 1e-4,
            init_steps: 2,
            seed: 42,
        }
    }

    /// Requests a single K.
    pub(crate) fn k(mut self, k: usize) -> Self {
        self.ks = vec![k];
        self
    }

    /// Requests multiple K values; all of them share one features scan per
    /// Lloyd iteration.
    pub(crate) fn k_values(mut self, ks: &[usize]) -> Self {
        self.ks = ks.to_vec();
        self
    }

    pub(crate) fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    pub(crate) fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub(crate) fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// k-means|| sampling rounds (Spark's `initializationSteps`).
    pub(crate) fn init_steps(mut self, init_steps: usize) -> Self {
        self.init_steps = init_steps;
        self
    }

    pub(crate) fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub(crate) async fn run(&self) -> Result<KMeansResult> {
        if self.ks.is_empty() {
            return plan_err!("k-means requires at least one K value");
        }
        if self.ks.iter().any(|&k| k == 0) {
            return plan_err!("k-means K values must be >= 1");
        }

        // Deduplicate K values, preserving request order.
        let mut ks: Vec<usize> = Vec::new();
        for &k in &self.ks {
            if !ks.contains(&k) {
                ks.push(k);
            }
        }
        let d = self.detect_d().await?;

        let mut rng = StdRng::seed_from_u64(self.seed);
        let centers_per_k = self.init_centers_multi(&ks, d, &mut rng).await?;
        let mut states: Vec<RunState> = ks
            .iter()
            .zip(centers_per_k)
            .map(|(&k, centers)| RunState {
                k_requested: k,
                k_eff: centers.len() / d,
                centers,
                delta: f64::MAX,
                l2_term: 0.0,
            })
            .collect();

        let mut iteration = 0usize;
        let mut t_sqnorm: Option<f64> = None;
        while iteration < self.max_iter {
            iteration += 1;
            let t = self.lloyd_step(&mut states, d).await?;
            if t_sqnorm.is_none() {
                t_sqnorm = t;
            }
            if states.iter().all(|s| s.delta < self.tol) {
                break;
            }
        }

        // Total metric per K.
        let metrics = if self.metric == DistanceMetric::L2 && iteration > 0 {
            match t_sqnorm {
                Some(t) => states.iter().map(|s| t - s.l2_term).collect(),
                None => self.total_metrics(&states, d).await?,
            }
        } else {
            self.total_metrics(&states, d).await?
        };

        Ok(KMeansResult {
            num_iterations: iteration,
            d,
            runs: states
                .into_iter()
                .zip(metrics)
                .map(|(s, metric)| KMeansRun {
                    k: s.k_eff,
                    total_metric: metric,
                    centers: s.centers,
                })
                .collect(),
        })
    }

    /// Feature dimension from the column type (`FixedSizeList`) or from the
    /// first row (`List`).
    async fn detect_d(&self) -> Result<usize> {
        let field = self
            .features
            .schema()
            .field_with_name(None, &self.feature_col)?;
        match field.data_type() {
            DataType::FixedSizeList(f, size) => {
                if f.data_type() != &DataType::Float32 {
                    return plan_err!(
                        "k-means feature column must hold Float32, got {:?}",
                        f.data_type()
                    );
                }
                Ok(*size as usize)
            }
            DataType::List(f) => {
                if f.data_type() != &DataType::Float32 {
                    return plan_err!(
                        "k-means feature column must hold Float32, got {:?}",
                        f.data_type()
                    );
                }
                let df = self
                    .features
                    .clone()
                    .select_columns(&[&self.feature_col])?
                    .limit(0, Some(1))?;
                let (flat, d) = collect_feature_flat(&df, &self.feature_col).await?;
                if flat.is_empty() {
                    return plan_err!("k-means requires at least one feature row");
                }
                Ok(d)
            }
            other => plan_err!(
                "k-means feature column must be FixedSizeList<Float32> or List<Float32>, got {other:?}"
            ),
        }
    }

    /// The first feature row: sorted by a seeded affine hash of the vertex
    /// id (deterministic given the seed; the frame always carries an Int64
    /// `id` column).
    async fn first_center(&self, r_a: i64, r_b: i64) -> Result<Vec<f32>> {
        let sorted = self.features.clone().sort(vec![
            finite_axpb(lit(r_a), col(VERTEX_ID), lit(r_b)).sort(true, true),
        ])?;
        let df = sorted
            .select_columns(&[&self.feature_col])?
            .limit(0, Some(1))?;
        let (flat, _) = collect_feature_flat(&df, &self.feature_col).await?;
        if flat.is_empty() {
            return plan_err!("k-means requires at least one feature row");
        }
        Ok(flat)
    }

    /// Spark MLlib `initKMeansParallel`, run for **all K values with every
    /// features scan shared**: one shared first center, then per sampling
    /// round one shared cost-sum scan and one shared sampling scan, one
    /// shared weighting scan — the number of init passes is independent of
    /// how many Ks are requested (per-K work is extra compute in the same
    /// scans, not extra scans).
    async fn init_centers_multi(
        &self,
        ks: &[usize],
        d: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<Vec<f32>>> {
        // Shared first center (one scan): per-K diversity comes from the
        // per-K sampling thresholds afterwards.
        let mut r_a = rng.random::<i64>();
        while r_a == 0 {
            r_a = rng.random::<i64>();
        }
        let r_b = rng.random::<i64>();
        let first = self.first_center(r_a, r_b).await?;
        let mut candidates: Vec<Vec<Vec<f32>>> = ks.iter().map(|_| vec![first.clone()]).collect();
        // A K becomes inactive once all its cost mass sits on its centers.
        let mut active: Vec<bool> = ks.iter().map(|_| true).collect();

        for _ in 0..self.init_steps {
            if !active.iter().any(|a| *a) {
                break;
            }

            // One scan: per-K cost sums under each K's accumulated centers.
            let cost_exprs: Vec<Expr> = candidates
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    sum(kmeans_cost_expr(
                        col(&self.feature_col),
                        c.len(),
                        d,
                        flatten(c),
                        self.metric,
                    ))
                    .alias(format!("__cs{i}"))
                })
                .collect();
            let batches = self
                .features
                .clone()
                .aggregate(vec![], cost_exprs)?
                .collect()
                .await?;
            let mut cost_sums = vec![0.0f64; ks.len()];
            for (i, s) in cost_sums.iter_mut().enumerate() {
                *s = batches[0]
                    .column(i)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("sum of Float64")
                    .value(0);
                if *s <= 0.0 {
                    active[i] = false;
                }
            }
            if !active.iter().any(|a| *a) {
                break;
            }

            // Thresholds as per-row expressions; the gate is their sum.
            let thresholds: Vec<Expr> = ks
                .iter()
                .enumerate()
                .map(|(i, &k)| {
                    if active[i] {
                        lit(2.0 * (k as f64))
                            * kmeans_cost_expr(
                                col(&self.feature_col),
                                candidates[i].len(),
                                d,
                                flatten(&candidates[i]),
                                self.metric,
                            )
                            / lit(cost_sums[i])
                    } else {
                        lit(0.0f64)
                    }
                })
                .collect();
            let gate = thresholds
                .iter()
                .fold(lit(0.0f64), |acc, t| acc + t.clone());
            let mut selection: Vec<Expr> = vec![col(&self.feature_col)];
            for (i, t) in thresholds.iter().enumerate() {
                selection.push(
                    random_expr()
                        .lt(t.clone() / gate.clone())
                        .alias(format!("__s{i}")),
                );
            }
            let df = self
                .features
                .clone()
                .filter(random_expr().lt(gate))?
                .select(selection)?;
            let batches = df.collect().await?;
            let mut added = 0usize;
            for b in &batches {
                let v = as_f32_list_like(b.column(0), "k_means", &self.feature_col)?;
                let flags: Vec<datafusion::arrow::array::BooleanArray> = (0..ks.len())
                    .map(|i| {
                        b.column(1 + i)
                            .as_any()
                            .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                            .expect("boolean flag")
                            .clone()
                    })
                    .collect();
                for row in 0..v.len() {
                    for (i, flag) in flags.iter().enumerate() {
                        if active[i] && !flag.is_null(row) && flag.value(row) {
                            candidates[i].push(v.value(row).to_vec());
                            added += 1;
                        }
                    }
                }
            }
            if added == 0 {
                break;
            }
        }

        // Distinct candidates per K; Ks with few enough candidates are
        // clamped right away.
        let distincts: Vec<Vec<Vec<f32>>> = candidates
            .iter()
            .map(|cs| {
                let mut distinct: Vec<Vec<f32>> = Vec::new();
                for c in cs {
                    if !distinct.iter().any(|e| e == c) {
                        distinct.push(c.clone());
                    }
                }
                distinct
            })
            .collect();
        let needs_weights: Vec<bool> = ks
            .iter()
            .zip(&distincts)
            .map(|(&k, ds)| ds.len() > k)
            .collect();

        // One shared weighting scan: group by all per-K assignment columns at
        // once and decompose the combination counts per K in the driver.
        // Clamped Ks contribute a constant column (their weights are unused).
        let group_exprs: Vec<Expr> = ks
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if needs_weights[i] {
                    kmeans_assign_expr(
                        col(&self.feature_col),
                        distincts[i].len(),
                        d,
                        flatten(&distincts[i]),
                        self.metric,
                    )
                    .alias(format!("__a{i}"))
                } else {
                    lit(0i32).alias(format!("__a{i}"))
                }
            })
            .collect();
        let batches = self
            .features
            .clone()
            .aggregate(group_exprs, vec![count(lit(1)).alias("__w")])?
            .collect()
            .await?;
        let mut weights: Vec<Vec<f64>> = ks
            .iter()
            .zip(&distincts)
            .map(|(_, ds)| vec![0.0f64; ds.len()])
            .collect();
        for b in &batches {
            let w = b
                .column(ks.len())
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("count");
            for row in 0..w.len() {
                if w.is_null(row) {
                    continue;
                }
                for (i, _) in ks.iter().enumerate() {
                    if !needs_weights[i] {
                        continue;
                    }
                    let a = b
                        .column(i)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .expect("assignment")
                        .value(row);
                    weights[i][a as usize] += w.value(row) as f64;
                }
            }
        }

        // Local weighted k-means++ per K (in-memory; candidate sets are tiny).
        let mut result = Vec::with_capacity(ks.len());
        for (i, &k) in ks.iter().enumerate() {
            if distincts[i].len() <= k {
                result.push(flatten(&distincts[i]));
            } else {
                result.push(flatten(&local_kmeans_pp(
                    rng,
                    &distincts[i],
                    &weights[i],
                    k,
                    d,
                    30,
                    self.metric,
                )));
            }
        }
        Ok(result)
    }

    /// One Lloyd iteration for all K values in a single features scan.
    ///
    /// Also fuses `sum(||x||^2)` (one norm per row — a fraction of the step's
    /// `k*d` distance work) so the final L2 metric needs no extra scan;
    /// returns that total (constant across iterations).
    async fn lloyd_step(&self, states: &mut [RunState], d: usize) -> Result<Option<f64>> {
        let (uniq, idx) = dedup_states(states);
        let mut exprs: Vec<Expr> = uniq
            .iter()
            .enumerate()
            .map(|(i, (k, c))| {
                kmeans_step_expr(col(&self.feature_col), *k, d, c.clone(), self.metric)
                    .alias(format!("__km{i}"))
            })
            .collect();
        exprs.push(
            sum(power_expr(
                l2_norm_expr(col(&self.feature_col)),
                lit(2.0f64),
            ))
            .alias("__sq"),
        );
        let batches = self
            .features
            .clone()
            .aggregate(vec![], exprs)?
            .collect()
            .await?;
        if batches.is_empty() || batches[0].num_rows() == 0 {
            return plan_err!("k-means step aggregate produced no rows");
        }
        let sq_col = batches[0].column(uniq.len());
        let t_sqnorm = sq_col
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("sum of Float64")
            .value(0);

        for (s, &ui) in states.iter_mut().zip(&idx) {
            let (_, centers) = &uniq[ui];
            let arr = batches[0].column(ui);
            let fsl = arr
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    datafusion::common::DataFusionError::Execution(
                        "k-means step result must be a FixedSizeList".to_string(),
                    )
                })?;
            let binding = fsl.value(0);
            let vals = binding
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("k_means_step returns Float64");
            let k = s.k_eff;
            let expected = k * d + k;
            if vals.len() != expected {
                return plan_err!(
                    "k-means step returned {expected} values expected, got {}",
                    vals.len()
                );
            }
            let mut new_centers = centers.clone();
            let mut delta = 0.0f64;
            let mut l2_term = 0.0f64;
            for c in 0..k {
                let cnt = vals.value(k * d + c);
                if cnt > 0.0 {
                    let mut sq_sum = 0.0f64;
                    for t in 0..d {
                        let sv = vals.value(c * d + t);
                        sq_sum += sv * sv;
                        let nv = (sv / cnt) as f32;
                        delta = delta.max((nv as f64 - centers[c * d + t] as f64).abs());
                        new_centers[c * d + t] = nv;
                    }
                    l2_term += sq_sum / cnt;
                } else {
                    log::warn!(
                        "k-means: empty cluster {c} (k={}) kept its previous center",
                        s.k_requested
                    );
                }
            }
            s.centers = new_centers;
            s.delta = delta;
            s.l2_term = l2_term;
        }
        Ok(Some(t_sqnorm))
    }

    /// Per-K total metric in one scan: `sum(k_means_cost(feat))` per distinct
    /// final state.
    async fn total_metrics(&self, states: &[RunState], d: usize) -> Result<Vec<f64>> {
        let (uniq, idx) = dedup_states(states);
        let exprs: Vec<Expr> = uniq
            .iter()
            .enumerate()
            .map(|(i, (k, c))| {
                sum(kmeans_cost_expr(
                    col(&self.feature_col),
                    *k,
                    d,
                    c.clone(),
                    self.metric,
                ))
                .alias(format!("__mt{i}"))
            })
            .collect();
        let batches = self
            .features
            .clone()
            .aggregate(vec![], exprs)?
            .collect()
            .await?;
        if batches.is_empty() || batches[0].num_rows() == 0 {
            return plan_err!("k-means metric aggregate produced no rows");
        }
        let mut metrics = vec![0.0f64; states.len()];
        for (m, &ui) in metrics.iter_mut().zip(&idx) {
            let arr = batches[0].column(ui);
            *m = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("sum of Float64")
                .value(0);
        }
        Ok(metrics)
    }
}

/// Collects the (single) selected feature column as a flat buffer plus the
/// row dimension `d` (0 rows -> empty buffer, `d == 0`).
async fn collect_feature_flat(df: &DataFrame, col_name: &str) -> Result<(Vec<f32>, usize)> {
    let batches = df.clone().collect().await?;
    let mut flat: Vec<f32> = Vec::new();
    let mut d = 0usize;
    for b in &batches {
        let v = as_f32_list_like(b.column(0), "k_means", col_name)?;
        for i in 0..v.len() {
            let row = v.value(i);
            if d == 0 {
                d = row.len();
            }
            flat.extend_from_slice(row);
        }
    }
    Ok((flat, d))
}

fn flatten(centers: &[Vec<f32>]) -> Vec<f32> {
    centers.concat()
}

/// Groups identical (k_eff, centers) states so the aggregate does not carry
/// duplicate expressions (which DataFusion would collapse into one column).
/// Returns the distinct states and, per input state, its index into them.
fn dedup_states(states: &[RunState]) -> (Vec<(usize, Vec<f32>)>, Vec<usize>) {
    let mut uniq: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut idx: Vec<usize> = Vec::with_capacity(states.len());
    for s in states {
        match uniq.iter().position(|(k, c)| {
            *k == s.k_eff
                && c.len() == s.centers.len()
                && c.iter()
                    .zip(&s.centers)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        }) {
            Some(i) => idx.push(i),
            None => {
                uniq.push((s.k_eff, s.centers.clone()));
                idx.push(uniq.len() - 1);
            }
        }
    }
    (uniq, idx)
}

/// Weighted pick: returns an index with probability proportional to `weights`.
fn pick_weighted(rng: &mut StdRng, weights: &[f64]) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return rng.random_range(0..weights.len());
    }
    let mut r = rng.random::<f64>() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// Spark MLlib `LocalKMeans.kMeansPlusPlus`: weighted k-means++ seeding
/// followed by weighted Lloyd iterations over the (small) candidate set.
fn local_kmeans_pp(
    rng: &mut StdRng,
    points: &[Vec<f32>],
    weights: &[f64],
    k: usize,
    d: usize,
    max_iter: usize,
    metric: DistanceMetric,
) -> Vec<Vec<f32>> {
    let n = points.len();

    // Weighted k-means++ seeding: first center with probability proportional
    // to its weight, each next center with probability proportional to
    // weight * D^2 under the current centers.
    let mut centers: Vec<Vec<f32>> = vec![points[pick_weighted(rng, weights)].clone()];
    while centers.len() < k {
        let flat = flatten(&centers);
        let mut costs = vec![0.0f64; n];
        let mut total = 0.0;
        for (i, p) in points.iter().enumerate() {
            let (_, dist) = nearest_centers(p, &flat, centers.len(), d, metric);
            costs[i] = dist as f64 * weights[i];
            total += costs[i];
        }
        if total <= 0.0 {
            // Every remaining point coincides with a center: fall back to a
            // weighted pick.
            centers.push(points[pick_weighted(rng, weights)].clone());
            continue;
        }
        let mut r = rng.random::<f64>() * total;
        let mut pick = n - 1;
        for (i, c) in costs.iter().enumerate() {
            r -= c;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        centers.push(points[pick].clone());
    }

    // Weighted Lloyd iterations.
    let mut assignment = vec![usize::MAX; n];
    for _ in 0..max_iter {
        let flat = flatten(&centers);
        let mut sums = vec![0.0f64; k * d];
        let mut cnts = vec![0.0f64; k];
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let (c, _) = nearest_centers(p, &flat, k, d, metric);
            if c != assignment[i] {
                changed = true;
                assignment[i] = c;
            }
            for t in 0..d {
                sums[c * d + t] += p[t] as f64 * weights[i];
            }
            cnts[c] += weights[i];
        }
        for c in 0..k {
            if cnts[c] > 0.0 {
                for t in 0..d {
                    centers[c][t] = (sums[c * d + t] / cnts[c]) as f32;
                }
            }
        }
        if !changed {
            break;
        }
    }
    centers
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int64Array, RecordBatch,
    };
    use datafusion::arrow::datatypes::{Field, Schema};
    use datafusion::datasource::MemTable;
    use std::sync::Arc;
    use std::time::Instant;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((*seed >> 8) % 1001) as f32 / 100.0 - 5.0
    }

    fn blobs(seed: u32, n_per: usize, means: &[[f32; 2]], noise_amp: f32) -> Vec<[f32; 2]> {
        let mut s = seed;
        let mut rows = Vec::new();
        for m in means {
            for _ in 0..n_per {
                let nx = lcg(&mut s) * noise_amp;
                let ny = lcg(&mut s) * noise_amp;
                rows.push([m[0] + nx, m[1] + ny]);
            }
        }
        rows
    }

    async fn features_df(ctx: &SessionContext, rows: &[[f32; 2]]) -> Result<DataFrame> {
        let schema = Schema::new(vec![
            Field::new(VERTEX_ID, DataType::Int64, false),
            Field::new(
                "feat",
                DataType::FixedSizeList(Arc::new(Field::new("el", DataType::Float32, false)), 2),
                false,
            ),
        ]);
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("el", DataType::Float32, false)),
            2,
            Arc::new(Float32Array::from(flat)),
            None,
        )?;
        let ids: Int64Array = (0..rows.len() as i64).collect();
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(ids) as ArrayRef, Arc::new(fsl) as ArrayRef],
        )?;
        let table = MemTable::try_new(Arc::new(schema), vec![vec![batch]])?;
        ctx.register_table("km_t", Arc::new(table))?;
        Ok(ctx.table("km_t").await?)
    }

    fn center_pairs(run: &KMeansRun) -> Vec<(f32, f32)> {
        let mut cs: Vec<(f32, f32)> = run.centers.chunks(2).map(|c| (c[0], c[1])).collect();
        cs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cs
    }

    #[tokio::test]
    async fn recovers_three_blobs() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(7, 30, &[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], 1.0);
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat").k(3).run().await?;
        assert_eq!(res.d, 2);
        assert_eq!(res.runs.len(), 1);
        let run = &res.runs[0];
        assert_eq!(run.k, 3);
        assert!(run.total_metric < 4000.0, "metric {}", run.total_metric);
        let want = [(-20.0f32, 20.0f32), (0.0, 0.0), (20.0, 20.0)];
        for (got, w) in center_pairs(run).iter().zip(want) {
            assert!(
                (got.0 - w.0).abs() < 2.0 && (got.1 - w.1).abs() < 2.0,
                "center {got:?} vs {w:?}"
            );
        }
        assert!(
            res.num_iterations < 20,
            "well-separated blobs must converge early, took {}",
            res.num_iterations
        );
        Ok(())
    }

    #[tokio::test]
    async fn multi_k_returns_run_per_k() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(11, 30, &[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], 1.0);
        let df = features_df(&ctx, &rows).await?;
        // Duplicate K values are deduplicated.
        let res = KMeansBuilder::new(&df, "feat")
            .k_values(&[1, 1, 3])
            .run()
            .await?;
        assert_eq!(res.runs.len(), 2);
        assert_eq!(res.runs[0].k, 1);
        assert_eq!(res.runs[1].k, 3);
        // The single-cluster center is the global mean (0, 40/3).
        let c = center_pairs(&res.runs[0]);
        assert_eq!(c.len(), 1);
        assert!((c[0].0 - 0.0).abs() < 1.0, "{:?}", c[0]);
        assert!((c[0].1 - 40.0 / 3.0).abs() < 1.0, "{:?}", c[0]);
        Ok(())
    }

    #[tokio::test]
    async fn max_iter_is_respected() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(13, 30, &[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], 1.0);
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat")
            .k(3)
            .max_iter(1)
            .run()
            .await?;
        assert_eq!(res.num_iterations, 1);
        Ok(())
    }

    #[tokio::test]
    async fn k1_converges_to_the_mean() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat")
            .seed(9)
            .tol(1e-6)
            .k(1)
            .run()
            .await?;
        let run = &res.runs[0];
        assert_eq!(run.k, 1);
        assert!((run.centers[0] - 3.0).abs() < 1e-3, "{:?}", run.centers);
        assert!((run.centers[1] - 4.0).abs() < 1e-3, "{:?}", run.centers);
        assert!(
            (run.total_metric - 16.0).abs() < 1e-3,
            "metric {}",
            run.total_metric
        );
        Ok(())
    }

    #[tokio::test]
    async fn identical_points_clamp_k_and_share_one_expression() -> Result<()> {
        // Fewer distinct points than K: K is clamped down (Spark returns the
        // distinct candidates), and both runs end up identical, which must not
        // confuse the multi-K aggregate (duplicate expressions collapse).
        let ctx = SessionContext::new();
        let rows = vec![[1.0f32, 2.0]; 6];
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat")
            .k_values(&[2, 3])
            .run()
            .await?;
        assert_eq!(res.runs.len(), 2);
        for run in &res.runs {
            assert_eq!(run.k, 1);
            // The L2 metric is algebraic (T - sum ||S_c||^2/n_c); `T` carries
            // a ~1e-7-relative sqrt-then-square round trip, so identical
            // points yield ~0, not bitwise 0.
            assert!(run.total_metric.abs() < 1e-4, "{}", run.total_metric);
            assert_eq!(run.centers, vec![1.0, 2.0]);
        }
        Ok(())
    }

    #[tokio::test]
    async fn empty_features_errors() -> Result<()> {
        let ctx = SessionContext::new();
        let df = features_df(&ctx, &[]).await?;
        let res = KMeansBuilder::new(&df, "feat").k(2).run().await;
        assert!(res.is_err(), "empty features must error");
        Ok(())
    }

    #[tokio::test]
    async fn single_init_step_still_recovers_blobs() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(19, 30, &[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], 1.0);
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat")
            .k(3)
            .init_steps(1)
            .run()
            .await?;
        let run = &res.runs[0];
        assert_eq!(run.k, 3);
        let want = [(-20.0f32, 20.0f32), (0.0, 0.0), (20.0, 20.0)];
        for (got, w) in center_pairs(run).iter().zip(want) {
            assert!((got.0 - w.0).abs() < 2.0 && (got.1 - w.1).abs() < 2.0);
        }
        Ok(())
    }

    /// The fused (algebraic) L2 metric must equal an explicit
    /// `sum(k_means_cost)` scan under the *returned* centers.
    #[tokio::test]
    async fn fused_l2_metric_matches_explicit_scan() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(23, 40, &[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], 2.0);
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat").k(3).run().await?;
        let run = &res.runs[0];
        let manual = df
            .clone()
            .aggregate(
                vec![],
                vec![
                    sum(kmeans_cost_expr(
                        col("feat"),
                        run.k,
                        res.d,
                        run.centers.clone(),
                        DistanceMetric::L2,
                    ))
                    .alias("m"),
                ],
            )?
            .collect()
            .await?;
        let explicit = manual[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        let scale = 1.0 + explicit.abs();
        assert!(
            (run.total_metric - explicit).abs() <= 1e-3 * scale,
            "fused {} vs explicit {explicit}",
            run.total_metric
        );
        Ok(())
    }

    /// The cosine fallback (final scan) is by construction the explicit
    /// definition — pin it.
    #[tokio::test]
    async fn cosine_metric_matches_explicit_scan() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(29, 30, &[[10.0, 0.0], [-5.0, 8.66], [-5.0, -8.66]], 0.1);
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat")
            .k(3)
            .metric(DistanceMetric::Cosine)
            .run()
            .await?;
        let run = &res.runs[0];
        let manual = df
            .clone()
            .aggregate(
                vec![],
                vec![
                    sum(kmeans_cost_expr(
                        col("feat"),
                        run.k,
                        res.d,
                        run.centers.clone(),
                        DistanceMetric::Cosine,
                    ))
                    .alias("m"),
                ],
            )?
            .collect()
            .await?;
        let explicit = manual[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        let scale = 1.0 + explicit.abs();
        assert!(
            (run.total_metric - explicit).abs() <= 1e-9 * scale,
            "cosine fallback {} vs explicit {explicit}",
            run.total_metric
        );
        Ok(())
    }

    /// Multi-K init produces a valid run per K and, after Lloyd, every
    /// K >= 3 covers all three blobs. (The scan *sharing* itself — one
    /// cost-sum scan, one sampling scan and one weighting scan per round
    /// regardless of how many Ks — is a property of the query plan; it is
    /// measured by the `kmeans_at_scale` bench, not assertable by wall clock
    /// at unit-test scale where per-query planning overhead dominates.)
    #[tokio::test]
    async fn multi_k_init_shared_scans() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(31, 40, &[[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]], 1.0);
        let df = features_df(&ctx, &rows).await?;

        let multi0 = KMeansBuilder::new(&df, "feat")
            .k_values(&[3, 5, 7])
            .max_iter(0)
            .run()
            .await?;
        assert_eq!(multi0.runs.len(), 3);

        // Quality after Lloyd: every K >= 3 must cover all three blobs (an
        // existence check per blob — larger K may split blobs into several
        // centers, so positional zipping would mismatch).
        let multi = KMeansBuilder::new(&df, "feat")
            .k_values(&[3, 5, 7])
            .run()
            .await?;
        assert_eq!(multi.runs.len(), 3);
        let want = [(-20.0f32, 20.0f32), (0.0, 0.0), (20.0, 20.0)];
        for run in &multi.runs {
            assert!(run.k >= 3, "k={}", run.k);
            for w in want {
                let covered = center_pairs(run)
                    .iter()
                    .any(|c| (c.0 - w.0).abs() < 3.0 && (c.1 - w.1).abs() < 3.0);
                assert!(
                    covered,
                    "k={} misses blob {w:?}: {:?}",
                    run.k,
                    center_pairs(run)
                );
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn cosine_metric_separates_directions() -> Result<()> {
        let ctx = SessionContext::new();
        let rows = blobs(17, 30, &[[10.0, 0.0], [-5.0, 8.66], [-5.0, -8.66]], 0.1);
        let df = features_df(&ctx, &rows).await?;
        let res = KMeansBuilder::new(&df, "feat")
            .k(3)
            .metric(DistanceMetric::Cosine)
            .run()
            .await?;
        let run = &res.runs[0];
        assert_eq!(run.k, 3);
        assert!(run.total_metric < 60.0, "metric {}", run.total_metric);
        Ok(())
    }
}
