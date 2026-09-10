//! FastRP (Fast Random Projection) vertex embeddings.
//!
//! A simplified FastRP (Chen & King, CIKM 2019): every vertex starts from a
//! deterministic sparse random projection vector `H_0 ∈ {−1, 0, +1}^D`
//! (see `fastrp_init`), then `K` rounds of message propagation sum the
//! source vectors into every destination:
//!
//! ```text
//! H_{t+1}(dst) = Σ_{(src, dst) ∈ E} H_t(src)
//! ```
//!
//! Normalization (per iteration): the propagated vector of each source may
//! be divided by its out-degree (linear, `L1`) or by the square
//! root of its out-degree. The out-degree is computed once and
//! attached to the edge table, so the loop needs no extra joins.
//!
//! Deviation from the paper, on purpose: no concatenation of iterates. The
//! final embedding is the weighted linear combination
//!
//! ```text
//! H = Σ_{t=1..K} w_t · H_t
//! ```
//!
//! (`iteration_weights`, length `K`, default all ones). The random init
//! `H_0` always stays out of the combination — it exists only as the message
//! source of the first iteration. A zero weight drops the corresponding
//! iterate from the pipeline entirely (it is never joined at the end).
//!
//! Unreached vertices keep no state at all during the loop (the aggregate
//! only emits vertices that received a message, which is the sum identity)
//! and are filled with the zero vector in one final streaming pass over the
//! output.

use datafusion::{
    arrow::datatypes::DataType,
    dataframe::DataFrameWriteOptions,
    error::{DataFusionError, Result},
    execution::object_store::ObjectStoreUrl,
    functions::math::sqrt,
    functions_aggregate::count::count,
    object_store::path::Path,
    prelude::*,
};

use crate::{
    EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID,
    expressions::{
        fastrp_init_expr, l2_norm_expr, vec_scale_expr, vec_sum_expr, vec_weighted_sum_expr,
        vec_zero_scalar,
    },
    memory::{CheckpointConfig, ParquetCheckpointer},
    utils::{GraphFramesConfig, scoped_ctx},
};

/// Name of the embedding column produced by the algorithm.
pub(crate) const EMBEDDING: &str = "embedding";

/// Name of the per-source out-degree column (only present when a per-step
/// normalization is enabled; attached to the edge table so the loop needs
/// no extra joins).
const DEGREE: &str = "__fastrp_degree";

/// Name of the temporary L2-norm column of the output pass.
const NORM: &str = "__fastrp_norm";

/// Name of the running accumulator column of the combine fold.
const ACC: &str = "__fastrp_acc";

/// Name of the accumulator's id column during the combine joins.
const ACC_VID: &str = "__fastrp_acc_vid";

/// Per-iteration normalization of the propagated vectors, applied to every
/// message: each source vector is divided by a function of the source
/// out-degree before the group-by sum (`H_{t+1} = S_norm · H_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastRPNormalization {
    /// Plain sum, no normalization (`S = A`).
    None,
    /// Linear normalization (`S = A · D⁻¹` in the paper): every propagated
    /// vector is divided by the out-degree of its source.
    L1,
    /// Square normalization (`S = A · D^(-1/2)` in the paper): every
    /// propagated vector is divided by the square root of the source
    /// out-degree.
    L2,
}

/// Builder for FastRP embeddings.
///
/// Required parameter: `dim` (`D`); `iterations` (`K`) defaults to 4.
pub struct FastRPBuilder<'a> {
    graph: &'a GraphFrame,
    dim: usize,
    iterations: usize,
    seed: u64,
    normalization: FastRPNormalization,
    norm_output: bool,
    iteration_weights: Option<Vec<f64>>,
    checkpoint_config: CheckpointConfig,
}

impl<'a> FastRPBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        FastRPBuilder {
            graph: graph,
            dim: 0,        // must be set explicitly
            iterations: 4, // a few hops mix the random projections well
            seed: 42,
            normalization: FastRPNormalization::None,
            norm_output: false,
            iteration_weights: None,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    /// Embedding dimension `D` (required).
    pub fn dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }

    /// Number of propagation iterations `K` (default: 4).
    pub fn iterations(mut self, v: usize) -> Self {
        self.iterations = v;
        self
    }

    /// Seed for the per-vertex random projections. The init is a pure
    /// function of `(seed, id)`, so runs are reproducible.
    pub fn seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Per-iteration normalization of the propagated vectors (default:
    /// [`FastRPNormalization::None`]).
    pub fn normalization(mut self, v: FastRPNormalization) -> Self {
        self.normalization = v;
        self
    }

    /// L2-normalize the final embeddings to unit length (default: `false`).
    /// Zero vectors stay zero, so downstream cosine/K-Means consumers never
    /// see NaN.
    pub fn norm_output(mut self, v: bool) -> Self {
        self.norm_output = v;
        self
    }

    /// Per-iterate weights of the final linear combination
    /// `H = Σ w_t · H_t` over `H_1..H_K`. Must have exactly `K` entries
    /// (default: all ones, i.e. sum every iterate). The random init `H_0`
    /// never enters the combination. Iterate `t` with `w_t = 0` is dropped
    /// from the pipeline entirely (not joined at the end).
    pub fn iteration_weights(mut self, v: Vec<f64>) -> Self {
        self.iteration_weights = Some(v);
        self
    }

    /// Set the object store URL
    pub fn with_checkpoint_store(mut self, store_url: ObjectStoreUrl) -> Self {
        self.checkpoint_config.store_url = store_url;
        self
    }

    /// Set the checkpoint directory
    pub fn set_checkpoint_dir(mut self, dir: Path) -> Self {
        self.checkpoint_config.dir = dir;
        self
    }

    /// Run FastRP and write `[VERTEX_ID, EMBEDDING]` parquet to `output`.
    ///
    /// The embedding column is a non-null `f32` vector column of length `D`
    /// (`FixedSizeList<Float32, D>`, surfaced as `List<Float32>` by a parquet
    /// round-trip — all vector consumers in this crate accept both).
    pub async fn run(self, ctx: &SessionContext, output: &str) -> Result<usize> {
        if self.dim == 0 {
            return Err(DataFusionError::Plan(
                "FastRP requires a positive embedding dimension: set `dim(D)`".to_string(),
            ));
        }
        let weights = match &self.iteration_weights {
            Some(w) => w.clone(),
            None => vec![1.0; self.iterations],
        };
        if weights.len() != self.iterations {
            return Err(DataFusionError::Plan(format!(
                "iteration_weights must have exactly {} entries (one per iteration, H_1..H_{}), got {}",
                self.iterations,
                self.iterations,
                weights.len()
            )));
        }

        let gf_config = ctx
            .state()
            .config()
            .options()
            .extensions
            .get::<GraphFramesConfig>()
            .cloned()
            .unwrap_or_default();

        let ctx = &scoped_ctx(ctx, gf_config.prefer_smj);
        self.checkpoint_config.validate_output(output)?;

        let run_id = uuid::Uuid::new_v4().to_string();
        log::info!("start FastRP with ID {run_id}");

        // Only the id (and the derived embedding) should travel through the
        // iterations; attribute columns would inflate every shuffle.
        let vertices = self.graph.vertices.clone().select_columns(&[VERTEX_ID])?;
        let mut edges = self
            .graph
            .edges
            .clone()
            .select_columns(&[EDGE_SRC, EDGE_DST])?;

        // Per-step normalization scales every propagated vector by a factor
        // derived from the source out-degree. The degree is computed once and
        // attached to the edge table (every edge has a source, so the inner
        // join is exact and no degree-zero division can happen).
        if self.normalization != FastRPNormalization::None {
            let degree_vid = "__fastrp_degree_vid";
            let degrees = edges.clone().aggregate(
                vec![col(EDGE_SRC).alias(degree_vid)],
                vec![count(col(EDGE_DST)).alias(DEGREE)],
            )?;
            edges = edges
                .join_on(
                    degrees,
                    JoinType::Inner,
                    vec![col(EDGE_SRC).eq(col(degree_vid))],
                )?
                .select(vec![col(EDGE_SRC), col(EDGE_DST), col(DEGREE)])?;
        }

        // Pre-sorted, co-partitioned checkpoints: the edges by `src`, every
        // iterate by `id` — the per-iteration join and the final K-way
        // combine are sort-merge joins without an extra sort.
        let mut edges_checkpointer = ParquetCheckpointer::new(
            self.checkpoint_config.store_url.clone(),
            self.checkpoint_config
                .dir
                .clone()
                .join(run_id.clone())
                .join("edges"),
        );
        let mut states_checkpointer = ParquetCheckpointer::new(
            self.checkpoint_config.store_url.clone(),
            self.checkpoint_config
                .dir
                .clone()
                .join(run_id.clone())
                .join("states"),
        );
        let mut vertices_checkpointer = ParquetCheckpointer::new(
            self.checkpoint_config.store_url.clone(),
            self.checkpoint_config
                .dir
                .clone()
                .join(run_id.clone())
                .join("vertices"),
        );

        let edges = edges_checkpointer
            .push_pre_sorted(&ctx, "edges", edges, EDGE_SRC)
            .await?;

        // H_0: the deterministic sparse random projections (message source of
        // iteration 1; never part of the final combination).
        let init = vertices.clone().select(vec![
            col(VERTEX_ID),
            fastrp_init_expr(col(VERTEX_ID), self.dim, self.seed).alias(EMBEDDING),
        ])?;
        let mut state = states_checkpointer
            .push_pre_sorted(&ctx, "state-0", init, VERTEX_ID)
            .await?;

        // Per-step normalization scales every propagated vector by a factor
        // derived from the source out-degree (which rides on the edge table).
        let message = match self.normalization {
            FastRPNormalization::None => col(EMBEDDING),
            FastRPNormalization::L1 => vec_scale_expr(
                col(EMBEDDING),
                lit(1.0f64) / cast(col(DEGREE), DataType::Float64),
            ),
            FastRPNormalization::L2 => vec_scale_expr(
                col(EMBEDDING),
                lit(1.0f64) / sqrt().call(vec![cast(col(DEGREE), DataType::Float64)]),
            ),
        };

        // Propagation: iterate `t` sums the vectors of all sources reached by
        // `t - 1` hops. The aggregate emits only vertices that received a
        // message (a missing state row *is* the zero vector), so no null
        // handling is needed inside the loop.
        let mut states: Vec<DataFrame> = Vec::with_capacity(self.iterations);
        for t in 1..=self.iterations {
            let triplets = edges.clone().join_on(
                state.clone(),
                JoinType::Inner,
                vec![col(EDGE_SRC).eq(col(VERTEX_ID))],
            )?;
            let messages = triplets.select(vec![
                col(EDGE_DST).alias(VERTEX_ID),
                message.clone().alias(EMBEDDING),
            ])?;
            let aggregated = messages.aggregate(
                vec![col(VERTEX_ID)],
                vec![vec_sum_expr(col(EMBEDDING), self.dim).alias(EMBEDDING)],
            )?;
            state = states_checkpointer
                .push_pre_sorted(&ctx, &format!("state-{t}"), aggregated, VERTEX_ID)
                .await?;
            states.push(state.clone());
        }

        // Final combination: `H = Σ w_t · H_t` over the iterates with a
        // non-zero weight, computed as a *sequential checkpointed fold*:
        //
        //   acc_0     = w_{t0} · H_{t0}
        //   acc_{i+1} = acc_i + w_t · H_t          (checkpointed every step)
        //
        // At any moment only two iterate columns (the running accumulator and
        // the iterate being folded in) are in memory; every intermediate
        // accumulator is offloaded to a pre-sorted parquet checkpoint, so the
        // combine never materializes the full `K · |V| · D` history.
        let base = vertices_checkpointer
            .push_pre_sorted(&ctx, "vertices", vertices, VERTEX_ID)
            .await?;

        let mut folds_checkpointer = ParquetCheckpointer::new(
            self.checkpoint_config.store_url.clone(),
            self.checkpoint_config
                .dir
                .clone()
                .join(run_id.clone())
                .join("folds"),
        );

        // iterates that actually participate (non-zero weight), in order
        let kept: Vec<(usize, f64)> = weights
            .iter()
            .enumerate()
            .filter(|(_, w)| **w != 0.0)
            .map(|(t, w)| (t, *w))
            .collect();

        let mut fixed = if self.iterations == 0 {
            // No propagation at all: the embedding is the raw random init.
            state.select(vec![col(VERTEX_ID), col(EMBEDDING)])?
        } else if kept.is_empty() {
            // Degenerate but valid: all weights are zero.
            let fsl_type = DataType::FixedSizeList(
                datafusion::arrow::datatypes::Field::new("el", DataType::Float32, false).into(),
                self.dim as i32,
            );
            let zero = vec_zero_scalar(&fsl_type, self.dim)?;
            base.select(vec![col(VERTEX_ID), lit(zero).alias(EMBEDDING)])?
        } else {
            let zero = vec_zero_scalar(
                &DataType::FixedSizeList(
                    datafusion::arrow::datatypes::Field::new("el", DataType::Float32, false).into(),
                    self.dim as i32,
                ),
                self.dim,
            )?;

            // Sequential checkpointed fold over the participating iterates:
            //
            //   acc       = w_{t0} · H_{t0} + w_{t1} · H_{t1}
            //   acc       = acc + w_t · H_t        (one join per iterate,
            //                                       checkpointed every step)
            //
            // Only two iterate columns are ever in memory (the running
            // accumulator and the iterate being folded in); every
            // intermediate accumulator is offloaded to a pre-sorted parquet
            // checkpoint, and all keys are covered by co-partitioned SMJs.
            let (t0, w0) = kept[0];
            let mut acc = if kept.len() >= 2 {
                let (t1, w1) = kept[1];
                let side_a = states[t0].clone().select(vec![
                    col(VERTEX_ID).alias(ACC_VID),
                    col(EMBEDDING).alias("__fastrp_ha"),
                ])?;
                let side_b = states[t1].clone().select(vec![
                    col(VERTEX_ID).alias("__fastrp_hb_vid"),
                    col(EMBEDDING).alias("__fastrp_hb"),
                ])?;
                side_a
                    .join_on(
                        side_b,
                        // full outer: a vertex may be present in one iterate
                        // and absent (zero contribution) in the other
                        JoinType::Full,
                        vec![col(ACC_VID).eq(col("__fastrp_hb_vid"))],
                    )?
                    .select(vec![
                        coalesce(vec![col(ACC_VID), col("__fastrp_hb_vid")]).alias(ACC_VID),
                        vec_weighted_sum_expr(
                            &[(lit(w0), col("__fastrp_ha")), (lit(w1), col("__fastrp_hb"))],
                            self.dim,
                        )
                        .alias(ACC),
                    ])?
            } else {
                // single kept iterate: it is the accumulator (scaled)
                states[t0].clone().select(vec![
                    col(VERTEX_ID).alias(ACC_VID),
                    vec_scale_expr(col(EMBEDDING), lit(w0)).alias(ACC),
                ])?
            };
            folds_checkpointer
                .push_pre_sorted(&ctx, "fold-0", acc.clone(), ACC_VID)
                .await?;

            // remaining iterates fold in one at a time: acc = acc + w_t · H_t.
            // The full outer join grows the key set as iterates reach new
            // vertices (an absent iterate row is a zero contribution).
            for (t, w) in kept.iter().skip(2) {
                let side = states[*t].clone().select(vec![
                    col(VERTEX_ID).alias("__fastrp_ht_vid"),
                    col(EMBEDDING).alias("__fastrp_ht"),
                ])?;
                acc = acc
                    .join_on(
                        side,
                        // full outer, same reason as the first fold step
                        JoinType::Full,
                        vec![col(ACC_VID).eq(col("__fastrp_ht_vid"))],
                    )?
                    .select(vec![
                        coalesce(vec![col(ACC_VID), col("__fastrp_ht_vid")]).alias(ACC_VID),
                        vec_weighted_sum_expr(
                            &[(lit(1.0), col(ACC)), (lit(*w), col("__fastrp_ht"))],
                            self.dim,
                        )
                        .alias(ACC),
                    ])?;
                acc = folds_checkpointer
                    .push_pre_sorted(&ctx, "fold", acc.clone(), ACC_VID)
                    .await?;
                // only the latest accumulator checkpoint is needed
                folds_checkpointer.evict_all_but_latest_n(&ctx, 1).await?;
            }

            // Left-join the accumulator onto the full vertex set: vertices
            // that were never reached have no accumulator row, and a missing
            // row is the zero vector.
            let embedding = when(col(ACC).is_null(), lit(zero))
                .otherwise(col(ACC))?
                .alias(EMBEDDING);
            base.join_on(acc, JoinType::Left, vec![col(VERTEX_ID).eq(col(ACC_VID))])?
                .select(vec![col(VERTEX_ID), embedding])?
        };

        // Optional: L2-normalize the final embeddings to unit length.
        // Zero vectors (norm == 0) stay zero, so the pass can not produce
        // NaN/Inf and downstream cosine/K-Means consumers are safe.
        if self.norm_output {
            fixed = fixed
                .with_column(NORM, l2_norm_expr(col(EMBEDDING)))?
                .select(vec![
                    col(VERTEX_ID),
                    when(
                        col(NORM).not_eq(lit(0.0f64)),
                        vec_scale_expr(col(EMBEDDING), lit(1.0f64) / col(NORM)),
                    )
                    .otherwise(col(EMBEDDING))?
                    .alias(EMBEDDING),
                ])?;
        }

        fixed
            .write_parquet(output, DataFrameWriteOptions::new(), None)
            .await?;
        log::info!("result was written into {output}");

        // clean up: all iterate checkpoints live under the run directory
        edges_checkpointer.purge(&ctx).await?;
        states_checkpointer.purge(&ctx).await?;
        vertices_checkpointer.purge(&ctx).await?;

        log::info!(
            "FastRP {run_id} finished after {} iterations, output: {output}",
            self.iterations
        );
        Ok(self.iterations)
    }
}

impl GraphFrame {
    /// Create a new FastRP algorithm builder
    pub fn fastrp(&self) -> FastRPBuilder<'_> {
        FastRPBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::expressions::as_f32_list_like;
    use crate::ml::fastrp_init_fill;
    use crate::utils::symmetrize;

    use datafusion::arrow::array::Int64Array;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Unique temp dir per test (same pattern as the pregel tests).
    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("graphframes_fastrp_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    /// RAII guard that recursively removes the temp directory when dropped.
    struct TempGuard(PathBuf);

    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn setup(label: &str) -> (SessionContext, Path, String, TempGuard) {
        let parent = unique_temp_dir(label);
        let checkpoint_root = parent.join("checkpoints");
        let output_root = parent.join("output");
        fs::create_dir_all(&checkpoint_root).expect("checkpoint dir");
        fs::create_dir_all(&output_root).expect("output dir");

        let checkpoint_dir = Path::from_filesystem_path(&checkpoint_root).unwrap();
        let output_uri = url::Url::from_directory_path(&output_root)
            .unwrap()
            .to_string();

        (
            SessionContext::new(),
            checkpoint_dir,
            output_uri,
            TempGuard(parent),
        )
    }

    fn create_graph(vertices: Vec<i64>, edges: Vec<Vec<i64>>) -> Result<GraphFrame> {
        let vertices_df = dataframe!(VERTEX_ID => Vec::<i64>::from(vertices))?;
        let edges_df = dataframe!(
            EDGE_SRC => Vec::<i64>::from(edges.iter().map(|e| e[0]).collect::<Vec<i64>>()),
            EDGE_DST => Vec::<i64>::from(edges.iter().map(|e| e[1]).collect::<Vec<i64>>()),
        )?;
        GraphFrame::try_new(vertices_df, edges_df)
    }

    /// Read `[id, embedding]` parquet into a map (accepts both vector types).
    async fn embeddings_map(df: DataFrame) -> Result<HashMap<i64, Vec<f32>>> {
        let mut map = HashMap::new();
        for batch in df.collect().await? {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let vecs = as_f32_list_like(batch.column(1), "test", "embedding")?;
            for i in 0..batch.num_rows() {
                map.insert(ids.value(i), vecs.value(i).to_vec());
            }
        }
        Ok(map)
    }

    /// The init kernel reference for one vertex.
    fn init_ref(id: i64, seed: u64, d: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; d];
        fastrp_init_fill(id, seed, d, &mut v);
        v
    }

    #[tokio::test]
    async fn fastrp_requires_positive_dim() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("no_dim");
        let g = create_graph(vec![1, 2], vec![vec![1, 2]])?;
        let result = g
            .fastrp()
            .iterations(1)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await;
        assert!(result.is_err(), "dim(0) must be rejected");
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_k1_on_path_sums_neighbor_inits() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("k1_path");
        let d = 2;
        let seed = 42u64;
        // path: 1 -> 2 -> 3 (vertex 1 has no in-edges)
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3]])?;

        g.fastrp()
            .dim(d)
            .iterations(1)
            .seed(seed)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;

        assert_eq!(got.len(), 3);
        assert_eq!(got[&1], vec![0.0, 0.0], "no in-edges -> zero vector");
        assert_eq!(got[&2], init_ref(1, seed, d));
        assert_eq!(got[&3], init_ref(2, seed, d));
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_k0_returns_random_init() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("k0");
        let d = 8;
        let seed = 7u64;
        let g = create_graph(vec![1, 2], vec![vec![1, 2]])?;

        let iterations = g
            .fastrp()
            .dim(d)
            .iterations(0)
            .seed(seed)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;
        assert_eq!(iterations, 0);

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        assert_eq!(got[&1], init_ref(1, seed, d));
        assert_eq!(got[&2], init_ref(2, seed, d));
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_k2_propagates_two_hops() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("k2_path");
        let d = 4;
        let seed = 9u64;
        // path: 1 -> 2 -> 3
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3]])?;

        g.fastrp()
            .dim(d)
            .iterations(2)
            .seed(seed)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        // default weights [1, 1]: H(v) = H_1(v) + H_2(v)
        // H_1 = {2: init(1), 3: init(2)}; H_2 = {3: init(1)}
        assert_eq!(
            got[&3],
            add(&init_ref(2, seed, d), &init_ref(1, seed, d)),
            "vertex 3 sums both iterates"
        );
        assert_eq!(got[&2], init_ref(1, seed, d), "H_2(2) is absent -> zero");
        assert_eq!(got[&1], vec![0.0; d]);
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_weights_can_drop_early_iterates() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("weights_drop");
        let d = 4;
        let seed = 9u64;
        // path: 1 -> 2 -> 3
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3]])?;

        // w = [0, 1]: only the last iterate survives — the classic
        // "last iterate" embedding
        g.fastrp()
            .dim(d)
            .iterations(2)
            .seed(seed)
            .iteration_weights(vec![0.0, 1.0])
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        assert_eq!(
            got[&3],
            init_ref(1, seed, d),
            "two hops reach vertex 1's init"
        );
        assert_eq!(got[&2], vec![0.0; d]);
        assert_eq!(got[&1], vec![0.0; d]);
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_weights_scale_the_iterates() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("weights_scale");
        let d = 2;
        let seed = 42u64;
        // triangle 1 -> 2 -> 3 -> 1: every vertex has an exact expression
        // H_1(v) = init(pred(v)), H_2(v) = init(pred(pred(v)))
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3], vec![3, 1]])?;

        g.fastrp()
            .dim(d)
            .iterations(2)
            .seed(seed)
            .iteration_weights(vec![1.0, 2.0])
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        let i1 = init_ref(1, seed, d);
        let i2 = init_ref(2, seed, d);
        let i3 = init_ref(3, seed, d);
        // H_1(1)=init(3), H_1(2)=init(1), H_1(3)=init(2)
        // edges into 1: 3->1, into 2: 1->2, into 3: 2->3
        // H_2(1)=H_1(3)=init(2), H_2(2)=H_1(1)=init(1), H_2(3)=H_1(2)=init(1)
        // H(1) = 1*H_1(1) + 2*H_2(1) = init(3) + 2*init(2), exact in f32
        assert_eq!(got[&1], add_scaled(&i3, 1.0, &i2, 2.0));
        assert_eq!(got[&2], add_scaled(&i1, 1.0, &i3, 2.0));
        assert_eq!(got[&3], add_scaled(&i2, 1.0, &i1, 2.0));
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_combine_folds_match_reference() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("fold3");
        let d = 2;
        let seed = 42u64;
        // triangle: H_t(v) = init(pred^t(v)) — exact expressions per vertex
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3], vec![3, 1]])?;

        g.fastrp()
            .dim(d)
            .iterations(3)
            .seed(seed)
            .iteration_weights(vec![0.5, 1.0, 2.0])
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        let i1 = init_ref(1, seed, d);
        let i2 = init_ref(2, seed, d);
        let i3 = init_ref(3, seed, d);

        // H_1(1)=i3, H_2(1)=i2, H_3(1)=i1  => H(1) = 0.5*i3 + 1*i2 + 2*i1
        assert_eq!(
            got[&1],
            add_scaled(&add_scaled(&i3, 0.5, &i2, 1.0), 1.0, &i1, 2.0)
        );
        // H_1(2)=i1, H_2(2)=i3, H_3(2)=i2  => H(2) = 0.5*i1 + 1*i3 + 2*i2
        assert_eq!(
            got[&2],
            add_scaled(&add_scaled(&i1, 0.5, &i3, 1.0), 1.0, &i2, 2.0)
        );
        // H_1(3)=i2, H_2(3)=i1, H_3(3)=i3  => H(3) = 0.5*i2 + 1*i1 + 2*i3
        assert_eq!(
            got[&3],
            add_scaled(&add_scaled(&i2, 0.5, &i1, 1.0), 1.0, &i3, 2.0)
        );
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_weights_length_is_validated() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("weights_len");
        let g = create_graph(vec![1, 2], vec![vec![1, 2]])?;
        let result = g
            .fastrp()
            .dim(4)
            .iterations(2)
            .iteration_weights(vec![1.0]) // 1 entry for K=2
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await;
        assert!(result.is_err(), "weight length must match iterations");
        Ok(())
    }

    /// `a*w_a + b*w_b`, exact in f32 for these integral values.
    fn add_scaled(a: &[f32], w_a: f32, b: &[f32], w_b: f32) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x * w_a + y * w_b).collect()
    }

    fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    #[tokio::test]
    async fn fastrp_symmetrized_graph_is_symmetric() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("undirected");
        let d = 3;
        let seed = 5u64;
        let base = create_graph(vec![1, 2], vec![vec![1, 2]])?;
        let edges = symmetrize(&base.edges, true, None)?;
        let g = GraphFrame {
            vertices: base.vertices,
            edges,
        };

        g.fastrp()
            .dim(d)
            .iterations(1)
            .seed(seed)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        assert_eq!(got[&1], init_ref(2, seed, d));
        assert_eq!(got[&2], init_ref(1, seed, d));
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_fills_isolated_vertices_with_zero() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("isolated");
        let d = 2;
        let seed = 1u64;
        // vertex 3 has no edges at all; vertex 1 has no in-edges
        let g = create_graph(vec![1, 2, 3], vec![vec![2, 1]])?;

        g.fastrp()
            .dim(d)
            .iterations(1)
            .seed(seed)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        assert_eq!(got[&3], vec![0.0; d]);
        assert_eq!(got[&2], vec![0.0; d]);
        assert_eq!(got[&1], init_ref(2, seed, d));
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_is_reproducible_for_a_seed() -> Result<()> {
        let d = 6;
        let seed = 77u64;

        let build = || async {
            let (ctx, checkpoint_dir, output_uri, _guard) = setup("repro");
            let g = create_graph(
                vec![1, 2, 3, 4],
                vec![vec![1, 2], vec![2, 3], vec![3, 4], vec![1, 4]],
            )?;
            g.fastrp()
                .dim(d)
                .iterations(2)
                .seed(seed)
                .set_checkpoint_dir(checkpoint_dir)
                .run(&ctx, &output_uri)
                .await?;
            embeddings_map(
                ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                    .await?,
            )
            .await
        };

        let first = build().await?;
        let second = build().await?;
        assert_eq!(first, second, "same seed must reproduce the embeddings");
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_l1_divides_by_source_degree() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("l1");
        let d = 2;
        let seed = 42u64;
        // star: 1 -> 2, 1 -> 3; out-degree of 1 is 2
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![1, 3]])?;

        g.fastrp()
            .dim(d)
            .iterations(1)
            .seed(seed)
            .normalization(FastRPNormalization::L1)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        // H(2) = H(3) = init(1) / deg(1), with the exact f32 factor
        let s = (1.0f64 / 2.0f64) as f32;
        let expected: Vec<f32> = init_ref(1, seed, d).iter().map(|x| x * s).collect();
        assert_eq!(got[&2], expected);
        assert_eq!(got[&3], expected);
        assert_eq!(got[&1], vec![0.0; d]);
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_l2_divides_by_sqrt_source_degree() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("l2");
        let d = 2;
        let seed = 42u64;
        let g = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![1, 3]])?;

        g.fastrp()
            .dim(d)
            .iterations(1)
            .seed(seed)
            .normalization(FastRPNormalization::L2)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        // the expression computes 1 / sqrt(deg) in f64, then scales in f32
        let s = (1.0f64 / (2.0f64).sqrt()) as f32;
        let expected: Vec<f32> = init_ref(1, seed, d).iter().map(|x| x * s).collect();
        assert_eq!(got[&2], expected);
        assert_eq!(got[&3], expected);
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_norm_output_gives_unit_norms() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("norm_output");
        let d = 4;
        let seed = 11u64;
        let g = create_graph(
            vec![1, 2, 3],
            vec![vec![1, 2], vec![2, 3], vec![3, 1], vec![1, 3]],
        )?;

        g.fastrp()
            .dim(d)
            .iterations(2)
            .seed(seed)
            .norm_output(true)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        assert_eq!(got.len(), 3);
        for (id, v) in &got {
            let norm_sq: f32 = v.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4 || norm == 0.0,
                "vertex {id}: expected unit norm (or zero), got {norm}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_norm_output_zero_vectors_stay_zero() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("norm_output_zero");
        let d = 3;
        let seed = 3u64;
        // vertex 1 has no in-edges -> zero embedding after K=1
        let g = create_graph(vec![1, 2], vec![vec![1, 2]])?;

        g.fastrp()
            .dim(d)
            .iterations(1)
            .seed(seed)
            .norm_output(true)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let got = embeddings_map(
            ctx.read_parquet(&output_uri, ParquetReadOptions::default())
                .await?,
        )
        .await?;
        assert_eq!(got[&1], vec![0.0; d], "zero vector must stay exactly zero");
        // and vertex 2 is unit-norm
        let norm: f32 = got[&2].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        Ok(())
    }
}
