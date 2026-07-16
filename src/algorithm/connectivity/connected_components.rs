//! Connected components via **randomized contraction**.
//!
//! Implementation inspired by the Spark GraphFrames Scala code distributed
//! under Apache License 2.0, ported to this project's out-of-core style.
//!
//! Algorithm: Bögeholz, Brand, Todor, "In-database connected component
//! analysis", ICDE 2020. The graph is contracted iteratively using fresh
//! random affine hash functions over GF(2^64) (`finite-axpb`); each
//! iteration every vertex relabels itself to the minimum hash in its
//! neighbourhood. A forward pass contracts until no edges remain, spilling
//! per-iteration component representatives to disk; a back-propagation pass
//! then unwinds the chain of affine hashes to express every original
//! vertex's component id in original-id space.

use crate::expressions::{axpb, finite_axpb};
use crate::memory::{CheckpointConfig, ParquetCheckpointer};
use crate::utils::{GraphFramesConfig, scoped_ctx, symmetrize};
use crate::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions_aggregate::expr_fn::min;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use uuid::Uuid;

pub const COMPONENT_COL: &str = "component";

// Internal column names used by the contraction. The `__cc_` prefix keeps
// them clear of any user vertex / edge attribute columns during joins.
const CC_V: &str = "__cc_v";
const CC_REP: &str = "__cc_rep";
const CC_NBR_REP: &str = "__cc_nbr_rep";
const CC_SELF_REP: &str = "__cc_self_rep";
const CC_FR_V: &str = "__cc_fr_v";
const CC_FR_REP: &str = "__cc_fr_rep";
const CC_COMP_KEY: &str = "__cc_comp_key";
const CC_NEW_COMPONENT: &str = "__cc_new_component";

/// Computes the per-iteration component representatives for every source
/// vertex under the affine hash `(rA, rB)`:
///   `rep(v) = least(axpb(rA, v, rB), min over neighbours of axpb(rA, u, rB))`.
///
/// Returns a 2-column frame `[CC_V, CC_REP]`.
fn compute_cc_reps(edges: &DataFrame, r_a: i64, r_b: i64) -> Result<DataFrame> {
    edges
        .clone()
        .aggregate(
            vec![col(EDGE_SRC)],
            vec![min(finite_axpb(lit(r_a), col(EDGE_DST), lit(r_b))).alias(CC_NBR_REP)],
        )?
        .with_column(CC_SELF_REP, finite_axpb(lit(r_a), col(EDGE_SRC), lit(r_b)))?
        .select(vec![
            col(EDGE_SRC).alias(CC_V),
            when(col(CC_SELF_REP).lt(col(CC_NBR_REP)), col(CC_SELF_REP))
                .otherwise(col(CC_NBR_REP))?
                .alias(CC_REP),
        ])
}

/// Relabels `edges` using the current representatives: each edge
/// `(u, w)` becomes `(rep(u), rep(w))`, with the resulting self-loops
/// (where `rep(u) == rep(w)`) dropped and the result deduplicated.
///
/// `cc_reps` is the `[CC_V, CC_REP]` frame produced by [`compute_cc_reps`].
fn relabel_edges(edges: &DataFrame, cc_reps: &DataFrame) -> Result<DataFrame> {
    // First relabel the source: join edges with reps on src == v.
    let edges_src_relabeled = edges
        .clone()
        .join(cc_reps.clone(), JoinType::Inner, &[EDGE_SRC], &[CC_V], None)?
        .select(vec![col(CC_REP).alias(EDGE_SRC), col(EDGE_DST)])?;

    // Then relabel the destination and drop self-loops created by the
    // contraction: keep (rep(u), rep(w)) only where rep(u) != rep(w).
    edges_src_relabeled
        .clone()
        .join_on(
            cc_reps.clone(),
            JoinType::Inner,
            vec![
                col(EDGE_DST).eq(col(CC_V)),
                col(EDGE_SRC).not_eq(col(CC_REP)),
            ],
        )?
        .select(vec![col(EDGE_SRC), col(CC_REP).alias(EDGE_DST)])?
        .distinct()
}

/// One step of the back-propagation pass.
///
/// Joins `older` (`cc_reps_{t}`, `[CC_V, CC_REP]`) against `frontier`
/// (`[CC_V, CC_REP]`, the reps of the next iteration already unwound) on
/// `older.rep == frontier.v`. For vertices whose rep was forwarded, take
/// the frontier's rep; otherwise apply the accumulated affine map
/// `(acc_a, acc_b)` to push this iteration's rep into final-id space.
///
/// Returns the new `[CC_V, CC_REP]` frontier.
fn back_prop_step(
    older: &DataFrame,
    frontier: &DataFrame,
    acc_a: i64,
    acc_b: i64,
) -> Result<DataFrame> {
    let frontier_renamed = frontier
        .clone()
        .with_column_renamed(CC_V, CC_FR_V)?
        .with_column_renamed(CC_REP, CC_FR_REP)?;

    older
        .clone()
        .join(
            frontier_renamed,
            JoinType::Left,
            &[CC_REP],
            &[CC_FR_V],
            None,
        )?
        .with_column(
            "__cc_new_rep",
            when(
                col(CC_FR_REP).is_null(),
                finite_axpb(lit(acc_a), col(CC_REP), lit(acc_b)),
            )
            .otherwise(col(CC_FR_REP))?,
        )?
        .select(vec![col(CC_V), col("__cc_new_rep").alias(CC_REP)])
}

/// Builder for connected-components computation.
///
/// The builder follows the same out-of-core contract as
/// [`crate::algorithm::pregel::Pregel`] and
/// [`crate::algorithm::centrality::pagerank::PageRankBuilder`]: the final
/// result is written as parquet into the user-supplied `output` directory
/// URI, intermediate state is spilled under the checkpoint dir, and
/// [`run`](ConnectedComponentsBuilder::run) returns the number of forward
/// contraction iterations.
pub struct ConnectedComponentsBuilder<'a> {
    graph_frame: &'a GraphFrame,
    use_labels_as_components: bool,
    random_seed: u64,
    checkpoint_config: CheckpointConfig,
}

impl<'a> ConnectedComponentsBuilder<'a> {
    pub fn new(graph_frame: &'a GraphFrame) -> Self {
        Self {
            graph_frame: graph_frame,
            use_labels_as_components: true,
            random_seed: 42,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    /// When `true` (the default), post-processes the result so every
    /// component id is the minimum *original* vertex label of its members.
    /// When `false`, emits the raw GF(2^64) hashed representatives instead
    /// (faster, but the ids are arbitrary hashed values, not real labels).
    pub fn use_labels_as_components(mut self, flag: bool) -> Self {
        self.use_labels_as_components = flag;
        self
    }

    pub fn with_checkpoint_store(mut self, store_url: ObjectStoreUrl) -> Self {
        self.checkpoint_config.store_url = store_url;
        self
    }

    pub fn set_checkpoint_dir(mut self, dir: Path) -> Self {
        self.checkpoint_config.dir = dir;
        self
    }

    pub fn set_seed(mut self, v: u64) -> Self {
        self.random_seed = v;
        self
    }

    /// Runs the algorithm, writing `[VERTEX_ID, COMPONENT_COL]` as parquet
    /// into `output` (a directory URI such as `file:///path/to/dir/`).
    ///
    /// Returns the number of forward contraction iterations performed.
    pub async fn run(
        self,
        ctx: &SessionContext,
        output: &str,
        _include_debug_columns: bool,
    ) -> Result<usize> {
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

        let vertices = self
            .graph_frame
            .vertices
            .clone()
            .select_columns(&vec![VERTEX_ID])?;
        let original_edges = self
            .graph_frame
            .edges
            .clone()
            .select_columns(&vec![EDGE_SRC, EDGE_DST])?;

        // ---- prepare: drop self-loops, symmetrize, deduplicate ----
        let run_id = Uuid::new_v4().to_string();
        log::info!("start WCC with run-id {run_id}");
        // it may look conter-intuitive
        // but IRL distinct is more expensive here.
        //
        // With distinct: full-scan over huge edges, slightly less on the first iter;
        // Without distinct: fast flow here, slightly slower on the first iter.
        //
        // TODO: benchmark it and see which one is better IRL, not in theory
        let prepared_edges = symmetrize(&original_edges, false)?;

        let ckpt_base = self.checkpoint_config.dir.clone().join(run_id.clone());
        let store_url = self.checkpoint_config.store_url.clone();

        // Three checkpointers, each rooted at its own subdirectory under
        // `<checkpoint_dir>/<run_id>/`:
        //  - edges:   the (mutated) edge set, only the latest iteration kept
        //  - forward: cc reps per forward iteration, peeled during back-pass
        //  - back:    rolling back-propagation frontier, latest kept
        let mut edges_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("edges"));
        let mut fwd_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("forward"));
        let mut back_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("backward"));

        // Offload the initial edges; from here on `edges` is disk-backed.
        let mut edges = edges_ckptr.push(ctx, "initial", prepared_edges).await?;
        let mut graph_size = edges.clone().count().await?;
        log::info!("after preparation graph has {graph_size} edges");

        // ---- forward pass: iterative randomized contraction ----
        let mut rng = StdRng::seed_from_u64(self.random_seed);
        let mut forward_reps: Vec<DataFrame> = Vec::new();

        // Per-iteration affine coefficients (rA, rB), used by the back pass.
        let mut affine_params: Vec<(i64, i64)> = Vec::new();
        let mut iteration = 0usize;

        while graph_size > 0 {
            iteration += 1;
            // Draw a non-zero `a` and a random `b` for this iteration's
            // affine hash. `a` must be invertible, hence non-zero.
            let mut r_a = rng.random::<i64>();
            while r_a == 0 {
                r_a = rng.random::<i64>();
            }
            let r_b = rng.random::<i64>();
            affine_params.push((r_a, r_b));

            let cc_reps = compute_cc_reps(&edges, r_a, r_b)?;
            // Spill reps to the forward checkpointer; keep the lazy,
            // disk-backed frame for the back pass.
            let cc_reps = fwd_ckptr.push(ctx, &iteration.to_string(), cc_reps).await?;
            forward_reps.push(cc_reps.clone());

            // Relabel edges: src -> rep, then dst -> rep, drop self-loops.
            let new_edges = relabel_edges(&edges, &cc_reps)?;

            edges = edges_ckptr
                .push(ctx, &iteration.to_string(), new_edges)
                .await?;
            // Count the new edge set BEFORE evicting the previous checkpoint.
            // When a contraction iteration empties the edge set (convergence),
            // `push` returns the in-memory frame without tracking it, and that
            // frame still references the about-to-be-deleted parquet; counting
            // first (while the old parquet still exists) avoids a NotFound.
            graph_size = edges.clone().count().await?;
            // Keep only the latest edges checkpoint on disk.
            edges_ckptr.evict(ctx, 1).await?;
            log::info!("cc forward iteration {iteration}, edges remaining: {graph_size}");
        }

        // Edges are no longer needed after the forward pass.
        edges_ckptr.purge(ctx).await?;

        // ---- back propagation: unwind the affine hashes ----
        let n = forward_reps.len();
        let frontier: Option<DataFrame> = if n == 0 {
            None
        } else {
            // Seed the back frontier with the last forward reps.
            let mut frontier = back_ckptr
                .push(ctx, "seed", forward_reps[n - 1].clone())
                .await?;
            // The last forward reps are now consumed; drop them from disk.
            fwd_ckptr.remove_last(ctx, 1).await?;

            // Accumulated affine coefficients of all iterations already undone.
            let mut acc_a = 1i64;
            let mut acc_b = 0i64;

            for t in (1..n).rev() {
                // Pop this back step's affine params (iteration t+1's).
                let (popped_a, popped_b) = affine_params.pop().unwrap();
                let old_acc_a = acc_a;
                acc_a = axpb(old_acc_a, popped_a, 0);
                acc_b = axpb(old_acc_a, popped_b, acc_b);

                let older = forward_reps[t - 1].clone();
                let composed = back_prop_step(&older, &frontier, acc_a, acc_b)?;

                frontier = back_ckptr.push(ctx, &t.to_string(), composed).await?;
                // Keep only the latest back checkpoint.
                back_ckptr.evict(ctx, 1).await?;
                // `older` (cc_reps_{t}) has been consumed into the new
                // frontier; remove it from the forward dir.
                fwd_ckptr.remove_last(ctx, 1).await?;

                log::info!("cc back propagation step t={t}");
            }
            Some(frontier)
        };

        // ---- final labeling ----
        let final_df = match frontier {
            Some(frontier_df) => {
                // Left-join every vertex id onto the frontier so that
                // isolated vertices (absent from the edge set) become their
                // own component.
                vertices
                    .clone()
                    .select(vec![col(VERTEX_ID)])?
                    .join_on(
                        frontier_df,
                        JoinType::Left,
                        vec![col(VERTEX_ID).eq(col(CC_V))],
                    )?
                    .with_column(
                        COMPONENT_COL,
                        when(col(CC_REP).is_null(), col(VERTEX_ID)).otherwise(col(CC_REP))?,
                    )?
                    .select(vec![col(VERTEX_ID), col(COMPONENT_COL)])?
            }
            None => {
                // No edges at all: every vertex is its own component.
                vertices
                    .clone()
                    .select(vec![col(VERTEX_ID), col(VERTEX_ID).alias(COMPONENT_COL)])?
            }
        };

        // `final_df` now holds (id, component) where `component` is the
        // hashed representative (or the id itself for isolated vertices).
        let result = if self.use_labels_as_components {
            // spill "raw" components to disk
            let components = fwd_ckptr.push(ctx, "final_df", final_df).await?;
            // Relabel each component to the minimum original id of its members.
            let labels = components.clone().aggregate(
                vec![col(COMPONENT_COL).alias(CC_COMP_KEY)],
                vec![min(col(VERTEX_ID)).alias(CC_NEW_COMPONENT)],
            )?;
            components
                .join(
                    labels,
                    JoinType::Inner,
                    &[COMPONENT_COL],
                    &[CC_COMP_KEY],
                    None,
                )?
                .select(vec![
                    col(VERTEX_ID),
                    when(col(CC_NEW_COMPONENT).is_null(), col(VERTEX_ID))
                        .otherwise(col(CC_NEW_COMPONENT))?
                        .alias(COMPONENT_COL),
                ])?
        } else {
            final_df
        };

        result
            .write_parquet(output, DataFrameWriteOptions::new(), None)
            .await?;

        log::info!("connected components written to {output} after {iteration} forward iterations");

        // Clean up all checkpoints.
        fwd_ckptr.purge(ctx).await?;
        back_ckptr.purge(ctx).await?;

        Ok(iteration)
    }
}

impl GraphFrame {
    /// Constructs a [`ConnectedComponentsBuilder`] for the current graph.
    ///
    /// Computes the weakly connected components using randomized
    /// contraction (Bögeholz et al., ICDE 2020). The result is written
    /// out-of-core to a user-supplied directory as parquet.
    ///
    /// # Example
    /// ```
    /// use datafusion::dataframe;
    /// use datafusion::prelude::SessionContext;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// # async fn run() -> datafusion::error::Result<()> {
    /// let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64, 3i64])?;
    /// let edges = dataframe!(EDGE_SRC => vec![1i64, 2i64, 3i64], EDGE_DST => vec![3i64, 1i64, 2i64])?;
    /// let graph = GraphFrame::try_new(vertices, edges)?;
    /// let ctx = SessionContext::new();
    /// // `output` must be a directory URI, e.g. "file:///tmp/cc_out/".
    /// graph
    ///     .connected_components()
    ///     .run(&ctx, "file:///tmp/cc_out/", false)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn connected_components(&self) -> ConnectedComponentsBuilder<'_> {
        ConnectedComponentsBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::create_ldbc_test_graph;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_cc_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Builds a fresh `SessionContext`, a non-overlapping checkpoint dir and
    /// `file://` output URI, and a `TempGuard` that cleans up on drop.
    fn setup(label: &str) -> Result<(SessionContext, Path, String, TempGuard)> {
        let parent = unique_temp_dir(label);
        let checkpoint_root = parent.join("checkpoints");
        let output_root = parent.join("output");
        fs::create_dir_all(&checkpoint_root).expect("failed to create checkpoint dir");
        fs::create_dir_all(&output_root).expect("failed to create output dir");

        let checkpoint_dir = Path::from_filesystem_path(&checkpoint_root)
            .expect("checkpoint dir must be convertible to object_store path");
        let output_uri = Url::from_directory_path(&output_root)
            .expect("output dir must be convertible to file:// URL")
            .to_string();

        let ctx = SessionContext::new();
        Ok((ctx, checkpoint_dir, output_uri, TempGuard(parent)))
    }

    /// Reads the result parquet written by `run` back into a DataFrame.
    async fn read_result(ctx: &SessionContext, output_uri: &str) -> Result<DataFrame> {
        ctx.read_parquet(output_uri, ParquetReadOptions::default())
            .await
    }

    #[tokio::test]
    async fn test_zero_vertices() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => Vec::<i64>::new())?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame { vertices, edges };

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("zero_vertices")?;
        let iters = graph
            .connected_components()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        // An empty graph has no vertices to label, so the forward loop runs
        // zero iterations and the output directory stays empty. We only assert
        // that `run` completes successfully and reports zero iterations; the
        // empty output directory cannot be read back as a parquet table.
        assert_eq!(iters, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_single_vertex() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64])?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame { vertices, edges };

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("single_vertex")?;
        graph
            .connected_components()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let result = read_result(&ctx, &output_uri).await?;
        assert_eq!(result.schema().fields().len(), 2);
        assert_eq!(result.clone().count().await?, 1);
        let collected = result.collect().await?;
        assert_eq!(
            collected[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            1i64
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_two_vertices() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64])?;
        let edges = dataframe!(EDGE_SRC => vec![1i64], EDGE_DST => vec![2i64])?;
        let graph = GraphFrame { vertices, edges };

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("two_vertices")?;
        graph
            .connected_components()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let result = read_result(&ctx, &output_uri).await?;
        assert_eq!(result.schema().fields().len(), 2);
        assert_eq!(result.clone().count().await?, 2);

        let sorted = result.sort_by(vec![col(VERTEX_ID)])?.collect().await?;
        let ids: Vec<i64> = sorted[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        let components: Vec<i64> = sorted[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(ids, vec![1, 2]);
        // Both vertices are connected, so they share a single component.
        assert_eq!(components[0], components[1]);
        Ok(())
    }

    #[tokio::test]
    async fn test_disconnected_vertices() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64])?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame { vertices, edges };

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("disconnected")?;
        graph
            .connected_components()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let result = read_result(&ctx, &output_uri).await?;
        assert_eq!(result.clone().count().await?, 2);

        let sorted = result.sort_by(vec![col(VERTEX_ID)])?.collect().await?;
        let components: Vec<i64> = sorted[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        // No edges: each vertex is its own component (== its own id, since
        // labels mode maps each isolated component to its single member).
        assert_eq!(components, vec![1, 2]);
        Ok(())
    }

    #[tokio::test]
    async fn test_use_labels_off_partitioning() -> Result<()> {
        // Two disjoint components: {1,2,3} and {10,11}.
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2, 3, 10, 11])?;
        let edges = dataframe!(
            EDGE_SRC => vec![1i64, 2, 10],
            EDGE_DST => vec![2i64, 3, 11],
        )?;
        let graph = GraphFrame { vertices, edges };

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("labels_off")?;
        graph
            .connected_components()
            .use_labels_as_components(false)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let result = read_result(&ctx, &output_uri).await?;
        let collected = result.collect().await?;

        // Build id -> component.
        let mut id_to_comp: HashMap<i64, i64> = HashMap::new();
        for batch in &collected {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let comps = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..ids.len() {
                id_to_comp.insert(ids.value(i), comps.value(i));
            }
        }

        // Partitioning must be correct even though the hashed component ids
        // are arbitrary: same-CC vertices share an id, cross-CC differ.
        assert_eq!(id_to_comp[&1], id_to_comp[&2]);
        assert_eq!(id_to_comp[&2], id_to_comp[&3]);
        assert_eq!(id_to_comp[&10], id_to_comp[&11]);
        assert_ne!(id_to_comp[&1], id_to_comp[&10]);
        Ok(())
    }

    async fn get_ldbc_wcc_results(dataset: &str) -> Result<DataFrame> {
        let ctx = SessionContext::new();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected_wcc_schema = Schema::new(vec![
            Field::new("vertex_id", DataType::Int64, false),
            Field::new("expected_component", DataType::Int64, false),
        ]);
        let expected_wcc_path = format!(
            "{}/testing/data/ldbc/{}/{}-WCC.csv",
            manifest_dir, dataset, dataset
        );
        let expected_sp = ctx
            .read_csv(
                &expected_wcc_path,
                CsvReadOptions::new()
                    .delimiter(b' ')
                    .has_header(false)
                    .schema(&expected_wcc_schema),
            )
            .await?;
        Ok(expected_sp)
    }

    #[tokio::test]
    async fn test_ldbc() -> Result<()> {
        let expected_components = get_ldbc_wcc_results("test-wcc-directed").await?;
        let graph = create_ldbc_test_graph("test-wcc-directed", false, false).await?;

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("ldbc")?;
        graph
            .connected_components()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let results = read_result(&ctx, &output_uri).await?;
        let diff = results
            .clone()
            .join(
                expected_components,
                JoinType::Left,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .select(vec![
                col(VERTEX_ID),
                col(COMPONENT_COL),
                col("expected_component"),
            ])?
            .filter(col(COMPONENT_COL).not_eq(col("expected_component")))?;

        assert_eq!(diff.count().await?, 0);
        Ok(())
    }

    // This test is "heavy" (17M edges) and requires --release to run in seconds;
    // to test `cargo test --release -- --ignored`
    //
    // Do not forget to Download data first:
    // https://ldbcouncil.org/benchmarks/graphalytics/datasets/
    //
    // Keep in mind that `create_ldbc_test_graph` expects:
    // 1) ./testing/data/ldbc/kgs/kgs.{v|e}.csv
    // 2) ./testing/data/ldbc/kgs/kgs-WCC.csv
    #[ignore]
    #[tokio::test]
    async fn test_ldbc_kgs() -> Result<()> {
        let expected_components = get_ldbc_wcc_results("kgs").await?;
        let graph = create_ldbc_test_graph("kgs", false, true).await?;

        let (ctx, checkpoint_dir, output_uri, _guard) = setup("kgs")?;
        graph
            .connected_components()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let results = read_result(&ctx, &output_uri).await?;
        let diff = results
            .clone()
            .join(
                expected_components,
                JoinType::Left,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .select(vec![
                col(VERTEX_ID),
                col(COMPONENT_COL),
                col("expected_component"),
            ])?
            .filter(col(COMPONENT_COL).not_eq(col("expected_component")))?;

        assert_eq!(diff.count().await?, 0);
        Ok(())
    }
}
