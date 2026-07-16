//! Maximal Independent Set
//!
//! Implementation inspired by the Spark GraphFrames Scala code distributed
//! under Apache License 2.0, ported to this project's out-of-core style.
//!
//! Algorithm: "An Improved Distributed Algorithm for Maximal Independent Set",
//! Mohsen Ghaffari
//!
//! https://arxiv.org/abs/1506.05093

use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions_aggregate;
use datafusion::functions_aggregate::expr_fn::bool_or;
use datafusion::prelude::*;
use datafusion::{object_store::path::Path, prelude::SessionContext};
use uuid::Uuid;

use crate::memory::ParquetCheckpointer;
use crate::utils::symmetrize;
use crate::{EDGE_DST, EDGE_SRC, VERTEX_ID};
use crate::{
    GraphFrame,
    memory::CheckpointConfig,
    utils::{GraphFramesConfig, scoped_ctx},
};

/// Internal vertex-id column carried by the per-vertex MIS state frames.
///
/// Deliberately distinct from [`VERTEX_ID`] so that joining two such frames
/// does not collide on a shared `id` column name.
const MIS_V: &str = "__mis_v";
/// Current selection probability `p` of an active vertex.
const MIS_PROB: &str = "__mis_prob";
/// Effective degree `d(v) = sum of p over v's neighbours`.
const MIS_DEG: &str = "__mis_deg";
/// Per-vertex "nominated this round" flag (drawn against the current `p`).
const MIS_NOM: &str = "__mis_nom";
/// Per-vertex "at least one neighbour nominated" flag.
const MIS_HAS_NBR_NOM: &str = "__mis_has_nbr_nom";

// Throw-away renamed keys used only to disambiguate the two sides of a join.
const MIS_NEW_V: &str = "__mis_new_v";
const MIS_NEW_FLAG: &str = "__mis_new_flag";
const MIS_REM_V: &str = "__mis_rem_v";

/// Final boolean flag column carried by the per-vertex MIS state frame:
/// `true` once a vertex has been added to the independent set.
pub const MIS_FLAG: &str = "mis";

/// Folds a set of newly-selected vertex ids (`members`, schema `[MIS_V]`)
/// into the per-vertex MIS state, setting [`MIS_FLAG`] to `true` for every
/// id present in `members`.
///
/// `members`' id is renamed to [`MIS_NEW_V`] before a left join so the result
/// schema carries no duplicate `__mis_v` field, and the joined flag is
/// coalesced to `false` so a non-matching left row cannot turn `flag.or(...)`
/// into SQL `NULL`.
fn or_into_mis(current_mis: DataFrame, members: DataFrame) -> Result<DataFrame> {
    let members = members
        .with_column_renamed(MIS_V, MIS_NEW_V)?
        .with_column(MIS_NEW_FLAG, lit(true))?;
    current_mis
        .join(members, JoinType::Left, &[MIS_V], &[MIS_NEW_V], None)?
        .with_column(
            MIS_FLAG,
            col(MIS_FLAG).or(coalesce(vec![col(MIS_NEW_FLAG), lit(false)])),
        )?
        .select(vec![col(MIS_V), col(MIS_FLAG)])
}

#[derive(Debug, Clone)]
pub struct MISBuilder<'a> {
    graph: &'a GraphFrame,
    /// Storage options
    checkpoint_config: CheckpointConfig,
}

impl<'a> MISBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        MISBuilder {
            graph,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
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

    /// Run the MIS.
    ///
    /// Each run is non-deterministic: nomination uses DataFusion's `random()`
    /// and there is currently no way to fix its seed.
    /// Tracking issue: https://github.com/apache/datafusion/issues/17686
    ///
    /// The result is written out-of-core as parquet into `output` (a directory
    /// URI such as `file:///path/to/dir/`) as a single [`VERTEX_ID`] column
    /// listing the vertices in the computed maximal independent set. Returns
    /// the number of iterations performed.
    pub async fn run(self, ctx: &SessionContext, output: &str) -> Result<usize> {
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

        let run_id = Uuid::new_v4().to_string();
        log::info!("start MIS with run-id {run_id}");

        let ckpt_base = self.checkpoint_config.dir.clone().join(run_id.clone());
        let store_url = self.checkpoint_config.store_url.clone();

        // a) edges checkpointer b) vertices left checkpointer c) current MIS checkpointer
        let mut edges_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("edges"));
        let mut vertex_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("vertices"));
        let mut mis_checkpointer =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("mis"));

        // current_mis: every vertex with a boolean `mis` flag (false initially).
        let mut current_mis = mis_checkpointer
            .push(
                ctx,
                "0",
                self.graph.vertices.clone().select(vec![
                    col(VERTEX_ID).alias(MIS_V),
                    lit(false).alias(MIS_FLAG),
                ])?,
            )
            .await?;

        // vertices_left: vertices still "active", each carrying its current
        // selection probability p (Ghaffari starts every vertex at 1/2).
        let mut vertices_left = vertex_ckptr
            .push(
                ctx,
                "0",
                self.graph.vertices.clone().select(vec![
                    col(VERTEX_ID).alias(MIS_V),
                    lit(0.5f64).alias(MIS_PROB),
                ])?,
            )
            .await?;

        // Symmetrize + dedup edges so the graph is treated as undirected and
        // simple. Dedup is required here: a duplicated edge would inflate the
        // effective-degree computation.
        //
        // Project to just (src, dst) first: the algorithm never reads edge
        // attributes, and `symmetrize`'s reversed half is a 2-column projection
        // so feeding it anything but [src, dst] is both wasted checkpoint bytes
        // and a hard union-schema error ("UNION queries have different number of
        // columns"). The only edge columns worth carrying are the endpoints.
        let mut edges = edges_ckptr
            .push(
                ctx,
                "initial",
                symmetrize(
                    &self
                        .graph
                        .edges
                        .clone()
                        .select_columns(&[EDGE_SRC, EDGE_DST])?,
                    true,
                )?,
            )
            .await?;

        let mut iteration = 0usize;
        let mut converged = false;

        while !converged {
            // ---- effective degree: d(v) = sum of p_t over v's neighbours ----
            //
            // see arXiv preprint for details.
            let effective_degrees = edges_ckptr
                .push(
                    ctx,
                    &format!("deg_{}", iteration),
                    edges
                        .clone()
                        .join(
                            vertices_left.clone(),
                            JoinType::Inner,
                            &[EDGE_DST],
                            &[MIS_V],
                            None,
                        )?
                        .aggregate(
                            vec![col(EDGE_SRC)],
                            vec![functions_aggregate::sum::sum(col(MIS_PROB)).alias(MIS_DEG)],
                        )?,
                )
                .await?;

            // ---- nominate (using p_t) and update p -> p_{t+1} ----
            // Per the paper, nomination must use the current p_t; only afterwards
            // is p advanced to p_{t+1}, which is what the next round reads.
            //
            // see arXiv preprint for details
            let probs = vertex_ckptr
                .push(
                    ctx,
                    &format!("probs_{}", iteration),
                    vertices_left
                        .clone()
                        .join(
                            effective_degrees.clone(),
                            JoinType::Inner,
                            &[MIS_V],
                            &[EDGE_SRC],
                            None,
                        )?
                        .with_column(MIS_NOM, random().lt_eq(col(MIS_PROB)))?
                        .with_column(
                            MIS_PROB,
                            when(col(MIS_DEG).gt_eq(lit(2.0)), col(MIS_PROB).div(lit(2.0)))
                                .when(
                                    lit(2.0).mul(col(MIS_PROB)).lt_eq(lit(0.5)),
                                    lit(2.0).mul(col(MIS_PROB)),
                                )
                                .otherwise(lit(0.5))?,
                        )?
                        .select(vec![col(MIS_V), col(MIS_PROB), col(MIS_NOM)])?,
                )
                .await?;

            // ---- isolated vertices: active vertices with no edges (degree 0) ----
            // Such a vertex never appears as an edge source, so it is absent from
            // `effective_degrees`;
            let isolated = vertices_left
                .clone()
                .join(
                    effective_degrees,
                    JoinType::LeftAnti,
                    &[MIS_V],
                    &[EDGE_SRC],
                    None,
                )?
                .select(vec![col(MIS_V)])?;

            // ---- for each vertex, does any neighbour nominate itself? ----
            let has_nom_nbr = edges
                .clone()
                .join(probs.clone(), JoinType::Inner, &[EDGE_DST], &[MIS_V], None)?
                .aggregate(
                    vec![col(EDGE_SRC)],
                    vec![bool_or(col(MIS_NOM)).alias(MIS_HAS_NBR_NOM)],
                )?;

            // ---- a nominated vertex with no nominated neighbour joins the MIS ----
            let joined_mis = probs
                .clone()
                .join(has_nom_nbr, JoinType::Inner, &[MIS_V], &[EDGE_SRC], None)?
                .filter(not(col(MIS_HAS_NBR_NOM)).and(col(MIS_NOM)))?
                .select(vec![col(MIS_V)])?;

            // ---- neighbours of freshly-joined MIS vertices (to be removed) ----
            // The symmetrized edge set makes a single direction sufficient: every
            // neighbour u of a joined vertex v is the source of edge (u, v).
            let neighbors_of_mis = edges
                .clone()
                .join(
                    joined_mis.clone(),
                    JoinType::Inner,
                    &[EDGE_DST],
                    &[MIS_V],
                    None,
                )?
                .select(vec![col(EDGE_SRC).alias(MIS_V)])?;

            // ---- vertices to drop from the active graph: joined MIS + their neighbours ----
            let removed = neighbors_of_mis
                .clone()
                .union(joined_mis.clone())?
                .distinct()?;

            // ---- update the MIS flag: isolated vertices + freshly joined vertices ----
            let new_mis_members = isolated.clone().union(joined_mis.clone())?.distinct()?;

            current_mis = mis_checkpointer
                .push(
                    ctx,
                    &format!("mis_{}", iteration),
                    or_into_mis(current_mis.clone(), new_mis_members)?,
                )
                .await?;

            // ---- new active vertex set: probs minus removed ----
            let removed_r = removed.clone().with_column_renamed(MIS_V, MIS_REM_V)?;
            vertices_left = vertex_ckptr
                .push(
                    ctx,
                    &format!("vertices-{}", iteration),
                    probs
                        .clone()
                        .join(
                            removed_r.clone(),
                            JoinType::LeftAnti,
                            &[MIS_V],
                            &[MIS_REM_V],
                            None,
                        )?
                        .select(vec![col(MIS_V), col(MIS_PROB)])?,
                )
                .await?;

            // ---- contract edges: drop any edge touching a removed vertex ----
            edges = edges_ckptr
                .push(
                    ctx,
                    &format!("contracted_{}", iteration),
                    edges
                        .clone()
                        .join(
                            removed_r.clone(),
                            JoinType::LeftAnti,
                            &[EDGE_SRC],
                            &[MIS_REM_V],
                            None,
                        )?
                        .join(
                            removed_r,
                            JoinType::LeftAnti,
                            &[EDGE_DST],
                            &[MIS_REM_V],
                            None,
                        )?,
                )
                .await?;

            // Count BEFORE evicting: when an iteration empties a frame, the
            // checkpointer returns the in-memory plan (no parquet files were
            // written) and that plan still references the about-to-be-deleted
            // checkpoints. Materialising the counts first keeps those reads valid.
            let cnt_v_left = vertices_left.clone().count().await?;
            let cnt_e_left = edges.clone().count().await?;
            log::info!(
                "iteration {iteration} done, {cnt_v_left} vertices, {cnt_e_left} edges left in the graph"
            );

            if cnt_e_left == 0 {
                // No edges remain among the active vertices, so the survivors are
                // pairwise non-adjacent: the whole remaining set is independent
                // and can join the MIS at once. Sweeping them here (rather than
                // looping once more) also avoids re-entering the loop with an
                // empty edge set whose in-memory frame references already-
                // evicted checkpoints.
                if cnt_v_left > 0 {
                    let remaining = vertices_left.clone().select(vec![col(MIS_V)])?;
                    current_mis = mis_checkpointer
                        .push(
                            ctx,
                            &format!("mis_sweep_{}", iteration),
                            or_into_mis(current_mis.clone(), remaining)?,
                        )
                        .await?;
                }
                converged = true;
            }

            vertex_ckptr.evict_all_but_latest_n(ctx, 1).await?;
            edges_ckptr.evict_all_but_latest_n(ctx, 1).await?;
            mis_checkpointer.evict_all_but_latest_n(ctx, 1).await?;

            iteration += 1;
        }

        log::info!("MIS converged after {iteration} iterations.");

        current_mis
            .filter(col(MIS_FLAG))?
            .select(vec![col(MIS_V).alias(VERTEX_ID)])?
            .write_parquet(output, DataFrameWriteOptions::new(), None)
            .await?;

        vertex_ckptr.purge(ctx).await?;
        edges_ckptr.purge(ctx).await?;
        mis_checkpointer.purge(ctx).await?;

        Ok(iteration)
    }
}

impl GraphFrame {
    /// Constructs an [`MISBuilder`] that computes a (randomized) maximal
    /// independent set using Ghaffari's algorithm.
    ///
    /// The result is written out-of-core to a user-supplied directory as
    /// parquet (a single `id` column with the selected vertices).
    ///
    /// # Example
    /// ```
    /// use datafusion::dataframe;
    /// use datafusion::prelude::SessionContext;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// # async fn run() -> datafusion::error::Result<()> {
    /// let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64, 3i64])?;
    /// let edges = dataframe!(EDGE_SRC => vec![1i64, 2i64], EDGE_DST => vec![2i64, 3i64])?;
    /// let graph = GraphFrame::try_new(vertices, edges)?;
    /// let ctx = SessionContext::new();
    /// // `output` must be a directory URI, e.g. "file:///tmp/mis_out/".
    /// graph.maximal_independent_set().run(&ctx, "file:///tmp/mis_out/").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn maximal_independent_set(&self) -> MISBuilder<'_> {
        MISBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EDGE_DST;
    use datafusion::arrow::array::Int64Array;
    use std::collections::HashSet;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_mis_test_{}_{n}_{label}", id()));
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

    /// Builds a fresh `SessionContext`, a non-overlapping checkpoint dir and a
    /// `file://` output URI, plus a `TempGuard` that cleans up on drop.
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

    /// Runs MIS on `graph`, then reads the result back and returns the set of
    /// selected vertex ids. For an empty graph (no output files written) this
    /// returns an empty set without attempting to read the empty directory.
    async fn run_and_collect(graph: &GraphFrame, label: &str) -> Result<HashSet<i64>> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup(label)?;
        graph
            .maximal_independent_set()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let output_path = Url::parse(&output_uri)
            .expect("valid file:// URL")
            .to_file_path()
            .expect("output URI is a file:// path");
        let has_files = fs::read_dir(&output_path)
            .map(|it| {
                it.filter_map(core::result::Result::ok)
                    .any(|e| e.file_name() != ".DS_Store")
            })
            .unwrap_or(false);
        if !has_files {
            return Ok(HashSet::new());
        }

        let df = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?;
        let batches = df.collect().await?;
        let mut ids = HashSet::new();
        for batch in &batches {
            let arr = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id column is Int64");
            for i in 0..arr.len() {
                ids.insert(arr.value(i));
            }
        }
        Ok(ids)
    }

    /// Collects all (src, dst) edges of `graph` into a Rust vector.
    async fn collect_edges(graph: &GraphFrame) -> Result<Vec<(i64, i64)>> {
        let batches = graph.edges().clone().collect().await?;
        let mut out = Vec::new();
        for batch in &batches {
            let src = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("src is Int64");
            let dst = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("dst is Int64");
            for i in 0..src.len() {
                out.push((src.value(i), dst.value(i)));
            }
        }
        Ok(out)
    }

    /// Collects all vertex ids of `graph` into a Rust set.
    async fn collect_vertices(graph: &GraphFrame) -> Result<HashSet<i64>> {
        let batches = graph.vertices().clone().collect().await?;
        let mut out = HashSet::new();
        for batch in &batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id is Int64");
            for i in 0..ids.len() {
                out.insert(ids.value(i));
            }
        }
        Ok(out)
    }

    /// Independence: no edge of `graph` has both endpoints in the set.
    /// (The graph is treated as undirected, so direction does not matter.)
    async fn is_independent(graph: &GraphFrame, mis: &HashSet<i64>) -> Result<bool> {
        for (s, d) in collect_edges(graph).await? {
            if mis.contains(&s) && mis.contains(&d) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Maximality: every vertex NOT in the set has at least one neighbour that
    /// IS in the set (so no further vertex can be added without breaking
    /// independence). Isolated vertices must therefore be in the set.
    async fn is_maximal(graph: &GraphFrame, mis: &HashSet<i64>) -> Result<bool> {
        let all = collect_vertices(graph).await?;
        let edges = collect_edges(graph).await?;
        for v in &all {
            if mis.contains(v) {
                continue;
            }
            let covered = edges
                .iter()
                .any(|(s, d)| (s == v && mis.contains(d)) || (d == v && mis.contains(s)));
            if !covered {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // --- deterministic-size cases -------------------------------------------

    #[tokio::test]
    async fn test_empty_graph() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => Vec::<i64>::new())?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "empty").await?;
        assert!(mis.is_empty(), "MIS of empty graph should be empty");
        Ok(())
    }

    #[tokio::test]
    async fn test_single_vertex() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![0i64])?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "single").await?;
        assert_eq!(mis.len(), 1, "MIS of single vertex graph has size 1");
        assert!(mis.contains(&0), "MIS should contain the only vertex 0");
        Ok(())
    }

    #[tokio::test]
    async fn test_disconnected_vertices() -> Result<()> {
        let n = 5i64;
        let ids: Vec<i64> = (0..n).collect();
        let vertices = dataframe!(VERTEX_ID => ids)?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "disconnected").await?;
        assert_eq!(
            mis.len(),
            n as usize,
            "MIS of an edge-less graph should contain every vertex"
        );
        assert!(is_independent(&graph, &mis).await?);
        assert!(is_maximal(&graph, &mis).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_complete_graph_k5() -> Result<()> {
        let ids: Vec<i64> = (0..5).collect();
        let vertices = dataframe!(VERTEX_ID => ids)?;
        let mut es = Vec::new();
        let mut ed = Vec::new();
        for i in 0..5i64 {
            for j in (i + 1)..5i64 {
                es.push(i);
                ed.push(j);
            }
        }
        let edges = dataframe!(EDGE_SRC => es, EDGE_DST => ed)?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "complete_k5").await?;
        assert_eq!(
            mis.len(),
            1,
            "MIS of a complete graph should contain exactly one vertex"
        );
        assert!(is_independent(&graph, &mis).await?);
        assert!(is_maximal(&graph, &mis).await?);
        Ok(())
    }

    // --- invariant-only cases (random outcome, but always valid) -------------

    #[tokio::test]
    async fn test_isolated_vertices_are_included() -> Result<()> {
        // Vertices 2 and 3 are isolated; 0 and 1 are linked by a single edge.
        let vertices = dataframe!(VERTEX_ID => vec![0i64, 1, 2, 3])?;
        let edges = dataframe!(EDGE_SRC => vec![0i64], EDGE_DST => vec![1i64])?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "isolated").await?;
        // The two isolated vertices must always be in the MIS, plus exactly one
        // endpoint of the lone edge.
        assert!(mis.contains(&2), "isolated vertex 2 should be in the MIS");
        assert!(mis.contains(&3), "isolated vertex 3 should be in the MIS");
        assert_eq!(
            mis.len(),
            3,
            "MIS should contain the 2 isolated + 1 of the edge"
        );
        assert!(is_independent(&graph, &mis).await?);
        assert!(is_maximal(&graph, &mis).await?);
        Ok(())
    }

    #[tokio::test]
    async fn test_path_graph_invariants() -> Result<()> {
        // A path 1-2-3-4-5-6: MIS size is 3, but the exact members depend on
        // the random draws. Assert only the structural invariants.
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2, 3, 4, 5, 6])?;
        let edges = dataframe!(
            EDGE_SRC => vec![1i64, 2, 3, 4, 5],
            EDGE_DST => vec![2i64, 3, 4, 5, 6],
        )?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "path").await?;
        assert!(
            is_independent(&graph, &mis).await?,
            "result must be independent"
        );
        assert!(is_maximal(&graph, &mis).await?, "result must be maximal");
        Ok(())
    }

    #[tokio::test]
    async fn test_friends_graph_invariants() -> Result<()> {
        // A small "social" graph with a triangle and a couple of branches.
        let vertices = dataframe!(
            VERTEX_ID => vec![1i64, 2, 3, 4, 5, 6, 7],
            "name" => vec!["a", "b", "c", "d", "e", "f", "g"],
        )?;
        let edges = dataframe!(
            EDGE_SRC => vec![1i64, 1, 2, 2, 3, 4, 4, 6],
            EDGE_DST => vec![2i64, 3, 3, 4, 5, 5, 6, 7],
        )?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "friends").await?;
        assert!(
            is_independent(&graph, &mis).await?,
            "result must be independent"
        );
        assert!(is_maximal(&graph, &mis).await?, "result must be maximal");
        assert!(
            !mis.is_empty(),
            "a non-empty connected graph should yield a non-empty MIS"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_cycle_graph_invariants() -> Result<()> {
        // A 6-cycle: no leaves, so every vertex has degree 2. Exercises the
        // probability-update path where d(v) < 2 keeps p pinned at 1/2.
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2, 3, 4, 5, 6])?;
        let edges = dataframe!(
            EDGE_SRC => vec![1i64, 2, 3, 4, 5, 6],
            EDGE_DST => vec![2i64, 3, 4, 5, 6, 1],
        )?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "cycle").await?;
        assert!(
            is_independent(&graph, &mis).await?,
            "result must be independent"
        );
        assert!(is_maximal(&graph, &mis).await?, "result must be maximal");
        // A 6-cycle admits maximal independent sets of size 2 (e.g. {1,4}) or 3
        // (e.g. {1,3,5}); the random outcome must fall in that range.
        assert!(
            (2..=3).contains(&mis.len()),
            "MIS of a 6-cycle should have 2 or 3 vertices, got {}",
            mis.len()
        );
        Ok(())
    }

    /// Regression test for the column-naming bug: the algorithm must actually
    /// run to completion on a graph with linked vertices (the original
    /// implementation failed at join planning with `DuplicateQualifiedField`).
    #[tokio::test]
    async fn test_runs_on_linked_graph() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2, 3])?;
        let edges = dataframe!(
            EDGE_SRC => vec![1i64, 2],
            EDGE_DST => vec![2i64, 3],
        )?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let iters = {
            let (ctx, checkpoint_dir, output_uri, _guard) = setup("linked")?;
            graph
                .maximal_independent_set()
                .set_checkpoint_dir(checkpoint_dir)
                .run(&ctx, &output_uri)
                .await?
        };
        assert!(
            iters > 0,
            "a linked graph should take at least one iteration"
        );
        Ok(())
    }

    /// Regression test for the edge-projection bug: `symmetrize` must not be
    /// fed the raw (attributed) edge frame, both to avoid a union-schema crash
    /// and to keep the edge checkpoint at just `[src, dst]`. MIS must run to
    /// completion on a graph whose edges carry extra attribute columns and
    /// still produce a valid (independent + maximal) set.
    #[tokio::test]
    async fn test_graph_with_edge_attributes() -> Result<()> {
        let vertices = dataframe!(
            VERTEX_ID => vec![1i64, 2, 3, 4, 5, 6],
            "name" => vec!["a", "b", "c", "d", "e", "f"],
        )?;
        let edges = dataframe!(
            EDGE_SRC => vec![1i64, 2, 3, 4, 5],
            EDGE_DST => vec![2i64, 3, 4, 5, 6],
            "weight" => vec![0.1f64, 0.2, 0.3, 0.4, 0.5],
            "label" => vec!["e1", "e2", "e3", "e4", "e5"],
        )?;
        let graph = GraphFrame::try_new(vertices, edges)?;

        let mis = run_and_collect(&graph, "attributed").await?;
        assert!(
            is_independent(&graph, &mis).await?,
            "result must be independent"
        );
        assert!(is_maximal(&graph, &mis).await?, "result must be maximal");
        assert!(!mis.is_empty());
        Ok(())
    }
}
