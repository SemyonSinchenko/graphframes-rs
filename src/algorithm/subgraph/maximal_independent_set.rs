//! Maximal Independent Set
//!
//! Implementation inspired by the Spark GraphFrames Scala code distributed
//! under Apache License 2.0, ported to this project's out-of-core style.
//!
//! Algorithm: "An Improved Distributed Algorithm for Maximal Independent Set",
//! Mohsen Ghaffari
//!
//! https://arxiv.org/abs/1506.05093
//!
//! # Round structure (fused, shuffle-minimal)
//!
//! One Ghaffari round is:
//!
//! 1. **gather** — one shuffle-free sort-merge pass over the frozen edges
//!    scatters `(p, nom)` of every active vertex to its neighbours, and a
//!    single aggregation computes both `d(v) = sum p(u)` and
//!    `has_nbr_nom(v) = bool_or(nom(u))` in one shuffle. The paper defines
//!    both over the same round-`t` values, so they share one message pass.
//! 2. **vertex program** — the aggregated messages are left-joined back onto
//!    the state. A vertex that received no messages has `d = NULL` (no active
//!    neighbour) and joins the MIS regardless of its draw; this replaces the
//!    classic "isolated vertices" anti-join *and* the final "no edges left"
//!    sweep, which both existed only to drain such vertices. The `p` update
//!    and the election flag are local projections.
//! 3. **removal** — `removed = elected ∪ N(elected)`. `N(elected)` is a
//!    src-keyed join over the frozen edges (mirror symmetry), and the set is
//!    deliberately *not* deduplicated: it only feeds an anti-join, which is
//!    insensitive to duplicate right-hand rows.
//! 4. **checkpoint** — the surviving state is written hash-partitioned and
//!    sorted by vertex id, drawing next round's `nom` at write time, so every
//!    join above sees co-partitioned, sorted inputs and skips both the
//!    shuffle and the sort.
//!
//! Elected vertices are appended to a per-round parquet log. A vertex can be
//! elected at most once (it leaves the active set in the same round), so the
//! log needs neither deduplication nor a join against a full-history flag
//! frame; the final result is the log projected to [`VERTEX_ID`].

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
/// Effective degree `d(v) = sum of p over v's *active* neighbours`.
const MIS_DEG: &str = "__mis_deg";
/// Per-vertex "nominated this round" flag (drawn against the current `p`).
const MIS_NOM: &str = "__mis_nom";
/// Per-vertex "at least one neighbour nominated" flag. `NULL` on the state
/// side of the left join (no active neighbour), coalesced to `false` below.
const MIS_HAS_NBR_NOM: &str = "__mis_has_nbr_nom";

// Throw-away renamed keys used only to disambiguate the two sides of a join.
/// Key of the aggregated-message frame entering the left join with the state.
const MIS_NEW_V: &str = "__mis_new_v";
/// Key of the `removed` frame entering the anti-join with the candidates.
const MIS_REM_V: &str = "__mis_rem_v";
/// Per-vertex "joins the MIS this round": nominated with no nominated
/// neighbour, or no active neighbour at all (`d = NULL`).
const MIS_ELECTED: &str = "__mis_elected";

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

        // a) frozen edge table b) per-vertex state c) per-round candidates
        // d) append-only elected log (never evicted within a run)
        let mut edges_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("edges"));
        let mut state_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("state"));
        let mut cand_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("cand"));
        let mut elected_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("elected"));

        // state_0: every vertex starts active with p = 1/2 (Ghaffari). `nom` is
        // drawn when the state is *written*, so the whole round reads one
        // stable draw per vertex -- `random()` is evaluated per row and per
        // execution, it must never live inside a lazily re-read frame.
        let mut state = state_ckptr
            .push_pre_sorted(
                ctx,
                "state-0",
                self.graph.vertices.clone().select(vec![
                    col(VERTEX_ID).alias(MIS_V),
                    lit(0.5f64).alias(MIS_PROB),
                    random().lt_eq(lit(0.5f64)).alias(MIS_NOM),
                ])?,
                MIS_V,
            )
            .await?;

        // The symmetrized + deduplicated edge set is the only edge data the
        // algorithm ever touches, and it is written exactly once:
        // hash-partitioned and sorted by EDGE_SRC. Dedup is required here: a
        // duplicated edge would inflate the effective-degree sum.
        //
        // Project to just (src, dst) first: the algorithm never reads edge
        // attributes, and `symmetrize`'s reversed half is a 2-column projection
        // so feeding it anything but [src, dst] is both wasted checkpoint bytes
        // and a hard union-schema error ("UNION queries have different number
        // of columns"). The only edge columns worth carrying are the endpoints.
        let edges = edges_ckptr
            .push_pre_sorted(
                ctx,
                "edges",
                symmetrize(
                    &self
                        .graph
                        .edges
                        .clone()
                        .select_columns(&[EDGE_SRC, EDGE_DST])?,
                    true,
                    None,
                )?,
                EDGE_SRC,
            )
            .await?;

        let mut iteration = 0usize;
        loop {
            // ---- gather: one shuffle-free pass over the frozen edges ----
            //
            // Scatter (p, nom) of every *active* source to its neighbours, then
            // compute both round reductions in a single shuffle:
            //   d(v)       = sum of p over active neighbours
            //   has_nom(v) = OR of nom over active neighbours
            // The inner join against the active state IS the edge contraction:
            // a removed vertex is absent from the state, hence sends nothing.
            let gathered = state
                .clone()
                .join(edges.clone(), JoinType::Inner, &[MIS_V], &[EDGE_SRC], None)?
                .select(vec![
                    col(EDGE_DST).alias(MIS_NEW_V),
                    col(MIS_PROB),
                    col(MIS_NOM),
                ])?
                .aggregate(
                    vec![col(MIS_NEW_V)],
                    vec![
                        functions_aggregate::sum::sum(col(MIS_PROB)).alias(MIS_DEG),
                        bool_or(col(MIS_NOM)).alias(MIS_HAS_NBR_NOM),
                    ],
                )?;

            // ---- vertex program: left join back + local projections ----
            //
            // The LEFT join subsumes the old `isolated` anti-join: a vertex
            // with no messages has d = NULL, i.e. no active neighbour, and
            // joins the MIS regardless of its draw. Per the paper, nomination
            // uses the current p_t; only afterwards is p advanced to p_{t+1},
            // which is what the next round reads.
            let candidates_df =
                state
                    .clone()
                    .join(gathered, JoinType::Left, &[MIS_V], &[MIS_NEW_V], None)?
                    .with_column(
                        MIS_ELECTED,
                        col(MIS_DEG)
                            .is_null()
                            .or(col(MIS_NOM)
                                .and(not(coalesce(vec![col(MIS_HAS_NBR_NOM), lit(false)])))),
                    )?
                    .with_column(
                        MIS_PROB,
                        when(col(MIS_DEG).gt_eq(lit(2.0)), col(MIS_PROB).div(lit(2.0)))
                            .when(
                                lit(2.0).mul(col(MIS_PROB)).lt_eq(lit(0.5)),
                                lit(2.0).mul(col(MIS_PROB)),
                            )
                            .otherwise(lit(0.5))?,
                    )?
                    .select(vec![col(MIS_V), col(MIS_PROB), col(MIS_ELECTED)])?;

            // Materialize the round exactly once; everything below is a cheap
            // filter over this frame. DataFusion plans are trees, not DAGs:
            // without this checkpoint every consumer of the candidate frame
            // would re-run the whole gather above.
            let candidates = cand_ckptr
                .push(ctx, &format!("cand_{iteration}"), candidates_df)
                .await?;

            // ---- elected log: append-only, no flag frame ----
            //
            // A vertex can be elected at most once (it leaves the active set in
            // the same round), so the log needs neither deduplication nor a
            // per-iteration left join against a full-history frame. A round
            // that elects nobody simply writes no files.
            let joined = elected_ckptr
                .push(
                    ctx,
                    &format!("joined_{iteration}"),
                    candidates
                        .clone()
                        .filter(col(MIS_ELECTED))?
                        .select(vec![col(MIS_V)])?,
                )
                .await?;

            // ---- removal: elected ∪ N(elected), *not* deduplicated ----
            //
            // The set only feeds an anti-join, which ignores duplicate
            // right-hand rows, so the old union-distinct shuffle is gone.
            //
            // N(elected) via a src-keyed join: the edge table is symmetric, so
            // {u : (u, x) ∈ E, x elected} == {dst : (x, dst) ∈ E, x elected},
            // and the src-keyed variant is co-partitioned with the frozen edge
            // table (no edge-side shuffle).
            let neighbors_of_elected = edges
                .clone()
                .join(joined.clone(), JoinType::Inner, &[EDGE_SRC], &[MIS_V], None)?
                .select(vec![col(EDGE_DST).alias(MIS_V)])?;
            let removed = neighbors_of_elected.union(joined)?;
            let removed_r = removed.with_column_renamed(MIS_V, MIS_REM_V)?;

            // ---- next state: survivors, drawing next round's nomination ----
            let next_state = candidates
                .clone()
                .filter(not(col(MIS_ELECTED)))?
                .join(removed_r, JoinType::LeftAnti, &[MIS_V], &[MIS_REM_V], None)?
                .select(vec![
                    col(MIS_V),
                    col(MIS_PROB),
                    random().lt_eq(col(MIS_PROB)).alias(MIS_NOM),
                ])?;

            state = state_ckptr
                .push_pre_sorted(ctx, &format!("state-{}", iteration + 1), next_state, MIS_V)
                .await?;

            // The active vertex set is the only thing that can shrink (edges
            // are frozen, elections are appended to the log). Convergence means
            // no active vertex is left; a surviving set that became pairwise
            // non-adjacent drains in the next round through the d = NULL
            // election rule, so no separate "no edges left" sweep is needed.
            let cnt_left = state.clone().count().await?;
            log::info!("iteration {iteration} done, {cnt_left} vertices left in the active graph");

            // Count BEFORE evicting: the count reads the freshly written state
            // checkpoint; after it materializes, older generations are dead.
            state_ckptr.evict_all_but_latest_n(ctx, 1).await?;
            cand_ckptr.evict_all_but_latest_n(ctx, 1).await?;
            // `edges` and the elected log are never evicted.

            iteration += 1;
            if cnt_left == 0 {
                break;
            }
        }

        log::info!("MIS converged after {iteration} iterations.");

        // The result is the elected log projected to the vertex-id column.
        // Rounds that elected nobody wrote no files (and are not tracked), so
        // `read_all` yields `None` for an empty MIS; rounds write
        // pairwise-disjoint id sets, so no dedup pass is needed.
        if let Some(elected) = elected_ckptr.read_all(ctx).await? {
            elected
                .select(vec![col(MIS_V).alias(VERTEX_ID)])?
                .write_parquet(output, DataFrameWriteOptions::new(), None)
                .await?;
        }

        state_ckptr.purge(ctx).await?;
        cand_ckptr.purge(ctx).await?;
        elected_ckptr.purge(ctx).await?;
        edges_ckptr.purge(ctx).await?;

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
