use crate::algorithm::pregel::{MessageDirection, pregel_default_msg, pregel_src};
use crate::expressions::kcore_merge_expr;
use crate::memory::CheckpointConfig;
use crate::utils::symmetrize;
use crate::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions_aggregate::array_agg::array_agg;
use datafusion::functions_aggregate::count::count;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;

/// Column name for the computed k-core number in the result.
pub const KCORE: &str = "kcore";

/// Builder for the k-core decomposition.
///
/// Each vertex seeds its core number with its degree, then repeatedly
/// recomputes it from its neighbours' current estimates until no value
/// changes. Core numbers are monotonically non-increasing, so the fixpoint
/// is reached in finitely many iterations.
///
/// The per-vertex update rule (`kcore_merge`) returns the largest `l` such
/// that at least `l` neighbours have a current core estimate `>= l`, capped
/// at the vertex's own previous core.
///
/// Reference: Mandal, Aritra, and Mohammad Al Hasan. "A distributed k-core
/// decomposition algorithm on spark." 2017 IEEE International Conference on
/// Big Data (Big Data). IEEE, 2017.
pub struct KCoreBuilder<'a> {
    graph: &'a GraphFrame,
    max_iter: usize,
    checkpoint_config: CheckpointConfig,
}

impl<'a> KCoreBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        KCoreBuilder {
            graph,
            max_iter: 0,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    /// Maximum number of Pregel iterations. `0` (the default) runs until the
    /// core numbers converge.
    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
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

    /// Runs the algorithm, writing `[VERTEX_ID, KCORE]` as parquet into
    /// `output` (a directory URI such as `file:///path/to/dir/`).
    ///
    /// Returns the number of Pregel iterations executed.
    pub async fn run(
        self,
        ctx: &SessionContext,
        output: &str,
        _include_debug_columns: bool,
    ) -> Result<usize> {
        const DEGREE: &str = "degree";

        // Treat the graph as a simple undirected graph: drop self-loops, add
        // the reverse of every edge, deduplicate. Symmetrizing lets us
        // communicate in both directions using only SrcToDst messages, which
        // is what makes `skip_dest_state` valid here.
        //
        // Overall it should worth it: skip_dest_state should allow
        // to benefit from pre-sorting of edges even if edges are x2
        // of the original size.
        let prepared_edges = symmetrize(
            &self
                .graph
                .edges
                .clone()
                .select_columns(&[EDGE_SRC, EDGE_DST])?,
            true,
        )?;

        // The undirected degree is the out-degree of the symmetrized graph.
        //
        // This op can be done with spilling, so not a big problem.
        //
        // TODO: in the future would be nice to refactor the pregel
        // so we can write edges first and then do some "preprocessing"
        // of vertices. Something like "prepare API".
        let degrees = prepared_edges.clone().aggregate(
            vec![col(EDGE_SRC)],
            vec![count(col(EDGE_DST)).alias(DEGREE)],
        )?;

        // Every original vertex participates; isolated ones get degree 0.
        let prepared_vertices = self
            .graph
            .vertices
            .clone()
            .select(vec![col(VERTEX_ID)])?
            .join(degrees, JoinType::Left, &[VERTEX_ID], &[EDGE_SRC], None)?
            .with_column(DEGREE, coalesce(vec![col(DEGREE), lit(0i64)]))?
            .select(vec![col(VERTEX_ID), col(DEGREE)])?;

        let prepared_graph = GraphFrame {
            vertices: prepared_vertices,
            edges: prepared_edges,
        };

        let new_core = kcore_merge_expr(pregel_default_msg(), col(KCORE));

        let mut pregel_builder = prepared_graph
            .pregel()
            .add_vertex_column(KCORE, col(DEGREE), new_core.clone())
            // Each vertex broadcasts its current core estimate to its
            // neighbours. All vertices send every iteration: the aggregate
            // rebuilds each neighbour list from scratch, so a vertex that
            // stopped sending would vanish from its neighbours' lists and
            // corrupt their estimates. Early stopping therefore relies on
            // the voting column alone, never on participation pruning.
            .add_message(pregel_src(KCORE), MessageDirection::SrcToDst)
            .add_aggregate_expr(array_agg(pregel_default_msg()))
            // A vertex votes "still active" exactly while its core number
            // changed this iteration; the run stops once nobody changed.
            .with_vertex_voting("active", col(KCORE).not_eq(new_core.clone()))
            .skip_dest_state()
            .with_checkpoint_store(self.checkpoint_config.store_url.clone())
            .set_checkpoint_dir(self.checkpoint_config.dir.clone());

        if self.max_iter > 0 {
            pregel_builder = pregel_builder.max_iterations(self.max_iter);
        }

        let num_iterations = pregel_builder.run(ctx, output, false).await?;
        Ok(num_iterations)
    }
}

impl GraphFrame {
    /// Constructs a [`KCoreBuilder`] computing the k-core (coreness) of every
    /// vertex.
    pub fn k_core(&self) -> KCoreBuilder<'_> {
        KCoreBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::collect_to_i64;
    use datafusion::arrow::array::Int64Array;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_kcore_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

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

    fn create_graph(vertices: Vec<i64>, edges: Vec<(i64, i64)>) -> Result<GraphFrame> {
        let vertices_df = dataframe!(VERTEX_ID => vertices)?;
        let (srcs, dsts): (Vec<i64>, Vec<i64>) = edges.into_iter().unzip();
        let edges_df = dataframe!(EDGE_SRC => srcs, EDGE_DST => dsts)?;
        Ok(GraphFrame {
            vertices: vertices_df,
            edges: edges_df,
        })
    }

    async fn read_result(ctx: &SessionContext, output_uri: &str) -> Result<DataFrame> {
        ctx.read_parquet(output_uri, ParquetReadOptions::default())
            .await
    }

    async fn kcore_map(ctx: &SessionContext, output_uri: &str) -> Result<HashMap<i64, i64>> {
        let result = read_result(ctx, output_uri)
            .await?
            .select(vec![col(VERTEX_ID), col(KCORE)])?;
        let collected = result.collect().await?;
        let mut map = HashMap::new();
        for batch in &collected {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let cores = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..ids.len() {
                map.insert(ids.value(i), cores.value(i));
            }
        }
        Ok(map)
    }

    #[tokio::test]
    async fn test_empty_graph() -> Result<()> {
        let graph = create_graph(vec![], vec![])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("empty")?;
        // An empty graph produces no rows; like the connected-components empty
        // case the output directory cannot be read back as parquet, so we only
        // assert that `run` completes successfully.
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_single_vertex() -> Result<()> {
        let graph = create_graph(vec![0], vec![])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("single_vertex")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let result = read_result(&ctx, &output_uri).await?;
        assert_eq!(result.clone().count().await?, 1);
        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map[&0], 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_two_connected_vertices() -> Result<()> {
        let graph = create_graph(vec![0, 1], vec![(0, 1)])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("two_connected")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map.len(), 2);
        // Both vertices form a 1-core.
        assert_eq!(map[&0], 1);
        assert_eq!(map[&1], 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_triangle_graph() -> Result<()> {
        let graph = create_graph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("triangle")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map.len(), 3);
        // A triangle is a 2-core.
        for v in 0..3 {
            assert_eq!(map[&v], 2, "vertex {v}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_star_graph() -> Result<()> {
        // Center (0) connected to three leaves. Each leaf has only one
        // neighbour, so no 2-core can form and everything collapses to 1.
        let graph = create_graph(vec![0, 1, 2, 3], vec![(0, 1), (0, 2), (0, 3)])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("star")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map.len(), 4);
        for v in 0..4 {
            assert_eq!(map[&v], 1, "vertex {v}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_chain_graph() -> Result<()> {
        // Open chain 0 - 1 - 2: endpoints have degree 1, so only a 1-core.
        let graph = create_graph(vec![0, 1, 2], vec![(0, 1), (1, 2)])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("chain")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map.len(), 3);
        for v in 0..3 {
            assert_eq!(map[&v], 1, "vertex {v}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_disconnected_vertices() -> Result<()> {
        let graph = create_graph(vec![0, 1, 2], vec![])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("disconnected")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map.len(), 3);
        for v in 0..3 {
            assert_eq!(map[&v], 0, "vertex {v}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_triangle_with_tail() -> Result<()> {
        // Triangle 1-2-3-1 (2-core) with a pendant tail 1-4-5 (1-core).
        // Degrees: 1->3, 2->2, 3->2, 4->2, 5->1 (degree != kcore for 1 and 4),
        // which is exactly what catches an implementation that converges too
        // early and returns kcore == degree for everyone.
        let graph = create_graph(
            vec![1, 2, 3, 4, 5],
            vec![(1, 2), (2, 3), (3, 1), (1, 4), (4, 5)],
        )?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("triangle_tail")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map[&1], 2);
        assert_eq!(map[&2], 2);
        assert_eq!(map[&3], 2);
        assert_eq!(map[&4], 1);
        assert_eq!(map[&5], 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_hierarchical_structure() -> Result<()> {
        // Core (k>=4): vertices 0-4 fully connected.
        // Mid layer (2<=k<=4): vertices 5-14 connect into the core.
        // Outer layer (k<=2): vertices 15-29 form sparse chains off the middle.
        let vertices: Vec<i64> = (0..30).collect();
        let mut core = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                core.push((i as i64, j as i64));
            }
        }
        let mid = vec![
            (5, 0),
            (5, 1),
            (5, 2),
            (6, 0),
            (6, 1),
            (6, 3),
            (7, 1),
            (7, 2),
            (7, 4),
            (8, 0),
            (8, 3),
            (8, 4),
            (9, 1),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 4),
            (11, 2),
            (11, 3),
            (12, 1),
            (12, 4),
            (13, 0),
            (13, 2),
            (14, 3),
            (14, 4),
        ];
        let outer = vec![
            (15, 5),
            (16, 6),
            (17, 7),
            (18, 8),
            (19, 9),
            (20, 10),
            (21, 11),
            (22, 12),
            (23, 13),
            (24, 14),
            (25, 15),
            (26, 16),
            (27, 17),
            (28, 18),
            (29, 19),
        ];
        let mut all_edges = Vec::new();
        all_edges.extend(core);
        all_edges.extend(mid);
        all_edges.extend(outer);

        let graph = create_graph(vertices, all_edges)?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("hierarchical")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map.len(), 30);
        for id in 0..5 {
            assert!(
                map[&id] >= 4,
                "core vertex {id} should be >= 4, got {}",
                map[&id]
            );
        }
        for id in 5..15 {
            assert!(
                (2..=4).contains(&map[&id]),
                "mid vertex {id} should be 2..=4, got {}",
                map[&id]
            );
        }
        for id in 15..30 {
            assert!(
                map[&id] <= 2,
                "outer vertex {id} should be <= 2, got {}",
                map[&id]
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_directed_input_treated_as_undirected() -> Result<()> {
        // A reciprocal edge (a->b and b->a) must count as a single undirected
        // connection: two vertices with degree 1 form a 1-core, not a 2-core.
        let graph = create_graph(vec![0, 1], vec![(0, 1), (1, 0)])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("reciprocal")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let map = kcore_map(&ctx, &output_uri).await?;
        assert_eq!(map[&0], 1);
        assert_eq!(map[&1], 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_max_iter_zero_runs_to_convergence() -> Result<()> {
        // Default (max_iter=0) must still converge: triangle -> 2.
        let graph = create_graph(vec![0, 1, 2], vec![(0, 1), (1, 2), (2, 0)])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("default_converge")?;
        graph
            .k_core()
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let result = read_result(&ctx, &output_uri)
            .await?
            .select(vec![col(VERTEX_ID), col(KCORE)])?;
        let values = collect_to_i64(&result.sort_by(vec![col(VERTEX_ID)])?, 1).await?;
        assert_eq!(values, &[2, 2, 2]);
        Ok(())
    }
}
