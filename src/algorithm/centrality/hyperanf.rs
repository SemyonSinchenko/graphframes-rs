use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::Result;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;
use datafusion::{execution::object_store::ObjectStoreUrl, prelude::SessionContext};

use crate::algorithm::pregel::{MessageDirection, pregel_default_msg, pregel_src};
use crate::expressions::{hll_long, hll_long_aggregate, hll_long_estimate, hll_long_union};
use crate::utils::symmetrize;
use crate::{EDGE_DST, EDGE_SRC, VERTEX_ID};
use crate::{GraphFrame, memory::CheckpointConfig};

const BALL: &str = "ball";

/// Output column: estimated number of vertices reachable within `n_hops` hops.
pub const NEIGHBORHOOD_SIZE: &str = "neighborhood_size";

/// Builder for the HyperANF approximate neighbourhood computation.
///
/// Each vertex `v` carries a HyperLogLog sketch `ball(v)` approximating the
/// set of vertices within the current number of hops. It is seeded with the
/// singleton `{v}` and grown each Pregel iteration by folding in the union of
/// its neighbours' sketches: `ball_{h+1}(v) = ball_h(v) ∪ ⋃_{u ∈ N(v)} ball_h(u)`.
/// After `n_hops` iterations the per-vertex result is the HLL estimate of its
/// ball (i.e. `HyperANF(n_hops)`). There is no convergence voting, so `n_hops`
/// is also the iteration budget — set it to at least the graph diameter to
/// obtain the converged ball.
///
/// Reference: Boldi, Paolo, Marco Rosa, and Sebastiano Vigna. "HyperANF:
/// Approximating the neighbourhood function of very large graphs on a budget."
/// Proceedings of the 20th international conference on World Wide Web. 2011.
pub struct HyperANFBuilder<'a> {
    graph: &'a GraphFrame,
    directed: bool,
    n_hops: usize,
    lg_k: u8,
    checkpoint_config: CheckpointConfig,
}

impl<'a> HyperANFBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        HyperANFBuilder {
            graph: graph,
            directed: true,
            n_hops: 2,
            lg_k: 12,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    pub fn n_hops(mut self, n_hops: usize) -> Self {
        self.n_hops = n_hops;
        self
    }

    pub fn lg_k(mut self, lg_k: u8) -> Self {
        if (lg_k < 4) || (lg_k > 21) {
            panic!("lg_k should be in [4, 21]!")
        }
        self.lg_k = lg_k;
        self
    }

    pub fn directed(mut self, flag: bool) -> Self {
        self.directed = flag;
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

    pub async fn run(
        self,
        ctx: &SessionContext,
        output: &str,
        _include_debug_columns: bool,
    ) -> Result<usize> {
        let edges = if self.directed {
            self.graph
                .edges
                .clone()
                .select_columns(&[EDGE_SRC, EDGE_DST])?
        } else {
            symmetrize(
                &self
                    .graph
                    .edges
                    .clone()
                    .select_columns(&[EDGE_SRC, EDGE_DST])?,
                true,
            )?
        };

        let vertices = self.graph.vertices.clone().select_columns(&[VERTEX_ID])?;
        let g = GraphFrame { vertices, edges };

        let intermediate_dir = self.checkpoint_config.dir.clone().join("_pregel_raw");
        let intermediate_uri = format!(
            "{}{}/",
            self.checkpoint_config.store_url.clone().as_str(),
            intermediate_dir
        );

        let pregel_builder = g
            .pregel()
            .skip_dest_state()
            // Hand Pregel its own sub-directory so the raw-output directory
            // (`_pregel_raw`, a sibling) does not overlap Pregel's checkpoint
            // dir — `validate_output` rejects nested paths.
            .set_checkpoint_dir(self.checkpoint_config.dir.clone().join("inner_checkpoint"))
            .with_checkpoint_store(self.checkpoint_config.store_url.clone())
            .add_vertex_column(
                BALL,
                hll_long(col(VERTEX_ID), self.lg_k),
                when(pregel_default_msg().is_null(), col(BALL))
                    .otherwise(hll_long_union(col(BALL), pregel_default_msg()))?,
            )
            .add_message(pregel_src(BALL), MessageDirection::SrcToDst)
            // Collapse the per-edge messages into one sketch per destination
            // vertex before the update.
            .add_aggregate_expr(hll_long_aggregate(pregel_default_msg()))
            .with_participation_column(
                "changed",
                lit(true),
                when(pregel_default_msg().is_null(), lit(false)).otherwise(
                    hll_long_union(col("ball"), pregel_default_msg()).not_eq(col("ball")),
                )?,
            )
            .max_iterations(self.n_hops);

        let num_iterations = pregel_builder
            .run(&ctx, &intermediate_uri, _include_debug_columns)
            .await?;

        let raw_balls = ctx
            .read_parquet(&intermediate_uri, ParquetReadOptions::default())
            .await?;
        let result = raw_balls.select(vec![
            col(VERTEX_ID),
            hll_long_estimate(col(BALL)).alias(NEIGHBORHOOD_SIZE),
        ])?;

        result
            .write_parquet(output, DataFrameWriteOptions::new(), None)
            .await?;

        Ok(num_iterations)
    }
}

impl GraphFrame {
    /// Constructs a [`HyperANFBuilder`] computing the approximate neighbourhood
    /// (HyperANF) ball size for every vertex within `n_hops` hops.
    ///
    /// The result parquet contains `[VERTEX_ID, NEIGHBORHOOD_SIZE]`, where the
    /// size is the HLL-estimated number of vertices within `n_hops` hops. In
    /// directed mode messages propagate source->destination, so each vertex
    /// accumulates its IN-neighbourhood (the set of vertices that can reach
    /// it); pass `directed(false)` for the symmetric (undirected) neighbourhood.
    pub fn hyperanf(&self) -> HyperANFBuilder<'_> {
        HyperANFBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Float64Array, Int64Array};
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("graphframes_hyperanf_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Builds a `SessionContext`, a non-overlapping checkpoint dir and `file://`
    /// output URI, plus a `TempGuard` that cleans up on drop. Mirrors the
    /// scaffolding used by the other algorithm tests.
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

    /// Reads the `[id, neighborhood_size]` result into a map. Columns are read
    /// by position so the helper does not depend on the estimate column's name.
    async fn neighborhood_sizes(
        ctx: &SessionContext,
        output_uri: &str,
    ) -> Result<HashMap<i64, f64>> {
        let df = ctx
            .read_parquet(output_uri, ParquetReadOptions::default())
            .await?;
        let batches = df.collect().await?;
        let mut map = HashMap::new();
        for batch in &batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let sizes = batch
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..ids.len() {
                map.insert(ids.value(i), sizes.value(i));
            }
        }
        Ok(map)
    }

    /// HLL is near-exact at these small cardinalities (lg_k=12), so a 0.5
    /// absolute tolerance both rounds to the true integer count and catches any
    /// participation/aggregation bug that drops a neighbour (a shift of >= 1).
    fn assert_size(map: &HashMap<i64, f64>, id: i64, expected: i64) {
        let got = *map
            .get(&id)
            .unwrap_or_else(|| panic!("vertex {id} missing from result"));
        assert!(
            (got - expected as f64).abs() < 0.5,
            "vertex {id}: expected ball size ~{expected}, got {got}"
        );
    }

    /// Directed chain `1->2->3->4`, `n_hops=1`: each ball is `{self} U in-neighbours`.
    /// Also pins the public output schema (`id`, `neighborhood_size`).
    #[tokio::test]
    async fn test_directed_chain_radius_1() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("directed_chain_r1")?;
        graph
            .hyperanf()
            .n_hops(1)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let result = ctx
            .read_parquet(&out, ParquetReadOptions::default())
            .await?;
        let schema = result.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), VERTEX_ID);
        assert_eq!(schema.field(1).name(), NEIGHBORHOOD_SIZE);

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        assert_size(&sizes, 1, 1);
        assert_size(&sizes, 2, 2);
        assert_size(&sizes, 3, 2);
        assert_size(&sizes, 4, 2);
        Ok(())
    }

    /// `n_hops=2` on the same chain: the frontier reaches two hops upstream.
    #[tokio::test]
    async fn test_directed_chain_radius_2() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("directed_chain_r2")?;
        graph
            .hyperanf()
            .n_hops(2)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        assert_size(&sizes, 1, 1);
        assert_size(&sizes, 2, 2);
        assert_size(&sizes, 3, 3);
        assert_size(&sizes, 4, 3);
        Ok(())
    }

    /// `n_hops=3` reaches the full directed diameter: vertex 4 sees {1,2,3,4}.
    #[tokio::test]
    async fn test_directed_chain_radius_3() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("directed_chain_r3")?;
        graph
            .hyperanf()
            .n_hops(3)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        assert_size(&sizes, 1, 1);
        assert_size(&sizes, 2, 2);
        assert_size(&sizes, 3, 3);
        assert_size(&sizes, 4, 4);
        Ok(())
    }

    /// `n_hops` well past the diameter must converge to the same balls. This is
    /// the correctness guard for participation pruning on overshoot: once a
    /// vertex's sketch stops changing it stops sending, and the now-redundant
    /// tail iterations must not corrupt the result.
    #[tokio::test]
    async fn test_directed_chain_overshoot_converges() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("directed_chain_overshoot")?;
        graph
            .hyperanf()
            .n_hops(10)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        // Identical to radius_3: the full directed reachability set.
        assert_size(&sizes, 1, 1);
        assert_size(&sizes, 2, 2);
        assert_size(&sizes, 3, 3);
        assert_size(&sizes, 4, 4);
        Ok(())
    }

    /// Diamond `1->2, 1->3, 2->4, 3->4`, `n_hops=2`. Vertex 4 has in-degree 2
    /// and must union BOTH upstream sketches into `{1,2,3,4}`. This is the test
    /// that catches a missing message aggregate: without one, vertex 4 is
    /// duplicated (one row per in-edge) and never sees the full set.
    #[tokio::test]
    async fn test_diamond_aggregates_multiple_neighbours() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (1, 3), (2, 4), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("diamond")?;
        graph
            .hyperanf()
            .n_hops(2)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        // Exactly one row per vertex (no duplication from the 2 in-edges at v4).
        assert_eq!(sizes.len(), 4);
        assert_size(&sizes, 1, 1);
        assert_size(&sizes, 2, 2);
        assert_size(&sizes, 3, 2);
        assert_size(&sizes, 4, 4);
        Ok(())
    }

    /// An isolated vertex never receives a message; its ball stays `{self}`.
    #[tokio::test]
    async fn test_isolated_vertex_stays_singleton() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3], vec![(1, 2)])?;
        let (ctx, ckpt, out, _g) = setup("isolated")?;
        graph
            .hyperanf()
            .n_hops(5)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        // Directed edge 1->2: messages flow SrcToDst, so each vertex accumulates
        // the balls of its IN-neighbours (who can reach it). v2 is reachable
        // from v1; v1 and the isolated v3 see only themselves.
        assert_size(&sizes, 1, 1); // {1}
        assert_size(&sizes, 2, 2); // {1,2}
        assert_size(&sizes, 3, 1); // {3} isolated
        Ok(())
    }

    /// Undirected chain (symmetrized internally), `n_hops=1`.
    #[tokio::test]
    async fn test_undirected_chain_radius_1() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("undirected_chain_r1")?;
        graph
            .hyperanf()
            .directed(false)
            .n_hops(1)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        assert_size(&sizes, 1, 2);
        assert_size(&sizes, 2, 3);
        assert_size(&sizes, 3, 3);
        assert_size(&sizes, 4, 2);
        Ok(())
    }

    /// Undirected chain, `n_hops` past the diameter: every vertex reaches all 4.
    #[tokio::test]
    async fn test_undirected_chain_converged() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("undirected_chain_converged")?;
        graph
            .hyperanf()
            .directed(false)
            .n_hops(10)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        for v in 1..=4 {
            assert_size(&sizes, v, 4);
        }
        Ok(())
    }

    /// `n_hops=0` runs zero Pregel iterations: every ball is the init `{self}`.
    #[tokio::test]
    async fn test_n_hops_zero_returns_singletons() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![(1, 2), (2, 3), (3, 4)])?;
        let (ctx, ckpt, out, _g) = setup("zero_hops")?;
        graph
            .hyperanf()
            .n_hops(0)
            .set_checkpoint_dir(ckpt)
            .run(&ctx, &out, false)
            .await?;

        let sizes = neighborhood_sizes(&ctx, &out).await?;
        for v in 1..=4 {
            assert_size(&sizes, v, 1);
        }
        Ok(())
    }

    /// `lg_k` outside `[4, 21]` fails fast at builder construction.
    #[test]
    #[should_panic(expected = "lg_k should be in [4, 21]!")]
    fn test_lg_k_out_of_range_panics() {
        let graph = create_graph(vec![1], vec![]).unwrap();
        let _ = graph.hyperanf().lg_k(3);
    }
}
