use datafusion::{
    error::Result, execution::object_store::ObjectStoreUrl, object_store::path::Path, prelude::*,
};

use crate::{
    EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID,
    algorithm::pregel::{MessageDirection, pregel_default_msg, pregel_src},
    expressions::most_common_expr,
    memory::CheckpointConfig,
    utils::symmetrize,
};
use datafusion::functions_aggregate::array_agg::array_agg;

pub const COMMUNITY: &str = "community";

pub struct ClassicalLPBuilder<'a> {
    graph: &'a GraphFrame,
    directed: bool,
    max_iter: usize,
    checkpoint_config: CheckpointConfig,
}

impl<'a> ClassicalLPBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        ClassicalLPBuilder {
            graph: graph,
            directed: true, // LDBC canonical implementation
            max_iter: 10,   // LDBC default
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
        self
    }

    /// Whether to treat the graph as directed.
    /// By default this library follows the LDBC semantic
    /// and treat all edges as bidirectional for directed graph.
    ///
    /// For undirected graph symmetrization is skipped.
    pub fn directed(mut self, directed: bool) -> Self {
        self.directed = directed;
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
                false,
            )?
        };

        let vertices = self.graph.vertices.clone().select_columns(&[VERTEX_ID])?;
        let g = GraphFrame { vertices, edges };

        let pregel = g
            .pregel()
            // Label starts as the vertex's own id. Each iteration a vertex
            // adopts the most common label among its neighbours; a vertex that
            // received no message (e.g. isolated) keeps its current label.
            .add_vertex_column(
                COMMUNITY,
                col(VERTEX_ID),
                coalesce(vec![pregel_default_msg(), col(COMMUNITY)]),
            )
            .add_message(pregel_src(COMMUNITY), MessageDirection::SrcToDst)
            // Nested expression: DataFusion splits `most_common(array_agg(...))`
            // into the array_agg aggregate plus a most_common projection on top,
            // so the checkpoint spill writes only the reduced O(|V|) result.
            .add_aggregate_expr(most_common_expr(array_agg(pregel_default_msg())))
            .max_iterations(self.max_iter)
            .skip_dest_state()
            .with_checkpoint_store(self.checkpoint_config.store_url.clone())
            .set_checkpoint_dir(self.checkpoint_config.dir.clone());

        let num_iterations = pregel.run(ctx, output, false).await?;
        Ok(num_iterations)
    }
}

impl GraphFrame {
    /// Constructs a [`ClassicalLPBuilder`] running classical (Raghavan-style)
    /// Label Propagation.
    pub fn classical_lp(&self) -> ClassicalLPBuilder<'_> {
        ClassicalLPBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::create_ldbc_test_graph;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_cdlp_test_{}_{n}_{label}", id()));
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

    /// Reads the LDBC reference CDLP communities (`<dataset>-CDLP.csv`, matching
    /// the `-WCC.csv` / `-BFS.csv` / `-PR.csv` convention): `vertex_id
    /// expected_community`, space-delimited.
    async fn get_ldbc_cdlp_results(dataset: &str) -> Result<DataFrame> {
        let ctx = SessionContext::new();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let schema = Schema::new(vec![
            Field::new("vertex_id", DataType::Int64, false),
            Field::new("expected_community", DataType::Int64, false),
        ]);
        let path = format!(
            "{}/testing/data/ldbc/{}/{}-CDLP.csv",
            manifest_dir, dataset, dataset
        );
        Ok(ctx
            .read_csv(
                &path,
                CsvReadOptions::new()
                    .delimiter(b' ')
                    .has_header(false)
                    .schema(&schema),
            )
            .await?)
    }

    /// Correctness against the LDBC `test-cdlp-directed` reference. CDLP runs
    /// on the undirected graph (`.directed(false)`), and crucially the
    /// symmetrization does NOT deduplicate: a mutual edge `u <-> v` contributes
    /// twice to each endpoint's label tally, which is what matches the LDBC
    /// reference (see `symmetrize(.., false)`). Labels are exact integers, so
    /// the check is an exact per-vertex match (no tolerance, unlike PageRank).
    #[tokio::test]
    async fn test_classical_lp_ldbc_directed() -> Result<()> {
        let test_dataset = "test-cdlp-directed";
        let graph = create_ldbc_test_graph(test_dataset, false, false).await?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("classical_lp_ldbc")?;
        graph
            .classical_lp()
            .directed(false)
            .max_iter(5)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let calculated = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?
            .sort(vec![col(VERTEX_ID).sort(true, true)])?
            .cache()
            .await?;
        let expected = get_ldbc_cdlp_results(test_dataset).await?;

        // Every vertex must be present.
        assert_eq!(
            calculated.clone().count().await?,
            8,
            "expected all 8 vertices"
        );

        // Exact match: no vertex may disagree with the reference community.
        let mismatches = calculated
            .clone()
            .join(
                expected,
                JoinType::Inner,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .filter(col(COMMUNITY).not_eq(col("expected_community")))?
            .count()
            .await?;
        assert_eq!(
            mismatches, 0,
            "{mismatches} vertices have a wrong community"
        );
        Ok(())
    }

    /// Builds a small `GraphFrame` from vertex ids and `(src, dst)` edges.
    fn create_graph(vertices: Vec<i64>, edges: Vec<(i64, i64)>) -> Result<GraphFrame> {
        let vertices_df = dataframe!(VERTEX_ID => vertices)?;
        let (srcs, dsts): (Vec<i64>, Vec<i64>) = edges.into_iter().unzip();
        let edges_df = dataframe!(EDGE_SRC => srcs, EDGE_DST => dsts)?;
        Ok(GraphFrame {
            vertices: vertices_df,
            edges: edges_df,
        })
    }

    /// Smoke test: two disconnected triangles. Synchronous CDLP with
    /// smallest-label tie-break converges each component to its minimum id, so
    /// triangle {1,2,3} -> 1 and triangle {4,5,6} -> 4. Exercises the engine
    /// end-to-end independently of the LDBC fixture.
    #[tokio::test]
    async fn test_classical_lp_two_triangles() -> Result<()> {
        let graph = create_graph(
            vec![1, 2, 3, 4, 5, 6],
            vec![
                (1, 2),
                (2, 1),
                (2, 3),
                (3, 2),
                (1, 3),
                (3, 1),
                (4, 5),
                (5, 4),
                (5, 6),
                (6, 5),
                (4, 6),
                (6, 4),
            ],
        )?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("two_triangles")?;
        graph
            .classical_lp()
            .directed(false)
            .max_iter(3)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let out = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?
            .select(vec![col(VERTEX_ID), col(COMMUNITY)])?
            .sort(vec![col(VERTEX_ID).sort(true, true)])?
            .collect()
            .await?;
        let ids = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let comms = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let want = [1i64, 1, 1, 4, 4, 4];
        for i in 0..want.len() {
            assert_eq!(comms.value(i), want[i], "vertex {}", ids.value(i));
        }
        Ok(())
    }

    /// No edges at all: every vertex is isolated and must keep its own id as its
    /// community, regardless of iteration count (the `coalesce` fallback in the
    /// update rule is what guarantees this).
    #[tokio::test]
    async fn test_classical_lp_isolated_vertices() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3, 4], vec![])?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("isolated")?;
        graph
            .classical_lp()
            .directed(false)
            .max_iter(5)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;

        let out = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?
            .select(vec![col(VERTEX_ID), col(COMMUNITY)])?
            .sort(vec![col(VERTEX_ID).sort(true, true)])?
            .collect()
            .await?;
        let ids = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let comms = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..ids.len() {
            assert_eq!(
                comms.value(i),
                ids.value(i),
                "isolated vertex {} must keep its own id as community",
                ids.value(i)
            );
        }
        Ok(())
    }

    /// Runs the chain `1-2-3-4-5-6` (undirected) for `max_iter` iterations and
    /// returns the per-vertex community in id order.
    async fn run_chain(max_iter: usize) -> Result<Vec<i64>> {
        let graph = create_graph(
            vec![1, 2, 3, 4, 5, 6],
            vec![(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        )?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup(&format!("chain_{max_iter}"))?;
        graph
            .classical_lp()
            .directed(false)
            .max_iter(max_iter)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        let out = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?
            .select(vec![col(VERTEX_ID), col(COMMUNITY)])?
            .sort(vec![col(VERTEX_ID).sort(true, true)])?
            .collect()
            .await?;
        let comms = out[0]
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        Ok((0..comms.len()).map(|i| comms.value(i)).collect())
    }

    /// On a path the minimum label (1) propagates one hop per iteration. After
    /// `max_iter = k`, vertex `k+1` carries label 1 while vertex `k+2` (when it
    /// exists) does not yet — i.e. the label-1 frontier sits at depth `k`.
    /// Increasing `max_iter` pushes that frontier one vertex further each time.
    #[tokio::test]
    async fn test_classical_lp_chain_propagation_depth() -> Result<()> {
        for max_iter in 1..=4 {
            let comms = run_chain(max_iter).await?;
            // vertex (max_iter + 1) is at index max_iter and must be labelled 1
            assert_eq!(
                comms[max_iter],
                1,
                "max_iter={max_iter}: vertex {} should carry label 1, got {comms:?}",
                max_iter + 1
            );
            // vertex (max_iter + 2), if present, must NOT yet carry label 1
            if max_iter + 1 < comms.len() {
                assert_ne!(
                    comms[max_iter + 1],
                    1,
                    "max_iter={max_iter}: vertex {} should not yet carry label 1, got {comms:?}",
                    max_iter + 2
                );
            }
        }
        Ok(())
    }
}
