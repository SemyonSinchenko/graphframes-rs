use crate::algorithm::pregel::{MessageDirection, pregel_msg, pregel_src};
use crate::memory::CheckpointConfig;
use crate::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions_aggregate::min_max::min;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;

/// Builder for configuring and running shortest paths computation from a set of source
/// vertices (landmarks) to all other vertices in the graph.
///
/// By default the builder computes distances **from** the landmarks **to** every vertex,
/// i.e. it follows edges in their declared `src -> dst` direction (forward BFS from each
/// landmark). This matches the natural Multi-Source Shortest Paths (MSSP) reading: the
/// landmarks are the sources, and the result is how far each vertex is from the nearest
/// source.
///
/// Use [`ShortestPathsBuilder::to_landmarks`] to compute the symmetric quantity — the
/// distance from each vertex back to the nearest landmark (following edges backwards).
#[derive(Debug, Clone)]
pub struct ShortestPathsBuilder<'a> {
    /// Reference to the graph frame containing vertices and edges
    graph_frame: &'a GraphFrame,
    /// Vector of vertex IDs designated as landmarks (sources)
    landmarks: Vec<i64>,
    /// Maximum number of iterations to run the algorithm
    max_iterations: usize,
    /// When `false` (default): distance from landmarks to vertices (forward `src -> dst`).
    /// When `true`: distance from vertices to landmarks (edges reversed internally).
    to_landmarks: bool,
    /// Storage options
    checkpoint_config: CheckpointConfig,
}

impl<'a> ShortestPathsBuilder<'a> {
    /// Creates a new ShortestPathsBuilder with the specified graph and landmarks.
    ///
    /// By default the builder computes distances **from** the landmarks to all vertices.
    ///
    /// # Arguments
    /// * `graph_frame` - The graph frame to compute the shortest paths on
    /// * `landmarks` - Vector of vertex IDs to use as landmarks (sources)
    pub fn new(graph_frame: &'a GraphFrame, landmarks: Vec<i64>) -> Self {
        let mut sorted_landmarks = landmarks.clone();
        sorted_landmarks.sort();
        Self {
            graph_frame,
            landmarks: sorted_landmarks,
            max_iterations: i32::MAX as usize,
            to_landmarks: false,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    /// Compute distances **from each vertex to the landmarks** instead of the default
    /// (landmarks to vertices).
    ///
    /// Internally this reverses every edge before running the Pregel iteration, so messages
    /// still propagate `src -> dst` but along the original in-edges of each vertex.
    pub fn to_landmarks(mut self) -> Self {
        self.to_landmarks = true;
        self
    }

    /// Sets the maximum number of iterations for the algorithm.
    ///
    /// # Arguments
    /// * `max_iterations` - Maximum number of iterations to run
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
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

    pub async fn run(
        self,
        ctx: &SessionContext,
        output: &str,
        include_debug_columns: bool,
    ) -> Result<usize> {
        const PARTICIPATING: &str = "participating";

        // Initialize participation: only landmarks participate initially
        let init_participating = self.landmarks.iter().fold(lit(false), |acc, &landmark| {
            acc.or(col(VERTEX_ID).eq(lit(landmark)))
        });

        // Update participation condition: a vertex participates if its distance to any
        // landmark would decrease given the incoming messages.
        let update_participating = self.landmarks.iter().fold(lit(false), |acc, &landmark| {
            let lm_str = landmark.to_string();
            let dist_col = format!("dist_{landmark}");
            acc.or(col(&dist_col).gt(pregel_msg(&lm_str)))
        });

        // The Pregel core always propagates messages SrcToDst using `pregel_src(...)`, which
        // computes distance **from landmark to vertex** on the declared edges. To compute the
        // opposite direction (distance from each vertex back to a landmark), we physically
        // reverse the edges before handing them to Pregel — no other wiring changes.
        //
        // While Pregel core supports reversed and bi-directional messages, this does not allow
        // to use skip_dest_state() optimization.
        let new_edges = if self.to_landmarks {
            self.graph_frame.edges.clone().select(vec![
                col(EDGE_DST).alias(EDGE_SRC),
                col(EDGE_SRC).alias(EDGE_DST),
            ])?
        } else {
            self.graph_frame
                .edges
                .clone()
                .select_columns(&[EDGE_SRC, EDGE_DST])?
        };
        let new_vertices = self
            .graph_frame
            .vertices
            .clone()
            .select_columns(&[VERTEX_ID])?;

        let prepared_gf = GraphFrame {
            vertices: new_vertices,
            edges: new_edges,
        };

        let mut pregel_builder = prepared_gf
            .pregel()
            .with_participation_column(
                PARTICIPATING,
                init_participating,
                update_participating.clone(),
            )
            .with_vertex_voting("active", update_participating)
            .max_iterations(self.max_iterations)
            .with_checkpoint_store(self.checkpoint_config.store_url)
            .set_checkpoint_dir(self.checkpoint_config.dir)
            .skip_dest_state();

        // One scalar vertex column, message, and min() aggregate per landmark.
        for &landmark in &self.landmarks {
            let lm_str = landmark.to_string();
            let dist_col = format!("dist_{landmark}");

            // For the landmark itself: distance is 0; for all others: infinity.
            let init_expr =
                when(col(VERTEX_ID).eq(lit(landmark)), lit(0i32)).otherwise(lit(i32::MAX))?;

            // If no message received (NULL from left join), keep existing distance.
            // Otherwise, keep the minimum of current and received distance.
            let update_expr = when(pregel_msg(&lm_str).is_null(), col(&dist_col)).otherwise(
                when(col(&dist_col).lt_eq(pregel_msg(&lm_str)), col(&dist_col))
                    .otherwise(pregel_msg(&lm_str))?,
            )?;

            // Message: source vertex's distance + 1, capped at i32::MAX.
            let msg_expr = when(
                pregel_src(&dist_col).lt(lit(i32::MAX)),
                pregel_src(&dist_col) + lit(1i32),
            )
            .otherwise(lit(i32::MAX))?;

            pregel_builder = pregel_builder
                .add_vertex_column(&dist_col, init_expr, update_expr)
                .add_named_message(&lm_str, msg_expr, MessageDirection::SrcToDst)
                .add_named_aggregate_expr(&lm_str, min(pregel_msg(&lm_str)));
        }

        let num_iterations = pregel_builder
            .run(ctx, output, include_debug_columns)
            .await?;

        Ok(num_iterations)
    }
}

impl GraphFrame {
    /// Computes shortest paths from all vertices to a set of landmark vertices.
    ///
    /// # Arguments
    /// * `landmarks` - Vector of vertex IDs to use as landmarks for computing the shortest paths
    ///
    /// # Returns
    /// a Builder object to configure and execute the shortest paths computation
    pub fn shortest_paths(&self, landmarks: Vec<i64>) -> ShortestPathsBuilder<'_> {
        ShortestPathsBuilder::new(self, landmarks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::create_ldbc_test_graph;
    use datafusion::arrow::array::{Int32Array, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::prelude::SessionContext;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Returns a unique directory under `std::env::temp_dir()` for this test run, creating it.
    /// Combining the PID with a process-wide counter guarantees uniqueness across parallel
    /// `cargo test` invocations and concurrent tests.
    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_sp_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    /// RAII guard that recursively removes the temp directory when dropped, so tests stay
    /// self-contained without depending on the `tempfile` crate.
    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Builds a `SessionContext`, an object_store checkpoint `Path`, an output `file://` URI,
    /// and a `TempGuard` that cleans up on drop. The parent temp dir contains two non-overlapping
    /// siblings — `checkpoints/` and `output/` — so `validate_output` is satisfied.
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

    fn create_small_test_graph(ctx: &SessionContext) -> Result<GraphFrame> {
        let vertices_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3, 4]))],
        )?;
        let vertices = ctx.read_batch(vertices_data)?;

        let edges_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 2, 3, 4, 4, 2, 3])),
                Arc::new(Int64Array::from(vec![2, 3, 4, 4, 1, 2, 1, 2])),
            ],
        )?;
        let edges = ctx.read_batch(edges_data)?;

        Ok(GraphFrame { vertices, edges })
    }

    #[tokio::test]
    async fn test_shortest_paths_single_landmark() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("sp_single_landmark")?;
        let graph = create_small_test_graph(&ctx)?;
        let landmarks = vec![1];
        graph
            .shortest_paths(landmarks)
            .to_landmarks() // distance from each vertex back to the landmark
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        let result = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?;

        // Create expected results with flat distance column
        let expected_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("expected_id", DataType::Int64, false),
                Field::new("expected_dist_1", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(Int32Array::from(vec![0, 1, 2, 1])),
            ],
        )?;
        let expected = ctx.read_batch(expected_data)?;

        // Join and compare results
        let comparison = result.join(
            expected,
            JoinType::Inner,
            &[VERTEX_ID],
            &["expected_id"],
            None,
        )?;

        let diff = comparison
            .filter(col("dist_1").not_eq(col("expected_dist_1")))?
            .select(vec![col(VERTEX_ID), col("dist_1"), col("expected_dist_1")])?;

        assert_eq!(
            diff.count().await?,
            0,
            "Found differences in shortest paths"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_shortest_paths_multiple_landmarks() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("sp_multiple_landmarks")?;
        let graph = create_small_test_graph(&ctx)?;
        let landmarks = vec![1, 4];
        graph
            .shortest_paths(landmarks)
            .to_landmarks() // distance from each vertex back to the nearest landmark
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        let result = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?;

        // Create expected results with flat distance columns
        let expected_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("expected_id", DataType::Int64, false),
                Field::new("expected_dist_1", DataType::Int32, false),
                Field::new("expected_dist_4", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(Int32Array::from(vec![0, 1, 2, 1])),
                Arc::new(Int32Array::from(vec![2, 1, 1, 0])),
            ],
        )?;
        let expected = ctx.read_batch(expected_data)?;

        // Join and compare results
        let comparison = result.join(
            expected,
            JoinType::Inner,
            &[VERTEX_ID],
            &["expected_id"],
            None,
        )?;

        let diff = comparison.filter(
            col("dist_1")
                .not_eq(col("expected_dist_1"))
                .or(col("dist_4").not_eq(col("expected_dist_4"))),
        )?;

        assert_eq!(
            diff.count().await?,
            0,
            "Found differences in shortest paths"
        );
        Ok(())
    }

    async fn get_ldbc_bfs_results(dataset: &str) -> Result<DataFrame> {
        let ctx = SessionContext::new();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected_pr_schema = Schema::new(vec![
            Field::new("vertex_id", DataType::Int64, false),
            Field::new("expected_distance", DataType::Int64, false),
        ]);
        let expected_sp_path = format!(
            "{}/testing/data/ldbc/{}/{}-BFS.csv",
            manifest_dir, dataset, dataset
        );
        let expected_sp = ctx
            .read_csv(
                &expected_sp_path,
                CsvReadOptions::new()
                    .delimiter(b' ')
                    .has_header(false)
                    .schema(&expected_pr_schema),
            )
            .await?;
        Ok(expected_sp)
    }

    #[tokio::test]
    async fn test_ldbc() -> Result<()> {
        let expected_distances = get_ldbc_bfs_results("test-bfs-directed").await?;
        let graph = create_ldbc_test_graph("test-bfs-directed", false, false).await?;
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("ldbc")?;

        graph
            .shortest_paths(vec![1])
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        let results = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?;

        let diff = results
            .join(
                expected_distances,
                JoinType::Left,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .select(vec![
                col(VERTEX_ID),
                col("dist_1").alias("got_distance"),
                when(
                    col("expected_distance").eq(lit(9223372036854775807i64)),
                    lit(i32::MAX as i64),
                )
                .otherwise(col("expected_distance"))
                .unwrap()
                .alias("expected_distance"),
            ])?
            .filter(col("got_distance").not_eq(col("expected_distance")))?;

        assert_eq!(diff.count().await?, 0);

        Ok(())
    }
}
