use crate::pregel::{MessageDirection, PREGEL_MSG, pregel_dst, pregel_src};
use crate::{GraphFrame, VERTEX_ID};
use arrow::compute::min;
use datafusion::arrow;
use datafusion::arrow::array::{ArrayRef, Int32Array, StructArray, as_struct_array};
use datafusion::arrow::datatypes::{DataType, Field, Fields};
use datafusion::common::ScalarValue;
use datafusion::common::cast::as_int32_array;
use datafusion::error::Result;
use datafusion::functions::core::expr_ext::FieldAccessor;
use datafusion::logical_expr::Volatility;
use datafusion::physical_plan::Accumulator;
use datafusion::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Column name for distances in the Shortest Paths algorithm
pub const DISTANCES: &str = "distances";

/// Internal accumulator type for maintaining the shortest path distances from landmarks.
///
/// This accumulator keeps track of the minimum distances from each vertex to a set of landmark vertices.
/// The distances are stored in a HashMap where keys are landmark IDs and values are minimum distances.
#[derive(Debug)]
struct DistancesMap {
    /// Maps landmark vertex IDs to their current minimum distances
    distances: HashMap<i64, i32>,
}

impl DistancesMap {
    fn new(landmarks: Arc<Vec<i64>>) -> Self {
        Self {
            distances: HashMap::from_iter(landmarks.iter().map(|&lm| (lm, i32::MAX))),
        }
    }
}

/// Implementation of DataFusion's Accumulator trait for DistancesMap.
///
/// The accumulator maintains and updates minimum distances from vertices to landmarks:
/// 1. Update batch processes with new distance values:
///    - Examines each landmark's distances from incoming struct arrays
///    - Updates the stored distance if the incoming value is smaller
/// 2. Evaluate converts the current state to ScalarValue
/// 3. State converts distances to struct array format with:
///    - Fields for each landmark ID
///    - Arrays containing current minimum distances
/// 4. Merge batch combines multiple states using the same logic as update
///
impl Accumulator for DistancesMap {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let array_of_structs = as_struct_array(&values[0]);
        for key in self.distances.clone().keys() {
            let array_for_key = array_of_structs.column_by_name(&key.to_string());
            let min_distance_from_incoming = array_for_key
                .map(|array| min(as_int32_array(array).ok()?))
                .unwrap_or(Some(i32::MAX))
                .unwrap_or(i32::MAX);

            if min_distance_from_incoming < self.distances[key] {
                self.distances.insert(*key, min_distance_from_incoming);
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.state().map(|state| state[0].clone())
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sorted_keys = self.distances.keys().clone().collect::<Vec<_>>();
        sorted_keys.sort();
        let fields = Fields::from(
            sorted_keys
                .iter()
                .map(|k| Field::new(k.to_string(), DataType::Int32, true))
                .collect::<Vec<_>>(),
        );
        let arrays = sorted_keys
            .iter()
            .map(|k| self.distances.get(k).unwrap())
            .map(|v| Int32Array::from(vec![*v]))
            .map(|arr| Arc::new(arr) as ArrayRef)
            .collect::<Vec<_>>();
        let nulls = None;
        let scalar_value = ScalarValue::Struct(Arc::new(StructArray::new(fields, arrays, nulls)));

        Ok(vec![scalar_value])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}

/// Builder for configuring and running shortest paths computation from vertices to landmarks.
///
/// This builder helps configure and execute a Pregel algorithm that computes the shortest paths
/// from all vertices in the graph to a specified set of landmark vertices.
pub struct ShortestPathsBuilder<'a> {
    /// Reference to the graph frame containing vertices and edges
    graph_frame: &'a GraphFrame,
    /// Vector of vertex IDs designated as landmarks
    landmarks: Vec<i64>,
    /// Maximum number of iterations to run the algorithm
    max_iterations: usize,
    /// Interval at which to checkpoint the computation state
    checkpoint_interval: usize,
    /// Apply a reversal shortest paths algorithm to get distances from landmarks to node
    reversed: bool,
}

impl<'a> ShortestPathsBuilder<'a> {
    /// Creates a new ShortestPathsBuilder with the specified graph and landmarks.
    ///
    /// # Arguments
    /// * `graph_frame` - The graph frame to compute the shortest paths on
    /// * `landmarks` - Vector of vertex IDs to use as landmarks
    pub fn new(graph_frame: &'a GraphFrame, landmarks: Vec<i64>) -> Self {
        let mut sorted_landmarks = landmarks.clone();
        sorted_landmarks.sort();
        Self {
            graph_frame,
            landmarks: sorted_landmarks,
            max_iterations: i32::MAX as usize,
            checkpoint_interval: 1,
            reversed: false,
        }
    }

    /// Sets the direction for shortest paths' computation.
    ///
    /// # Arguments
    /// * `reversed` - If true, computes shortest paths from landmarks to vertices.
    ///               If false, computes the shortest paths from vertices to landmarks.
    pub fn reversed(mut self, reversed: bool) -> Self {
        self.reversed = reversed;
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

    /// Sets the interval at which to checkpoint computation state.
    ///
    /// # Arguments
    /// * `checkpoint_interval` - Number of iterations between checkpoints
    pub fn checkpoint_interval(mut self, checkpoint_interval: usize) -> Self {
        self.checkpoint_interval = checkpoint_interval;
        self
    }

    pub async fn run(self) -> Result<DataFrame> {
        // The data type for storing distances to landmarks;
        // A struct in a form lm -> distance
        let internal_distance_data_type = DataType::Struct(Fields::from(
            self.landmarks
                .clone()
                .into_iter()
                .map(|lm| Field::new(lm.to_string(), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));

        // For landmarks:
        // - distance to itself is 0
        // - distance to other landmarks is infinity
        // For non-landmarks:
        // - distance to other landmarks is infinity
        let init_distances = named_struct(
            self.landmarks
                .clone()
                .iter()
                .flat_map(|lm| {
                    vec![
                        lit(lm.to_string()),
                        when(col(VERTEX_ID).eq(lit(lm.clone())), lit(0i32))
                            .otherwise(lit(i32::MAX))
                            .unwrap(),
                    ]
                })
                .collect::<Vec<Expr>>(),
        );

        // The logic of updating distances:
        // - If no message received (PREGEL_MSG is null), keep existing distances
        // - Otherwise, for each landmark:
        //   - Compare current distance with received distance
        //   - Keep the minimum of the two values
        let update_distances =
            when(col(PREGEL_MSG).is_null(), col(DISTANCES)).otherwise(named_struct(
                self.landmarks
                    .clone()
                    .iter()
                    .flat_map(|lm| {
                        let left_value = col(DISTANCES).field(lm.to_string());
                        let right_value = col(PREGEL_MSG).field(lm.to_string());
                        vec![
                            lit(lm.to_string()),
                            when(left_value.clone().lt_eq(right_value.clone()), left_value)
                                .otherwise(right_value)
                                .unwrap(),
                        ]
                    })
                    .collect::<Vec<Expr>>(),
            ))?;

        const PARTICIPATING: &str = "participating";

        let landmarks_copy = Arc::new(self.landmarks.clone());

        let aggregate_expr_udaf = create_udaf(
            "_merge_distance_maps",
            vec![internal_distance_data_type.clone()],
            Arc::new(internal_distance_data_type.clone()),
            Volatility::Immutable,
            Arc::new(move |_| Ok(Box::new(DistancesMap::new(landmarks_copy.clone())))),
            Arc::new(vec![internal_distance_data_type.clone()]),
        );

        // Initialize participation: only landmarks participate initially
        let init_participating = self.landmarks.iter().fold(lit(false), |acc, &landmark| {
            acc.or(col(VERTEX_ID).eq(lit(landmark)))
        });

        // Update participation condition: a vertex participates if it already participates or
        // if it receives a message (meaning it's a neighbor of a participating vertex)
        let update_participating =
            self.landmarks
                .clone()
                .iter()
                .fold(lit(false), |acc, &landmark| {
                    acc.or(col(DISTANCES)
                        .field(landmark.to_string())
                        .gt(col(PREGEL_MSG).field(landmark.to_string())))
                });

        // Message to send: current distances map + 1
        let message_expr = named_struct(
            self.landmarks
                .clone()
                .iter()
                .flat_map(|lm| {
                    let col_name = lm.to_string();
                    let d_col = if self.reversed {
                        pregel_src(DISTANCES).field(col_name.clone())
                    } else {
                        pregel_dst(DISTANCES).field(col_name.clone())
                    };
                    vec![
                        lit(col_name),
                        when(d_col.clone().lt(lit(i32::MAX)), d_col + lit(1i32))
                            .otherwise(lit(i32::MAX))
                            .unwrap(),
                    ]
                })
                .collect(),
        );

        // Run Pregel algorithm
        let result = self
            .graph_frame
            .pregel()
            // Add vertex columns
            .add_vertex_column(DISTANCES, init_distances.clone(), update_distances)
            // Set participation condition
            .with_participation_column(
                PARTICIPATING,
                init_participating,
                update_participating.clone(),
            )
            // Add a message
            .add_message(
                message_expr,
                if self.reversed {
                    MessageDirection::SrcToDst
                } else {
                    MessageDirection::DstToSrc
                },
            )
            // Set aggregate expression
            .with_aggregate_expr(aggregate_expr_udaf.call(vec![col(PREGEL_MSG)]))
            // Set voting condition
            .with_vertex_voting("active", update_participating)
            // Set max iterations if provided
            .max_iterations(self.max_iterations)
            // Set checkpoint interval
            .checkpoint_interval(2)
            // Run the algorithm
            .run(false)
            .await?;

        // Return the result with vertex ID and distances map
        Ok(result.data.select(vec![col(VERTEX_ID), col(DISTANCES)])?)
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
    pub fn shortest_paths(&self, landmarks: Vec<i64>) -> ShortestPathsBuilder {
        ShortestPathsBuilder::new(self, landmarks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::create_ldbc_test_graph;
    use datafusion::arrow::array::{Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::prelude::SessionContext;
    use std::sync::Arc;

    fn create_small_test_graph() -> Result<GraphFrame> {
        let ctx = SessionContext::new();

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
        let ctx = SessionContext::new();
        let graph = create_small_test_graph()?;
        let landmarks = vec![1];
        let result = graph.shortest_paths(landmarks).run().await?;

        // Create expected results
        let expected_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("expected_id", DataType::Int64, false),
                Field::new(
                    "expected_distances",
                    DataType::Struct(Fields::from(vec![Field::new("1", DataType::Int32, true)])),
                    false,
                ),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StructArray::new(
                    Fields::from(vec![Field::new("1", DataType::Int32, true)]),
                    vec![Arc::new(Int32Array::from(vec![0, 1, 2, 1]))],
                    None,
                )),
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
            .filter(
                col(DISTANCES)
                    .field("1")
                    .not_eq(col("expected_distances").field("1")),
            )?
            .select(vec![
                col(VERTEX_ID),
                col(DISTANCES).field("1"),
                col("expected_distances").field("1"),
            ])?;

        assert_eq!(
            diff.count().await?,
            0,
            "Found differences in shortest paths"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_shortest_paths_multiple_landmarks() -> Result<()> {
        let ctx = SessionContext::new();
        let graph = create_small_test_graph()?;
        let landmarks = vec![1, 4];
        let result = graph.shortest_paths(landmarks).run().await?;

        // Create expected results
        let expected_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("expected_id", DataType::Int64, false),
                Field::new(
                    "expected_distances",
                    DataType::Struct(Fields::from(vec![
                        Field::new("1", DataType::Int32, true),
                        Field::new("4", DataType::Int32, true),
                    ])),
                    false,
                ),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                Arc::new(StructArray::new(
                    Fields::from(vec![
                        Field::new("1", DataType::Int32, true),
                        Field::new("4", DataType::Int32, true),
                    ]),
                    vec![
                        Arc::new(Int32Array::from(vec![0, 1, 2, 1])),
                        Arc::new(Int32Array::from(vec![2, 1, 1, 0])),
                    ],
                    None,
                )),
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
            col(DISTANCES)
                .field("1")
                .not_eq(col("expected_distances").field("1"))
                .or(col(DISTANCES)
                    .field("4")
                    .not_eq(col("expected_distances").field("4"))),
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

        let results = graph
            .shortest_paths(vec![1])
            .reversed(true) // In LDBC the task is formulated as find distance from the root
            .checkpoint_interval(1)
            .run()
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
                col(DISTANCES).field("1").alias("got_distance"),
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
