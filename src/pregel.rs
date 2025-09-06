use crate::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use datafusion::error::Result;
use datafusion::functions::core::expr_ext::FieldAccessor;
use datafusion::prelude::*;
use std::vec;

/// Message column name used in Pregel algorithm
pub const PREGEL_MSG: &str = "__pregel_msg";
pub const PREGEL_MSG_SRC: &str = "__pregel_msg_src";
pub const PREGEL_MSG_DST: &str = "__pregel_msg_dst";
pub const PREGEL_MSG_EDGE: &str = "__pregel_msg_edge";

/// Direction of message passing in Pregel
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageDirection {
    /// Send messages from source to destination
    SrcToDst,
    /// Send messages from destination to source
    DstToSrc,
    /// Send messages in both directions
    Bidirectional,
}

/// Configuration for a vertex column in Pregel
#[derive(Debug, Clone)]
pub struct VertexColumn {
    /// Name of the column
    pub name: String,
    /// Initial expression for the column
    pub init_expr: Expr,
    /// Update expression for the column
    pub update_expr: Expr,
}

/// Configuration for a message in Pregel
#[derive(Debug, Clone)]
pub struct Message {
    /// Expression to generate the message
    pub expr: Expr,
    /// Direction of the message
    pub direction: MessageDirection,
}

/// Builder for Pregel algorithm configuration
#[derive(Debug)]
pub struct PregelBuilder {
    /// Graph itself
    graph: GraphFrame,
    /// Maximum number of iterations
    max_iterations: Option<usize>,
    /// Whether to use vertex voting for early stopping
    use_vertex_voting: bool,
    /// Column name for an activity flag
    activity_column: Option<String>,
    /// Update condition for vertex voting
    voting_condition: Option<Expr>,
    /// Vertex columns to update
    vertex_columns: Vec<VertexColumn>,
    /// Edge columns to include
    edge_columns: Vec<String>,
    /// Column indicating whether a vertex participates in message generation
    participation_column: Option<VertexColumn>,
    /// Messages to send
    messages: Vec<Message>,
    /// Expression to aggregate messages
    aggregate_expr: Option<Expr>,
    /// Checkpoint interval for intermediate results
    checkpoint_interval: usize,
}

#[derive(Debug)]
pub struct PregelOutput {
    pub data: DataFrame,
    pub num_iterations: usize,
}

pub fn pregel_src(col_name: &str) -> Expr {
    col(PREGEL_MSG_SRC).field(col_name)
}

pub fn pregel_dst(col_name: &str) -> Expr {
    col(PREGEL_MSG_DST).field(col_name)
}

pub fn pregel_edge(col_name: &str) -> Expr {
    col(PREGEL_MSG_EDGE).field(col_name)
}

impl PregelBuilder {
    /// Create a new PregelBuilder
    pub fn new(graph: GraphFrame) -> Self {
        Self {
            graph,
            max_iterations: None,
            use_vertex_voting: false,
            activity_column: None,
            voting_condition: None,
            vertex_columns: Vec::new(),
            edge_columns: Vec::new(),
            participation_column: None,
            messages: Vec::new(),
            aggregate_expr: None,
            checkpoint_interval: 0,
        }
    }

    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    /// Set the checkpoint interval for intermediate results
    pub fn checkpoint_interval(mut self, checkpoint_interval: usize) -> Self {
        self.checkpoint_interval = checkpoint_interval;
        self
    }

    /// Enable vertex voting for early stopping
    pub fn with_vertex_voting(mut self, activity_column: &str, voting_condition: Expr) -> Self {
        self.use_vertex_voting = true;
        self.activity_column = Some(activity_column.to_string());
        self.voting_condition = Some(voting_condition);
        self
    }

    /// Add a vertex column to update
    pub fn add_vertex_column(mut self, name: &str, init_expr: Expr, update_expr: Expr) -> Self {
        self.vertex_columns.push(VertexColumn {
            name: name.to_string(),
            init_expr,
            update_expr,
        });
        self
    }

    /// Add an edge column to include
    pub fn add_edge_column(mut self, name: &str) -> Self {
        self.edge_columns.push(name.to_string());
        self
    }

    /// Set the participation column
    pub fn with_participation_column(
        mut self,
        column: &str,
        initial_expr: Expr,
        update_condition: Expr,
    ) -> Self {
        self.participation_column = Some(VertexColumn {
            name: column.to_string(),
            init_expr: initial_expr,
            update_expr: update_condition,
        });
        self
    }

    /// Add a message to send
    pub fn add_message(mut self, expr: Expr, direction: MessageDirection) -> Self {
        self.messages.push(Message { expr, direction });
        self
    }

    /// Set the expression to aggregate messages
    pub fn with_aggregate_expr(mut self, expr: Expr) -> Self {
        self.aggregate_expr = Some(expr);
        self
    }

    /// Build and run the Pregel algorithm
    ///
    /// Returns the DataFrame with the vertex columns after running the algorithm.
    /// If include_debug_columns is true, it will also include activity flags, voting, etc.
    pub async fn run(self, include_debug_columns: bool) -> Result<PregelOutput> {
        // Validate configuration
        if self.messages.is_empty() {
            return Err(datafusion::error::DataFusionError::Plan(
                "No messages defined for Pregel algorithm".to_string(),
            ));
        }

        if self.aggregate_expr.is_none() && self.messages.len() > 1 {
            return Err(datafusion::error::DataFusionError::Plan(
                "Aggregate expression is required when multiple messages are defined".to_string(),
            ));
        }

        // Initialize vertices with initial expressions
        let mut current_vertices = self.graph.vertices.clone();
        for column in &self.vertex_columns {
            current_vertices =
                current_vertices.with_column(&column.name, column.init_expr.clone())?;
        }

        let mut iteration = 0;
        let max_iterations = self.max_iterations.unwrap_or(usize::MAX);

        let mut messages = Vec::new();

        for message in &self.messages {
            match message.direction {
                MessageDirection::SrcToDst => messages.push(named_struct(vec![
                    lit(VERTEX_ID),
                    pregel_dst(VERTEX_ID),
                    lit("msg"),
                    message.expr.clone(),
                ])),
                MessageDirection::DstToSrc => messages.push(named_struct(vec![
                    lit(VERTEX_ID),
                    pregel_src(VERTEX_ID),
                    lit("msg"),
                    message.expr.clone(),
                ])),
                MessageDirection::Bidirectional => {
                    messages.push(named_struct(vec![
                        lit(VERTEX_ID),
                        pregel_dst(VERTEX_ID),
                        lit("msg"),
                        message.expr.clone(),
                    ]));
                    messages.push(named_struct(vec![
                        lit(VERTEX_ID),
                        pregel_src(VERTEX_ID),
                        lit("msg"),
                        message.expr.clone(),
                    ]));
                }
            }
        }

        let mut update_columns = self
            .vertex_columns
            .iter()
            .map(|column| column.update_expr.clone().alias(column.name.clone()))
            .collect::<Vec<Expr>>();

        if self.use_vertex_voting {
            let activity_column = self.activity_column.as_ref().unwrap();
            current_vertices = current_vertices.with_column(activity_column, lit(true))?;
            update_columns.push(
                self.voting_condition
                    .unwrap_or(lit(true))
                    .alias(self.activity_column.clone().unwrap()),
            );
        }
        // Initialize participation column if specified
        if self.participation_column.is_some() {
            let participation_column = self.participation_column.as_ref().unwrap();
            current_vertices = current_vertices.with_column(
                &participation_column.name,
                participation_column.init_expr.clone(),
            )?;
            update_columns.push(
                participation_column
                    .update_expr
                    .clone()
                    .alias(participation_column.name.clone()),
            );
        }
        update_columns.push(col(VERTEX_ID).alias(VERTEX_ID));
        let edges_struct = self
            .graph
            .edges
            .clone()
            .select(vec![
                named_struct(
                    self.graph
                        .edges
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| field.name())
                        .flat_map(|name| vec![lit(name), col(name)])
                        .collect(),
                )
                .alias(PREGEL_MSG_EDGE),
            ])?
            .cache()
            .await?;

        // Main Pregel loop
        while iteration < max_iterations {
            iteration += 1;
            let vertices_struct = current_vertices.clone().select(vec![
                named_struct(
                    current_vertices
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| field.name())
                        .flat_map(|name| vec![lit(name), col(name)])
                        .collect(),
                )
                .alias("_vertex_struct"),
            ])?;

            let mut triplets = vertices_struct
                .clone()
                .with_column_renamed("_vertex_struct", PREGEL_MSG_SRC)?
                .join_on(
                    edges_struct.clone(),
                    JoinType::Inner,
                    vec![pregel_src(VERTEX_ID).eq(pregel_edge(EDGE_SRC))],
                )?
                .join_on(
                    vertices_struct
                        .clone()
                        .with_column_renamed("_vertex_struct", PREGEL_MSG_DST)?,
                    JoinType::Inner,
                    vec![pregel_dst(VERTEX_ID).eq(pregel_edge(EDGE_DST))],
                )?;

            // If participation_column is present, skip triplets where both src and dst
            // are not participating in iteration.
            if self.participation_column.is_some() {
                let participation_column = self.participation_column.as_ref().unwrap();
                triplets = triplets.filter(
                    pregel_src(&participation_column.name)
                        .or(pregel_dst(&participation_column.name)),
                )?;
            }

            // Unfortunately, "unnest" does not allow passing to it an array of expression;
            // It throws "NotImplementedError"
            // "Unnest should be rewritten to LogicalPlan::Unnest before type coercion"
            //
            // Union should be considered as a workaround here.
            let messages_df = messages
                .iter()
                .map(|message| {
                    triplets.clone().select(vec![
                        message.clone().field(VERTEX_ID).alias(VERTEX_ID),
                        message.clone().field("msg").alias(PREGEL_MSG),
                    ])
                })
                .reduce(|a, b| match (a, b) {
                    (Ok(adf), Ok(bdf)) => adf.union(bdf),
                    _ => Err(datafusion::error::DataFusionError::Plan(
                        "Error in Pregel algorithm".to_string(),
                    )),
                })
                .ok_or(datafusion::error::DataFusionError::Plan(
                    "generated messages_df is None".to_string(),
                ))??
                .filter(col(PREGEL_MSG).is_not_null())?;

            let aggregated_messages = if self.aggregate_expr.is_some() {
                messages_df.aggregate(
                    vec![col(VERTEX_ID)],
                    vec![self.aggregate_expr.clone().unwrap().alias(PREGEL_MSG)],
                )?
            } else {
                messages_df
            };

            let vertices_with_messages = current_vertices
                .clone()
                .join_on(
                    aggregated_messages.with_column_renamed(VERTEX_ID, "am_vid")?,
                    JoinType::Left,
                    vec![col(VERTEX_ID).eq(col("am_vid"))],
                )?
                .drop_columns(&["am_vid"])?;

            let new_vertices = vertices_with_messages
                .clone()
                .select(update_columns.clone())?;

            current_vertices =
                if self.checkpoint_interval > 0 && iteration % self.checkpoint_interval == 0 {
                    new_vertices.clone().cache().await?
                } else {
                    new_vertices.clone()
                };
            if self.use_vertex_voting {
                if new_vertices
                    .clone()
                    .filter(col(self.activity_column.clone().unwrap()))?
                    .count()
                    .await?
                    == 0
                {
                    break;
                }
            }
        }

        // If include_debug_columns is false, drop the debug columns
        if !include_debug_columns {
            let mut required_columns = self
                .vertex_columns
                .iter()
                .map(|column| col(column.clone().name))
                .collect::<Vec<Expr>>();
            required_columns.push(col(VERTEX_ID));
            current_vertices = current_vertices.select(required_columns)?;
        }

        Ok(PregelOutput {
            data: current_vertices,
            num_iterations: iteration,
        })
    }
}

impl GraphFrame {
    /// Create a new Pregel algorithm builder
    pub fn pregel(&self) -> PregelBuilder {
        PregelBuilder::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Array, Int32Array, Int64Array};
    use datafusion::functions_aggregate::min_max::max;
    use datafusion::functions_aggregate::sum::sum;

    fn create_graph(vertices: Vec<i64>, edges: Vec<Vec<i64>>) -> Result<GraphFrame> {
        let vertices_df = dataframe!(
            VERTEX_ID => Vec::<i64>::from(vertices),
        )?;
        let edges_df = dataframe!(EDGE_SRC => Vec::<i64>::from(
            edges.iter().map(|e| e[0]).collect::<Vec<i64>>()
        ), EDGE_DST => Vec::<i64>::from(
            edges.iter().map(|e| e[1]).collect::<Vec<i64>>()
        ))?;

        Ok(GraphFrame {
            vertices: vertices_df,
            edges: edges_df,
        })
    }

    #[tokio::test]
    async fn test_pregel_zero_iterations() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3]])?;
        let result = graph
            .pregel()
            .max_iterations(0)
            .with_participation_column("participation", lit(true), lit(true))
            .with_vertex_voting("activity", lit(true))
            .add_vertex_column("value", lit(0), col("value"))
            .add_message(lit(1), MessageDirection::SrcToDst)
            .run(true)
            .await?
            .data;

        let result_schema = result.schema();
        assert_eq!(result_schema.fields().len(), 4);
        assert_eq!(result_schema.fields()[0].name(), "id");
        assert_eq!(result_schema.fields()[1].name(), "value");
        assert_eq!(result_schema.fields()[2].name(), "activity");
        assert_eq!(result_schema.fields()[3].name(), "participation");
        Ok(())
    }

    #[tokio::test]
    async fn test_pregel_in_degree() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3], vec![1, 3]])?;
        let result = graph
            .pregel()
            .max_iterations(1)
            .add_vertex_column("in_degree", lit(0), col("in_degree") + col(PREGEL_MSG))
            .add_message(lit(1), MessageDirection::SrcToDst)
            .with_aggregate_expr(sum(col(PREGEL_MSG)))
            .run(false)
            .await?
            .data;

        let counts = result
            .select(vec![col("id"), col("in_degree")])?
            .sort(vec![col("id").sort(true, false)])?
            .collect()
            .await?;

        assert_eq!(
            counts[0]
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 1, 2]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_pregel_out_degree() -> Result<()> {
        let graph = create_graph(vec![1, 2, 3], vec![vec![1, 2], vec![2, 3], vec![1, 3]])?;
        let result = graph
            .pregel()
            .max_iterations(1)
            .add_vertex_column("out_degree", lit(0), col("out_degree") + col(PREGEL_MSG))
            .add_message(lit(1), MessageDirection::DstToSrc)
            .with_aggregate_expr(sum(col(PREGEL_MSG)))
            .run(false)
            .await?
            .data;

        let counts = result
            .select(vec![col("id"), col("out_degree")])?
            .sort(vec![col("id").sort(true, false)])?
            .collect()
            .await?;

        assert_eq!(
            counts[0]
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[2, 1, 0]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_pregel_loop() -> Result<()> {
        let graph = create_graph(vec![1], vec![vec![1, 1]])?;
        let result = graph
            .pregel()
            .max_iterations(1)
            .add_vertex_column("loop", lit(0), col("loop") + col(PREGEL_MSG))
            .add_message(lit(1), MessageDirection::SrcToDst)
            .with_aggregate_expr(sum(col(PREGEL_MSG)))
            .run(false)
            .await?
            .data;

        let counts = result.select(vec![col("loop")])?.collect().await?;

        assert_eq!(
            counts[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[1]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_pregel_no_edges() -> Result<()> {
        let graph = create_graph(vec![1, 2], vec![])?;
        let result = graph
            .pregel()
            .max_iterations(1)
            .add_vertex_column("value", lit(0), col("value") + col(PREGEL_MSG))
            .add_message(lit(1), MessageDirection::SrcToDst)
            .with_aggregate_expr(sum(col(PREGEL_MSG)))
            .run(false)
            .await?
            .data;

        let values = result.select(vec![col("value")])?.collect().await?;

        assert_eq!(
            values[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 0]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_pregel_chain_propagation() -> Result<()> {
        // Create a chain graph: 1 -> 2 -> 3 -> 4
        let graph = create_graph(vec![1, 2, 3, 4], vec![vec![1, 2], vec![2, 3], vec![3, 4]])?;

        let result = graph
            .pregel()
            .max_iterations(100) // We should check that voting works
            .add_vertex_column(
                "value",
                when(col("id").eq(lit(1)), lit(1)).otherwise(lit(0))?,
                when(col(PREGEL_MSG).gt(col("value")), col(PREGEL_MSG)).otherwise(col("value"))?,
            )
            .with_vertex_voting("active", col("value").not_eq(col(PREGEL_MSG)))
            .add_message(pregel_src("value"), MessageDirection::SrcToDst)
            .with_aggregate_expr(max(col(PREGEL_MSG)))
            .run(false)
            .await?;

        // In four iterations it should converge
        assert_eq!(result.num_iterations, 4);

        let values = result.data.select(vec![col("value")])?.collect().await?;

        let mut values_vec: Vec<i32> = Vec::new();
        for batch in values.iter() {
            values_vec.append(
                &mut batch
                    .column(0)
                    .clone()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec(),
            );
        }

        // All vertices should have value 1
        assert!(values_vec.iter().all(|&x| x == 1));
        Ok(())
    }

    #[tokio::test]
    async fn test_pregel_back_chain_propagation() -> Result<()> {
        // Create a chain graph: 1 -> 2 -> 3 -> 4
        let graph = create_graph(vec![1, 2, 3, 4], vec![vec![1, 2], vec![2, 3], vec![3, 4]])?;

        let result = graph
            .pregel()
            .max_iterations(100) // We should check that voting works
            .add_vertex_column(
                "value",
                when(col("id").eq(lit(4)), lit(1)).otherwise(lit(0))?,
                when(col(PREGEL_MSG).gt(col("value")), col(PREGEL_MSG)).otherwise(col("value"))?,
            )
            .with_vertex_voting("active", col("value").not_eq(col(PREGEL_MSG)))
            .add_message(pregel_dst("value"), MessageDirection::DstToSrc)
            .with_aggregate_expr(max(col(PREGEL_MSG)))
            .run(false)
            .await?;

        // In four iterations it should converge
        assert_eq!(result.num_iterations, 4);

        let values = result.data.select(vec![col("value")])?.collect().await?;

        let mut values_vec: Vec<i32> = Vec::new();
        for batch in values.iter() {
            values_vec.append(
                &mut batch
                    .column(0)
                    .clone()
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec(),
            );
        }

        // All vertices should have value 1
        assert!(values_vec.iter().all(|&x| x == 1));
        Ok(())
    }

    fn create_circle_edges(n: i64) -> Vec<Vec<i64>> {
        let mut edges = Vec::new();
        for i in 0..n {
            edges.push(vec![i, (i + 1) % n]);
            edges.push(vec![i, (i + n - 1) % n]);
        }
        edges
    }

    #[tokio::test]
    async fn test_pregel_long_iterations() -> Result<()> {
        let n = 100;
        let vertices: Vec<i64> = (0..n).collect();
        let edges = create_circle_edges(n);
        let graph = create_graph(vertices, edges)?;

        let result = graph
            .pregel()
            .max_iterations(40)
            .checkpoint_interval(2)
            .add_vertex_column("value", lit(0), col("value") + col(PREGEL_MSG))
            .add_message(lit(1), MessageDirection::Bidirectional)
            .with_aggregate_expr(sum(col(PREGEL_MSG)))
            .run(false)
            .await?;

        let values = result.data.select(vec![col("value")])?.collect().await?;

        let mut values_vec: Vec<i64> = Vec::new();
        for batch in values.iter() {
            values_vec.append(
                &mut batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec(),
            );
        }

        assert!(values_vec.iter().all(|&x| x == 160));
        Ok(())
    }
}
