mod connected_components;
mod pagerank;
mod pregel;
mod shortest_paths;
pub mod util;

use datafusion::error::Result;
use datafusion::functions_aggregate::count::count;
use datafusion::prelude::*;

/// Column names for the vertex id column.
pub const VERTEX_ID: &str = "id";
/// Column names for the edge source column.
pub const EDGE_SRC: &str = "src";
/// Column names for the edge destination column.
pub const EDGE_DST: &str = "dst";
/// Column names for the edge column in triplet representation.
pub const EDGE_COL: &str = "edge";
/// Column names for the source vertex in triplet representation.
pub const SRC_VERTEX: &str = "src_vertex";
/// Column names for the destination vertex in triplet representation.
pub const DST_VERTEX: &str = "dst_vertex";

/// A data structure representing a graph in the form of vertices and edges.
///
/// The `GraphFrame` struct is designed to hold a graph's data where vertices
/// (nodes) and edges (connections) are represented as `DataFrame` structures.
///
/// # Fields
///
/// * `vertices` - A `DataFrame` that contains information about the graph's vertices.
///                Each row in the `DataFrame` represents a vertex (`VERTEX_ID`), and additional
///                columns can store attributes (e.g., labels or properties) for
///                each vertex.
///
/// * `edges` - A `DataFrame` that contains information about the graph's edges.
///             Each row in the `DataFrame` represents an edge, with columns
///             typically storing the source vertex (`EDGE_SRC`), destination vertex (`EDGE_DST`), and
///             any additional attributes (e.g., weights or labels) associated
///             with the edge.
///
/// # Example
///
/// ```
/// use datafusion::dataframe;
/// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
/// let vertices = dataframe!(
///   VERTEX_ID => vec![1i64, 2i64, 3i64],
///   "attr" => vec!["a", "b", "c"]
/// ).unwrap();
/// let edges = dataframe!(
///   EDGE_SRC => vec![1i64, 2i64, 3i64],
///   EDGE_DST => vec![3i64, 1i64, 2i64],
///   "attr" => vec!["d", "j", "h"]
/// ).unwrap();
///
/// let graph = GraphFrame { vertices, edges };
/// ```
#[derive(Debug, Clone)]
pub struct GraphFrame {
    pub vertices: DataFrame,
    pub edges: DataFrame,
}

impl GraphFrame {
    /// Returns the total number of nodes in the graph.
    ///
    /// # Returns
    ///
    /// This function returns a `Result<i64>`:
    /// - `Ok(i64)`: The total number of nodes (vertices) in the graph, represented as a 64-bit signed integer.
    /// - `Err`: If an error occurs during the computation or retrieval of the node count.
    ///
    /// # Example
    ///
    /// ```
    /// use datafusion::dataframe;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// let vertices = dataframe!(
    ///   VERTEX_ID => vec![1i64, 2i64, 3i64],
    ///   "attr" => vec!["a", "b", "c"]
    /// ).unwrap();
    /// let edges = dataframe!(
    ///   EDGE_SRC => vec![1i64, 2i64, 3i64],
    ///   EDGE_DST => vec![3i64, 1i64, 2i64],
    ///   "attr" => vec!["d", "j", "h"]
    /// ).unwrap();
    ///
    /// let graph = GraphFrame { vertices, edges };
    /// let node_count = graph.num_nodes();
    /// ```
    pub async fn num_nodes(&self) -> Result<i64> {
        let count = self.vertices.clone().count().await?;
        Ok(count as i64)
    }

    /// Returns the total number of edges in the graph.
    ///
    /// # Returns
    ///
    /// This function returns a `Result<i64>`:
    /// - `Ok(i64)` - The total number of edges, represented as a 64-bit integer.
    /// - `Err(E)` - If an error occurs during the computation, the error is propagated.
    ///
    /// # Examples
    ///
    /// ```
    /// use datafusion::dataframe;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// let vertices = dataframe!(
    ///   VERTEX_ID => vec![1i64, 2i64, 3i64],
    ///   "attr" => vec!["a", "b", "c"]
    /// ).unwrap();
    /// let edges = dataframe!(
    ///   EDGE_SRC => vec![1i64, 2i64, 3i64],
    ///   EDGE_DST => vec![3i64, 1i64, 2i64],
    ///   "attr" => vec!["d", "j", "h"]
    /// ).unwrap();
    ///
    /// let graph = GraphFrame { vertices, edges };
    /// let edge_count = graph.num_edges();
    /// ```
    pub async fn num_edges(&self) -> Result<i64> {
        let count = self.edges.clone().count().await?;
        Ok(count as i64)
    }

    /// Computes the in-degrees for each vertex in the graph.
    ///
    /// This function calculates the in-degree of each vertex by counting the number of
    /// incoming edges. It returns a `DataFrame`
    /// containing two columns:
    /// - `VERTEX_ID`: The unique identifier of the vertex (derived from the destination of the edges).
    /// - `in_degree`: The count of incoming edges (in-degrees) for each vertex.
    ///
    /// # Returns
    /// An asynchronous function that returns:
    /// - `Ok(DataFrame)` containing the vertex IDs and their corresponding in-degrees.
    /// - `Err` if the aggregation or selection operation fails.
    ///
    /// # Example
    /// ```rust
    /// use datafusion::dataframe;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// let vertices = dataframe!(
    ///   VERTEX_ID => vec![1i64, 2i64, 3i64],
    ///   "attr" => vec!["a", "b", "c"]
    /// ).unwrap();
    /// let edges = dataframe!(
    ///   EDGE_SRC => vec![1i64, 2i64, 3i64],
    ///   EDGE_DST => vec![3i64, 1i64, 2i64],
    ///   "attr" => vec!["d", "j", "h"]
    /// ).unwrap();
    ///
    /// let graph = GraphFrame { vertices, edges };
    /// let edge_count = graph.in_degrees();
    /// ```
    pub async fn in_degrees(&self) -> Result<DataFrame> {
        let df = self.edges.clone().aggregate(
            vec![col(EDGE_DST)],
            vec![count(col(EDGE_SRC)).alias("in_degree")],
        )?;
        Ok(df.select(vec![col(EDGE_DST).alias(VERTEX_ID), col("in_degree")])?)
    }
    /// Computes the out-degrees for each vertex in the graph.
    ///
    /// This function calculates the out-degree of each vertex by counting the number of
    /// outcoming edges. It returns a `DataFrame`
    /// containing two columns:
    /// - `VERTEX_ID`: The unique identifier of the vertex (derived from the destination of the edges).
    /// - `in_degree`: The count of incoming edges (in-degrees) for each vertex.
    ///
    /// # Returns
    /// An asynchronous function that returns:
    /// - `Ok(DataFrame)` containing the vertex IDs and their corresponding in-degrees.
    /// - `Err` if the aggregation or selection operation fails.
    ///
    /// # Example
    /// ```rust
    /// use datafusion::dataframe;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// let vertices = dataframe!(
    ///   VERTEX_ID => vec![1i64, 2i64, 3i64],
    ///   "attr" => vec!["a", "b", "c"]
    /// ).unwrap();
    /// let edges = dataframe!(
    ///   EDGE_SRC => vec![1i64, 2i64, 3i64],
    ///   EDGE_DST => vec![3i64, 1i64, 2i64],
    ///   "attr" => vec!["d", "j", "h"]
    /// ).unwrap();
    ///
    /// let graph = GraphFrame { vertices, edges };
    /// let edge_count = graph.in_degrees();
    /// ```
    pub async fn out_degrees(&self) -> Result<DataFrame> {
        let df = self.edges.clone().aggregate(
            vec![col(EDGE_SRC)],
            vec![count(col(EDGE_DST)).alias("out_degree")],
        )?;
        Ok(df.select(vec![col(EDGE_SRC).alias(VERTEX_ID), col("out_degree")])?)
    }

    /// Creates a symmetric graph by duplicating all edges in the reverse direction.
    /// For each edge (a,b) in the graph, adds the edge (b,a) if it doesn't exist.
    /// Any additional edge attributes are preserved in the reversed edges.
    ///
    /// # Returns
    /// A new `GraphFrame` containing the original vertices and symmetrized edges.
    ///
    /// # Errors
    /// Return a DataFusion error if the edge transformation operations fail.
    pub fn symmetrize(&self) -> Result<GraphFrame> {
        let vertices = self.vertices.clone();
        let edges_cols = self
            .edges
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>();
        let new_edges_cols = edges_cols
            .clone()
            .iter()
            .map(|c| {
                if c == EDGE_SRC {
                    col(EDGE_SRC).alias(EDGE_DST)
                } else if c == EDGE_DST {
                    col(EDGE_DST).alias(EDGE_SRC)
                } else {
                    col(c)
                }
            })
            .collect::<Vec<_>>();

        // We need to:
        // - swap dst and src
        // - preserve the order of columns for union
        let edges = self.edges.clone().union(
            self.edges
                .clone()
                .select(new_edges_cols)?
                .select_columns(&edges_cols.iter().map(|c| c.as_str()).collect::<Vec<_>>())?,
        )?;
        Ok(GraphFrame { vertices, edges })
    }

    /// Generates a DataFrame containing "triplets" by combining information from edges and vertices.
    ///
    /// This method aggregates data about source vertices, edges, and destination vertices,
    /// producing a combined representation of these relationships as triplets.
    /// It constructs structured representations of edges and vertices, then performs
    /// joins to associate source and destination vertices with their respective edges.
    ///
    /// # Returns
    ///
    /// Returns a `Result<DataFrame>` which can either:
    /// - Contain the `DataFrame` representing the triplets (source vertex, edge, destination vertex).
    /// - Return an error if an operation (e.g., selection or join) fails during the process.
    ///
    /// Output `DataFrame` contains the following columns:
    /// - `SRC_VERTEX` - struct with all the columns of vertices, associated with a source of the triple
    /// - `EDGE_COL` - struct with all the columns of edges, associated with an edge
    /// - `DST_VERTEX` - struct with all the columns of vertices, associated with a destination of the triplet
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - Either the source vertices or destination vertices cannot be joined with edges due to schema mismatches.
    /// - Any selection or transformation process internally fails due to invalid queries.
    ///
    /// # Example
    ///
    /// ```
    /// use datafusion::dataframe;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// let vertices = dataframe!(
    ///   VERTEX_ID => vec![1i64, 2i64, 3i64],
    ///   "attr" => vec!["a", "b", "c"]
    /// ).unwrap();
    /// let edges = dataframe!(
    ///   EDGE_SRC => vec![1i64, 2i64, 3i64],
    ///   EDGE_DST => vec![3i64, 1i64, 2i64],
    ///   "attr" => vec!["d", "j", "h"]
    /// ).unwrap();
    ///
    /// let graph = GraphFrame { vertices, edges };
    /// let triplets = graph.triplets();
    /// ```
    pub async fn triplets(&self) -> Result<DataFrame> {
        let edges_struct = self.edges.clone().select(vec![
            col(EDGE_SRC),
            col(EDGE_DST),
            named_struct(
                self.edges
                    .clone()
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .flat_map(|name| vec![lit(name), col(name)])
                    .collect(),
            )
            .alias(EDGE_COL),
        ])?;
        let vertices_struct = self.vertices.clone().select(vec![
            col(VERTEX_ID),
            named_struct(
                self.vertices
                    .clone()
                    .schema()
                    .fields()
                    .iter()
                    .map(|field| field.name())
                    .flat_map(|name| vec![lit(name), col(name)])
                    .collect(),
            )
            .alias("_vertex_struct"),
        ])?;
        edges_struct
            .join_on(
                vertices_struct.clone().select(vec![
                    col(VERTEX_ID),
                    col("_vertex_struct").alias(SRC_VERTEX),
                ])?,
                JoinType::Left,
                vec![col(EDGE_SRC).eq(col(VERTEX_ID))],
            )?
            .select(vec![col(SRC_VERTEX), col(EDGE_DST), col(EDGE_COL)])?
            .join_on(
                vertices_struct.select(vec![
                    col(VERTEX_ID),
                    col("_vertex_struct").alias(DST_VERTEX),
                ])?,
                JoinType::Left,
                vec![col(EDGE_DST).eq(col(VERTEX_ID))],
            )?
            .select(vec![col(SRC_VERTEX), col(EDGE_COL), col(DST_VERTEX)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Fields};
    use std::collections::HashMap;

    pub(crate) fn create_test_graph() -> Result<GraphFrame> {
        let vertices = dataframe!(
            VERTEX_ID => vec![1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64, 8i64, 9i64, 10i64],
            "name" => vec!["Hub", "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Ivy"]
        )?;

        let edges = dataframe!(
            EDGE_SRC => Vec::<i64>::from(
                vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10,]
            ),
            EDGE_DST => Vec::<i64>::from(
                vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 8, 9, 9, 10, 10, 1,]
            ),
        )?;

        Ok(GraphFrame { vertices, edges })
    }

    #[tokio::test]
    async fn test_num_nodes() -> Result<()> {
        let graph = create_test_graph()?;
        let num_nodes = graph.num_nodes().await?;
        assert_eq!(num_nodes, 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_num_edges() -> Result<()> {
        let graph = create_test_graph()?;
        let num_edges = graph.num_edges().await?;
        assert_eq!(num_edges, 30);
        Ok(())
    }

    #[tokio::test]
    async fn test_in_degrees() -> Result<()> {
        let graph = create_test_graph()?;
        let df = graph.in_degrees().await?;
        let batches = df.collect().await?;

        let mut degree_map = HashMap::new();
        for batch in batches.iter() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let degrees = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..ids.len() {
                degree_map.insert(ids.value(i), degrees.value(i));
            }
        }

        assert_eq!(degree_map.get(&1), Some(&1));
        assert_eq!(degree_map.get(&2), Some(&1));
        assert_eq!(degree_map.get(&3), Some(&2));
        assert_eq!(degree_map.get(&4), Some(&3));
        assert_eq!(degree_map.get(&5), Some(&4));
        assert_eq!(degree_map.get(&6), Some(&5));
        assert_eq!(degree_map.get(&7), Some(&4));
        assert_eq!(degree_map.get(&8), Some(&4));
        assert_eq!(degree_map.get(&9), Some(&3));
        assert_eq!(degree_map.get(&10), Some(&3));

        Ok(())
    }

    #[tokio::test]
    async fn test_out_degrees() -> Result<()> {
        let graph = create_test_graph()?;
        let df = graph.out_degrees().await?;
        let batches = df.collect().await?;

        let mut degree_map = HashMap::new();
        for batch in batches.iter() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let degrees = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();

            for i in 0..ids.len() {
                degree_map.insert(ids.value(i), degrees.value(i));
            }
        }

        assert_eq!(degree_map.get(&1), Some(&9));
        assert_eq!(degree_map.get(&2), Some(&4));
        assert_eq!(degree_map.get(&3), Some(&3));
        assert_eq!(degree_map.get(&4), Some(&3));
        assert_eq!(degree_map.get(&5), Some(&3));
        assert_eq!(degree_map.get(&6), Some(&2));
        assert_eq!(degree_map.get(&7), Some(&2));
        assert_eq!(degree_map.get(&8), Some(&2));
        assert_eq!(degree_map.get(&9), Some(&1));
        assert_eq!(degree_map.get(&10), Some(&1));

        Ok(())
    }

    #[tokio::test]
    async fn test_triplets() -> Result<()> {
        let vertices =
            dataframe!(VERTEX_ID => vec![1i64, 2i64, 3i64], "attr" => vec!["a", "b", "c"])?;
        let edges = dataframe!(EDGE_SRC => vec![1i64, 2i64, 3i64], EDGE_DST => vec![3i64, 1i64, 2i64], "attr" => vec!["d", "j", "h"])?;

        let graph = GraphFrame { vertices, edges };
        let triplets = graph.triplets().await?;

        // Check schema
        let schema = triplets.schema();
        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), SRC_VERTEX);
        assert_eq!(schema.field(1).name(), EDGE_COL);
        assert_eq!(schema.field(2).name(), DST_VERTEX);
        assert!(
            schema
                .field(0)
                .data_type()
                .eq(&DataType::Struct(Fields::from(vec![
                    Field::new(VERTEX_ID, DataType::Int64, true),
                    Field::new("attr", DataType::Utf8, true)
                ])))
        );
        assert!(
            schema
                .field(1)
                .data_type()
                .eq(&DataType::Struct(Fields::from(vec![
                    Field::new(EDGE_SRC, DataType::Int64, true),
                    Field::new(EDGE_DST, DataType::Int64, true),
                    Field::new("attr", DataType::Utf8, true),
                ])))
        );
        assert!(
            schema
                .field(2)
                .data_type()
                .eq(&DataType::Struct(Fields::from(vec![
                    Field::new(VERTEX_ID, DataType::Int64, true),
                    Field::new("attr", DataType::Utf8, true)
                ])))
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_symmetrize() -> Result<()> {
        let graph = create_test_graph()?;
        let sym_graph = graph.symmetrize()?;

        // Original vertices should be preserved
        assert_eq!(graph.num_nodes().await?, sym_graph.num_nodes().await?);

        // Number of edges should double
        let orig_edges = graph.num_edges().await?;
        let sym_edges = sym_graph.num_edges().await?;
        assert_eq!(sym_edges, orig_edges * 2);

        // In and out degrees should be equal for all vertices in symmetric graph
        let in_degrees = sym_graph.in_degrees().await?.collect().await?;
        let out_degrees = sym_graph.out_degrees().await?.collect().await?;

        let mut in_degree_map = HashMap::new();
        let mut out_degree_map = HashMap::new();

        for batch in in_degrees.iter() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let degrees = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..ids.len() {
                in_degree_map.insert(ids.value(i), degrees.value(i));
            }
        }

        for batch in out_degrees.iter() {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let degrees = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..ids.len() {
                out_degree_map.insert(ids.value(i), degrees.value(i));
            }
        }

        for id in 1..=10 {
            assert_eq!(in_degree_map.get(&id), out_degree_map.get(&id));
        }

        Ok(())
    }
}
