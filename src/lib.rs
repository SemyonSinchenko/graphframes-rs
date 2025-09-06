mod pagerank;
mod pregel;
mod shortest_paths;
pub mod util;

use datafusion::error::Result;
use datafusion::functions_aggregate::count::count;
use datafusion::prelude::*;

pub const VERTEX_ID: &str = "id";
pub const EDGE_SRC: &str = "src";
pub const EDGE_DST: &str = "dst";
pub const EDGE_COL: &str = "edge";
pub const SRC_VERTEX: &str = "src_vertex";
pub const DST_VERTEX: &str = "dst_vertex";

#[derive(Debug, Clone)]
pub struct GraphFrame {
    pub vertices: DataFrame,
    pub edges: DataFrame,
}

impl GraphFrame {
    pub async fn num_nodes(&self) -> Result<i64> {
        let count = self.vertices.clone().count().await?;
        Ok(count as i64)
    }

    pub async fn num_edges(&self) -> Result<i64> {
        let count = self.edges.clone().count().await?;
        Ok(count as i64)
    }

    pub async fn in_degrees(&self) -> Result<DataFrame> {
        let df = self.edges.clone().aggregate(
            vec![col(EDGE_DST)],
            vec![count(col(EDGE_SRC)).alias("in_degree")],
        )?;
        Ok(df.select(vec![col(EDGE_DST).alias(VERTEX_ID), col("in_degree")])?)
    }

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
    /// // Assuming `edges_df` and `vertices_df` are initialized DataFrames for
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
    use datafusion::arrow::array::{Int64Array, RecordBatch, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use std::collections::HashMap;
    use std::sync::Arc;

    fn create_test_graph() -> Result<GraphFrame> {
        let ctx = SessionContext::new();

        let vertices_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
                Arc::new(StringArray::from(vec![
                    "Hub", "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
                    "Ivy",
                ])),
            ],
        );
        let vertices = ctx.read_batch(vertices_data?)?;

        let edges_data = RecordBatch::try_new(
            SchemaRef::from(Schema::new(vec![
                Field::new("src", DataType::Int64, false),
                Field::new("dst", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7,
                    8, 8, 9, 10,
                ])),
                Arc::new(Int64Array::from(vec![
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 3, 4, 5, 6, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 8, 9,
                    9, 10, 10, 1,
                ])),
            ],
        );
        let edges = ctx.read_batch(edges_data?)?;

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
