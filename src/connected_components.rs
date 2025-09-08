//! Implementation inspired by GraphFrames Spark/Scala code
//! that is distributed under Apache License 2.0.
//! https://github.com/graphframes/graphframes/blob/master/core/src/main/scala/org/graphframes/lib/ConnectedComponents.scala

use crate::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use datafusion::arrow::array::{Array, Decimal128Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::functions_aggregate::expr_fn::{min, sum};
use datafusion::prelude::*;

pub const COMPONENT_COL: &str = "component";
const MIN_NBR: &str = "min_nbr";

/// Computes the minimum neighbor for each vertex in a graph edge list.
///
/// This function takes a DataFrame representing the edges of a graph. It computes the minimum destination vertex
/// (neighbor) for each source vertex in the graph. Optionally, the graph can be symmetrized before
/// computing the minimum neighbors. Symmetrization ensures that for every directed edge `(u, v)`,
/// the reverse edge `(v, u)` is also included in the edge list.
fn min_neighbours(edges: &DataFrame, symmetrize: bool) -> Result<DataFrame> {
    // symmetrize edges if needed
    let ee = if symmetrize {
        edges.clone().union(edges.clone().select(vec![
            col(EDGE_DST).alias(EDGE_SRC),
            col(EDGE_SRC).alias(EDGE_DST),
        ])?)?
    } else {
        edges.clone()
    };
    ee.aggregate(
        vec![col(EDGE_SRC).alias(VERTEX_ID)],
        vec![min(col(EDGE_DST)).alias(MIN_NBR)],
    )?
    .select(vec![
        col(VERTEX_ID),
        when(col(VERTEX_ID).lt(col(MIN_NBR)), col(VERTEX_ID))
            .otherwise(col(MIN_NBR))?
            .alias(MIN_NBR),
    ])
}

/// Calculate the sum of all the minimum neighbor values in a DataFrame.
async fn min_nbr_sum(min_neighbours: &DataFrame) -> Result<i128> {
    min_neighbours
        .clone()
        .aggregate(
            vec![],
            vec![sum(cast(col(MIN_NBR), DataType::Decimal128(38, 0))).alias(MIN_NBR)],
        )?
        .collect()
        .await?
        .first()
        .ok_or(datafusion::error::DataFusionError::Internal(
            "failed to calculate and collect min_nbr_sum: result is empty".to_string(),
        ))?
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or(datafusion::error::DataFusionError::Internal(
            "failed to get min_nbr_sum as Decimal128Array".to_string(),
        ))
        .map(|a| a.value(0))
}

#[derive(Debug, Clone)]
pub struct ConnectedComponentsOutput {
    pub data: DataFrame,
    pub num_iterations: usize,
    pub min_nbr_sum: Vec<i128>,
}

#[derive(Debug, Clone)]
pub struct ConnectedComponentsBuilder<'a> {
    graph_frame: &'a GraphFrame,
}

impl<'a> ConnectedComponentsBuilder<'a> {
    pub fn new(graph_frame: &'a GraphFrame) -> Self {
        Self { graph_frame }
    }

    pub async fn run(self) -> Result<ConnectedComponentsOutput> {
        // Preparation of the graph:
        // - removing self-loops
        // - changing edge direction so SRC < DST
        // - de-duplicate edges
        let vertices = self.graph_frame.vertices.clone();
        let original_edges = self.graph_frame.edges.clone();

        let no_loops_edges = original_edges.filter(col(EDGE_SRC).not_eq(col(EDGE_DST)))?;
        let ordered_by_direction_edges = no_loops_edges.select(vec![
            when(col(EDGE_SRC).lt(col(EDGE_DST)), col(EDGE_SRC))
                .otherwise(col(EDGE_DST))?
                .alias(EDGE_SRC),
            when(col(EDGE_SRC).lt(col(EDGE_DST)), col(EDGE_DST))
                .otherwise(col(EDGE_SRC))?
                .alias(EDGE_DST),
        ])?;
        let deduped_edges = ordered_by_direction_edges.distinct()?;

        let mut iteration = 0usize;
        let mut metrics = Vec::<i128>::new();
        let mut converged = false;

        let mut minimal_neighbours_1 = min_neighbours(&deduped_edges.clone(), true)?;
        let mut last_iter_nbr_sum = min_nbr_sum(&minimal_neighbours_1.clone()).await?;
        metrics.push(last_iter_nbr_sum);
        let mut current_edges = deduped_edges.clone().cache().await?;

        while !converged {
            iteration += 1;
            // large-star step:
            // connects all strictly larger neighbors to the min neighbor (including self)
            current_edges = current_edges
                .join_on(
                    minimal_neighbours_1.clone(),
                    JoinType::Inner,
                    vec![col(EDGE_SRC).eq(col(VERTEX_ID))],
                )?
                .select(vec![
                    col(EDGE_DST).alias(EDGE_SRC),
                    col(MIN_NBR).alias(EDGE_DST),
                ])?
                .distinct()?
                .cache()
                .await?;

            // small-star step:
            // computes min neighbors (excluding self-min)
            let minimal_neighbours_2 = min_neighbours(&current_edges.clone(), false)?
                .cache()
                .await?;

            // connect all smaller neighbors to the min neighbor
            current_edges = current_edges
                .clone()
                .join_on(
                    minimal_neighbours_2.clone(),
                    JoinType::Inner,
                    vec![col(EDGE_SRC).eq(col(VERTEX_ID))],
                )?
                .select(vec![col(MIN_NBR).alias(EDGE_SRC), col(EDGE_DST)])?
                .filter(col(EDGE_SRC).not_eq(col(EDGE_DST)))?
                .union(minimal_neighbours_2.select(vec![
                    col(MIN_NBR).alias(EDGE_SRC),
                    col(VERTEX_ID).alias(EDGE_DST),
                ])?)?
                .distinct()?
                .cache()
                .await?;
            minimal_neighbours_1 = min_neighbours(&current_edges.clone(), true)?
                .cache()
                .await?;
            let current_sum = min_nbr_sum(&minimal_neighbours_1.clone()).await?;

            if current_sum == last_iter_nbr_sum {
                converged = true;
            } else {
                last_iter_nbr_sum = current_sum;
                metrics.push(current_sum);
            }
        }

        Ok(ConnectedComponentsOutput {
            data: vertices
                .join_on(
                    current_edges,
                    JoinType::Left,
                    vec![col(VERTEX_ID).eq(col(EDGE_DST))],
                )?
                .select(vec![
                    col(VERTEX_ID),
                    when(col(EDGE_SRC).is_null(), col(VERTEX_ID))
                        .otherwise(col(EDGE_SRC))?
                        .alias(COMPONENT_COL),
                ])?,
            num_iterations: iteration,
            min_nbr_sum: metrics,
        })
    }
}

impl GraphFrame {
    /// Constructs a new `ConnectedComponentsBuilder` for the current graph.
    ///
    /// This method is used to initialize the process of finding weakly connected components
    /// within the graph. It creates a `ConnectedComponentsBuilder` instance
    /// associated with the current graph, allowing further configuration or direct
    /// computation of connected components.
    ///
    /// An implementation is based on the "large star - small star" algorithm:
    /// Kiveris, Raimondas, et al. "Connected components in mapreduce and beyond."
    /// Proceedings of the ACM Symposium on Cloud Computing. 2014.
    /// https://dl.acm.org/doi/10.1145/2670979.2670997
    ///
    /// ### Returns
    /// * `ConnectedComponentsBuilder`: A builder object for configuring or
    ///   computing connected components.
    ///
    /// ### Example
    /// ```
    /// use datafusion::dataframe;
    /// use graphframes_rs::{GraphFrame, VERTEX_ID, EDGE_SRC, EDGE_DST};
    /// let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64, 3i64]).unwrap();
    /// let edges = dataframe!(EDGE_SRC => vec![1i64, 2i64, 3i64], EDGE_DST => vec![3i64, 1i64, 2i64]).unwrap();
    /// let graph = GraphFrame { vertices, edges };
    /// let components = graph.connected_components().run();
    /// ```
    pub fn connected_components(&self) -> ConnectedComponentsBuilder<'_> {
        ConnectedComponentsBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::create_test_graph;
    use crate::util::create_ldbc_test_graph;
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{Field, Schema};

    #[tokio::test]
    async fn test_min_nbr() -> Result<()> {
        let graph = create_test_graph()?;
        let min_nbrs = min_neighbours(&graph.edges.clone(), true)?;
        assert_eq!(min_nbrs.schema().fields().len(), 2);
        assert_eq!(min_nbrs.clone().count().await?, 10);
        let collected = min_nbrs.collect().await?;
        let min_neighbours = collected
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name(MIN_NBR)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .map(|v| v.unwrap())
                    .collect::<Vec<i64>>()
            })
            .collect::<Vec<i64>>();
        min_neighbours
            .iter()
            .zip(vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            .for_each(|(nbr, exp)| assert_eq!(*nbr, exp));
        Ok(())
    }

    #[tokio::test]
    async fn test_min_nbr_sum() -> Result<()> {
        let graph = create_test_graph()?;
        let min_nbrs = min_neighbours(&graph.edges.clone(), true)?;
        println!("min_nbrs schema: {:?}", min_nbrs.clone().schema());
        let first = min_nbrs
            .clone()
            .select(vec![cast(col(MIN_NBR), DataType::Decimal128(38, 0))])?;
        first.collect().await?;
        let sum = min_nbr_sum(&min_nbrs).await?;
        assert_eq!(sum, 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_zero_vertices() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => Vec::<i64>::new())?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame { vertices, edges };
        let cc = ConnectedComponentsBuilder::new(&graph).run().await?;
        assert_eq!(cc.data.schema().fields().len(), 2);
        assert_eq!(cc.data.count().await?, 0);
        assert_eq!(cc.num_iterations, 1);
        assert_eq!(cc.min_nbr_sum.len(), 1);
        assert_eq!(cc.min_nbr_sum[0], 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_single_vertex() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64])?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame { vertices, edges };
        let cc = ConnectedComponentsBuilder::new(&graph).run().await?;
        assert_eq!(cc.data.schema().fields().len(), 2);
        assert_eq!(cc.data.clone().count().await?, 1);
        assert_eq!(cc.num_iterations, 1);
        assert_eq!(cc.min_nbr_sum.len(), 1);
        assert_eq!(cc.min_nbr_sum[0], 0);
        assert_eq!(
            cc.data
                .collect()
                .await?
                .first()
                .unwrap()
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            1i64
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_two_vertices() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64])?;
        let edges = dataframe!(EDGE_SRC => vec![1i64], EDGE_DST => vec![2i64])?;
        let graph = GraphFrame { vertices, edges };
        let cc = ConnectedComponentsBuilder::new(&graph).run().await?;
        assert_eq!(cc.data.schema().fields().len(), 2);
        assert_eq!(cc.data.clone().count().await?, 2);
        assert_eq!(cc.num_iterations, 1);
        assert_eq!(cc.min_nbr_sum.len(), 1);
        assert_eq!(cc.min_nbr_sum[0], 2);
        let batches = cc.data.sort_by(vec![col(VERTEX_ID)])?.collect().await?;
        let result = batches.iter().fold(Vec::<i64>::new(), |mut acc, batch| {
            acc.append(
                &mut batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec(),
            );
            acc
        });
        assert_eq!(result[0], 1i64);
        assert_eq!(result[1], 1i64);

        Ok(())
    }

    #[tokio::test]
    async fn test_disconnected_vertices() -> Result<()> {
        let vertices = dataframe!(VERTEX_ID => vec![1i64, 2i64])?;
        let edges = dataframe!(EDGE_SRC => Vec::<i64>::new(), EDGE_DST => Vec::<i64>::new())?;
        let graph = GraphFrame { vertices, edges };
        let cc = ConnectedComponentsBuilder::new(&graph).run().await?;
        assert_eq!(cc.data.schema().fields().len(), 2);
        assert_eq!(cc.data.clone().count().await?, 2);
        assert_eq!(cc.num_iterations, 1);
        assert_eq!(cc.min_nbr_sum.len(), 1);
        assert_eq!(cc.min_nbr_sum[0], 0);
        let batches = cc.data.sort_by(vec![col(VERTEX_ID)])?.collect().await?;
        let result = batches.iter().fold(Vec::<i64>::new(), |mut acc, batch| {
            acc.append(
                &mut batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec(),
            );
            acc
        });
        assert_eq!(result[0], 1i64);
        assert_eq!(result[1], 2i64);

        Ok(())
    }

    async fn get_ldbc_wcc_results(dataset: &str) -> Result<DataFrame> {
        let ctx = SessionContext::new();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected_wcc_schema = Schema::new(vec![
            Field::new("vertex_id", DataType::Int64, false),
            Field::new("expected_component", DataType::Int64, false),
        ]);
        let expected_wcc_path = format!(
            "{}/testing/data/ldbc/{}/{}-WCC.csv",
            manifest_dir, dataset, dataset
        );
        let expected_sp = ctx
            .read_csv(
                &expected_wcc_path,
                CsvReadOptions::new()
                    .delimiter(b' ')
                    .has_header(false)
                    .schema(&expected_wcc_schema),
            )
            .await?;
        Ok(expected_sp)
    }

    #[tokio::test]
    async fn test_ldbc() -> Result<()> {
        let expected_components = get_ldbc_wcc_results("test-wcc-directed").await?;
        let graph = create_ldbc_test_graph("test-wcc-directed", false, false).await?;

        let results = graph.connected_components().run().await?.data;
        let diff = results
            .clone()
            .join(
                expected_components,
                JoinType::Left,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .select(vec![
                col(VERTEX_ID),
                col(COMPONENT_COL),
                col("expected_component"),
            ])?
            .filter(col(COMPONENT_COL).not_eq(col("expected_component")))?;

        assert_eq!(diff.count().await?, 0);

        Ok(())
    }
}
