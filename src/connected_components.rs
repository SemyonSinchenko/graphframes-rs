//! Implementation inspired by GraphFrames Spark/Scala code
//! that is distributed under Apache License 2.0.
//! https://github.com/graphframes/graphframes/blob/master/core/src/main/scala/org/graphframes/lib/ConnectedComponents.scala

use crate::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use datafusion::arrow::array::{Array, Decimal128Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::functions_aggregate::expr_fn::{count, min, sum};
use datafusion::prelude::*;

pub const COMPONENT_COL: &str = "component";
const MIN_NBR: &str = "min_nbr";
const CNT_OF_NBR: &str = "cnt_of_nbr";

fn min_neighbours(edges: &DataFrame, including_self: bool) -> Result<DataFrame> {
    let res = edges
        .clone()
        .union(edges.clone().select(vec![
            col(EDGE_DST).alias(EDGE_SRC),
            col(EDGE_SRC).alias(EDGE_DST),
        ])?)?
        .aggregate(
            vec![col(EDGE_SRC).alias(VERTEX_ID)],
            vec![min(col(EDGE_DST)).alias(MIN_NBR), count(col(EDGE_DST))],
        );

    if including_self {
        res?.with_column(
            MIN_NBR,
            when(col(VERTEX_ID).lt(col(MIN_NBR)), col(VERTEX_ID)).otherwise(col(MIN_NBR))?,
        )
    } else {
        res
    }
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

#[derive(Debug)]
pub struct ConnectedComponentsOutput {
    pub data: DataFrame,
    pub num_iterations: usize,
    pub min_nbr_sum: Vec<i128>,
}

#[derive(Debug)]
pub struct ConnectedComponentsBuilder<'a> {
    graph_frame: &'a GraphFrame,
    checkpoint_interval: i32,
}

impl<'a> ConnectedComponentsBuilder<'a> {
    pub fn new(graph_frame: &'a GraphFrame) -> Self {
        Self {
            graph_frame,
            checkpoint_interval: 1,
        }
    }

    pub fn checkpoint_interval(mut self, checkpoint_interval: i32) -> Self {
        self.checkpoint_interval = checkpoint_interval;
        self
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

        let cc_graph = GraphFrame {
            vertices: vertices.clone(),
            edges: deduped_edges.clone(),
        };

        let mut iteration = 0usize;
        let mut metrics = Vec::<i128>::new();
        let mut converged = false;

        let mut minimal_neighbours_1 = min_neighbours(&cc_graph.edges.clone(), true)?;
        let mut last_iter_nbr_sum = min_nbr_sum(&minimal_neighbours_1.clone()).await?;
        metrics.push(last_iter_nbr_sum);
        let mut current_edges = deduped_edges.clone();

        while !converged {
            iteration += 1;
            // large-star step:
            // connects all strictly larger neighbors to the min neighbor (including self)
            current_edges = current_edges
                .join_on(
                    minimal_neighbours_1.clone(),
                    JoinType::Inner,
                    vec![col(VERTEX_ID).eq(col(EDGE_SRC))],
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
                    vec![col(VERTEX_ID).eq(col(EDGE_SRC))],
                )?
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
            data: vertices.join_on(
                current_edges,
                JoinType::Inner,
                vec![col(VERTEX_ID).eq(col(EDGE_DST))],
            )?,
            num_iterations: iteration,
            min_nbr_sum: metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::create_test_graph;
    use datafusion::arrow::array::Int64Array;

    #[tokio::test]
    async fn test_min_nbr() -> Result<()> {
        let graph = create_test_graph()?;
        let min_nbrs = min_neighbours(&graph.edges.clone(), true)?;
        assert_eq!(min_nbrs.schema().fields().len(), 3);
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
}
