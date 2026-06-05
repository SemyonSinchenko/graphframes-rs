use crate::pregel::{MessageDirection, PREGEL_MSG, PregelBuilder, pregel_src};
use crate::{GraphFrame, VERTEX_ID};
use datafusion::error::Result;
use datafusion::functions_aggregate::sum::sum;
use datafusion::prelude::*;

/// Column name for pagerank in the Page Rank algorithm
pub const PAGERANK: &str = "pagerank";

/// A builder for the PageRank algorithm.
///
#[derive(Debug, Clone)]
pub struct PageRankBuilder<'a> {
    graph: &'a GraphFrame,
    max_iter: usize,
    reset_prob: f64,
    include_debug_columns: bool,
    checkpoint_interval: usize,
    tol: f64,
}

impl<'a> PageRankBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        PageRankBuilder {
            graph,
            max_iter: 0,
            reset_prob: 0.15,
            include_debug_columns: false,
            checkpoint_interval: 2, // Revisit once default is finalized. temp for now.
            tol: 0.01,
        }
    }

    pub fn max_iter(mut self, iter: usize) -> Self {
        self.max_iter = iter;
        self
    }

    pub fn reset_prob(mut self, prob: f64) -> Self {
        self.reset_prob = prob;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn checkpoint_interval(mut self, checkpoint_interval: usize) -> Self {
        self.checkpoint_interval = checkpoint_interval;
        self
    }

    /// Execute PageRank algorithm.
    /// Execution via generic Pregel engine
    pub async fn run(&self) -> Result<DataFrame> {
        let num_vertices = self.graph.num_nodes().await? as f64;
        // Defaults to 0.85 if default reset_prob is used: 0.15
        let alpha = 1.0 - self.reset_prob;
        let reset_prob_per_vertices = self.reset_prob / num_vertices;

        // --- Step 1: Pre-calculate out-degrees ---
        // PageRank needs the out-degree of each vertex to distribute its rank.
        //contains vertex_id and put_degree for each vertices
        let vertices_with_degrees = self.graph.out_degrees().await?;

        // Create a temp graph that has vertices with degrees that will be used in Pregel Execution
        let graph_with_degrees = GraphFrame {
            vertices: vertices_with_degrees,
            edges: self.graph.edges.clone(),
        };

        // --- Step 2: Create Pregel builder ---
        let pregel_builder = PregelBuilder::new(graph_with_degrees)
            .checkpoint_interval(self.checkpoint_interval)
            .add_vertex_column(
                PAGERANK,
                lit(reset_prob_per_vertices), // All vertices start with a rank of 1/N
                lit(reset_prob_per_vertices) + lit(alpha) * col(PREGEL_MSG), // PageRank calculation
            )
            .add_vertex_column("out_degree", col("out_degree"), col("out_degree")) // out_degrees are static
            .add_message(
                pregel_src(PAGERANK) / pregel_src("out_degree"),
                MessageDirection::SrcToDst,
            )
            .with_aggregate_expr(sum(col(PREGEL_MSG)))
            .skip_dest_state();

        // --- Step 3: Run Pregel Engine ---
        let result = if self.max_iter > 0 {
            pregel_builder
                .max_iterations(self.max_iter)
                .run(self.include_debug_columns)
                .await?
        } else {
            pregel_builder
                .with_vertex_voting(
                    "rank_changed",
                    abs(col(PAGERANK)
                        - (lit(reset_prob_per_vertices) + lit(alpha) * col(PREGEL_MSG)))
                    .lt(lit(self.tol)),
                )
                .run(self.include_debug_columns)
                .await?
        };
        let calculated_page_ranks = result.data;

        // Calculating the pagerank aggregation
        let aggregated_rank = calculated_page_ranks
            .clone()
            .aggregate(vec![], vec![sum(col(PAGERANK)).alias("pagerank_sum")])?
            .cache()
            .await?;

        // Cross join with calculated pageranks df
        let final_page_ranks =
            calculated_page_ranks.join(aggregated_rank, JoinType::Inner, &[], &[], None)?;
        Ok(final_page_ranks.select(vec![
            col(VERTEX_ID),
            (col(PAGERANK) / col("pagerank_sum")).alias(PAGERANK),
        ])?)
    }
}

impl GraphFrame {
    /// Create a new PageRank algorithm builder
    pub fn pagerank(&self) -> PageRankBuilder<'_> {
        PageRankBuilder::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::create_ldbc_test_graph;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    // Gets the expected pagerank results from the mentioned ldbc dataset
    async fn get_ldbc_pr_results(dataset: &str) -> Result<DataFrame> {
        let ctx = SessionContext::new();
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let expected_pr_schema = Schema::new(vec![
            Field::new("vertex_id", DataType::Int64, false),
            Field::new("expected_pr", DataType::Float64, false),
        ]);
        let expected_pr_path = format!(
            "{}/testing/data/ldbc/{}/{}-PR.csv",
            manifest_dir, dataset, dataset
        );
        let expected_pr = ctx
            .read_csv(
                &expected_pr_path,
                CsvReadOptions::new()
                    .delimiter(b' ')
                    .has_header(false)
                    .schema(&expected_pr_schema),
            )
            .await?;
        Ok(expected_pr)
    }

    #[tokio::test]
    async fn test_pagerank_run() -> Result<()> {
        let test_dataset: &str = "test-pr-directed";
        let graph = create_ldbc_test_graph(test_dataset, false, false).await?;
        let calculated_page_rank = graph
            .pagerank()
            .max_iter(14)
            .reset_prob(0.15)
            .checkpoint_interval(1)
            .run()
            .await?;
        let ldbc_page_rank = get_ldbc_pr_results(test_dataset).await?;

        let comparison_df = calculated_page_rank
            .join(
                ldbc_page_rank,
                JoinType::Left,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .with_column("difference", abs(col(PAGERANK) - col("expected_pr")))?
            .filter(col("difference").gt(lit(0.0015)))?;

        // comparison_df.clone().show().await?;

        // Check if there are no pageranks with difference more than 0.0015
        assert_eq!(comparison_df.count().await?, 0);

        Ok(())
    }

    /// Verifies the early-stopping path in `PageRankBuilder::run`.
    ///
    /// When `max_iter` is `0` (the default, or set explicitly), the builder
    /// uses `with_vertex_voting` to stop once the per-vertex rank change
    /// falls below `tol`. The resulting ranks must still match the LDBC
    /// reference within the same tolerance used by `test_pagerank_run`.
    #[tokio::test]
    async fn test_pagerank_run_early_stopping() -> Result<()> {
        let test_dataset: &str = "test-pr-directed";
        let graph = create_ldbc_test_graph(test_dataset, false, false).await?;
        // `max_iter(0)` exercises the `with_vertex_voting` branch in
        // `PageRankBuilder::run`. A tight `tol` keeps the converged ranks
        // well within the comparison threshold below.
        let calculated_page_rank = graph
            .pagerank()
            .max_iter(0)
            .reset_prob(0.15)
            .checkpoint_interval(1)
            .tol(0.0001)
            .run()
            .await?;
        let ldbc_page_rank = get_ldbc_pr_results(test_dataset).await?;

        let comparison_df = calculated_page_rank
            .join(
                ldbc_page_rank,
                JoinType::Left,
                &[VERTEX_ID],
                &["vertex_id"],
                None,
            )?
            .with_column("difference", abs(col(PAGERANK) - col("expected_pr")))?
            .filter(col("difference").gt(lit(0.01)))?;

        // No pagerank should differ from the LDBC reference by more than 0.0015.
        assert_eq!(comparison_df.count().await?, 0);

        Ok(())
    }
}
