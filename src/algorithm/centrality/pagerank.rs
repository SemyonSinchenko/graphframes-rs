use crate::algorithm::pregel::{MessageDirection, pregel_default_msg, pregel_src};
use crate::memory::CheckpointConfig;
use crate::{GraphFrame, VERTEX_ID};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions_aggregate::sum::sum;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;
use futures::{StreamExt, TryStreamExt};

/// Column name for pagerank in the Page Rank algorithm
pub const PAGERANK: &str = "pagerank";

/// A builder for the PageRank algorithm.
///
#[derive(Debug, Clone)]
pub struct PageRankBuilder<'a> {
    graph: &'a GraphFrame,
    max_iter: usize,
    reset_prob: f64,
    tol: f64,

    /// Storage options
    checkpoint_config: CheckpointConfig,
}

impl<'a> PageRankBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        PageRankBuilder {
            graph,
            max_iter: 0,
            reset_prob: 0.15,
            tol: 0.01,
            checkpoint_config: CheckpointConfig::default_local_fs(),
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

    /// Execute PageRank algorithm.
    /// Execution via generic Pregel engine.
    ///
    /// Runs the Pregel engine to produce raw pageranks, then reads them back,
    /// normalizes so the ranks sum to 1, and writes the final `id` + `pagerank` projection
    /// to `output`. The intermediate raw Pregel output is deleted before returning.
    /// Returns the number of Pregel iterations executed.
    pub async fn run(
        self,
        ctx: &SessionContext,
        output: &str,
        _include_debug_columns: bool,
    ) -> Result<usize> {
        // Defaults to 0.85 if default reset_prob is used: 0.15
        let alpha = 1.0 - self.reset_prob;
        let reset_prob_per_vertices = self.reset_prob;

        // PageRank needs the out-degree of each vertex to distribute its rank.
        let vertices_with_degrees = self.graph.out_degrees().await?;

        // Create a temp graph that has vertices with degrees that will be used in Pregel Execution
        let graph_with_degrees = GraphFrame {
            vertices: vertices_with_degrees,
            edges: self.graph.edges.clone(),
        };

        let store_url = self.checkpoint_config.store_url.clone();
        let intermediate_dir = self.checkpoint_config.dir.clone().join("_pregel_raw");
        let intermediate_uri = format!("{}{}/", store_url.as_str(), intermediate_dir);

        let pregel_builder = graph_with_degrees
            .pregel()
            .add_vertex_column(
                PAGERANK,
                lit(reset_prob_per_vertices), // All vertices start with a rank of 1/N
                lit(reset_prob_per_vertices)
                    + lit(alpha) * coalesce(vec![pregel_default_msg(), lit(0.0)]),
            )
            .add_vertex_column("out_degree", col("out_degree"), col("out_degree"))
            .add_message(
                pregel_src(PAGERANK) / pregel_src("out_degree"),
                MessageDirection::SrcToDst,
            )
            .add_aggregate_expr(sum(pregel_default_msg()))
            .skip_dest_state()
            .with_checkpoint_store(store_url.clone())
            .set_checkpoint_dir(self.checkpoint_config.dir.join("inner_checkpoint"));

        let num_iterations = if self.max_iter > 0 {
            pregel_builder
                .max_iterations(self.max_iter)
                .run(ctx, &intermediate_uri, false)
                .await?
        } else {
            pregel_builder
                .with_vertex_voting(
                    "rank_changed",
                    abs(col(PAGERANK)
                        - (lit(reset_prob_per_vertices)
                            + lit(alpha) * coalesce(vec![pregel_default_msg(), lit(0.0)])))
                    .gt(lit(self.tol)),
                )
                .run(ctx, &intermediate_uri, false)
                .await?
        };

        let calculated_page_ranks = ctx
            .read_parquet(&intermediate_uri, ParquetReadOptions::default())
            .await?;

        let aggregated_rank = calculated_page_ranks
            .clone()
            .aggregate(vec![], vec![sum(col(PAGERANK)).alias("pagerank_sum")])?
            .cache()
            .await?;

        let final_page_ranks =
            calculated_page_ranks.join(aggregated_rank, JoinType::Inner, &[], &[], None)?;
        let final_page_ranks = final_page_ranks.select(vec![
            col(VERTEX_ID),
            (col(PAGERANK) / col("pagerank_sum")).alias(PAGERANK),
        ])?;

        final_page_ranks
            .write_parquet(output, DataFrameWriteOptions::new(), None)
            .await?;

        // Clean up the intermediate raw Pregel output now that the normalized ranks have been
        // written to `output`. The Pregel engine already purges its own internal checkpoints
        // (edges + state), but its final output — which we wrote to `_pregel_raw` — is the
        // caller's responsibility. We use the object_store API (list + delete_stream) rather
        // than std::fs so this works against any backing store, not just the local filesystem.
        let store = ctx.runtime_env().object_store(&store_url)?;
        let intermediate_path = intermediate_dir.clone();
        let objects = store
            .list(Some(&intermediate_path))
            .map_ok(|m| m.location)
            .boxed();
        store.delete_stream(objects).try_collect::<Vec<_>>().await?;

        Ok(num_iterations)
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
    use crate::utils::create_ldbc_test_graph;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Returns a unique directory under `std::env::temp_dir()` for this test run, creating it.
    /// Combining the PID with a process-wide counter guarantees uniqueness across parallel
    /// `cargo test` invocations and concurrent tests.
    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_pr_test_{}_{n}_{label}", id()));
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
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("pagerank_run")?;
        graph
            .pagerank()
            .max_iter(14)
            .reset_prob(0.15)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        let calculated_page_rank = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
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
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("pagerank_early_stopping")?;
        // `max_iter(0)` exercises the `with_vertex_voting` branch in
        // `PageRankBuilder::run`. A tight `tol` keeps the converged ranks
        // well within the comparison threshold below.
        graph
            .pagerank()
            .max_iter(0)
            .reset_prob(0.15)
            .tol(0.0001)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri, false)
            .await?;
        let calculated_page_rank = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
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

        // No pagerank should differ from the LDBC reference by more than 0.01.
        assert_eq!(comparison_df.count().await?, 0);

        Ok(())
    }
}
