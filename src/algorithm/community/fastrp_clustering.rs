//! Graph clustering via FastRP embeddings + K-Means.
//!
//! A thin pipeline on top of the two existing building blocks: build
//! unit-length FastRP embeddings ([`crate::algorithm::embeddings::fastrp`],
//! output normalization forced on) and cluster them with K-Means
//! ([`crate::ml::KMeansBuilder`], k-means|| init, one column per requested
//! `k`).
//!
//! The output parquet contains `[id, embedding, center_{k}...]`: the unit
//! norm embedding plus the assigned center index for every requested `k`.

use datafusion::{
    dataframe::DataFrameWriteOptions,
    error::{DataFusionError, Result},
    execution::object_store::ObjectStoreUrl,
    object_store::path::Path,
    prelude::*,
};
use futures::{StreamExt, TryStreamExt};

use crate::{
    GraphFrame, VERTEX_ID,
    algorithm::embeddings::fastrp::{EMBEDDING, FastRPNormalization},
    expressions::kmeans_assign_expr,
    memory::CheckpointConfig,
    ml::{DistanceMetric, KMeansBuilder, KMeansResult},
    utils::{GraphFramesConfig, scoped_ctx},
};

/// Builder for the FastRP + K-Means clustering pipeline.
pub struct FastRPClusteringBuilder {
    graph: GraphFrame,
    dim: usize,
    iterations: usize,
    seed: u64,
    normalization: FastRPNormalization,
    iteration_weights: Option<Vec<f64>>,
    k: Vec<usize>,
    metric: DistanceMetric,
    max_iter: usize,
    tol: f64,
    kmeans_init_steps: usize,
    checkpoint_config: CheckpointConfig,
}

impl FastRPClusteringBuilder {
    pub fn new(graph: GraphFrame) -> Self {
        FastRPClusteringBuilder {
            graph: graph,
            dim: 0,
            iterations: 4,
            seed: 42,
            normalization: FastRPNormalization::None,
            iteration_weights: None,
            k: vec![2],
            metric: DistanceMetric::L2,
            max_iter: 20,
            tol: 1e-4,
            kmeans_init_steps: 2,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
    }

    /// Embedding dimension `D`.
    pub fn set_dim(mut self, v: usize) -> Self {
        self.dim = v;
        self
    }

    /// FastRP propagation iterations `K` (default: 4).
    pub fn set_iterations(mut self, v: usize) -> Self {
        self.iterations = v;
        self
    }

    /// Seed for the random projections and the k-means|| init (default: 42).
    pub fn set_seed(mut self, v: u64) -> Self {
        self.seed = v;
        self
    }

    /// Per-iteration normalization of the propagated vectors (default:
    /// [`FastRPNormalization::None`]).
    pub fn set_normalization(mut self, v: FastRPNormalization) -> Self {
        self.normalization = v;
        self
    }

    /// Per-iterate weights of the FastRP linear combination
    /// `H = Σ w_t · H_t` over `H_1..H_K` (must have exactly `K` entries;
    /// default: all ones). The random init is never part of the combination.
    pub fn set_iteration_weights(mut self, v: Vec<f64>) -> Self {
        self.iteration_weights = Some(v);
        self
    }

    /// Set a single `k`: number of clusters.
    pub fn set_k(mut self, k: usize) -> Self {
        self.k = vec![k];
        self
    }

    /// Set multiple k values: one `center_{k}` column per K in the output.
    pub fn set_multiple_k(mut self, kk: Vec<usize>) -> Self {
        self.k = kk;
        self
    }

    /// K-Means distance metric (default: L2; embeddings are unit norm, so
    /// L2 on the sphere ranks neighbors like cosine would).
    pub fn set_metric(mut self, v: DistanceMetric) -> Self {
        self.metric = v;
        self
    }

    /// Maximum Lloyd iterations (default: 20).
    pub fn set_max_iter(mut self, v: usize) -> Self {
        self.max_iter = v;
        self
    }

    /// Convergence tolerance on the center shift (default: 1e-4).
    pub fn set_tol(mut self, v: f64) -> Self {
        self.tol = v;
        self
    }

    /// k-means|| initialization steps (default: 2).
    pub fn set_kmeans_init_steps(mut self, v: usize) -> Self {
        self.kmeans_init_steps = v;
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

    /// Run the pipeline and write `[id, embedding, center_{k}...]` parquet
    /// to `output`. Returns the [`KMeansResult`] of the K-Means stage.
    pub async fn run(self, ctx: &SessionContext, output: &str) -> Result<KMeansResult> {
        if self.dim == 0 {
            return Err(DataFusionError::Plan(
                "FastRP clustering requires a positive embedding dimension: set `set_dim(D)`"
                    .to_string(),
            ));
        }
        if self.k.is_empty() {
            return Err(DataFusionError::Plan(
                "FastRP clustering requires at least one k value: set `set_k(k)`".to_string(),
            ));
        }
        let gf_config = ctx
            .state()
            .config()
            .options()
            .extensions
            .get::<GraphFramesConfig>()
            .cloned()
            .unwrap_or_default();

        let ctx = &scoped_ctx(ctx, gf_config.prefer_smj);
        self.checkpoint_config.validate_output(output)?;

        let run_id = uuid::Uuid::new_v4().to_string();
        log::info!("start FastRP clustering with ID {run_id}");

        let run_dir = self.checkpoint_config.dir.clone().join(run_id.clone());
        let embeddings_dir = run_dir.clone().join("embeddings");
        let embeddings_uri = format!(
            "{}{}/",
            self.checkpoint_config.store_url.as_str(),
            embeddings_dir
        );

        // Stage 1: unit-length FastRP embeddings.
        self.graph
            .fastrp()
            .dim(self.dim)
            .iterations(self.iterations)
            .seed(self.seed)
            .normalization(self.normalization)
            .iteration_weights(self.iteration_weights.clone().unwrap_or_else(|| {
                // mirror the FastRP default: all ones over H_1..H_K
                vec![1.0; self.iterations]
            }))
            .norm_output(true)
            .set_checkpoint_dir(run_dir.join("fastrp_checkpoints"))
            .with_checkpoint_store(self.checkpoint_config.store_url.clone())
            .run(&ctx, &embeddings_uri)
            .await?;

        // Stage 2: K-Means over the embeddings.
        let raw = ctx
            .read_parquet(&embeddings_uri, ParquetReadOptions::default())
            .await?;
        let features = raw.select(vec![col(VERTEX_ID), col(EMBEDDING)])?;

        let kmeans = KMeansBuilder::new(&features, EMBEDDING)
            .seed(self.seed)
            .k_values(&self.k)
            .metric(self.metric)
            .max_iter(self.max_iter)
            .tol(self.tol)
            .init_steps(self.kmeans_init_steps);

        log::info!("run KMeans on the FastRP embedding...");
        let result = kmeans.run().await?;
        log::info!(
            "KMeans converged after {} iterations",
            result.num_iterations
        );

        // Final table: keep the embedding (handy for inspection) and assign
        // the center index per requested k.
        let mut final_columns = vec![col(VERTEX_ID), col(EMBEDDING)];
        for (kk, run) in self.k.iter().zip(&result.runs) {
            final_columns.push(
                kmeans_assign_expr(
                    col(EMBEDDING),
                    run.k, // effective k
                    result.d,
                    run.centers.clone(),
                    self.metric,
                )
                .alias(format!("center_{kk}")),
            );
        }

        features
            .select(final_columns)?
            .write_parquet(output, DataFrameWriteOptions::new(), None)
            .await?;
        log::info!("result was written into {output}");

        // clean up the intermediate embeddings
        let store = ctx
            .runtime_env()
            .object_store(&self.checkpoint_config.store_url)?;
        let paths = store
            .list(Some(&embeddings_dir))
            .map_ok(|m| m.location)
            .boxed();
        store.delete_stream(paths).try_collect::<Vec<_>>().await?;

        Ok(result)
    }
}

impl GraphFrame {
    /// Create a new FastRP + K-Means clustering builder.
    pub fn fastrp_clustering(&self) -> FastRPClusteringBuilder {
        FastRPClusteringBuilder::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::utils::symmetrize;
    use crate::{EDGE_DST, EDGE_SRC, VERTEX_ID};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("graphframes_fastrp_clu_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    struct TempGuard(PathBuf);

    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn setup(label: &str) -> (SessionContext, Path, String, TempGuard) {
        let parent = unique_temp_dir(label);
        let checkpoint_root = parent.join("checkpoints");
        let output_root = parent.join("output");
        fs::create_dir_all(&checkpoint_root).expect("checkpoint dir");
        fs::create_dir_all(&output_root).expect("output dir");

        let checkpoint_dir = Path::from_filesystem_path(&checkpoint_root).unwrap();
        let output_uri = url::Url::from_directory_path(&output_root)
            .unwrap()
            .to_string();

        (
            SessionContext::new(),
            checkpoint_dir,
            output_uri,
            TempGuard(parent),
        )
    }

    /// Two 4-cliques {1..4}, {5..8} joined by the single bridge 4-5,
    /// symmetrized.
    fn two_cliques() -> Result<GraphFrame> {
        let vertices = dataframe!(VERTEX_ID => Vec::<i64>::from(vec![1, 2, 3, 4, 5, 6, 7, 8]))?;
        let mut edges = Vec::new();
        for (a, b) in [
            (1, 2),
            (1, 3),
            (2, 3),
            (1, 4),
            (2, 4),
            (3, 4), // clique A
            (5, 6),
            (5, 7),
            (6, 7),
            (5, 8),
            (6, 8),
            (7, 8), // clique B
            (4, 5), // bridge
        ] {
            edges.push(vec![a, b]);
        }
        let edges_df = dataframe!(
            EDGE_SRC => Vec::<i64>::from(edges.iter().map(|e| e[0]).collect::<Vec<i64>>()),
            EDGE_DST => Vec::<i64>::from(edges.iter().map(|e| e[1]).collect::<Vec<i64>>()),
        )?;
        let edges_df = symmetrize(&edges_df, true, None)?;
        GraphFrame::try_new(vertices, edges_df)
    }

    /// Read `[id, <col_name>]` into a map (column looked up by name).
    async fn read_clusters(df: DataFrame, col_name: &str) -> Result<HashMap<i64, i64>> {
        let mut map = HashMap::new();
        for batch in df.collect().await? {
            let schema = batch.schema();
            let id_idx = schema.index_of(VERTEX_ID)?;
            let c_idx = schema.index_of(col_name)?;
            let ids = batch
                .column(id_idx)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int64Array>()
                .unwrap();
            let clusters = batch
                .column(c_idx)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::Int32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                map.insert(ids.value(i), clusters.value(i) as i64);
            }
        }
        Ok(map)
    }

    #[tokio::test]
    async fn fastrp_clustering_separates_cliques() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("cliques");
        let g = two_cliques()?;

        g.fastrp_clustering()
            .set_dim(8)
            .set_iterations(3)
            .set_seed(42)
            .set_k(2)
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let out = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?;
        let clusters = read_clusters(out, "center_2").await?;
        assert_eq!(clusters.len(), 8);

        // at most the (symmetrized) bridge may cross the cluster boundary
        let mut crossing = 0;
        for (a, b) in [
            (1, 2),
            (1, 3),
            (2, 3),
            (1, 4),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (6, 7),
            (5, 8),
            (6, 8),
            (7, 8),
            (4, 5),
        ] {
            if clusters[&a] != clusters[&b] {
                crossing += 1;
            }
        }
        assert!(
            crossing <= 2,
            "the two cliques must land in different clusters, crossing={crossing}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_clustering_multiple_k_columns() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("multi_k");
        let g = two_cliques()?;

        g.fastrp_clustering()
            .set_dim(8)
            .set_iterations(2)
            .set_seed(42)
            .set_multiple_k(vec![2, 3])
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await?;

        let out = ctx
            .read_parquet(&output_uri, ParquetReadOptions::default())
            .await?;
        let names: Vec<String> = out
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect();
        assert!(names.contains(&"center_2".to_string()), "cols={names:?}");
        assert!(names.contains(&"center_3".to_string()), "cols={names:?}");
        assert!(names.contains(&"embedding".to_string()), "cols={names:?}");
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_clustering_is_deterministic_for_a_seed() -> Result<()> {
        let run = || async {
            let (ctx, checkpoint_dir, output_uri, _guard) = setup("det");
            let g = two_cliques()?;
            g.fastrp_clustering()
                .set_dim(8)
                .set_iterations(3)
                .set_seed(5)
                .set_k(2)
                .set_checkpoint_dir(checkpoint_dir)
                .run(&ctx, &output_uri)
                .await?;
            let out = ctx
                .read_parquet(&output_uri, ParquetReadOptions::default())
                .await?;
            read_clusters(out, "center_2").await
        };

        let first = run().await?;
        let second = run().await?;
        assert_eq!(first, second, "same seed must reproduce the clustering");
        Ok(())
    }

    #[tokio::test]
    async fn fastrp_clustering_requires_dim_and_k() -> Result<()> {
        let (ctx, checkpoint_dir, output_uri, _guard) = setup("validation");
        let g = two_cliques()?;

        let no_dim = g
            .fastrp_clustering()
            .set_checkpoint_dir(checkpoint_dir.clone())
            .run(&ctx, &output_uri)
            .await;
        assert!(no_dim.is_err(), "dim(0) must be rejected");

        let no_k = g
            .fastrp_clustering()
            .set_dim(4)
            .set_multiple_k(vec![])
            .set_checkpoint_dir(checkpoint_dir)
            .run(&ctx, &output_uri)
            .await;
        assert!(no_k.is_err(), "empty k must be rejected");
        Ok(())
    }
}
