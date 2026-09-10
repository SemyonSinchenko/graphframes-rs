#[cfg(feature = "cli")]
#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::object_store::path::Path as ObjectPath;
use datafusion::prelude::*;
use graphframes_rs::GraphFramesConfig;
use graphframes_rs::{
    DistanceMetric, EmbeddingMode, FastRPNormalization, InitStrategy, KMeansBuilder,
    WeightsStrategy, kmeans_assign_expr,
};
use graphframes_rs::{EDGE_DST, EDGE_SRC, GraphFrame, VERTEX_ID};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
enum Format {
    Parquet,
    Csv,
    Json,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FastrpNormalization {
    /// Plain sum, no normalization.
    None,
    /// Linear normalization: divide by the source out-degree.
    L1,
    /// Square normalization: divide by the square root of the source out-degree.
    L2,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PicInit {
    /// Degree vector d/Σd — the paper's recommended (and MLlib's "degree") init.
    Degree,
    /// Seeded pseudo-random v0, L1-normalized in expectation.
    Random,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PicWeights {
    /// Use the weight column (or unit weights) as-is.
    None,
    /// Positive pointwise mutual information: max(0, ln(w·W/(d_i·d_j))).
    Ppmi,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PicEmbedding {
    /// Paper mode (default): cluster on the final iterate only.
    Last,
    /// Extended mode: cluster on the full iterate history [v1..vm].
    Trajectory,
}

#[derive(Args, Debug)]
struct CommonArgs {
    /// Path (or URI) to the vertices file or directory.
    ///
    /// Must contain an Int64 vertex-id column (`id` by default; override with
    /// `--id-col-name`). Extra attribute columns are allowed but ignored.
    #[arg(long)]
    vertices: String,

    /// Path (or URI) to the edges file or directory.
    ///
    /// Must contain Int64 source/destination columns (`src`/`dst` by default;
    /// override with `--src-col-name` / `--dst-col-name`).
    #[arg(long)]
    edges: String,

    /// Output directory as a `file://` URI, e.g. `file:///tmp/out/`.
    ///
    /// The result is always written as parquet regardless of `--format`;
    /// parent directories are created automatically.
    #[arg(long)]
    output: String,

    /// Input file format for `--vertices` and `--edges`.
    ///
    /// Parquet by default. `csv` / `json` use DataFusion's default schema
    /// inference and parsing options.
    #[arg(long, value_enum, default_value_t = Format::Parquet)]
    format: Format,

    /// Name of the vertex-id column in the input; renamed to the library's `id`.
    #[arg(long, default_value = "id")]
    id_col_name: String,

    /// Name of the edge source column in the input; renamed to the library's `src`.
    #[arg(long, default_value = "src")]
    src_col_name: String,

    /// Name of the edge destination column in the input; renamed to `dst`.
    #[arg(long, default_value = "dst")]
    dst_col_name: String,

    /// Symmetrize the input graph after loading (add the reverse of every edge).
    ///
    /// Useful for algorithms defined on undirected graphs when the input stores
    /// each edge only once (LDBC undirected datasets). Off by default: some
    /// algorithms (e.g. PageRank) are only defined on directed graphs, and the
    /// per-algorithm decision belongs to the caller.
    #[arg(long)]
    symmetrize: bool,

    /// Keep the edge weight column in the graph.
    ///
    /// Currently no algorithm consumes edge weights, but keeping the column
    /// makes the loaded `GraphFrame` ready for weighted algorithms (e.g. SSSP).
    #[arg(long)]
    weighted: bool,

    /// Name of the edge weight column; only used with `--weighted`.
    #[arg(long, default_value = "weight")]
    weight_col_name: String,

    /// DataFusion spill-pool memory limit, e.g. `4G` or `512M`.
    ///
    /// A command-line value takes precedence over the environment variable.
    #[arg(long, env = "GRAPHFRAMES_MAX_MEMORY", default_value = "4G")]
    max_memory: String,

    /// Parallelism (= DataFusion `target_partitions`).
    ///
    /// A command-line value takes precedence over the environment variable.
    #[arg(long, env = "GRAPHFRAMES_NUM_WORKERS", default_value_t = 2)]
    num_workers: usize,

    /// Base working directory for checkpoints and DataFusion spill files.
    ///
    /// Relative paths resolve against the current directory; the directory and
    /// its `checkpoints/` and `df-spill/` subdirectories are created if missing.
    /// A command-line value takes precedence over the environment variable.
    #[arg(long, env = "GRAPHFRAMES_WORKDIR", default_value = "gf_workdir")]
    checkpoint_dir: String,

    /// Upper bound on the total size of DataFusion's temporary spill directory,
    /// e.g. `200G`.
    ///
    /// A command-line value takes precedence over the environment variable.
    #[arg(long, env = "GRAPHFRAMES_MAX_TEMP_FILE", default_value = "200G")]
    max_temp_file: String,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// PageRank via Pregel.
    PageRank {
        #[command(flatten)]
        common: CommonArgs,

        #[arg(long)]
        tol: f64,

        #[arg(long, default_value_t = 0.15)]
        reset_prob: f64,

        #[arg(long, default_value_t = 0)]
        max_iter: usize,
    },

    /// Weakly connected components via randomized contraction.
    Wcc {
        #[command(flatten)]
        common: CommonArgs,

        /// Random seed for the per-iteration affine hashes.
        #[arg(long)]
        seed: u64,
    },

    /// Maximal independent set (Ghaffari's algorithm).
    Mis {
        #[command(flatten)]
        common: CommonArgs,
    },

    /// k-core (coreness) decomposition.
    Kcore {
        #[command(flatten)]
        common: CommonArgs,

        /// 0 (default) = run until convergence.
        #[arg(long, default_value_t = 0)]
        max_iter: usize,
    },

    /// HyperANF approximate neighbourhood function.
    Hyperanf {
        #[command(flatten)]
        common: CommonArgs,

        /// Number of hops (also the iteration budget).
        #[arg(long)]
        n_hops: usize,

        /// HLL log2 of k. Must be in 4..=21.
        #[arg(long, default_value_t = 12, value_parser = clap::value_parser!(u8).range(4..=21))]
        lg_k: u8,

        /// Directed propagation (default). Pass `false` for the symmetric neighbourhood.
        #[arg(long, default_value_t = true)]
        directed: bool,
    },

    /// Multi-source shortest paths.
    ShortestPath {
        #[command(flatten)]
        common: CommonArgs,

        /// Landmark vertex ids, comma-separated (e.g. --landmarks 1,4,7).
        #[arg(long, value_delimiter = ',')]
        landmarks: Vec<i64>,

        /// Iteration cap; default = effectively unbounded.
        #[arg(long)]
        max_iterations: Option<usize>,

        /// Compute distance *to* the landmarks (reverse edges) instead of from them.
        #[arg(long)]
        to_landmarks: bool,
    },

    /// Power Iteration Clustering (Lin & Cohen 2010; Spark MLlib-inspired).
    Pic {
        #[command(flatten)]
        common: CommonArgs,

        /// Number of clusters; one cluster column per K (comma-separated).
        #[arg(long, value_delimiter = ',', default_values_t = vec![2usize])]
        k: Vec<usize>,

        /// Maximum power iterations. PIC is *truncated* power iteration: the
        /// cluster signal lives in early iterates, so on slow-mixing graphs a
        /// lower budget gives a *better* embedding (paper average: 13).
        #[arg(long, default_value_t = 20)]
        max_iter: usize,

        /// Convergence threshold on the (mass-normalized) acceleration.
        /// Flat across graph sizes; smaller = more iterations.
        #[arg(long, default_value_t = 1e-5)]
        tol: f64,

        /// Initial vector for the power iteration.
        #[arg(long, value_enum, default_value_t = PicInit::Degree)]
        init: PicInit,

        /// Edge weight transform.
        #[arg(long, value_enum, default_value_t = PicWeights::None)]
        weights: PicWeights,

        /// What the embedding column holds (paper clusters the final iterate).
        #[arg(long, value_enum, default_value_t = PicEmbedding::Last)]
        embedding: PicEmbedding,
    },

    /// Raw (graph-free) machine-learning algorithms.
    Mllib {
        #[command(subcommand)]
        cmd: MllibCommand,
    },

    /// FastRP (Fast Random Projection) vertex embeddings.
    Fastrp {
        #[command(flatten)]
        common: CommonArgs,

        /// Embedding dimension D.
        #[arg(long)]
        dim: usize,

        /// Number of propagation iterations K. The embedding is the last
        /// iterate H_K; 0 returns the raw random projections.
        #[arg(long, default_value_t = 4)]
        iterations: usize,

        /// Seed for the per-vertex sparse random projections.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Per-iteration normalization of the propagated vectors:
        /// `l1` divides each vector by the source out-degree (paper's S⁻¹),
        /// `l2` by the square root of it (S^(-1/2)).
        #[arg(long, value_enum, default_value_t = FastrpNormalization::None)]
        normalization: FastrpNormalization,

        /// L2-normalize the final embeddings to unit length.
        #[arg(long)]
        norm_output: bool,

        /// Per-iterate weights of the final linear combination
        /// H = Σ w_t·H_t over H_1..H_K (must have K entries; default: all
        /// ones). The random init is never part of the combination; a zero
        /// weight drops the iterate entirely.
        #[arg(long, value_delimiter = ',')]
        iteration_weights: Option<Vec<f64>>,
    },

    /// Graph clustering: FastRP embeddings (unit norm) + K-Means.
    FastrpClustering {
        #[command(flatten)]
        common: CommonArgs,

        /// Embedding dimension D.
        #[arg(long)]
        dim: usize,

        /// FastRP propagation iterations K.
        #[arg(long, default_value_t = 4)]
        iterations: usize,

        /// Seed for the random projections and the k-means|| init.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// Per-iteration normalization of the propagated vectors
        /// (see fastrp).
        #[arg(long, value_enum, default_value_t = FastrpNormalization::None)]
        normalization: FastrpNormalization,

        /// Per-iterate weights of the FastRP linear combination
        /// H = Σ w_t·H_t over H_1..H_K (must have K entries; default: all
        /// ones).
        #[arg(long, value_delimiter = ',')]
        iteration_weights: Option<Vec<f64>>,

        /// Number of clusters; one center column per K (comma-separated).
        #[arg(long, value_delimiter = ',', default_values_t = vec![2usize])]
        k: Vec<usize>,

        /// K-Means distance metric (embeddings are L2-normalized).
        #[arg(long, value_enum, default_value_t = KmeansArgsMetric::L2)]
        metric: KmeansArgsMetric,

        /// Maximum Lloyd iterations.
        #[arg(long, default_value_t = 20)]
        max_iter: usize,

        /// Convergence tolerance on the center shift.
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,

        /// k-means|| initialization steps.
        #[arg(long, default_value_t = 2)]
        init_steps: usize,
    },

    /// Classical Label Propagation (CDLP).
    ClassicalLp {
        #[command(flatten)]
        common: CommonArgs,

        /// Maximum iterations (LDBC default).
        #[arg(long, default_value_t = 10)]
        max_iter: usize,

        /// Treat the graph as undirected.
        /// CDLP is defined on the undirected graph:
        /// the default (`false`) treat graph as directed
        /// symmetrizes the edge set (LDBC semantics);
        /// pass `true` to skip symmetrization.
        #[arg(long, default_value_t = false)]
        undirected: bool,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum KmeansArgsMetric {
    L2,
    Cosine,
}

#[derive(Subcommand, Debug)]
enum MllibCommand {
    /// K-Means (k-means|| init, Lloyd iterations) over a feature column.
    ///
    /// The features (vertices) file must contain an Int64 `id` column and a
    /// Float32 feature column of the shape `List<Float32>` or
    /// `FixedSizeList<Float32>`.
    Kmeans {
        /// Path (or URI) to the features (vertices) file or directory.
        #[arg(long)]
        features: String,

        /// Output directory as a `file://` URI.
        #[arg(long)]
        output: String,

        /// Input file format for `--vertices`.
        #[arg(long, value_enum, default_value_t = Format::Parquet)]
        format: Format,

        /// Name of the vertex-id column in the input.
        #[arg(long, default_value = "id")]
        id_col_name: String,

        /// Name of the feature column: List<Float32> / FixedSizeList<Float32>.
        #[arg(long)]
        feature_col: String,

        /// Number of clusters; one cluster column per K (comma-separated).
        #[arg(long, value_delimiter = ',', default_values_t = vec![2usize])]
        k: Vec<usize>,

        /// Distance metric.
        #[arg(long, value_enum, default_value_t = KmeansArgsMetric::L2)]
        metric: KmeansArgsMetric,

        /// Maximum Lloyd iterations.
        #[arg(long, default_value_t = 20)]
        max_iter: usize,

        /// Convergence tolerance on the center shift.
        #[arg(long, default_value_t = 1e-4)]
        tol: f64,

        /// k-means|| initialization steps.
        #[arg(long, default_value_t = 2)]
        init_steps: usize,

        /// Seed for the (deterministic) sampling and Lloyd loop.
        #[arg(long, default_value_t = 42)]
        seed: u64,

        /// DataFusion spill-pool memory limit, e.g. `4G` or `512M`.
        #[arg(long, env = "GRAPHFRAMES_MAX_MEMORY", default_value = "4G")]
        max_memory: String,

        /// Parallelism (= DataFusion `target_partitions`).
        #[arg(long, env = "GRAPHFRAMES_NUM_WORKERS", default_value_t = 2)]
        num_workers: usize,

        /// Base working directory for spill files.
        #[arg(long, env = "GRAPHFRAMES_WORKDIR", default_value = "gf_workdir")]
        checkpoint_dir: String,

        /// Upper bound on the total size of DataFusion's temporary spill dir.
        #[arg(long, env = "GRAPHFRAMES_MAX_TEMP_FILE", default_value = "200G")]
        max_temp_file: String,
    },
}

#[derive(Parser, Debug)]
#[command(
    name = "graphframes",
    version,
    about = "Out-of-core graph algorithms over DataFusion"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

struct ResolvedDir {
    fs: PathBuf,
    object: ObjectPath,
}

fn ensure_dir(p: impl AsRef<Path>) -> Result<ResolvedDir> {
    let p = p.as_ref();
    std::fs::create_dir_all(p)?;
    let fs = std::fs::canonicalize(p)?;
    let object = ObjectPath::from_absolute_path(&fs)?;
    Ok(ResolvedDir { fs, object })
}

fn parse_mem(s: &str) -> Result<usize> {
    let s = s.trim();

    let split = s
        .find(|c: char| !c.is_ascii_digit())
        .ok_or_else(|| DataFusionError::Execution(format!("missing size unit in '{s}'")))?;
    let (digits, unit) = s.split_at(split);

    let num: usize = digits
        .parse()
        .map_err(|e| DataFusionError::Execution(format!("invalid number in '{s}': {e}")))?;

    let mult = match unit.to_ascii_lowercase().as_str() {
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        other => {
            return Err(DataFusionError::Execution(format!(
                "wrong memory unit '{other}'"
            )));
        }
    };
    Ok(num * mult)
}

fn build_context(
    work: &ResolvedDir,
    max_pool_mem: &str,
    num_workers: usize,
    max_temp_file_size: &str,
) -> Result<SessionContext> {
    let max_pool_mem = parse_mem(max_pool_mem)?;
    let max_temp_file_size = parse_mem(max_temp_file_size)? as u64;
    let spill = ensure_dir(work.fs.join("df-spill"))?;

    // The crate's stated default is "SMJ by default" (GraphFramesConfig::prefer_smj).
    // DataFusion's own default is the opposite (`datafusion.optimizer.prefer_hash_join
    // = true`), and it applies at *planning* time to every DataFrame of this session —
    // including the pre-Pregel `out_degrees` join, which is planned under the base
    // session (not under `scoped_ctx`). Mirror the crate default here so all joins are
    // planned consistently; otherwise a hash-join build side over sliced aggregate
    // output can reserve the whole memory pool (buffer-capacity accounting on
    // batch_size-sliced group-by arrays) and exhaust a 4G pool on small graphs.
    let gf_config = GraphFramesConfig::default();
    let config = SessionConfig::from_env()?
        .with_target_partitions(num_workers)
        .set_bool(
            "datafusion.optimizer.prefer_hash_join",
            !gf_config.prefer_smj,
        )
        .with_option_extension(gf_config);

    let runtime_env = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(max_pool_mem)))
        .with_temp_file_path(spill.fs)
        .with_max_temp_directory_size(max_temp_file_size)
        .build_arc()?;

    let session_state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime_env)
        .with_default_features()
        .build();

    Ok(SessionContext::from(session_state))
}

async fn read_data(ctx: &SessionContext, path: &str, format: Format) -> Result<DataFrame> {
    let r = match format {
        Format::Parquet => ctx.read_parquet(path, ParquetReadOptions::new()).await?,
        Format::Csv => ctx.read_csv(path, CsvReadOptions::new()).await?,
        Format::Json => ctx.read_json(path, JsonReadOptions::default()).await?,
    };

    Ok(r)
}

async fn build_graph(
    ctx: &SessionContext,
    vertices: &str,
    edges: &str,
    id_col: &str,
    src_col: &str,
    dst_col: &str,
    format: Format,
    weighted: bool,
    weight_col_name: &str,
    symmetrize: bool,
) -> Result<GraphFrame> {
    let r_vertices = read_data(ctx, vertices, format.clone()).await?;
    let r_edges = read_data(ctx, edges, format.clone()).await?;

    let vertices = r_vertices.select(vec![col(id_col).alias(VERTEX_ID)])?;

    let mut edge_cols = vec![col(src_col).alias(EDGE_SRC), col(dst_col).alias(EDGE_DST)];
    if weighted {
        // Keep the weight column under its original name: no canonical name is
        // defined yet, and no algorithm consumes it. Column selection fails
        // with a clear DataFusion error if the input has no such column.
        edge_cols.push(col(weight_col_name));
    }
    let edges = r_edges.select(edge_cols)?;

    let g = GraphFrame::try_new(vertices, edges)?;
    if symmetrize { g.symmetrize() } else { Ok(g) }
}

async fn setup(common: &CommonArgs) -> Result<(SessionContext, GraphFrame, ObjectPath)> {
    let work = ensure_dir(&common.checkpoint_dir)?;
    let checkpoints = ensure_dir(work.fs.join("checkpoints"))?;

    let ctx = build_context(
        &work,
        &common.max_memory,
        common.num_workers,
        &common.max_temp_file,
    )?;

    let g = build_graph(
        &ctx,
        &common.vertices,
        &common.edges,
        &common.id_col_name,
        &common.src_col_name,
        &common.dst_col_name,
        common.format.clone(),
        common.weighted,
        &common.weight_col_name,
        common.symmetrize,
    )
    .await?;

    Ok((ctx, g, checkpoints.object))
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("graphframes_rs=info"),
    )
    // Use env_logger's default format (timestamp + colored level + message)
    // but drop the module path;
    .format_target(false)
    .init();

    let cli = Cli::parse();

    match cli.command {
        Command::PageRank {
            common,
            tol,
            reset_prob,
            max_iter,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let _ = g
                .pagerank()
                .reset_prob(reset_prob)
                .tol(tol)
                .max_iter(max_iter)
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output, false)
                .await?;
        }
        Command::Wcc { common, seed } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let _ = g
                .connected_components()
                .set_seed(seed)
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output, false)
                .await?;
        }
        Command::Mis { common } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let _ = g
                .maximal_independent_set()
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output)
                .await?;
        }
        Command::Kcore { common, max_iter } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let _ = g
                .k_core()
                .max_iter(max_iter)
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output, false)
                .await?;
        }
        Command::Hyperanf {
            common,
            n_hops,
            lg_k,
            directed,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let _ = g
                .hyperanf()
                .n_hops(n_hops)
                .lg_k(lg_k)
                .directed(directed)
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output, false)
                .await?;
        }
        Command::ShortestPath {
            common,
            landmarks,
            max_iterations,
            to_landmarks,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let mut b = g.shortest_paths(landmarks);
            if to_landmarks {
                b = b.to_landmarks();
            }
            if let Some(mi) = max_iterations {
                b = b.max_iterations(mi);
            }
            let _ = b
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output, false)
                .await?;
        }
        Command::Pic {
            common,
            k,
            max_iter,
            tol,
            init,
            weights,
            embedding,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let mut b = g
                .pic()
                .set_multiple_k(k)
                .set_max_iterations(max_iter)
                .set_tol(tol)
                .set_checkpoint_dir(ckpt);
            if common.weighted {
                b = b.set_edge_weight_col(&common.weight_col_name);
            }
            b = match init {
                PicInit::Degree => b.set_init_strategy(InitStrategy::DegreeBased),
                PicInit::Random => b.set_init_strategy(InitStrategy::Random),
            };
            b = match weights {
                PicWeights::None => b.set_weights_strategy(WeightsStrategy::None),
                PicWeights::Ppmi => b.set_weights_strategy(WeightsStrategy::PPMI),
            };
            b = match embedding {
                PicEmbedding::Last => b.set_embedding_mode(EmbeddingMode::LastIterate),
                PicEmbedding::Trajectory => b.set_embedding_mode(EmbeddingMode::FullTrajectory),
            };
            let res = b.run(&ctx, &common.output).await?;
            log::info!(
                "PIC embedding dim = {}, KMeans Lloyd iterations = {}",
                res.d,
                res.num_iterations
            );
        }
        Command::Mllib { cmd } => match cmd {
            MllibCommand::Kmeans {
                features,
                output,
                format,
                id_col_name,
                feature_col,
                k,
                metric,
                max_iter,
                tol,
                init_steps,
                seed,
                max_memory,
                num_workers,
                checkpoint_dir,
                max_temp_file,
            } => {
                let work = ensure_dir(&checkpoint_dir)?;
                let ctx = build_context(&work, &max_memory, num_workers, &max_temp_file)?;

                let raw = read_data(&ctx, &features, format).await?;
                let features =
                    raw.select(vec![col(&id_col_name).alias(VERTEX_ID), col(&feature_col)])?;

                let metric = match metric {
                    KmeansArgsMetric::L2 => DistanceMetric::L2,
                    KmeansArgsMetric::Cosine => DistanceMetric::Cosine,
                };

                // Deduplicate K values preserving order (KMeansBuilder does the
                // same), so runs zip 1:1 with the requested list.
                let mut ks: Vec<usize> = Vec::new();
                for &kk in &k {
                    if !ks.contains(&kk) {
                        ks.push(kk);
                    }
                }

                let res = KMeansBuilder::new(&features, &feature_col)
                    .k_values(&ks)
                    .metric(metric)
                    .max_iter(max_iter)
                    .tol(tol)
                    .init_steps(init_steps)
                    .seed(seed)
                    .run()
                    .await?;
                log::info!(
                    "k-means finished after {} Lloyd iterations, d = {}",
                    res.num_iterations,
                    res.d
                );

                // One cluster column per requested K (named by the requested
                // K; computed with the effective centers, see KMeansRun.k).
                let mut columns = vec![col(VERTEX_ID)];
                for (kk, run) in ks.iter().zip(&res.runs) {
                    log::info!(
                        "k = {} (effective {}), total metric = {}",
                        kk,
                        run.k,
                        run.total_metric
                    );
                    columns.push(
                        kmeans_assign_expr(
                            col(&feature_col),
                            run.k,
                            res.d,
                            run.centers.clone(),
                            metric,
                        )
                        .alias(format!("cluster_{kk}")),
                    );
                }
                features
                    .select(columns)?
                    .write_parquet(&output, DataFrameWriteOptions::new(), None)
                    .await?;
                log::info!("result was written into {output}");
            }
        },
        Command::Fastrp {
            common,
            dim,
            iterations,
            seed,
            normalization,
            norm_output,
            iteration_weights,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let normalization = match normalization {
                FastrpNormalization::None => FastRPNormalization::None,
                FastrpNormalization::L1 => FastRPNormalization::L1,
                FastrpNormalization::L2 => FastRPNormalization::L2,
            };
            let mut builder = g
                .fastrp()
                .dim(dim)
                .iterations(iterations)
                .seed(seed)
                .normalization(normalization)
                .norm_output(norm_output);
            if let Some(w) = iteration_weights {
                builder = builder.iteration_weights(w);
            }
            let iterations = builder
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output)
                .await?;
            log::info!("FastRP finished after {iterations} iterations");
        }
        Command::FastrpClustering {
            common,
            dim,
            iterations,
            seed,
            normalization,
            iteration_weights,
            k,
            metric,
            max_iter,
            tol,
            init_steps,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let normalization = match normalization {
                FastrpNormalization::None => FastRPNormalization::None,
                FastrpNormalization::L1 => FastRPNormalization::L1,
                FastrpNormalization::L2 => FastRPNormalization::L2,
            };
            let metric = match metric {
                KmeansArgsMetric::L2 => DistanceMetric::L2,
                KmeansArgsMetric::Cosine => DistanceMetric::Cosine,
            };
            let mut builder = g
                .fastrp_clustering()
                .set_dim(dim)
                .set_iterations(iterations)
                .set_seed(seed)
                .set_normalization(normalization);
            if let Some(w) = iteration_weights {
                builder = builder.set_iteration_weights(w);
            }
            let res = builder
                .set_multiple_k(k)
                .set_metric(metric)
                .set_max_iter(max_iter)
                .set_tol(tol)
                .set_kmeans_init_steps(init_steps)
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output)
                .await?;
            log::info!(
                "FastRP clustering: d = {}, KMeans iterations = {}",
                res.d,
                res.num_iterations
            );
        }
        Command::ClassicalLp {
            common,
            max_iter,
            undirected,
        } => {
            let (ctx, g, ckpt) = setup(&common).await?;
            let _ = g
                .classical_lp()
                .directed(undirected)
                .max_iter(max_iter)
                .set_checkpoint_dir(ckpt)
                .run(&ctx, &common.output, false)
                .await?;
        }
    }

    Ok(())
}
