#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;
use graphframes_rs::{EDGE_DST, EDGE_SRC, GraphFrame};
use std::env;
use std::sync::Arc;

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

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args: Vec<String> = env::args().collect();

    let vertices = &args[1];
    let edges = &args[2];
    let algorithm = &args[3];
    let params = &args[4];
    let out = &args[5];
    let max_memory = &args[6];
    let num_partition: usize = args[7]
        .parse()
        .map_err(|e| DataFusionError::Execution(format!("invalid number of partitions: {e}")))?;

    let max_pool_mem = parse_mem(max_memory)?;
    let env = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(max_pool_mem)))
        .with_temp_file_path(std::env::current_dir()?.join("gf_df_tmp"))
        .build_arc()?;
    let session_state = SessionStateBuilder::new()
        .with_config(SessionConfig::new().with_target_partitions(num_partition))
        .with_runtime_env(env)
        .with_default_features()
        .build();
    let ctx = SessionContext::from(session_state);
    let vertices = ctx
        .read_parquet(vertices, ParquetReadOptions::new())
        .await?;

    let edges = ctx
        .read_parquet(edges, ParquetReadOptions::new())
        .await?
        .select(vec![
            col("source").alias(EDGE_SRC),
            col("target").alias(EDGE_DST),
        ])?;

    let graph = GraphFrame::try_new(vertices, edges)?;

    if algorithm == "pagerank" {
        // PageRank via Pregel. Algorithm-specifi parameter
        // is convergence tolerance.
        let tol = params.parse::<f64>().unwrap();
        let pr = graph
            .pagerank()
            .reset_prob(0.15)
            .max_iter(0)
            .tol(tol)
            .set_checkpoint_dir(Path::from(
                std::env::current_dir()?
                    .join("gf_checkpoints")
                    .to_string_lossy()
                    .as_ref(),
            ))
            .run(&ctx, out, false)
            .await?;

        println!("num-iterations: {}", pr);
    } else if algorithm == "wcc" {
        // Weakly connected components via randomized contraction. The only
        // algorithm-specific parameter is the random seed (used to seed the
        // per-iteration affine hashes).
        let seed: u64 = params
            .parse()
            .map_err(|e| DataFusionError::Execution(format!("invalid random seed: {e}")))?;
        let cc = graph
            .connected_components()
            .set_checkpoint_dir(Path::from(
                std::env::current_dir()?
                    .join("gf_checkpoints")
                    .to_string_lossy()
                    .as_ref(),
            ))
            .set_seed(seed)
            .run(&ctx, out, false)
            .await?;

        println!("num-iterations: {}", cc);
    };

    Ok(())
}
