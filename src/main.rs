use datafusion::error::Result;
use datafusion::{
    dataframe::DataFrameWriteOptions,
    prelude::{ParquetReadOptions, SessionContext},
};
use graphframes_rs::GraphFrame;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    let vertices = &args[1];
    let edges = &args[2];
    let algorithm = &args[3];
    let params = &args[4];
    let out = &args[5];

    let ctx = SessionContext::new();
    let vertices = ctx
        .read_parquet(vertices, ParquetReadOptions::new())
        .await?;
    let edges = ctx.read_parquet(edges, ParquetReadOptions::new()).await?;

    let graph = GraphFrame::try_new(vertices, edges)?;

    if algorithm == "pagerank" {
        let tol = params.parse::<f64>().unwrap();
        let pr = graph
            .pagerank()
            .reset_prob(0.15)
            .max_iter(0) // zero iters mean until convergence
            .tol(tol)
            .run(&ctx, "", false)
            .await?;
        //pr.write_csv(out, DataFrameWriteOptions::new(), Option::None)
        //    .await?;
    };

    Ok(())
}
