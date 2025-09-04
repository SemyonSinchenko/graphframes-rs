use criterion::{Criterion, criterion_group, criterion_main};
use graphframes_rs::util::create_ldbc_test_graph;
use std::env;
use tokio::runtime::Runtime;

fn benchmark_pagerank(c: &mut Criterion) {
    let dataset_name =
        env::var("BENCHMARK_DATASET").expect("BENCHMARK_DATASET environment variable not set");

    let checkpoint_interval: usize = env::var("CHECKPOINT_INTERVAL")
        .expect("BENCHMARK_DATASET environment variable not set")
        .parse()
        .expect("CHECKPOINT_INTERVAL is not a valid int");

    let mut group = c.benchmark_group("PageRank");

    // Create a Tokio runtime to execute the async graph loading function.
    let rt = Runtime::new().unwrap();

    // Load the graph data once before running the benchmark.
    let graph = rt
        .block_on(create_ldbc_test_graph(&dataset_name, true, false))
        .expect("Failed to create test graph");

    // Creating pagerank_builder here so to exclude the time of generation in each iteration
    let pagerank_builder = graph
        .pagerank()
        .max_iter(10)
        .checkpoint_interval(checkpoint_interval)
        .reset_prob(0.15);

    // Define the benchmark.
    // Criterion runs the code inside the closure many times to get a reliable measurement.
    group.bench_function(
        String::from(
            "pagerank-".to_owned() + &dataset_name + "-cp-" + &checkpoint_interval.to_string(),
        ),
        |b| {
            // Use the `to_async` adapter to benchmark an async function.
            b.to_async(&rt).iter(|| async {
                let _ = pagerank_builder
                    .clone()
                    .run()
                    .await
                    .unwrap()
                    .collect()
                    .await;
            })
        },
    );

    group.finish();
}

criterion_group!(benches, benchmark_pagerank);
criterion_main!(benches);
