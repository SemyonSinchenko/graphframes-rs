use criterion::{Criterion, criterion_group, criterion_main};
use graphframes_rs::util::create_ldbc_test_graph;
use std::env;
use tokio::runtime::Runtime;

fn benchmark_cc(c: &mut Criterion) {
    let dataset_name =
        env::var("BENCHMARK_DATASET").expect("BENCHMARK_DATASET environment variable not set");
    let is_weighted = match env::var("WEIGHTED").expect("WEIGHTED environment variable not set") {
        s if s == "true" => true,
        _ => false,
    };

    let mut group = c.benchmark_group("Connected Components");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(200));

    // Create a Tokio runtime to execute the async graph loading function.
    let rt = Runtime::new().unwrap();

    // Load the graph data once before running the benchmark.
    let graph = rt
        .block_on(create_ldbc_test_graph(&dataset_name, true, is_weighted))
        .expect("Failed to create test graph");

    // Creating cc_builder here so to exclude the time of generation in each iteration
    let cc_builder = graph.connected_components();

    // Define the benchmark.
    // Criterion runs the code inside the closure many times to get a reliable measurement.
    group.bench_function(String::from("cc-".to_owned() + &dataset_name), |b| {
        // Use the `to_async` adapter to benchmark an async function.
        b.to_async(&rt).iter(|| async {
            let _ = cc_builder.clone().run().await.unwrap().data.collect().await;
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_cc);
criterion_main!(benches);
