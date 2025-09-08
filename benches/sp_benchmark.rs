use criterion::{Criterion, criterion_group, criterion_main};
use graphframes_rs::util::create_ldbc_test_graph;
use std::env;
use tokio::runtime::Runtime;

fn benchmark_sp(c: &mut Criterion) {
    let dataset_name =
        env::var("BENCHMARK_DATASET").expect("BENCHMARK_DATASET environment variable not set");
    let checkpoint_interval: usize = env::var("CHECKPOINT_INTERVAL")
        .expect("BENCHMARK_DATASET environment variable not set")
        .parse()
        .expect("CHECKPOINT_INTERVAL is not a valid int");

    let is_weighted = match env::var("WEIGHTED").expect("WEIGHTED environment variable not set") {
        s if s == "true" => true,
        _ => false,
    };
    let mut group = c.benchmark_group("ShortestPath");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(200));

    let rt = Runtime::new().unwrap();
    let graph = rt
        .block_on(create_ldbc_test_graph(&dataset_name, true, is_weighted))
        .expect("Failed to create test graph");

    let sp_builder = graph
        .shortest_paths(vec![2i64]) // TODO: replace to read from props
        .checkpoint_interval(checkpoint_interval);

    group.bench_function(
        String::from("sp-".to_owned() + &dataset_name + "-cp-" + &checkpoint_interval.to_string()),
        |b| {
            // Use the `to_async` adapter to benchmark an async function.
            b.to_async(&rt).iter(|| async {
                let _ = sp_builder.clone().run().await.unwrap().collect().await;
            })
        },
    );

    group.finish();
}

criterion_group!(benches, benchmark_sp);
criterion_main!(benches);
