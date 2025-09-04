# Running Benchmarks for Graphframe-rs

Benchmarking for Graphframe-rs are currently done on LDBC Graphalytics [datasets](https://ldbcouncil.org/benchmarks/graphalytics/datasets/).
Benchmarking runs and reports are executed/generated as html-reports using Rust Criterion crate.

## How to run benchmarks ?

`run_benchmarks.py` file is the main source for running the benchmarks.

### Dependencies

Running benchmarks using python requires:

```text
Python Package:
- requests # for downloading dataset
```

```text
CLI utility:
- zstd # For decompressing the downloaded dataset
- tar # For unzipping the decompressed dataset
```

### Parameters for `run_benchmarks.py`

- `--dataset`: [MANDATORY] LDBC dataset name on which user want to run the benchmark (for e.g. test-pr-directed, cit-Patents). Dataset name are exactly same as mentioned in LDBC website.
- `--checkpoint_interval`: If user wants to define a specific number of checkpoints for Algorithms to run on. `default: 1`
- `--name`: If a particular benchmark needs to run. Name should be same as the `[[bench]]` names present in `Cargo.toml`

```bash
# Running all the benchmarks
python3 run_benchmarks.py --dataset cit-Patents --checkpoint_interval 2

# Running an individual benchmark
python3 run_benchmarks.py --dataset cit-Patents --checkpoint_interval 2 --name pagerank_benchmark
```

## Benchmarking Reports

Criterion benchmarking html-reports can be seen by opening `target/criterion/report/index.html` in any browser after benchmarking completes.
