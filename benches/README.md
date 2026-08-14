# Benchmarks for graphframes-rs

Benchmarking is done by running the release `graphframes` CLI against LDBC
Graphalytics datasets and monitoring wall time, peak RSS and disk usage.
Benchmarks are pure performance analysis: correctness is covered by the
in-crate LDBC test suites (`src/**/tests`).

## Layout

    benches/
    ├── python/            # the benchmark runner (pure stdlib Python)
    │   ├── main.py        #   entry point: CLI args, orchestration, JSON output
    │   ├── datasets.py    #   LDBC dataset catalog + parquet download
    │   ├── monitor.py     #   subprocess + RSS/disk sampling thread
    │   ├── stats.py       #   percentiles, t-CIs, DTW shape alignment
    │   └── plotting.py    #   gnuplot script generation + rendering
    ├── data/ldbc/<ds>/    # downloaded parquet datasets (gitignored)
    └── results/           # benchmark reports (committed to git):
        └── <algorithm>/<size-class>/<dataset>/max_mem_<mem>_workers_<n>/
            ├── benchmark.json
            ├── wall_time.{dat,gnuplot,png}
            ├── rss.{dat,gnuplot,png}
            └── disk.{dat,gnuplot,png}

## Usage

    # list the available datasets
    python3 benches/python/main.py --list-datasets

    # run all seven algorithms on the default dataset (cit-Patents)
    python3 benches/python/main.py

    # a single algorithm on a specific dataset with custom tuning knobs
    python3 benches/python/main.py --dataset wiki-Talk --algorithms pagerank \
        --num-workers 8 --max-memory 16G

    # undirected datasets: symmetrize input for algorithms defined on
    # undirected graphs (shortest-path, hyperanf)
    python3 benches/python/main.py --dataset graph500-22 --undirected

    # weighted datasets: keep the edge weight column while loading
    python3 benches/python/main.py --dataset kgs --weighted

    # use a prebuilt binary instead of `cargo build --release`
    python3 benches/python/main.py --binary ./target/release/graphframes

## What each run produces

For every (algorithm, dataset) the runner:

1. performs **one warmup run** (discarded — warms the OS page cache and
   calibrates the monitor sampling interval to ~300 samples per run),
2. performs `--runs` (default 5) measured runs, each a fresh process: every
   run is a "first run", no cross-run state,
3. computes wall-time statistics (median/mean/std/min/max/p90/p95), peak RSS
   (`VmHWM` from `/proc`, Linux) and peak disk consumption,
4. aligns the RSS/disk series across runs on a fraction-of-run axis — by
   default via **DTW shape warping** (dynamic time warping of the z-scored,
   smoothed RSS curve of each run against the median-duration reference run,
   Sakoe-Chiba band 0.25; pure python, ~10 ms per 300-sample run), falling
   back to plain duration normalization when a run has too few samples; the
   method actually used is recorded in the JSON (`--align duration` forces
   the simple method),
5. writes `benchmark.json` (params, per-run results, stats, raw series,
   environment) plus three gnuplot scripts/plots: wall-time **kernel density
   estimate** (PDF: probability density on Y vs wall time on X, Gaussian
   kernel with Silverman bandwidth computed in pure python, vertical
   median/p90/p95 lines, individual runs as a baseline rug), RSS line with
   95% CI band, disk line with 95% CI band.

## Fixed hyperparameters

The runner hardcodes LDBC-inspired parameters (no properties/manifest files):

| algorithm | CLI subcommand | parameters |
|---|---|---|
| pagerank | `page-rank`   | `--max-iter 10` (damping 0.85) |
| wcc      | `wcc`         | `--seed 42` |
| kcore    | `kcore`       | `--max-iter 10` |
| hyperanf | `hyperanf`    | `--n-hops 5` |
| sp       | `shortest-path` | landmark = 25th percentile vertex id (0.25 * vertices) |
| cdlp     | `classical-lp` | `--max-iter 10` (LDBC bidirectional-edge semantic) |
| mis      | `mis`         | — |

`--undirected` is honoured per algorithm: PageRank is defined only on directed
graphs (ignored), WCC/KCore/MIS and ClassicalLP symmetrize internally
(ignored), and shortest-path + hyperanf get a symmetrized input.

## Monitoring notes

* RSS sampling reads `/proc/<pid>/status` (`VmRSS` per poll, `VmHWM` for the
  peak) — Linux only by design; the JSON simply records no RSS series on other
  platforms.
* Disk sampling defaults to `--disk-mode du`: the size of the workdir tree
  itself (checkpoints, spills, output), measured with a pure-python `os.walk`
  — exact and cheap (the workdir only holds a handful of parquet files), so it
  samples at the same rate as everything else. This is what makes the disk
  curve reflect the algorithm (e.g. PageRank's `edges` checkpoint appears as
  an immediate step up that lives for the whole run and drops on purge at the
  end). `--disk-mode statvfs` remains as an O(1) alternative (filesystem-wide
  used-space delta) but it is blind to the workdir on filesystems with lazy
  free-space accounting (e.g. btrfs) and can read ~0 while the tree holds
  hundreds of MB. Deltas are floored at 0 (disk consumption is never
  negative), and the RSS/disk plots cap the mean and CI band at 0 as well.
* Point `--checkpoint-dir` at fast local storage (NVMe/SSD); with
  `--disk-mode du` (the default) the measurement is the workdir tree itself,
  so a dedicated mount is no longer required for clean disk numbers.

## Datasets

The catalog in `datasets.py` mirrors the official table at
https://ldbcouncil.org/benchmarks/graphalytics/datasets/ (name, scale class,
node/edge counts, package size). Datasets are downloaded on demand as parquet
(`.../graphalytics-parquet/<ds>-{v,e}.parquet`) into `benches/data/ldbc/<ds>/`
and reused across runs. The default dataset is `cit-Patents`.
