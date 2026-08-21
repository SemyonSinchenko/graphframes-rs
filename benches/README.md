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

## Reference benchmark: Spark GraphFrames (local mode)

The third leg of the comparison is the reference *distributed* engine the
project is named after: Apache Spark + GraphFrames (the `graphframes-rs`
name is a play on this stack). `main_graphframe.py` runs the same LDBC flow
with one local SparkSession per algorithm:

    spark.read.parquet -> GraphFrame(vertices, edges) -> algorithm
    -> result.parquet

Files:

    benches/python/main_graphframe.py      # runner: same CLI, monitor, stats and
                                           # report layout as main.py/main_lbdb.py
    benches/python/lbdb-requirements.txt   # adds pyspark==4.1.2 + graphframes-py==0.12.1

Stack (pinned): pyspark/spark **4.1.2** (the version GraphFrames 0.12.1
targets — its POM depends on spark-sql_2.13:4.1.2), graphframes-py **0.12.1**
and the JVM jar `io.graphframes:graphframes-spark4_2.13:0.12.1`, resolved by
Spark via `--packages` (`spark.jars.packages`) at session-creation time
(first run downloads it from Maven Central; later runs use the Ivy cache).
The session is created **before** any measurement — the runner then wraps
each algorithm run in the shared monitor, so session/JVM startup and jar
download are never part of the reported wall time.

GraphFrames is slow, so the defaults are the opposite of the other runners:
**no warmup, 3 runs** (`--warmup 1` restores the discarded calibration run).

Usage (the runner bootstraps `benches/python/.lbdb-venv` from
`lbdb-requirements.txt` on demand, or pass an interpreter that has them):

    python3 benches/python/main_graphframe.py                       # wiki-Talk (2XS) smoke test
    python3 benches/python/main_graphframe.py --dataset cit-Patents --runs 5
    python3 benches/python/main_graphframe.py --dataset wiki-Talk \
        --algorithms pagerank,wcc,cdlp,sssp --max-memory 8G --num-workers 8 \
        --storage-level disk --gf-version 0.12.1

Reports land next to the other two runners' results:

    benches/results/gf/<algorithm>/<size-class>/<dataset>/
        spark_driver_<mem>_shuffle_<n>_storage_<memory|disk>/
            benchmark.json + wall_time/rss/disk .gnuplot/.png

### Knobs

| flag | Spark setting | notes |
|---|---|---|
| `--max-memory` | `spark.driver.memory` | local mode: one JVM holds driver + executors |
| `--num-workers` | `spark.sql.shuffle.partitions` | master is `local[*]`, so all cores are used; give it at least 4 |
| `--storage-level` | `StorageLevel` passed to the algorithms | `memory` -> `MEMORY_AND_DISK` (default), `disk` -> `DISK_ONLY` (wcc/cdlp/sssp; PageRank has no storage-level knob) |
| `--gf-version` | `spark.jars.packages` | selects the jar `io.graphframes:graphframes-spark4_2.13:<version>` (default 0.12.1) |

Per run the workdir tree is watched by the shared monitor (RSS is summed
over the process tree, so the driver JVM is included in `peak_rss_kb`; disk
is the workdir tree, so Spark's shuffle spill under `spark_local` and the
`spark_checkpoints` folder show up in the disk curve):

    <checkpoint-dir>/gf_<algo>_<dataset>/
        spark_local/                  spark.local.dir (shuffle/block spill)
        run_<i>/spark_checkpoints/    sc.setCheckpointDir() target
        run_<i>/output/result.parquet algorithm result

### Algorithm mapping

The session and algorithm configuration follow the official GraphFrames
benchmark (https://graphframes.io/01-about/03-benchmarks.html):
KryoSerializer, local (in-executor) checkpoints, raw directed edges.

| algorithm | GraphFrames call | notes |
|---|---|---|
| pagerank | `pageRank(resetProbability=0.15, maxIter=10)` | the GraphX implementation (the only one exposed); mirrors `page-rank --max-iter 10` (damping 0.85) |
| wcc | `connectedComponents(algorithm="randomized_contraction")` | fastest of the four CC implementations; symmetrizes internally and writes intermediate tables to the per-run `spark_checkpoints` folder |
| cdlp | `labelPropagation(maxIter=10, algorithm="graphframes")` | GraphFrames pregel implementation (faster than GraphX here); run on the raw directed edges like the official benchmark — **not** symmetrized, because the pregel aggregates messages by receiver and the LDBC bidirectional semantic would be O(degree x labels) per receiver (~25x slower on degree-skewed graphs like wiki-Talk, max out-degree 100k vs max in-degree 3k) |
| sssp | `shortestPaths(landmarks=[0.25 * vertices], algorithm="graphframes")` | landmark = 25th percentile vertex id (main.py convention); GraphFrames implementation, faster than GraphX on large graphs |

The JSON adds `engine: spark-graphframes`, the Spark/python/java versions and
jar coordinates under `environment`, per-run `worker` phase timings
(`read_parquet_s`, `graph_build_s`, `algorithm_s`, `write_parquet_s`) plus
the effective params (`shuffle_partitions`, `driver_memory`, `storage_level`,
`impl`, `symmetrized_input`, `use_local_checkpoints`).

On wiki-Talk (2XS, 2M vertices / 5M edges, local[*] on a 12-core box, 4G
driver, Kryo, local checkpoints) the smoke run gives roughly: pagerank
~30 s, wcc ~30 s, cdlp ~66 s, sssp ~44 s wall per cold run at ~5 GiB peak
RSS — matching the official GraphFrames benchmark numbers (CDLP ~78 s on a
4-core GitHub Actions runner).

## Reference benchmark: in-memory CSR (icebug)

To quantify what the out-of-core mode costs relative to a plain in-memory
CSR engine, `main_lbdb.py` re-runs the same LDBC flow with the reference
stack recommended by the [icebug](https://github.com/Ladybug-Memory/icebug)
developers (icebug = a NetworKit fork over zero-copy Arrow CSR arrays):

    pyarrow read -> IcebugMemGraph.from_arrow_tables(nodes, rels)
    -> nk.graph.Graph (fromCSR / fromIcebugMemGraph, zero-copy) -> algorithm
    -> parquet result

Files:

    benches/python/main_lbdb.py        # runner: same CLI, monitor, stats and
                                       # report layout as main.py
    benches/python/lbdb_algorithms.py  # worker: one cold process per run,
                                       #   read -> CSR -> graph -> algorithm
                                       #   -> result.parquet, plus a phase
                                       #   timing report; applies the DuckDB
                                       #   memory cap / spill dir / threads
    benches/python/lbdb-requirements.txt  # icebug==13.0, icebug-format[convert-duckdb], pyarrow

Usage (the runner creates `benches/python/.lbdb-venv` on demand via `uv`,
or pass `--python <interpreter>` with the requirements pre-installed):

    python3 benches/python/main_lbdb.py                       # wiki-Talk (2XS) smoke test
    python3 benches/python/main_lbdb.py --dataset cit-Patents --runs 5
    python3 benches/python/main_lbdb.py --dataset graph500-24 --threads 4 --max-memory 8G

Reports land next to the graphframes results, following the same
`<algorithm>/<size-class>/<dataset>/<knob>/` structure:

    benches/results/ldbd/<algorithm>/<size-class>/<dataset>/icebug_threads_<n>/
    benches/results/ldbd/<algorithm>/<size-class>/<dataset>/icebug_mem_<mem>_threads_<n>/

The leaf is the reference counterpart of `max_mem_<mem>_workers_<n>`: it
records the knobs of the measured engine — the OpenMP thread count of the
NetworKit core, plus the DuckDB memory cap when `--max-memory` is given.
The JSON adds an `engine: icebug` field, the reference package versions
under `environment.ref_*`, and per-run `worker` phase timings (`import_s`,
`read_parquet_s`, `csr_and_graph_s`, `algorithm_s`, `write_parquet_s`) plus
the effective `duckdb` config (`memory_limit`, `temp_directory`, `threads`),
so interpreter/CSR overhead can be separated from pure algorithm time when
comparing with graphframes.

### Reference memory control (`--max-memory`)

The CSR conversion inside `IcebugMemGraph.from_arrow_tables` runs on
DuckDB, whose default `memory_limit` is 80% of RAM. On M-class graphs that
default does not force spilling early enough and the process dies at the
OS level: on graph500-24 (8.87M vertices / 260.4M edges) the *directed*
pagerank build (two `from_arrow_tables` passes over m rows) peaked at
**23.2 GiB RSS** with a DuckDB working set of ~15–19 GiB through the CSR
builds (see `ldbd/pagerank/M/graph500-24/icebug_threads_4/`), while the
*symmetrized* wcc/cdlp builds (one pass over ~2m rows — a 7.7 GiB relation
table, 3.9 GiB index output) roughly double that working set and get
OOM-killed before NetworKit ever runs.

`--max-memory` caps DuckDB via its `memory_limit` (injected into the
connections the library opens, see `lbdb_algorithms.py`), which makes the
conversion a bounded-memory pipeline: overflowing sorts spill and the
in-memory relation-table blocks are evicted into `temp_directory` — the
runner points that at `<workdir>/run_<i>/duckdb_tmp`, so the spill shows up
in the du-mode disk curve. `--threads` is applied to NetworKit *and*
DuckDB, so one knob describes the whole run. Sizing from the graph500-24
numbers (unavoidable pyarrow-resident memory is ~4 GiB of input + ~4 GiB
of CSR output + interpreter on top of the cap):

| cap | expected effect on M-class |
|---|---|
| none (DuckDB default ≈ 80% RAM) | pagerank peaks ~23 GiB; wcc/cdlp OOM |
| `--max-memory 12G` | relation table fully resident + sort headroom, moderate spill — fastest survivable setting |
| `--max-memory 8G` | relation table ~resident, sorts spill; peak RSS ≈ cap + ~9 GiB — recommended default for a shared ~38 GiB box |
| `--max-memory 4G` | matches graphframes' default spill pool; the 7.7 GiB relation table churns and re-reads constantly — slow but very safe |

Verified on wiki-Talk with an intentionally tiny cap (`--max-memory 128M`):
the run succeeds, spills ~150 MB into the workdir (visible in the disk
curve), RSS drops from 1.3 to 0.9 GiB, the CSR phase slows 3.4s → 5.7s,
and the resulting PageRank is bit-identical to the uncapped run.

Algorithm mapping (parameters mirror the table above):

| algorithm | graphframes CLI | reference (NetworKit) | notes |
|---|---|---|---|
| pagerank | `page-rank --max-iter 10` | `centrality.PageRank`, damp 0.85, `maxIterations=10`, `tol=0` | fixed 10-iteration budget on both sides; the directed zero-copy graph gets its in-edge CSR from a second transposed `from_arrow_tables` call (icebug's CSR graph only stores out-adjacency) |
| wcc | `wcc --seed 42` | `components.ParallelConnectedComponents` (parallel label propagation) | both symmetrize the input internally with duplicates kept |
| cdlp | `classical-lp --max-iter 10` | `community.PLP`, `maxIterations=10` | both on the symmetrized undirected graph; PLP breaks label ties randomly while LDBC CDLP picks the smallest label, and PLP may stop before the cap once labels stabilize — partitions can therefore differ slightly from the LDBC reference while the workload stays comparable |

Semantic caveats worth remembering when reading the numbers:

* measured wall time is the whole cold process (interpreter start + imports
  + parquet read + DuckDB CSR build + zero-copy graph + algorithm + write),
  exactly like main.py times a cold `graphframes` process;
* `IcebugMemGraph.from_arrow_tables` materializes the CSR with DuckDB SQL —
  on wiki-Talk that conversion is ~2/3 of the total runtime, which is the
  honest cost of the recommended "parquet in, CSR out" flow;
* for WCC/CDLP the reference keeps self-loops once (icebug) while graphframes
  drops them during symmetrization; PageRank on both sides uses the raw
  directed edges.

On wiki-Talk (2XS, 2.4M vertices / 5.0M edges, 12 threads, icebug 13.0)
the smoke run gives: pagerank ≈ 4.8 s, wcc ≈ 3.2 s, cdlp ≈ 4.6 s wall per
cold run at ~1.3–1.7 GiB peak RSS (see
`results/ldbd/*/2XS/wiki-Talk/icebug_threads_12/`). On graph500-24 (M,
--threads 4) pagerank needs ~188 s per run, 84% of it in the CSR
conversion itself — the honest price of the recommended parquet→CSR flow
(see `results/ldbd/pagerank/M/graph500-24/icebug_threads_4/`).

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
