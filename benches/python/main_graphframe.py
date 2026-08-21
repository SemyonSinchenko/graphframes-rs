#!/usr/bin/env python3
"""Spark GraphFrames benchmark runner (local mode).

The third leg of the LDBC benchmark: graphframes-rs (``main.py``) and the
in-memory CSR reference stack (``main_lbdb.py``) both run the *same* LDBC
Graphalytics flow on parquet data; this runner does the same with
Apache Spark + GraphFrames, all inside one local JVM:

    spark.read.parquet -> GraphFrame(vertices, edges) -> algorithm
    -> parquet result

For every (algorithm, dataset) the runner creates **one** SparkSession
(``spark.jars.packages`` resolves the GraphFrames jar at this point) and
then performs `--runs` measured algorithm runs.  Session creation is *not*
measured: GraphFrames is slow, so by default there is no warmup and only 3
runs; pass ``--warmup 1`` to get the page-cache-warming discarded run used by
the other two runners.

Algorithms and parameters mirror ``benches/python/main.py``:

    pagerank  g.pageRank(resetProbability=0.15, maxIter=10) -- the GraphX
              implementation (the only one GraphFrames exposes), mirroring
              page-rank --max-iter 10 (damping 0.85)
    wcc       g.connectedComponents(algorithm="randomized_contraction") --
              the fastest of the four implementations (graphx / two_phase /
              randomized_contraction); it symmetrizes the edge set internally
              (union + distinct) and checkpoints intermediate tables into the
              per-run spark_checkpoints folder
    cdlp      g.labelPropagation(maxIter=10, algorithm="graphframes") -- the
              GraphFrames (pregel) implementation is faster than GraphX on
              this workload; run on the raw directed edges like the official
              benchmark (NOT symmetrized: the pregel aggregates messages by
              receiver, so the LDBC bidirectional semantic would be
              O(degree x labels) per receiver and ~25x slower on
              degree-skewed graphs such as wiki-Talk)
    sssp      g.shortestPaths(landmarks=[0.25 * vertices],
              algorithm="graphframes", is_directed=True) -- the GraphFrames
              implementation is faster than GraphX on large graphs; landmark
              = 25th percentile vertex id, like main.py's shortest-path

Session configuration follows the official GraphFrames benchmark
(https://graphframes.io/01-about/03-benchmarks.html): KryoSerializer,
local (in-executor) checkpoints, master local[*] and
spark.sql.shuffle.partitions = --num-workers.

Knobs:

    --max-memory    goes entirely to spark.driver.memory (local mode: the
                    single JVM holds driver + executors)
    --num-workers   goes directly to spark.sql.shuffle.partitions (the
                    master is local[*], so the JVM uses all cores)
    --storage-level memory -> StorageLevel.MEMORY_AND_DISK (default),
                    disk -> StorageLevel.DISK_ONLY; passed to the algorithms
                    that accept a storage_level (wcc/cdlp/sssp; PageRank has
                    no storage-level knob)

Per run the workdir tree is watched by the shared ``monitor`` module (RSS is
summed over the process tree so the driver JVM is included; disk is the
workdir tree itself, so Spark's shuffle spill under ``spark_local`` and the
``spark_checkpoints`` folder show up in the disk curve):

    <checkpoint-dir>/gf_<algo>_<dataset>/
        spark_local/                  spark.local.dir (shuffle/block spill)
        run_<i>/spark_checkpoints/    sc.setCheckpointDir() target
        run_<i>/output/result.parquet algorithm result
        run_<i>/worker_report.json    per-run phase timings + params

Reports land next to the ladybug/icebug results, following the same layout:

    benches/results/gf/<algorithm>/<size-class>/<dataset>/
        spark_driver_<mem>_shuffle_<n>_storage_<memory|disk>/
            benchmark.json + wall_time/rss/disk .gnuplot/.png

The runner needs pyspark + graphframes-py; they are part of
``lbdb-requirements.txt`` and the runner bootstraps the shared
``benches/python/.lbdb-venv`` on demand (re-executing itself under it).

Usage:
    python3 benches/python/main_graphframe.py --list-datasets
    python3 benches/python/main_graphframe.py                        # wiki-Talk smoke test
    python3 benches/python/main_graphframe.py --dataset cit-Patents --runs 5
    python3 benches/python/main_graphframe.py --dataset wiki-Talk \
        --algorithms pagerank,wcc --max-memory 8G --num-workers 8 --storage-level disk
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import datasets  # noqa: E402
import monitor   # noqa: E402
import plotting  # noqa: E402
import stats     # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "benches" / "data" / "ldbc"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benches" / "results"
DEFAULT_WORKDIR = REPO_ROOT / "gf_workdir"
DEFAULT_VENV = Path(__file__).resolve().parent / ".lbdb-venv"
REQUIREMENTS = Path(__file__).resolve().parent / "lbdb-requirements.txt"
RESULT_SUBDIR = "gf"  # spark results live next to the ldbd reference results

# jar coordinates for --packages / spark.jars.packages
JAR_GROUP = "io.graphframes"
JAR_ARTIFACT = "graphframes-spark4_2.13"  # Spark 4 / Scala 2.13 artifact

# Fixed LDBC-inspired hyperparameters mirroring benches/python/main.py, plus
# the configuration of the official GraphFrames benchmark
# (https://graphframes.io/01-about/03-benchmarks.html): KryoSerializer for the
# session and local (in-executor) checkpoints for the graphframes algorithms.
# `symmetrize_input` marks algorithms for which the edge set is symmetrized
# before building the GraphFrame (LDBC bidirectional-edge semantic, matching
# the internal symmetrization of the Rust classical-lp and the icebug
# add_reverse_edges=True reference build).
ALGORITHMS = {
    "pagerank": {
        "impl": "graphx",                 # the only implementation GraphFrames exposes
        "max_iter": 10,                   # mirrors page-rank --max-iter 10
        "reset_probability": 0.15,        # damping 0.85
        "symmetrize_input": False,
        "storage_level_knob": False,      # pageRank() has no storage-level knob
        "use_local_checkpoints": False,   # GraphX impl: no DataFrame checkpoints
    },
    "wcc": {
        "impl": "randomized_contraction", # fastest of the four CC implementations
        "symmetrize_input": False,        # randomized_contraction symmetrizes internally
        "storage_level_knob": True,
        "use_local_checkpoints": True,    # local (in-executor) checkpoints, like the
                                          # official benchmark; the intermediate
                                          # ccreps parquet tables still go to the
                                          # per-run spark_checkpoints folder
    },
    "cdlp": {
        "impl": "graphframes",            # faster than the GraphX label propagation
        "max_iter": 10,
        # Raw directed edges, exactly like the official GraphFrames benchmark.
        # Symmetrizing the input (the LDBC "edges are bidirectional" semantic
        # used by main.py's classical-lp) would make the GraphFrames pregel
        # aggregate messages by *receiver* for the full (in+out) degree of
        # every vertex: on degree-skewed graphs (wiki-Talk max out-degree
        # ~100k vs max in-degree ~3k) that is O(degree x labels) per receiver
        # and runs ~25x slower, so it is not benchmarked here.
        "symmetrize_input": False,
        "storage_level_knob": True,
        "use_local_checkpoints": True,
    },
    "sssp": {
        "impl": "graphframes",            # faster than the GraphX shortest paths
        "landmark_frac": 0.25,            # landmark = 25th percentile vertex id
        "symmetrize_input": False,
        "storage_level_knob": True,
        "use_local_checkpoints": True,
    },
}

STORAGE_LEVELS = {
    "memory": "MEMORY_AND_DISK",
    "disk": "DISK_ONLY",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Spark GraphFrames on LDBC Graphalytics parquet datasets "
                    "(local mode, one SparkSession per algorithm, session creation excluded "
                    "from the measurement).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="wiki-Talk",
                   help="LDBC dataset name (see --list-datasets); wiki-Talk (2XS) for smoke tests")
    p.add_argument("--algorithms", default=",".join(ALGORITHMS),
                   help=f"comma-separated subset of: {', '.join(ALGORITHMS)}")
    p.add_argument("--runs", type=int, default=3,
                   help="number of measured runs per algorithm (GraphFrames is slow: no warmup, 3 runs)")
    p.add_argument("--warmup", type=int, default=0,
                   help="discarded warmup runs before the measured ones (0 = none; 1 = like main.py)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="spark.sql.shuffle.partitions (Spark's shuffle parallelism knob; "
                        "the master is local[*] so the driver JVM uses all cores; "
                        "give it at least 4)")
    p.add_argument("--max-memory", default="4G",
                   help="spark.driver.memory, e.g. 4G / 512M (local mode: driver + executors share one JVM)")
    p.add_argument("--storage-level", choices=["memory", "disk"], default="memory",
                   help="storage level for intermediate/final DataFrames: memory -> "
                        "MEMORY_AND_DISK, disk -> DISK_ONLY (wcc/cdlp/sssp)")
    p.add_argument("--gf-version", default="0.12.1",
                   help="GraphFrames version; selects the jar "
                        "io.graphframes:graphframes-spark4_2.13:<version> via --packages")
    p.add_argument("--checkpoint-dir", default=str(DEFAULT_WORKDIR),
                   help="base workdir for the per-run spark_checkpoints folder, spark.local.dir "
                        "and monitor logs")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help=f"root of the results tree (reports go into <root>/{RESULT_SUBDIR}/...)")
    p.add_argument("--results-subdir", default=RESULT_SUBDIR,
                   help="subdirectory for the spark results inside --results-dir")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="dataset download location")
    p.add_argument("--sample-interval", type=float, default=0.5,
                   help="monitor sampling interval in seconds (used when --warmup 0; "
                        "otherwise calibrated from the warmup run)")
    p.add_argument("--target-samples", type=int, default=300,
                   help="target number of monitor samples per measured run (warmup calibration only)")
    p.add_argument("--disk-mode", choices=["statvfs", "du"], default="du",
                   help="disk monitoring mode (see monitor.py; du = workdir tree size)")
    p.add_argument("--align", choices=["dtw", "duration"], default="dtw",
                   help="series alignment: DTW shape warping or plain duration normalization")
    p.add_argument("--list-datasets", action="store_true", help="print the dataset catalog and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Interpreter bootstrap (pyspark + graphframes-py live in the shared venv)
# ---------------------------------------------------------------------------

def ensure_runner_env() -> Path | None:
    """Return the interpreter to re-exec under, or None when the current one works.

    The runner itself needs pyspark + graphframes-py (the SparkSession and the
    monitored algorithm runs share this process).  If they are importable in
    the current interpreter, nothing to do.  Otherwise create the shared
    ``benches/python/.lbdb-venv`` from ``lbdb-requirements.txt`` (same venv as
    main_lbdb.py, same creation strategy: uv when available, venv+pip else)
    and return its python so ``main()`` can re-exec under it.
    """
    import importlib.util

    if importlib.util.find_spec("pyspark") is not None and importlib.util.find_spec("graphframes") is not None:
        return None

    venv_python = DEFAULT_VENV / "bin" / "python"
    needs_install = not venv_python.exists()
    if not needs_install:
        probe = subprocess.run([str(venv_python), "-c", "import pyspark, graphframes"],
                               capture_output=True, text=True)
        needs_install = probe.returncode != 0

    if needs_install:
        print(f"Creating benchmark venv at {DEFAULT_VENV} (from {REQUIREMENTS.name})")
        DEFAULT_VENV.parent.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if uv:
            r = subprocess.run([uv, "venv", "--python", "3", str(DEFAULT_VENV)])
            if r.returncode != 0:
                raise SystemExit("uv venv failed")
            r = subprocess.run([uv, "pip", "install", "-r", str(REQUIREMENTS),
                                "--python", str(venv_python)])
        else:
            r = subprocess.run([sys.executable, "-m", "venv", str(DEFAULT_VENV)])
            if r.returncode != 0:
                raise SystemExit("python -m venv failed")
            r = subprocess.run([str(venv_python), "-m", "pip", "install", "-U", "pip"])
            r = subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        if r.returncode != 0:
            raise SystemExit("failed to install benchmark dependencies; install them manually "
                             "into a venv and run this script with that interpreter")
    return venv_python


# ---------------------------------------------------------------------------
# Spark session + algorithm execution
# ---------------------------------------------------------------------------

def create_spark_session(args: argparse.Namespace, workdir: Path):
    """Create the SparkSession for one (algorithm, dataset).

    NOT measured: this is where spark-submit starts the driver JVM and
    ``spark.jars.packages`` resolves the GraphFrames jar (downloading it from
    Maven Central on first use).  All Spark state (local dir, shuffle
    partitions, driver heap) is fixed here.
    """
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    spark_local = workdir / "spark_local"
    spark_local.mkdir(parents=True, exist_ok=True)

    conf = SparkConf() \
        .setAppName("graphframes-bench") \
        .setMaster("local[*]") \
        .set("spark.driver.memory", args.max_memory) \
        .set("spark.sql.shuffle.partitions", str(args.num_workers)) \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .set("spark.local.dir", str(spark_local)) \
        .set("spark.jars.packages", f"{JAR_GROUP}:{JAR_ARTIFACT}:{args.gf_version}")
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_one_algorithm(algo: str, info: dict, args: argparse.Namespace,
                      run_workdir: Path, spark) -> dict:
    """Run one algorithm on the dataset inside `spark`, inside the measured window.

    Mirrors the reference worker flow: read parquet -> GraphFrame -> algorithm
    -> write result.parquet, with per-phase timings for the report.  Everything
    is lazy except the final parquet write, which is the terminal action.
    """
    import time

    from pyspark.sql import functions as F
    from graphframes import GraphFrame
    from pyspark.storagelevel import StorageLevel

    conf = ALGORITHMS[algo]
    data_dir = Path(args.data_dir)
    vpath = data_dir / args.dataset / f"{args.dataset}-v.parquet"
    epath = data_dir / args.dataset / f"{args.dataset}-e.parquet"
    storage_level = getattr(StorageLevel, STORAGE_LEVELS[args.storage_level])

    phases = {}
    wall0 = time.perf_counter()

    t0 = time.perf_counter()
    vertices = spark.read.parquet(str(vpath))          # id
    edges = spark.read.parquet(str(epath))             # source, target
    phases["read_parquet_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    edges = edges.select(F.col("source").alias("src"), F.col("target").alias("dst"))
    if conf["symmetrize_input"]:
        edges = edges.union(edges.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
    g = GraphFrame(vertices, edges)
    phases["graph_build_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if algo == "pagerank":
        res = g.pageRank(resetProbability=conf["reset_probability"], maxIter=conf["max_iter"]).vertices
        res_col = "pagerank"
    elif algo == "wcc":
        res = g.connectedComponents(algorithm=conf["impl"], storage_level=storage_level,
                                    use_local_checkpoints=conf["use_local_checkpoints"])
        res_col = "component"
    elif algo == "cdlp":
        res = g.labelPropagation(maxIter=conf["max_iter"], algorithm=conf["impl"],
                                 storage_level=storage_level,
                                 use_local_checkpoints=conf["use_local_checkpoints"])
        res_col = "label"
    else:  # sssp
        n_vertices = info["vertices"] or 0
        landmark = int(n_vertices * conf["landmark_frac"])
        res = g.shortestPaths(landmarks=[landmark], algorithm=conf["impl"],
                              storage_level=storage_level, is_directed=True,
                              use_local_checkpoints=conf["use_local_checkpoints"])
        res_col = "distances"
    phases["algorithm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    outdir = run_workdir / "output"
    outdir.mkdir(parents=True, exist_ok=True)
    res.write.mode("overwrite").parquet(str(outdir / "result.parquet"))
    phases["write_parquet_s"] = time.perf_counter() - t0
    phases["wall_in_process_s"] = time.perf_counter() - wall0

    report = {
        "algorithm": algo,
        "vertices": info["vertices"],
        "edges": info["edges"],
        "edges_input": 2 * (info["edges"] or 0) if conf["symmetrize_input"] else info["edges"],
        # directed = the graphframe is traversed along edge direction only:
        # wcc (randomized_contraction) symmetrizes internally; pagerank/cdlp/sssp
        # run on the raw directed edge set
        "directed": algo != "wcc",
        "impl": conf["impl"],
        "result_column": res_col,
        "phases_s": phases,
        "params": {
            "max_iter": conf.get("max_iter"),
            "reset_probability": conf.get("reset_probability"),
            "landmark_frac": conf.get("landmark_frac"),
            "landmarks": [int((info["vertices"] or 0) * conf["landmark_frac"])] if algo == "sssp" else None,
            "storage_level": STORAGE_LEVELS[args.storage_level],
            "symmetrized_input": conf["symmetrize_input"],
            "shuffle_partitions": args.num_workers,
            "driver_memory": args.max_memory,
        },
    }
    with open(run_workdir / "worker_report.json", "w") as f:
        json.dump(report, f)
    return report


def environment_info(spark, args: argparse.Namespace) -> dict:
    import platform
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_runner": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gnuplot": shutil.which("gnuplot") is not None,
        "engine": "spark graphframes (local mode)",
        "spark_version": spark.version,
        "master": spark.sparkContext.master,
        "graphframes_jar": f"{JAR_GROUP}:{JAR_ARTIFACT}:{args.gf_version}",
        "driver_memory": args.max_memory,
        "shuffle_partitions": args.num_workers,
        "storage_level": STORAGE_LEVELS[args.storage_level],
    }
    try:
        import pyspark
        import graphframes
        env["pyspark_version"] = pyspark.__version__
        env["graphframes_py_version"] = getattr(graphframes, "__version__", "unknown")
    except Exception:
        pass
    try:
        env["java_version"] = spark.sparkContext._jvm.System.getProperty("java.version")
    except Exception:
        pass
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        if r.returncode == 0:
            env["git_commit"] = r.stdout.strip()
    except OSError:
        pass
    return env


# ---------------------------------------------------------------------------
# Run orchestration (same flow and report layout as main_lbdb.py)
# ---------------------------------------------------------------------------

def run_algorithm(algo: str, info: dict, args: argparse.Namespace,
                  results_root: Path, spark) -> None:
    storage = args.storage_level  # "memory" / "disk" (the --storage-level flag value)
    leaf = f"spark_driver_{args.max_memory}_shuffle_{args.num_workers}_storage_{storage}"
    run_dir = (results_root / args.results_subdir / algo / info["scale"] / args.dataset / leaf)
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.checkpoint_dir) / f"gf_{algo}_{args.dataset}"
    workdir.mkdir(parents=True, exist_ok=True)

    def execute(run_i, interval) -> tuple[monitor.RunResult, dict | None]:
        run_workdir = workdir / f"run_{run_i}"
        # fresh run dir every time: stale checkpoints/spill from an earlier
        # invocation would flatten the disk-consumption delta towards zero
        if run_workdir.exists():
            shutil.rmtree(run_workdir)
        run_workdir.mkdir(parents=True, exist_ok=True)
        # per-run checkpoint folder (spark_checkpoints) inside the checkpoint dir
        ckpt_dir = run_workdir / "spark_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        spark.sparkContext.setCheckpointDir(str(ckpt_dir))

        res = monitor.run_in_process(
            lambda: run_one_algorithm(algo, info, args, run_workdir, spark),
            str(workdir), interval, args.disk_mode)
        return res, _worker_report(run_workdir)

    # --- warmup (optional): discarded; warms page cache and calibrates the interval ---
    warm = None
    if args.warmup > 0:
        try:
            warm, _ = execute("warmup", 0.5)
        except Exception as e:
            raise SystemExit(f"{algo}/{args.dataset}: warmup run failed:\n{e}")
        interval = max(0.02, min(1.0, warm.wall_time_s / args.target_samples))
        print(f"  warmup: {warm.wall_time_s:.2f}s -> sampling interval {interval:.3f}s")
    else:
        interval = args.sample_interval

    # --- measured runs ---
    results, worker_reports = [], []
    for i in range(args.runs):
        try:
            res, report = execute(i, interval)
        except Exception as e:
            print(f"  run {i} failed: {e}", file=sys.stderr)
            raise SystemExit(f"{algo}/{args.dataset}: run {i} failed:\n{e}")
        results.append(res)
        worker_reports.append(report)
        print(f"  run {i}: {res.wall_time_s:.3f}s  peak_rss={res.peak_rss_kb} kB  "
              f"peak_disk={res.peak_disk_bytes / 1e6:.1f} MB  samples={len(res.samples)}")

    # --- statistics (same helpers as main.py / main_lbdb.py) ---
    times = [r.wall_time_s for r in results]
    time_stats = stats.describe(times)
    rss_peaks = [r.peak_rss_kb for r in results if r.peak_rss_kb]
    rss_stats = stats.describe(rss_peaks) if rss_peaks else None
    disk_peaks = [r.peak_disk_bytes for r in results]
    disk_stats = stats.describe(disk_peaks)
    medges = (info["edges"] / time_stats["median"] / 1e6) if info["edges"] else None

    # --- series alignment ---
    series = None
    samples_list = [r.samples for r in results]
    if all(len(s) >= 2 for s in samples_list):
        align = stats.align_series(samples_list, method=args.align)
        series = {
            "alignment": align["method"],
            "ref_run": align["ref_run"],
            "band_frac": align["band_frac"],
            "grid_size": len(align["grid"]),
            "rss_bands_gib": [[m / 1048576, lo / 1048576, hi / 1048576] for m, lo, hi in align["rss_bands"]],
            "disk_bands_gib": [[m / 1073741824, lo / 1073741824, hi / 1073741824] for m, lo, hi in align["disk_bands"]],
        }

    # --- JSON ---
    payload = {
        "algorithm": algo,
        "dataset": args.dataset,
        "size_class": info["scale"],
        "engine": "spark-graphframes",
        "params": {
            "driver_memory": args.max_memory,
            "shuffle_partitions": args.num_workers,
            "storage_level": STORAGE_LEVELS[args.storage_level],
            "gf_version": args.gf_version,
            "jar": f"{JAR_GROUP}:{JAR_ARTIFACT}:{args.gf_version}",
            "impl": ALGORITHMS[algo]["impl"],
            "runs": args.runs,
            "warmup": args.warmup,
            "sample_interval_s": interval,
            "target_samples": args.target_samples,
            "align": args.align,
            "disk_mode": args.disk_mode,
            "checkpoint_dir": str(workdir),
            "algo_params": {k: v for k, v in ALGORITHMS[algo].items() if k != "impl"},
        },
        "graph": {
            "vertices": info["vertices"],
            "edges": info["edges"],
            "nodes_str": info["nodes_str"],
            "edges_str": info["edges_str"],
        },
        "warmup": {
            "wall_time_s": warm.wall_time_s if warm else None,
            "sampling_interval_s": interval,
        },
        "monitor": {"disk_mode": args.disk_mode, "target_samples": args.target_samples},
        "runs": [
            {"index": i, "wall_time_s": r.wall_time_s, "peak_rss_kb": r.peak_rss_kb,
             "peak_disk_bytes": r.peak_disk_bytes, "returncode": r.returncode,
             "n_samples": len(r.samples), "worker": w}
            for i, (r, w) in enumerate(zip(results, worker_reports))
        ],
        "stats": {
            "wall_time_s": time_stats,
            "peak_rss_kb": rss_stats,
            "peak_disk_bytes": disk_stats,
            "medges_per_sec": medges,
        },
        "series": series,
        "raw_series": [
            {"t": [s[0] for s in r.samples],
             "rss_kb": [s[1] for s in r.samples],
             "disk_bytes": [s[2] for s in r.samples]}
            for r in results
        ],
        "environment": environment_info(spark, args),
    }
    with open(run_dir / "benchmark.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --- plots ---
    title = (f"spark graphframes {algo} / {args.dataset} ({info['scale']}) — "
             f"driver_{args.max_memory} shuffle_{args.num_workers} storage_{storage}")
    plotting.write_wall_time(run_dir, title, times, time_stats)
    if series is not None:
        grid = [i / (series["grid_size"] - 1) for i in range(series["grid_size"])]
        plotting.write_series(run_dir, "rss", title + " — RSS", "RSS (GiB)",
                              grid, [tuple(b) for b in series["rss_bands_gib"]])
        plotting.write_series(run_dir, "disk", title + " — disk usage", "disk consumed (GiB)",
                              grid, [tuple(b) for b in series["disk_bands_gib"]])
    rendered = sum(plotting.render(s) for s in run_dir.glob("*.gnuplot"))
    print(f"  -> {run_dir}  (gnuplot: {'rendered' if rendered else 'not available'})")


def _worker_report(run_workdir: Path) -> dict | None:
    """Parse the per-run worker JSON report written by run_one_algorithm."""
    path = run_workdir / "worker_report.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.loads(f.read().strip())
    except (OSError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        print(datasets.list_datasets())
        return

    # pyspark + graphframes-py live in the shared venv: re-exec under it on demand
    venv_python = ensure_runner_env()
    if venv_python is not None:
        os.execv(str(venv_python), [str(venv_python), os.path.abspath(__file__)] + sys.argv[1:])

    if args.dataset not in datasets.CATALOG:
        raise SystemExit(f"unknown dataset {args.dataset!r}; use --list-datasets")
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in algos:
        if a not in ALGORITHMS:
            raise SystemExit(f"unknown algorithm {a!r}; available: {', '.join(ALGORITHMS)}")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    info = datasets.info(args.dataset)
    print(f"dataset: {args.dataset} ({info['scale']}, {info['nodes_str']} nodes, "
          f"{info['edges_str']} edges, {info['size']})")
    datasets.ensure_dataset(args.dataset, Path(args.data_dir))

    results_root = Path(args.results_dir)
    for algo in algos:
        print(f"== spark graphframes {algo} / {args.dataset} ==")
        spark = create_spark_session(args, Path(args.checkpoint_dir) / f"gf_{algo}_{args.dataset}")
        print(f"  spark {spark.version} session ready "
              f"(driver={args.max_memory}, shuffle={args.num_workers}, "
              f"storage={STORAGE_LEVELS[args.storage_level]}, "
              f"jar {JAR_GROUP}:{JAR_ARTIFACT}:{args.gf_version})")
        try:
            run_algorithm(algo, info, args, results_root, spark)
        finally:
            spark.stop()

    print(f"\nDone. Spark results under {results_root / args.results_subdir}")


if __name__ == "__main__":
    main()
