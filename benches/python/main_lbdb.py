#!/usr/bin/env python3
"""Reference benchmark runner: CSR in-memory graph analytics (icebug) vs graphframes-rs.

Runs the *same* LDBC Graphalytics flow as ``main.py`` but with the in-memory
CSR reference stack recommended by the icebug developers:

    pyarrow read -> IcebugMemGraph.from_arrow_tables -> nk.graph.Graph
    (fromCSR/fromIcebugMemGraph) -> NetworKit algorithm -> parquet output

For every (algorithm, dataset) it performs one discarded warmup run plus
`--runs` measured runs, each a **fresh Python process** (no pre-caching, no
cross-run state, exactly like the graphframes runner measures fresh CLI
processes), monitors wall time / RSS / disk via the shared ``monitor``
module, and writes the same report layout as ``main.py``:

    <results-dir>/ldbd/<algorithm>/<size-class>/<dataset>/icebug_threads_<n>/
        benchmark.json + wall_time/rss/disk .dat/.gnuplot/.png

The results leaf plays the role of graphframes' `max_mem_<mem>_workers_<n>`:
`icebug_threads_<n>` by default, or `icebug_mem_<mem>_threads_<n>` when
`--max-memory` is given.

Algorithms and parameters mirror ``main.py``:

    pagerank  PageRank, damp 0.85, exactly 10 iterations (icebug's
              PageRank.maxIterations property = graphframes --max-iter 10)
    wcc       weakly connected components via parallel label propagation
              (nk ParallelConnectedComponents) on the symmetrized graph
    cdlp      label propagation community detection (nk PLP),
              maxIterations=10, on the symmetrized graph

The measured subprocess wall time covers interpreter start + imports +
parquet read + CSR build + graph construction + algorithm + result write;
``benchmark.json`` additionally carries the in-process phase breakdown
reported by the worker, so startup overhead can be subtracted when comparing.

Usage:
    python3 benches/python/main_lbdb.py --list-datasets
    python3 benches/python/main_lbdb.py                       # wiki-Talk smoke test
    python3 benches/python/main_lbdb.py --dataset cit-Patents --runs 5
    python3 benches/python/main_lbdb.py --dataset wiki-Talk --algorithms pagerank,wcc,cdlp
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
WORKER = Path(__file__).resolve().parent / "lbdb_algorithms.py"
RESULT_SUBDIR = "ldbd"   # reference results live next to the main results tree

# The same fixed hyperparameters as benches/python/main.py (see ALGORITHMS
# there). `symmetrize` marks algorithms for which graphframes symmetrizes the
# edge set internally (WCC via `symmetrize(edges, false)`, CDLP via the LDBC
# bidirectional-edge semantic); the reference stack mirrors that with
# IcebugMemGraph(add_reverse_edges=True) inside the measured run.
ALGORITHMS = {
    "pagerank": {"max_iter": 10, "symmetrize": False},
    "wcc":      {"max_iter": None, "symmetrize": True},
    "cdlp":     {"max_iter": 10, "symmetrize": True},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reference (icebug in-memory CSR) LDBC benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="wiki-Talk",
                   help="LDBC dataset name (see --list-datasets); wiki-Talk (2XS) for smoke tests")
    p.add_argument("--algorithms", default=",".join(ALGORITHMS),
                   help=f"comma-separated subset of: {', '.join(ALGORITHMS)}")
    p.add_argument("--runs", type=int, default=5, help="number of measured runs per algorithm")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help=f"root of the results tree (reports go into <root>/{RESULT_SUBDIR}/...)")
    p.add_argument("--results-subdir", default=RESULT_SUBDIR,
                   help="subdirectory for the reference results inside --results-dir")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="dataset download location")
    p.add_argument("--checkpoint-dir", default=str(DEFAULT_WORKDIR),
                   help="base workdir for per-run outputs and monitor logs")
    p.add_argument("--python", default=None,
                   help="python interpreter with the lbdb-requirements.txt deps "
                        f"(default: {DEFAULT_VENV}/bin/python, created on demand)")
    p.add_argument("--threads", type=int, default=None,
                   help="NetworKit/OpenMP and DuckDB threads per run (default: all cores)")
    p.add_argument("--max-memory", default=None,
                   help="DuckDB memory limit for the CSR conversion, e.g. 8G "
                        "(same syntax as main.py --max-memory). Forces the "
                        "conversion to spill to disk instead of OOMing; the "
                        "results leaf becomes icebug_mem_<mem>_threads_<n>. "
                        "Default: none (DuckDB default = 80%% of RAM)")
    p.add_argument("--seed", type=int, default=42, help="NetworKit RNG seed (mirrors --seed 42)")
    p.add_argument("--target-samples", type=int, default=300,
                   help="target number of monitor samples per measured run")
    p.add_argument("--disk-mode", choices=["statvfs", "du"], default="du",
                   help="disk monitoring mode (see monitor.py; du = workdir tree size)")
    p.add_argument("--align", choices=["dtw", "duration"], default="dtw",
                   help="series alignment: DTW shape warping or plain duration normalization")
    p.add_argument("--list-datasets", action="store_true", help="print the dataset catalog and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Environment / interpreter bootstrap
# ---------------------------------------------------------------------------

def ensure_python(requested: str | None) -> tuple[Path, dict]:
    """Return the worker interpreter and a versions dict.

    With no explicit --python the dedicated venv `benches/python/.lbdb-venv`
    is created on demand (uv when available, venv+pip otherwise) and kept
    around between runs.
    """
    if requested:
        py = shutil.which(requested) or requested
        py = Path(py)
        if not py.exists():
            raise SystemExit(f"interpreter not found: {py}")
        return py, _versions(py)

    venv_python = DEFAULT_VENV / "bin" / "python"
    if not venv_python.exists():
        print(f"Creating reference venv at {DEFAULT_VENV} (from {REQUIREMENTS})")
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
            raise SystemExit(
                "failed to install reference dependencies; install them manually into a "
                "venv and pass --python <venv>/bin/python")
    return venv_python, _versions(venv_python)


def _versions(py: Path) -> dict:
    probe = (
        "import sys, importlib.metadata as m; "
        "print(sys.version.split()[0]); "
        "print(m.version('icebug')); "
        "print(m.version('icebug-format')); "
        "print(m.version('pyarrow')); "
        "print(m.version('duckdb')); "
        "import networkit as nk; print(nk.getMaxNumberOfThreads())"
    )
    r = subprocess.run([str(py), "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"reference interpreter probe failed:\n{r.stderr[-800:]}")
    lines = r.stdout.split()
    return {
        "python": lines[0], "icebug": lines[1], "icebug_format": lines[2],
        "pyarrow": lines[3], "duckdb": lines[4], "max_threads": int(lines[5]),
    }


def environment_info(versions: dict) -> dict:
    import platform
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_runner": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gnuplot": shutil.which("gnuplot") is not None,
        "engine": "icebug (in-memory CSR via icebug-format + networkit fork)",
    }
    env.update({f"ref_{k}": v for k, v in versions.items()})
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        if r.returncode == 0:
            env["git_commit"] = r.stdout.strip()
    except OSError:
        pass
    return env


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------

def make_command(py: Path, algo: str, dataset: str, info: dict, args: argparse.Namespace,
                 run_workdir: Path) -> list[str]:
    conf = ALGORITHMS[algo]
    output_dir = run_workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    cmd = [
        str(py), str(WORKER),
        "--algorithm", algo,
        "--vertices", str(data_dir / dataset / f"{dataset}-v.parquet"),
        "--edges", str(data_dir / dataset / f"{dataset}-e.parquet"),
        "--src-col", "source",
        "--dst-col", "target",
        "--seed", str(args.seed),
        "--output", str(output_dir),
        "--report", str(run_workdir / "worker_report.json"),
    ]
    if conf["max_iter"] is not None:
        cmd += ["--max-iter", str(conf["max_iter"])]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    # spill location inside the run workdir: the du-mode disk monitor walks
    # the whole workdir tree, so DuckDB spill files show up in the disk curve
    cmd += ["--temp-dir", str(run_workdir / "duckdb_tmp")]
    if args.max_memory:
        cmd += ["--max-memory", args.max_memory]
    return cmd


def _worker_report(run_workdir: Path) -> dict | None:
    """Parse the worker's one-line JSON report from the run directory."""
    path = run_workdir / "worker_report.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.loads(f.read().strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return None


def run_algorithm(py: Path, versions: dict, algo: str, dataset: str, info: dict,
                  args: argparse.Namespace, results_root: Path) -> None:
    threads = args.threads or versions["max_threads"]
    leaf = (f"icebug_mem_{args.max_memory}_threads_{threads}" if args.max_memory
            else f"icebug_threads_{threads}")
    run_dir = (results_root / args.results_subdir / algo / info["scale"] / dataset / leaf)
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.checkpoint_dir) / f"lbdb_{algo}_{dataset}"
    workdir.mkdir(parents=True, exist_ok=True)

    def execute(run_i, interval) -> monitor.RunResult:
        run_workdir = workdir / f"run_{run_i}"
        # fresh run dir every time: stale outputs from an earlier invocation
        # would flatten the disk-consumption delta towards zero
        if run_workdir.exists():
            shutil.rmtree(run_workdir)
        run_workdir.mkdir(parents=True, exist_ok=True)
        cmd = make_command(py, algo, dataset, info, args, run_workdir)
        print("  $ " + " ".join(cmd))
        return monitor.run_with_monitor(
            cmd, str(workdir), interval, args.disk_mode,
            stdout_log=str(run_workdir / "stdout.log"),
            stderr_log=str(run_workdir / "stderr.log"),
        )

    # --- warmup: discarded; warms page cache and calibrates the interval ---
    warm = execute("warmup", 0.5)
    if warm.returncode != 0:
        raise SystemExit(f"{algo}/{dataset}: warmup run failed (rc={warm.returncode}):\n{warm.stderr_tail[-1200:]}")
    interval = max(0.02, min(1.0, warm.wall_time_s / args.target_samples))
    print(f"  warmup: {warm.wall_time_s:.2f}s -> sampling interval {interval:.3f}s")

    # --- measured runs: fresh process each, no pre-caching ---
    results, worker_reports = [], []
    for i in range(args.runs):
        res = execute(i, interval)
        if res.returncode != 0:
            raise SystemExit(f"{algo}/{dataset}: run {i} failed (rc={res.returncode}):\n{res.stderr_tail[-1200:]}")
        results.append(res)
        worker_reports.append(_worker_report(workdir / f"run_{i}"))
        print(f"  run {i}: {res.wall_time_s:.3f}s  peak_rss={res.peak_rss_kb} kB  "
              f"peak_disk={res.peak_disk_bytes / 1e6:.1f} MB  samples={len(res.samples)}")

    # --- statistics (same helpers as main.py) ---
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
        "dataset": dataset,
        "size_class": info["scale"],
        "engine": "icebug",
        "params": {
            "icebug_threads": threads,
            "duckdb_max_memory": args.max_memory,
            "seed": args.seed,
            "runs": args.runs,
            "warmup": 1,
            "target_samples": args.target_samples,
            "align": args.align,
            "disk_mode": args.disk_mode,
            "max_iter": ALGORITHMS[algo]["max_iter"],
            "symmetrized_input": ALGORITHMS[algo]["symmetrize"],
        },
        "graph": {
            "vertices": info["vertices"],
            "edges": info["edges"],
            "nodes_str": info["nodes_str"],
            "edges_str": info["edges_str"],
        },
        "warmup": {"wall_time_s": warm.wall_time_s, "sampling_interval_s": interval},
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
        "environment": environment_info(versions),
    }
    with open(run_dir / "benchmark.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --- plots ---
    title = (f"icebug {algo} / {dataset} ({info['scale']}) — "
             + (f"mem_{args.max_memory}_" if args.max_memory else "")
             + f"threads_{threads}")
    plotting.write_wall_time(run_dir, title, times, time_stats)
    if series is not None:
        grid = [i / (series["grid_size"] - 1) for i in range(series["grid_size"])]
        plotting.write_series(run_dir, "rss", title + " — RSS", "RSS (GiB)",
                              grid, [tuple(b) for b in series["rss_bands_gib"]])
        plotting.write_series(run_dir, "disk", title + " — disk usage", "disk consumed (GiB)",
                              grid, [tuple(b) for b in series["disk_bands_gib"]])
    rendered = sum(plotting.render(s) for s in run_dir.glob("*.gnuplot"))
    print(f"  -> {run_dir}  (gnuplot: {'rendered' if rendered else 'not available'})")


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        print(datasets.list_datasets())
        return
    if args.dataset not in datasets.CATALOG:
        raise SystemExit(f"unknown dataset {args.dataset!r}; use --list-datasets")
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in algos:
        if a not in ALGORITHMS:
            raise SystemExit(f"unknown algorithm {a!r}; available: {', '.join(ALGORITHMS)}")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    info = datasets.info(args.dataset)
    print(f"dataset: {args.dataset} ({info['scale']}, {info['nodes_str']} nodes, "
          f"{info['edges_str']} edges, {info['size']})")
    datasets.ensure_dataset(args.dataset, Path(args.data_dir))

    py, versions = ensure_python(args.python)
    print(f"reference interpreter: {py} (icebug {versions['icebug']}, "
          f"icebug-format {versions['icebug_format']}, pyarrow {versions['pyarrow']}, "
          f"duckdb {versions['duckdb']}, {versions['max_threads']} threads)")

    results_root = Path(args.results_dir)
    for algo in algos:
        print(f"== icebug {algo} / {args.dataset} ==")
        run_algorithm(py, versions, algo, args.dataset, info, args, results_root)

    print(f"\nDone. Reference results under {results_root / args.results_subdir}")


if __name__ == "__main__":
    main()
