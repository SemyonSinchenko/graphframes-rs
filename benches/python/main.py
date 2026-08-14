#!/usr/bin/env python3
"""Benchmark runner for graphframes-rs.

Runs the release `graphframes` CLI against LDBC Graphalytics parquet
datasets, monitoring wall time / RSS / disk usage per run, and writes JSON +
gnuplot reports under:

    benches/results/<algorithm>/<size-class>/<dataset>/max_mem_<mem>_workers_<n>/

Workflow per (algorithm, dataset):
  1. one warmup run (discarded; warms the page cache and calibrates the
     monitor sampling interval),
  2. `--runs` measured runs (fresh process each: every run is a "first run"),
  3. statistics (median/mean/std/min/max/p90/p95, Student-t CIs, optional
     DTW shape alignment of the RSS/disk series),
  4. benchmark.json + wall_time/rss/disk .gnuplot (+ .png when gnuplot exists).

Usage:
    python3 benches/python/main.py --list-datasets
    python3 benches/python/main.py --dataset wiki-Talk
    python3 benches/python/main.py --dataset cit-Patents --algorithms pagerank,wcc
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

# Fixed (LDBC-inspired) hyperparameters. `undirected` marks algorithms that
# are defined on undirected graphs and should get a symmetrized input when the
# user passes --undirected. PageRank is directed-only; WCC/KCore/MIS and
# ClassicalLP already handle symmetrization internally (CDLP always follows
# the LDBC "edges are bidirectional" semantic).
ALGORITHMS = {
    "pagerank": {"cli": "page-rank", "args": ["--max-iter", "10", "--tol", "0.01"], "undirected": False},
    "wcc":      {"cli": "wcc", "args": ["--seed", "42"], "undirected": False},
    "kcore":    {"cli": "kcore", "args": ["--max-iter", "10"], "undirected": False},
    "hyperanf": {"cli": "hyperanf", "args": ["--n-hops", "5"], "undirected": True},
    "sp":       {"cli": "shortest-path", "args": [], "undirected": True},
    "cdlp":     {"cli": "classical-lp", "args": ["--max-iter", "10"], "undirected": False},
    "mis":      {"cli": "mis", "args": [], "undirected": False},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the graphframes-rs CLI on LDBC Graphalytics parquet datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="cit-Patents", help="LDBC dataset name (see --list-datasets)")
    p.add_argument("--algorithms", default=",".join(ALGORITHMS),
                   help=f"comma-separated subset of: {', '.join(ALGORITHMS)}")
    p.add_argument("--runs", type=int, default=5, help="number of measured runs per algorithm")
    p.add_argument("--num-workers", type=int, default=2, help="DataFusion target partitions")
    p.add_argument("--max-memory", default="4G", help="DataFusion spill-pool memory limit (e.g. 4G, 512M)")
    p.add_argument("--checkpoint-dir", default=str(DEFAULT_WORKDIR),
                   help="base workdir for checkpoints, spills and per-run outputs")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="root of the committed results tree")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="dataset download location")
    p.add_argument("--undirected", action="store_true",
                   help="dataset is undirected: symmetrize input for algorithms defined on undirected graphs")
    p.add_argument("--weighted", action="store_true", help="keep the edge weight column while loading")
    p.add_argument("--target-samples", type=int, default=300,
                   help="target number of monitor samples per measured run")
    p.add_argument("--disk-mode", choices=["statvfs", "du"], default="du",
                   help="disk monitoring: du = exact size of the workdir tree "
                        "(pure-python walk, default); statvfs = filesystem-wide "
                        "used-space delta (O(1) but blind to the workdir on "
                        "filesystems with lazy free-space accounting)")
    p.add_argument("--align", choices=["dtw", "duration"], default="dtw",
                   help="series alignment: DTW shape warping or plain duration normalization")
    p.add_argument("--binary", default=None,
                   help="path to a prebuilt graphframes binary (skips cargo build)")
    p.add_argument("--list-datasets", action="store_true", help="print the dataset catalog and exit")
    return p.parse_args()


def build_release() -> Path:
    print("Building release binary: cargo build --release --bin graphframes")
    r = subprocess.run(["cargo", "build", "--release", "--bin", "graphframes"], cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit("cargo build failed")
    return REPO_ROOT / "target" / "release" / "graphframes"


def environment_info() -> dict:
    import platform
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gnuplot": shutil.which("gnuplot") is not None,
    }
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        if r.returncode == 0:
            env["git_commit"] = r.stdout.strip()
    except OSError:
        pass
    return env


def make_command(binary: Path, algo: str, dataset: str, info: dict, args: argparse.Namespace,
                 run_workdir: Path, run_i) -> list[str]:
    conf = ALGORITHMS[algo]
    output_dir = run_workdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    cmd = [
        str(binary), conf["cli"],
        "--vertices", str(data_dir / dataset / f"{dataset}-v.parquet"),
        "--edges", str(data_dir / dataset / f"{dataset}-e.parquet"),
        "--src-col-name", "source",
        "--dst-col-name", "target",
        "--num-workers", str(args.num_workers),
        "--max-memory", args.max_memory,
        "--checkpoint-dir", str(run_workdir),
        "--output", output_dir.resolve().as_uri(),
    ]
    if args.weighted:
        cmd.append("--weighted")
    if algo == "sp":
        vertices = info["vertices"] or 0
        cmd += ["--landmarks", str(int(vertices * 0.25))]
    cmd += conf["args"]
    if args.undirected and conf["undirected"]:
        cmd.append("--symmetrize")
    return cmd


def run_algorithm(binary: Path, algo: str, dataset: str, info: dict,
                  args: argparse.Namespace, results_root: Path) -> None:
    conf = ALGORITHMS[algo]
    run_dir = (results_root / algo / info["scale"] / dataset /
               f"max_mem_{args.max_memory}_workers_{args.num_workers}")
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.checkpoint_dir) / f"{algo}_{dataset}"
    workdir.mkdir(parents=True, exist_ok=True)

    def execute(run_i, interval) -> monitor.RunResult:
        run_workdir = workdir / f"run_{run_i}"
        run_workdir.mkdir(parents=True, exist_ok=True)
        cmd = make_command(binary, algo, dataset, info, args, run_workdir, run_i)
        print("  $ " + " ".join(cmd))
        return monitor.run_with_monitor(
            cmd, str(workdir), interval, args.disk_mode,
            stdout_log=str(run_workdir / "stdout.log"),
            stderr_log=str(run_workdir / "stderr.log"),
        )

    # --- warmup: discarded; warms page cache and calibrates the interval ---
    warm = execute("warmup", 0.5)
    if warm.returncode != 0:
        raise SystemExit(f"{algo}/{dataset}: warmup run failed (rc={warm.returncode}):\n{warm.stderr_tail[-800:]}")
    interval = max(0.02, min(1.0, warm.wall_time_s / args.target_samples))
    print(f"  warmup: {warm.wall_time_s:.2f}s -> sampling interval {interval:.3f}s")

    # --- measured runs ---
    results = []
    for i in range(args.runs):
        res = execute(i, interval)
        if res.returncode != 0:
            raise SystemExit(f"{algo}/{dataset}: run {i} failed (rc={res.returncode}):\n{res.stderr_tail[-800:]}")
        results.append(res)
        print(f"  run {i}: {res.wall_time_s:.3f}s  peak_rss={res.peak_rss_kb} kB  "
              f"peak_disk={res.peak_disk_bytes / 1e6:.1f} MB  samples={len(res.samples)}")

    # --- statistics ---
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
        "params": {
            "max_memory": args.max_memory,
            "num_workers": args.num_workers,
            "checkpoint_dir": str(workdir),
            "undirected": args.undirected,
            "weighted": args.weighted,
            "runs": args.runs,
            "warmup": 1,
            "target_samples": args.target_samples,
            "align": args.align,
            "disk_mode": args.disk_mode,
            "cli_args": conf["args"],
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
             "n_samples": len(r.samples)}
            for i, r in enumerate(results)
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
        "environment": environment_info(),
    }
    with open(run_dir / "benchmark.json", "w") as f:
        json.dump(payload, f, indent=2)

    # --- plots ---
    title = f"{algo} / {dataset} ({info['scale']}) — max_mem_{args.max_memory} workers_{args.num_workers}"
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

    binary = Path(args.binary) if args.binary else build_release()
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}")

    results_root = Path(args.results_dir)
    for algo in algos:
        print(f"== {algo} / {args.dataset} ==")
        run_algorithm(binary, algo, args.dataset, info, args, results_root)

    print(f"\nDone. Results committed under {results_root}")


if __name__ == "__main__":
    main()
