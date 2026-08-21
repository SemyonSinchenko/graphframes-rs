#!/usr/bin/env python3
"""Subprocess/process monitoring for benchmark runs (pure stdlib).

Two entry points:

  run_with_monitor(cmd, ...)  launches the `graphframes` CLI as a child
                              process and samples, from a background thread:
                                * wall time    -- time.perf_counter()
                                * RSS         -- /proc/<pid>/status VmRSS /
                                                 VmHWM (Linux only; degrades
                                                 to None elsewhere)
                                * disk        -- os.statvfs delta on the
                                                 workdir filesystem
                                                 ("statvfs" mode) or the
                                                 workdir tree size ("du"
                                                 mode, exact but slower)
                              Sampling does not disturb the child:
                              stdout/stderr are redirected to files (never
                              pipes, so a full pipe buffer cannot deadlock a
                              chatty child).

  run_in_process(fn, ...)     runs a callable in the current process (used by
                              main_graphframe.py, where the SparkSession lives
                              in the benchmark process) with the same wall /
                              RSS / disk sampling.  RSS is summed over the
                              process tree (python -> spark-submit -> java),
                              so the driver JVM is included even though it is
                              a child of the measured process.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class RunResult:
    wall_time_s: float
    returncode: int
    samples: list = field(default_factory=list)  # (t, rss_kb, disk_bytes)
    peak_rss_kb: float | None = None
    peak_disk_bytes: float | None = None
    stderr_tail: str = ""


def _read_proc_status(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            return f.read()
    except OSError:
        return None


def _vm_kb(status: str | None, key: str) -> int | None:
    if status is None:
        return None
    for line in status.splitlines():
        if line.startswith(key + ":"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
    return None


def _statvfs_used(path: str) -> int:
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    avail = st.f_bavail * st.f_frsize
    return total - avail


def _tree_size_kb(path: str) -> int:
    """Total size in KiB of every regular file under `path` (pure python).

    The workdir only ever holds a handful of checkpoint/output parquet files,
    so walking it per poll is cheap (µs-ms) and gives an *exact* per-run disk
    footprint. This replaces the shell `du` subprocess (slow, 0.5s poll floor)
    and the filesystem-wide `statvfs` delta, which cannot see the workdir's own
    writes on filesystems with lazy free-space accounting.
    """
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return total // 1024


def _sample(pid: int, workdir: str, disk_mode: str, baseline_disk: float) -> tuple | None:
    """Sample (rss_kb, disk_bytes). Disk is floored at 0 (never negative)."""
    status = _read_proc_status(pid)
    rss = _vm_kb(status, "VmRSS")
    if disk_mode == "statvfs":
        # Filesystem-wide used-space delta: other activity on the same
        # filesystem (or lazy free-space accounting) can push it below the
        # baseline -> floor at 0, disk consumption is never negative.
        disk = max(0.0, _statvfs_used(workdir) - baseline_disk)
    else:
        # Exact: size of the workdir tree itself (checkpoints, spills, output).
        disk = max(0.0, _tree_size_kb(workdir) * 1024.0 - baseline_disk)
    if rss is None and disk is None:
        return None
    return (rss, disk)


def run_with_monitor(
    cmd: list[str],
    workdir: str,
    interval: float,
    disk_mode: str = "statvfs",
    stdout_log: str | None = None,
    stderr_log: str | None = None,
) -> RunResult:
    """Run `cmd`, sampling RSS/disk every `interval` seconds.

    `workdir` must exist and is the root whose filesystem (or tree, in
    "du" mode) is watched for disk consumption.
    """
    out_f = open(stdout_log, "w") if stdout_log else subprocess.DEVNULL
    err_f = open(stderr_log, "w") if stderr_log else subprocess.DEVNULL
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)

    baseline_disk = (
        _statvfs_used(workdir)
        if disk_mode == "statvfs"
        else _tree_size_kb(workdir) * 1024.0
    )

    samples: list[tuple[float, float, float]] = []
    stop = threading.Event()
    lock = threading.Lock()

    def loop():
        while not stop.is_set():
            t = time.perf_counter() - start
            s = _sample(proc.pid, workdir, disk_mode, baseline_disk)
            if s is not None:
                rss, disk = s
                with lock:
                    samples.append((t, rss if rss is not None else 0.0,
                                    disk if disk is not None else 0.0))
            stop.wait(interval)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    returncode = proc.wait()
    t_end = time.perf_counter() - start

    # one final sample attempt (the process is gone, so this only captures disk)
    s = _sample(proc.pid, workdir, disk_mode, baseline_disk)
    if s is not None:
        rss, disk = s
        with lock:
            samples.append((t_end, rss if rss is not None else 0.0,
                            disk if disk is not None else 0.0))
    stop.set()
    th.join(timeout=interval * 2 + 1)

    # peak RSS: VmHWM is a high-water mark, so the max seen while alive is the peak
    peak_rss = None
    for t, rss, _ in samples:
        if rss > 0 and (peak_rss is None or rss > peak_rss):
            peak_rss = rss
    peak_disk = max((d for _, _, d in samples), default=0.0)

    tail = ""
    if stderr_log and os.path.exists(stderr_log):
        with open(stderr_log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")

    if out_f is not subprocess.DEVNULL:
        out_f.close()
    if err_f is not subprocess.DEVNULL:
        err_f.close()

    return RunResult(
        wall_time_s=t_end,
        returncode=returncode,
        samples=samples,
        peak_rss_kb=peak_rss,
        peak_disk_bytes=peak_disk,
        stderr_tail=tail,
    )


def _tree_pids(pid: int) -> list[int]:
    """`pid` plus every descendant pid, via /proc/<pid>/task/<tid>/children.

    Used to monitor process trees (e.g. a pyspark driver: python -> spark-submit
    -> java). Returns at least `[pid]`; on non-Linux systems the walk yields
    only `pid` itself.
    """
    found: list[int] = []
    seen: set[int] = set()
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        found.append(p)
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{p}/task/{tid}/children") as f:
                    for tok in f.read().split():
                        c = int(tok)
                        if c not in seen:
                            stack.append(c)
            except (OSError, ValueError):
                continue
    return found


def _tree_rss_kb(pid: int) -> int | None:
    """Sum of VmRSS over `pid` and its descendants (None when unreadable)."""
    total = 0
    saw = False
    for p in _tree_pids(pid):
        rss = _vm_kb(_read_proc_status(p), "VmRSS")
        if rss is not None:
            total += rss
            saw = True
    return total if saw else None


def _sample_tree(pid: int, workdir: str, disk_mode: str, baseline_disk: float) -> tuple | None:
    """Sample (rss_kb, disk_bytes) for a whole process tree. Same semantics as
    `_sample`, but RSS is summed over `pid` and all its descendants (Spark's
    driver JVM lives in a child process even though the session is created
    in-process)."""
    rss = _tree_rss_kb(pid)
    if disk_mode == "statvfs":
        disk = max(0.0, _statvfs_used(workdir) - baseline_disk)
    else:
        disk = max(0.0, _tree_size_kb(workdir) * 1024.0 - baseline_disk)
    if rss is None and disk is None:
        return None
    return (rss, disk)


def run_in_process(
    fn,
    workdir: str,
    interval: float,
    disk_mode: str = "statvfs",
) -> RunResult:
    """Run `fn()` in the current process, sampling RSS/disk every `interval` s.

    Wall time covers `fn()` only (the caller is responsible for excluding any
    one-time setup, e.g. creating a SparkSession, from the measurement). RSS is
    summed over the current process *tree* (`_sample_tree`) so that a pyspark
    driver JVM spawned as a child process is included. Exceptions from `fn()`
    propagate to the caller (the sampling thread is stopped first).
    """
    start = time.perf_counter()
    baseline_disk = (
        _statvfs_used(workdir)
        if disk_mode == "statvfs"
        else _tree_size_kb(workdir) * 1024.0
    )

    samples: list[tuple[float, float, float]] = []
    stop = threading.Event()
    lock = threading.Lock()
    pid = os.getpid()

    def loop():
        while not stop.is_set():
            t = time.perf_counter() - start
            s = _sample_tree(pid, workdir, disk_mode, baseline_disk)
            if s is not None:
                rss, disk = s
                with lock:
                    samples.append((t, rss if rss is not None else 0.0,
                                    disk if disk is not None else 0.0))
            stop.wait(interval)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    try:
        fn()
    finally:
        t_end = time.perf_counter() - start
        stop.set()
        th.join(timeout=interval * 2 + 1)
        # one final sample: the JVM may still be alive, so RSS keeps its peak
        s = _sample_tree(pid, workdir, disk_mode, baseline_disk)
        if s is not None:
            rss, disk = s
            with lock:
                samples.append((t_end, rss if rss is not None else 0.0,
                                disk if disk is not None else 0.0))

    peak_rss = None
    for t, rss, _ in samples:
        if rss > 0 and (peak_rss is None or rss > peak_rss):
            peak_rss = rss
    peak_disk = max((d for _, _, d in samples), default=0.0)

    return RunResult(
        wall_time_s=t_end,
        returncode=0,
        samples=samples,
        peak_rss_kb=peak_rss,
        peak_disk_bytes=peak_disk,
        stderr_tail="",
    )
