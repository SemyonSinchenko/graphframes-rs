#!/usr/bin/env python3
"""gnuplot script generation and rendering (pure stdlib).

Every benchmark run produces three .dat + .gnuplot pairs (and, when gnuplot
is on PATH, the corresponding .png):

  wall_time  -- kernel density estimate (PDF) of the measured run times:
                probability density on Y vs wall time on X, vertical
                median/p90/p95 lines, individual runs as a baseline rug
  rss        -- mean RSS across runs vs fraction-of-run with 95% CI band
  disk       -- mean disk consumption across runs vs fraction-of-run with CI

The scripts are plain text templates with numbers injected as constants, so
they are fully reproducible without re-running anything.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _write_dat(path: Path, header: str, rows: list[list], fmt: str) -> None:
    with open(path, "w") as f:
        f.write("# " + header + "\n")
        for row in rows:
            f.write(" ".join(fmt % v for v in row) + "\n")


def _quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _escape_gnuplot_string(s: str) -> str:
    """Escape a string for embedding in a double-quoted gnuplot string.

    Real newlines become the two-character escape `\n` (gnuplot renders it as
    a line break inside strings); quotes and backslashes are escaped.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def wall_time_script(
    png_path: Path, dat_path: Path, title: str, st: dict, n_runs: int,
    ymax: float, xlo: float, xhi: float, rug_times: list[float],
) -> str:
    """KDE (PDF) plot of the run times.

    X = wall time, Y = probability density of the Gaussian-kernel estimate
    (computed in pure python, `stats.kde`); the median / p90 / p95 are vertical
    lines and the individual runs appear as a rug along the baseline so small-n
    samples stay visible instead of being hidden inside the smooth curve.

    The x range is the full KDE grid extent (data + 4*bandwidth on each side):
    clipping to a tight data range would cut through the curve where an outlier
    run still carries non-negligible density.
    """
    med, mean = st["median"], st["mean"]
    std, mn, mx = st["std"], st["min"], st["max"]
    p90, p95 = st["p90"], st["p95"]
    ytop = ymax * 1.3
    rug_h = ymax * 0.04
    title_text = (
        f"{title}\n"
        f"median={med:.3f}s  mean={mean:.3f}s  std={std:.3f}s  min={mn:.3f}s  "
        f"max={mx:.3f}s  p90={p90:.3f}s  p95={p95:.3f}s  runs={n_runs}"
    )
    lines = [
        "set terminal pngcairo size 1400,700 enhanced font 'Sans,11'",
        f"set output '{_quote(str(png_path))}'",
        f"set title \"{_escape_gnuplot_string(title_text)}\"",
        "set xlabel 'wall time (s)'",
        "set ylabel 'probability density (1/s)'",
        # room at the top for the 2-line title + the percentile labels
        "set tmargin 5",
        f"set yrange [0:{ytop:.6g}]",
        f"set xrange [{xlo:.6f}:{xhi:.6f}]",
        "set grid y",
        "set key top right",
        # vertical percentile lines; labels sit in the top margin (below the title)
        f"set arrow from {med:.6f},0 to {med:.6f},{ytop:.6g} nohead lc rgb 'red' lw 2",
        f"set label 'median {med:.3f}s' at {med:.6f},{ytop:.6g} offset char 0,1 tc rgb 'red'",
        f"set arrow from {p90:.6f},0 to {p90:.6f},{ytop:.6g} nohead lc rgb 'orange' lw 1 dt 2",
        f"set label 'p90 {p90:.3f}s' at {p90:.6f},{ytop:.6g} offset char 0,1 tc rgb 'orange'",
        f"set arrow from {p95:.6f},0 to {p95:.6f},{ytop:.6g} nohead lc rgb 'orange' lw 1 dt 3",
        f"set label 'p95 {p95:.3f}s' at {p95:.6f},{ytop:.6g} offset char 0,1 tc rgb 'orange'",
    ]
    # rug marks: one tick per measured run along the baseline
    for t in rug_times:
        lines.append(
            f"set arrow from {t:.6f},0 to {t:.6f},{rug_h:.6g} nohead lc rgb '#666666' lw 1"
        )
    lines += [
        # KDE curve: filled area + line
        f"plot '{_quote(str(dat_path))}' using 1:2 with filledcurves y=0 "
        "lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \\",
        f"     '{_quote(str(dat_path))}' using 1:2 with lines lw 2 "
        "lc rgb '#4682b4' title 'KDE (runs)'",
    ]
    return "\n".join(lines) + "\n"


def write_wall_time(
    results_dir: Path, title: str, times: list[float], st: dict
) -> None:
    """Gaussian-kernel density estimate of the run times + percentile lines.

    .dat layout: `time_s density` (grid of ~256 points; the density integrates
    to 1 over the grid). The median/p90/p95 are drawn as vertical lines and the
    individual runs as a rug along the baseline.
    """
    import stats

    xs, dens = stats.kde(times)
    rows = [[x, d] for x, d in zip(xs, dens)]
    dat = results_dir / "wall_time.dat"
    gpl = results_dir / "wall_time.gnuplot"
    png = results_dir / "wall_time.png"
    _write_dat(dat, "time_s density", rows, "%.9g")
    gpl.write_text(wall_time_script(
        png, dat, title, st, len(times), max(dens), xs[0], xs[-1], times,
    ))


def series_script(
    png_path: Path, dat_path: Path, title: str, ylabel: str
) -> str:
    return (
        "set terminal pngcairo size 1400,700 enhanced font 'Sans,11'\n"
        f"set output '{_quote(str(png_path))}'\n"
        f"set title \"{_quote(title)}\"\n"
        "set xlabel 'fraction of run (%)'\n"
        f"set ylabel '{ylabel}'\n"
        "set grid\n"
        "set key top left\n"
        # RSS/disk are non-negative: never let the axis go below 0,
        # even if a CI band was floored at 0 in the .dat.
        "set yrange [0:*]\n"
        f"plot '{_quote(str(dat_path))}' using 1:3:4 with filledcurves "
        "lc rgb '#cccccc' title '95% CI', \\\n"
        f"     '' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'mean'\n"
    )


def write_series(
    results_dir: Path, name: str, title: str, ylabel: str,
    grid: list[float], bands: list[tuple[float, float, float]],
) -> None:
    """Mean + CI band vs fraction-of-run (%); used for RSS and disk.

    RSS/disk are non-negative: the mean and the CI band are floored at 0 so a
    small-mean + wide-CI grid point (e.g. the ramp start with few runs) never
    renders below zero.
    """
    rows = [[f * 100.0, max(0.0, m), max(0.0, lo), max(0.0, hi)]
            for f, (m, lo, hi) in zip(grid, bands)]
    dat = results_dir / f"{name}.dat"
    gpl = results_dir / f"{name}.gnuplot"
    png = results_dir / f"{name}.png"
    _write_dat(dat, "fraction_pct mean ci_low ci_high", rows, "%.6f")
    gpl.write_text(series_script(png, dat, title, ylabel))


def render(script_path: Path) -> bool:
    """Run gnuplot on the script; returns False when gnuplot is unavailable."""
    if shutil.which("gnuplot") is None:
        return False
    r = subprocess.run(["gnuplot", str(script_path)], capture_output=True, text=True)
    return r.returncode == 0
