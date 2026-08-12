#!/usr/bin/env python3
"""Statistical helpers for benchmark post-processing (pure stdlib).

Everything here is deliberately dependency-free: percentiles, Student-t
confidence intervals, resampling onto a fraction-of-run grid, and dynamic
time warping (DTW) used to align runs whose phases drift in wall time.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left

# --------------------------------------------------------------------------
# Basic descriptive statistics
# --------------------------------------------------------------------------


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default method) on sorted data."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty data")
    if n == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = lo + 1
    frac = rank - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def kde(
    values: list[float],
    grid_points: int = 256,
    bandwidth: float | None = None,
) -> tuple[list[float], list[float]]:
    """Gaussian kernel density estimate of `values` on a uniform grid.

    Pure stdlib (no numpy/scipy). Bandwidth defaults to Silverman's rule of
    thumb ``h = 1.06 * min(std, IQR/1.34) * n^-1/5``, clamped so degenerate
    (all-equal or single-run) samples still render as a narrow spike.

    Returns ``(xs, density)`` with ``density`` a true PDF on the grid
    (trapezoidal integral over the returned grid equals 1).
    """
    n = len(values)
    if n == 0:
        raise ValueError("kde of empty data")
    xs = sorted(values)
    vmin, vmax = xs[0], xs[-1]
    spread = vmax - vmin

    if bandwidth is None:
        sd = statistics.stdev(xs) if n > 1 else 0.0
        q75, q25 = percentile(xs, 75), percentile(xs, 25)
        iqr = q75 - q25
        sigma = min(sd, iqr / 1.34) if iqr > 0 else sd
        h = 1.06 * sigma * (n ** -0.2) if sigma > 0 else 0.0
        if h <= 0.0:  # degenerate sample (all equal): base the kernel on scale
            h = spread * 0.1 if spread > 0 else max(abs(vmin) * 0.1, 1.0)
        h = max(h, spread * 1e-6)
    else:
        h = bandwidth
    if h <= 0.0:
        h = 1.0

    pad = 4.0 * h  # cover ~4 sigma on each side so the curve falls to ~0
    lo, hi = vmin - pad, vmax + pad
    xs_grid = [lo + (hi - lo) * i / (grid_points - 1) for i in range(grid_points)]

    const = 1.0 / (n * h * math.sqrt(2.0 * math.pi))
    inv_h2 = -0.5 / (h * h)
    density = []
    for x in xs_grid:
        s = 0.0
        for v in xs:
            d = x - v
            s += math.exp(d * d * inv_h2)
        density.append(const * s)

    # renormalize via the trapezoidal rule: integral over the grid == 1
    total = 0.0
    for i in range(grid_points - 1):
        total += (density[i] + density[i + 1]) * (xs_grid[i + 1] - xs_grid[i]) / 2.0
    if total > 0.0:
        density = [d / total for d in density]
    return xs_grid, density


def describe(values: list[float]) -> dict:
    """Median/mean/std/min/max/p90/p95 of a list of measurements."""
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "median": percentile(s, 50),
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
    }


# Two-sided 97.5% critical values of Student's t for df = 1..30.
_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical(n: int) -> float:
    """97.5% two-sided t critical value for n samples; 1.96 fallback for n > 30."""
    if n < 2:
        return 0.0
    return _T_TABLE.get(n, 1.96)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) using mean ± t * SEM."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = statistics.fmean(values)
    if n < 2:
        return (mean, mean, mean)
    sem = statistics.stdev(values) / (n ** 0.5)
    t = t_critical(n)
    return (mean, mean - t * sem, mean + t * sem)


# --------------------------------------------------------------------------
# Resampling and signal processing
# --------------------------------------------------------------------------


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    """Piecewise-linear interpolation with endpoint clamping."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect_left(xs, x)
    x0, x1 = xs[i - 1], xs[i]
    y0, y1 = ys[i - 1], ys[i]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def resample_fraction(
    samples: list[tuple[float, float]], grid_size: int
) -> tuple[list[float], list[float]]:
    """Map (t, v) samples onto a [0, 1] fraction-of-run grid via linear interpolation.

    Returns (fractions, values) each of length `grid_size`.
    """
    ts = [t for t, _ in samples]
    vs = [v for _, v in samples]
    duration = ts[-1] if ts else 1.0
    if duration <= 0.0:
        duration = 1.0
    fracs = [i / (grid_size - 1) for i in range(grid_size)]
    vals = [_interp(ts, vs, f * duration) for f in fracs]
    return fracs, vals


def smooth(values: list[float], window: int = 5) -> list[float]:
    """Moving average (window-sized, centered)."""
    if window < 2:
        return list(values)
    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def find_peaks(values: list[float], prominence_frac: float = 0.2) -> list[int]:
    """Indices of local maxima with prominence >= prominence_frac of the range.

    Adjacent maxima (within 1 index) are merged, keeping the strongest.
    """
    if len(values) < 3:
        return []
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span <= 0.0:
        return []
    threshold = span * prominence_frac
    peaks: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            base = min(values[i - 1], values[i + 1])
            if values[i] - base >= threshold:
                peaks.append(i)
    merged: list[int] = []
    for p in peaks:
        if merged and p - merged[-1] <= 1:
            if values[p] > values[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged


# --------------------------------------------------------------------------
# Shape-based time-warping alignment (dynamic time warping)
# --------------------------------------------------------------------------


def _dtw_matrix(
    a: list[float], b: list[float], band_frac: float = 0.25
) -> tuple[float, list[tuple[int, int]]]:
    """DTW distance matrix + optimal path between two series (O(n*m) time).

    A Sakoe-Chiba band limits |i - j| to `band_frac` of the series length: it
    keeps warps physically plausible (a phase of one run cannot slide more than
    ~band_frac of its duration relative to the reference) and cuts the cost.
    Pure-python is fast enough: 300x300 cells run in ~10ms, 1000x1000 in ~0.2s.
    """
    n, m = len(a), len(b)
    band = max(1, int(band_frac * max(n, m)))
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        Di, Dip = D[i], D[i - 1]
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = (ai - b[j - 1]) ** 2
            Di[j] = c + min(Dip[j], Di[j - 1], Dip[j - 1])
    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(
                (D[i - 1][j], (i - 1, j)),
                (D[i][j - 1], (i, j - 1)),
                (D[i - 1][j - 1], (i - 1, j - 1)),
            )
            i, j = best[1]
    path.reverse()
    return D[n][m], path


def _dtw_warp(
    ref: list[float], run: list[float], band_frac: float = 0.25
) -> dict[int, float]:
    """Map each reference index i to the aligned run index j.

    When the optimal path visits a reference index several times (a plateau or
    a steep ramp of the other series), the mean of the matched run indices is
    used; the resulting piecewise-constant map is smoothed into a monotone
    warp below.
    """
    _, path = _dtw_matrix(ref, run, band_frac)
    warp: dict[int, list[float]] = {}
    for i, j in path:
        warp.setdefault(i, []).append(float(j))
    out = {}
    for i, js in warp.items():
        out[i] = statistics.fmean(js)
    # enforce monotonicity (DTW can emit tiny non-monotone jitter after the
    # mean-collapse); a strictly increasing map keeps the resampler well-behaved
    prev = -1.0
    for i in sorted(out):
        out[i] = max(out[i], prev)
        prev = out[i]
    return out


def _warp_curve(
    warp: dict[int, float],
    grid: list[float],
    ts: list[float],
    vs: list[float],
) -> list[float]:
    """Resample a raw (t, v) series onto the grid through a DTW warp.

    `warp[i]` is the run index (fraction-of-run grid coordinate) aligned to
    grid point i of the reference. The raw series is re-interpolated at the
    warped *wall times*, so the result keeps the run's own RSS/disk levels.
    """
    dur = ts[-1] if ts else 1.0
    if dur <= 0.0:
        dur = 1.0
    idx = list(range(len(grid)))
    out = []
    for i, f in enumerate(grid):
        j = warp.get(i, i)  # identity when the run is the reference
        # j is a grid index (possibly fractional): map it back to a fraction
        f_run = _interp(idx, grid, j)
        out.append(_interp(ts, vs, f_run * dur))
    return out


def align_series(
    runs: list[list[tuple[float, float, float]]],
    grid_size: int = 100,
    method: str = "dtw",
    band_frac: float = 0.25,
    smooth_window: int = 7,
) -> dict:
    """Align per-run (t, rss, disk) sample lists onto a common fraction grid.

    `method` is one of "duration" (plain t/T normalization) or "dtw"
    (shape-based dynamic-time-warping: each run's RSS curve is z-scored and
    DTW-aligned to the median-duration reference run, then the warp is applied
    to both RSS and disk; falls back to "duration" when a run has too few
    samples for a meaningful warp).

    Returns a dict with:
      method       -- method actually used
      ref_run      -- index of the reference run (median duration)
      band_frac    -- Sakoe-Chiba band used (dtw mode only)
      grid         -- fraction values on [0, 1]
      rss          -- list of per-run resampled RSS values (kB on the grid)
      disk         -- list of per-run resampled disk values (bytes on the grid)
      rss_mean/ci  -- mean and t-CI across runs per grid point
      disk_mean/ci
    """
    durations = [samples[-1][0] if samples else 0.0 for samples in runs]
    median_dur = statistics.median(durations)
    ref_idx = min(range(len(durations)), key=lambda i: abs(durations[i] - median_dur))

    grid = [i / (grid_size - 1) for i in range(grid_size)]

    # resample every run onto the fraction grid (duration normalization) first;
    # the DTW warp then corrects phase differences between runs
    def grid_values(samples, warp):
        ts = [t for t, _, _ in samples]
        rss = [v for _, v, _ in samples]
        disk = [v for _, _, v in samples]
        if warp is not None:
            rss = _warp_curve(warp, grid, ts, rss)
            disk = _warp_curve(warp, grid, ts, disk)
        else:
            _, rss = resample_fraction([(t, v) for t, v, _ in samples], grid_size)
            _, disk = resample_fraction([(t, v) for t, _, v in samples], grid_size)
        return rss, disk

    use_dtw = method == "dtw" and all(len(s) >= 8 for s in runs)
    warps: list = [None] * len(runs)
    if use_dtw:
        # DTW on z-scored, lightly smoothed shapes so the warp follows the
        # *shape* (ramp/plateau phases) and not the absolute RSS levels, which
        # legitimately differ between runs (e.g. 1264 vs 1241 MiB plateaus).
        _, ref_curve = resample_fraction(
            [(t, v) for t, v, _ in runs[ref_idx]], grid_size
        )
        ref_shape = smooth(
            [(x - statistics.fmean(ref_curve)) / (statistics.stdev(ref_curve) or 1.0)
             for x in ref_curve],
            smooth_window,
        )
        for i, samples in enumerate(runs):
            if i == ref_idx:
                continue
            _, curve = resample_fraction([(t, v) for t, v, _ in samples], grid_size)
            shape = smooth(
                [(x - statistics.fmean(curve)) / (statistics.stdev(curve) or 1.0)
                 for x in curve],
                smooth_window,
            )
            warps[i] = _dtw_warp(ref_shape, shape, band_frac)

    all_rss: list[list[float]] = []
    all_disk: list[list[float]] = []
    for i, samples in enumerate(runs):
        rss, disk = grid_values(samples, warps[i])
        all_rss.append(rss)
        all_disk.append(disk)

    rss_means = [statistics.fmean(vals) for vals in zip(*all_rss)] if len(all_rss) > 1 else all_rss[0]
    disk_means = [statistics.fmean(vals) for vals in zip(*all_disk)] if len(all_disk) > 1 else all_disk[0]

    def bands(cols):
        out = []
        for vals in zip(*cols):
            m, lo, hi = mean_ci(list(vals))
            out.append((m, lo, hi))
        return out

    return {
        "method": "dtw" if use_dtw else "duration",
        "ref_run": ref_idx,
        "band_frac": band_frac if use_dtw else 0.0,
        "grid": grid,
        "rss": all_rss,
        "disk": all_disk,
        "rss_bands": bands(all_rss),
        "disk_bands": bands(all_disk),
    }
