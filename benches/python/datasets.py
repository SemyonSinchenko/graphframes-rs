#!/usr/bin/env python3
"""LDBC Graphalytics dataset catalog + parquet download (pure stdlib).

The catalog mirrors the official dataset table
(https://ldbcouncil.org/benchmarks/graphalytics/datasets/): name, scale
class, node/edge counts and package size. Counts are stored both as the
human-readable strings from the page ("3M") and as parsed integers used for
e.g. the SSSP landmark (25th percentile vertex id) and throughput numbers.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://datasets.ldbcouncil.org/graphalytics-parquet"

# name -> (scale, nodes_str, edges_str, size_str)
_CATALOG = [
    ("cit-Patents", "XS", "3M", "16M", "119.1 MB"),
    ("com-friendster", "XL", "65M", "1B", "6.7 GB"),
    ("datagen-7_5-fb", "S", "633k", "34M", "162.3 MB"),
    ("datagen-7_6-fb", "S", "754k", "42M", "200.0 MB"),
    ("datagen-7_7-zf", "S", "13M", "32M", "434.5 MB"),
    ("datagen-7_8-zf", "S", "16M", "41M", "544.3 MB"),
    ("datagen-7_9-fb", "S", "1M", "85M", "401.2 MB"),
    ("datagen-8_0-fb", "M", "1M", "107M", "502.5 MB"),
    ("datagen-8_1-fb", "M", "2M", "134M", "625.4 MB"),
    ("datagen-8_2-zf", "M", "43M", "106M", "1.4 GB"),
    ("datagen-8_3-zf", "M", "53M", "130M", "1.7 GB"),
    ("datagen-8_4-fb", "M", "3M", "269M", "1.2 GB"),
    ("datagen-8_5-fb", "L", "4M", "332M", "1.5 GB"),
    ("datagen-8_6-fb", "L", "5M", "421M", "1.9 GB"),
    ("datagen-8_7-zf", "L", "145M", "340M", "4.6 GB"),
    ("datagen-8_8-zf", "L", "168M", "413M", "5.3 GB"),
    ("datagen-8_9-fb", "L", "10M", "848M", "3.7 GB"),
    ("datagen-9_0-fb", "XL", "12M", "1B", "4.6 GB"),
    ("datagen-9_1-fb", "XL", "16M", "1B", "5.8 GB"),
    ("datagen-9_2-zf", "XL", "434M", "1B", "13.7 GB"),
    ("datagen-9_3-zf", "XL", "555M", "1B", "17.4 GB"),
    ("datagen-9_4-fb", "XL", "29M", "2B", "14.0 GB"),
    ("datagen-sf3k-fb", "XL", "33M", "2B", "12.7 GB"),
    ("datagen-sf10k-fb", "2XL", "100M", "9B", "40.5 GB"),
    ("dota-league", "S", "61k", "50M", "114.3 MB"),
    ("graph500-22", "S", "2M", "64M", "202.4 MB"),
    ("graph500-23", "M", "4M", "129M", "410.6 MB"),
    ("graph500-24", "M", "8M", "260M", "847.7 MB"),
    ("graph500-25", "L", "17M", "523M", "1.7 GB"),
    ("graph500-26", "XL", "32M", "1B", "3.4 GB"),
    ("graph500-27", "XL", "63M", "2B", "7.1 GB"),
    ("graph500-28", "2XL", "121M", "4B", "14.4 GB"),
    ("graph500-29", "2XL", "232M", "8B", "29.6 GB"),
    ("graph500-30", "3XL", "447M", "17B", "60.8 GB"),
    ("kgs", "XS", "832k", "17M", "65.7 MB"),
    ("twitter_mpi", "XL", "52M", "1B", "5.7 GB"),
    ("wiki-Talk", "2XS", "2M", "5M", "34.9 MB"),
    ("example-directed", "-", "10", "17", "1.0 KB"),
    ("example-undirected", "-", "9", "12", "1.0 KB"),
    ("test-bfs-directed", "-", "<100", "<100", "<2.0 KB"),
    ("test-bfs-undirected", "-", "<100", "<100", "<2.0 KB"),
    ("test-cdlp-directed", "-", "<100", "<100", "<2.0 KB"),
    ("test-cdlp-undirected", "-", "<100", "<100", "<2.0 KB"),
    ("test-pr-directed", "-", "<100", "<100", "<2.0 KB"),
    ("test-pr-undirected", "-", "<100", "<100", "<2.0 KB"),
    ("test-lcc-directed", "-", "<100", "<100", "<2.0 KB"),
    ("test-lcc-undirected", "-", "<100", "<100", "<2.0 KB"),
    ("test-wcc-directed", "-", "<100", "<100", "<2.0 KB"),
    ("test-wcc-undirected", "-", "<100", "<100", "<2.0 KB"),
    ("test-sssp-directed", "-", "<100", "<100", "<2.0 KB"),
    ("test-sssp-undirected", "-", "<100", "<100", "<2.0 KB"),
]

CATALOG = {name: (scale, nodes_s, edges_s, size_s) for name, scale, nodes_s, edges_s, size_s in _CATALOG}


def parse_count(s: str) -> int | None:
    """Parse '3M' -> 3_000_000, '633k' -> 633_000, '1B' -> 1_000_000_000, '10' -> 10."""
    s = s.strip()
    if not s or s.startswith("<"):
        return None
    mult = 1
    if s[-1] in "kK":
        mult = 1_000
        s = s[:-1]
    elif s[-1] in "mM":
        mult = 1_000_000
        s = s[:-1]
    elif s[-1] in "bB":
        mult = 1_000_000_000
        s = s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return None


def info(name: str) -> dict:
    scale, nodes_s, edges_s, size_s = CATALOG[name]
    return {
        "name": name,
        "scale": scale,
        "nodes_str": nodes_s,
        "edges_str": edges_s,
        "size": size_s,
        "vertices": parse_count(nodes_s),
        "edges": parse_count(edges_s),
    }


def list_datasets() -> str:
    rows = [["name", "scale", "nodes", "edges", "size"]]
    for name, scale, nodes_s, edges_s, size_s in _CATALOG:
        rows.append([name, scale, nodes_s, edges_s, size_s])
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = []
    for r in rows:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(r, widths)))
    return "\n".join(lines)


def ensure_dataset(name: str, data_dir: Path, retries: int = 3) -> None:
    """Download <name>-v.parquet / <name>-e.parquet into data_dir/<name>/ if missing."""
    ds_dir = data_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("v", "e"):
        target = ds_dir / f"{name}-{suffix}.parquet"
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{BASE_URL}/{name}-{suffix}.parquet"
        print(f"Downloading {url}")
        tmp = target.with_suffix(".parquet.part")
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "graphframes-rs-bench"})
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.rename(target)
                print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB)")
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                print(f"  attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(5)
                else:
                    raise SystemExit(f"failed to download {url}")
