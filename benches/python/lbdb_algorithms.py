#!/usr/bin/env python3
"""Reference (in-memory CSR) implementations of the LDBC benchmark algorithms.

This module is the worker used by ``main_lbdb.py``.  It follows the flow
recommended by the icebug developers for LDBC parquet data:

  1. read the vertex/edge parquet files with pyarrow,
  2. build an ``icebug_format.IcebugMemGraph`` (a CSR over Arrow arrays,
     materialised by DuckDB) with ``IcebugMemGraph.from_arrow_tables``,
  3. wrap it into a NetworKit graph zero-copy with
     ``nk.graph.Graph.fromCSR`` / ``fromIcebugMemGraph``,
  4. run the algorithm and write the result back to parquet.

Everything (read + CSR build + graph construction + algorithm + write)
happens inside one process per run: nothing is pre-cached between runs.

Algorithms (parameters mirror `benches/python/main.py` for graphframes-rs):

  pagerank  nk.centrality.PageRank, damp=0.85, maxIterations=10 (fixed
            budget, like `page-rank --max-iter 10`).  Because the zero-copy
            CSR graph only carries out-edges, a second transposed CSR is
            built from the same edge table (source/target swapped) and fed
            to `Graph.fromCSR` as the in-edge arrays - the directed graph
            then has correct in- *and* out-adjacency.
  wcc       nk.components.ParallelConnectedComponents (parallel label
            propagation) on the symmetrized (add_reverse_edges=True)
            undirected graph - graphframes' `wcc` symmetrizes internally
            the same way (duplicates kept, self-loops dropped there /
            kept once here).
  cdlp      nk.community.PLP (parallel Raghavan label propagation),
            maxIterations=10, on the same symmetrized undirected graph -
            matching `classical-lp --max-iter 10` (LDBC bidirectional-edge
            semantic).  Note PLP breaks label ties randomly while the
            LDBC/graphframes rule is "smallest label", so partitions can
            differ on symmetric ties; the performance comparison is
            unaffected.

Output parquet schema (single file, one row per vertex):

  pagerank -> id:int64, pagerank:float64 (sum-normalized like graphframes)
  wcc      -> id:int64, component:int64  (nk-internal component ids)
  cdlp     -> id:int64, community:int64  (nk-internal label ids)

A one-line JSON report (phase timings, iterations, graph sizes) is printed
to stdout and written to --report when given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# Heavy imports are timed explicitly: the interpreter spends real time in
# pyarrow/duckdb/networkit imports and the benchmark measures the whole
# read->write pipeline of a cold process anyway.


def _import_stack():
    t0 = time.perf_counter()
    import pyarrow as pa          # noqa: F401
    import pyarrow.parquet as pq  # noqa: F401
    import networkit as nk
    from icebug_format import IcebugMemGraph
    return pa, pq, nk, IcebugMemGraph, time.perf_counter() - t0


PA, PQ, NK, IcebugMemGraph, IMPORT_TIME = _import_stack()

ALGORITHMS = ("pagerank", "wcc", "cdlp")

# Fixed LDBC-inspired hyperparameters, mirroring benches/python/main.py.
PAGERANK_DAMP = 0.85    # graphframes: reset_prob 0.15 -> alpha 0.85
PAGERANK_MAX_ITER = 10  # graphframes: --max-iter 10
PAGERANK_TOL = 0.0      # disable convergence: fixed iteration budget,
                        # exactly like the graphframes fixed-budget mode
CDLP_MAX_ITER = 10      # graphframes: classical-lp --max-iter 10


def _single_u64_array(x):
    """pyarrow ChunkedArray/column -> single contiguous UInt64Array."""
    pa = PA
    if isinstance(x, pa.ChunkedArray):
        x = pa.concat_arrays(x.chunks) if x.num_chunks > 1 else x.chunk(0)
    if x.type != pa.uint64():
        x = x.cast(pa.uint64())
    return x


def _read_inputs(vertices: str, edges: str, id_col: str, src_col: str, dst_col: str):
    """Read the LDBC parquet files and normalize column names."""
    nodes = PQ.read_table(vertices, columns=[id_col]).rename_columns(["id"])
    rels = PQ.read_table(edges, columns=[src_col, dst_col]).rename_columns(["source", "target"])
    return nodes, rels


def _directed_graph(nodes, rels):
    """Zero-copy directed nk.Graph with correct out- AND in-adjacency.

    IcebugMemGraph stores one CSR indexed by source.  The zero-copy nk graph
    built from it has out-edges only (iterInNeighbors is empty), which would
    silently break every in-edge based algorithm (PageRank).  So a second
    IcebugMemGraph is built from the transposed edge table and handed to
    Graph.fromCSR as the in-edge arrays - still the documented
    from_arrow_tables -> CSR -> nk.Graph flow, just in both directions.
    """
    rels_t = rels.select(["target", "source"]).rename_columns(["source", "target"])
    csr_out = IcebugMemGraph.from_arrow_tables(nodes, rels)
    csr_in = IcebugMemGraph.from_arrow_tables(nodes, rels_t)
    return NK.graph.Graph.fromCSR(
        nodes.num_rows, True,
        _single_u64_array(csr_out.indices.column("target")),
        _single_u64_array(csr_out.indptr.column("ptr")),
        _single_u64_array(csr_in.indices.column("target")),
        _single_u64_array(csr_in.indptr.column("ptr")),
    )


def _undirected_graph(nodes, rels):
    """Zero-copy undirected nk.Graph over the symmetrized adjacency.

    add_reverse_edges=True mirrors graphframes' internal symmetrization for
    WCC/CDLP on directed datasets (every non-self edge in both directions,
    duplicates kept); directed=False makes nk interpret the doubly-stored
    CSR as one undirected adjacency.
    """
    csr = IcebugMemGraph.from_arrow_tables(nodes, rels, add_reverse_edges=True)
    return NK.graph.Graph.fromIcebugMemGraph(csr, directed=False)


def run_algorithm(name: str, g, max_iter: int | None = None, damp: float = PAGERANK_DAMP,
                  tol: float = PAGERANK_TOL):
    """Run one algorithm; returns (values, n_iterations, extra_metrics)."""
    if name == "pagerank":
        pr = NK.centrality.PageRank(g, damp=damp, tol=tol)
        if not hasattr(type(pr), "maxIterations"):
            raise RuntimeError(
                "icebug PageRank has no maxIterations property (removed in "
                "icebug 13.0); pin icebug==12.9 in lbdb-requirements.txt")
        pr.maxIterations = max_iter if max_iter is not None else PAGERANK_MAX_ITER
        t0 = time.perf_counter()
        pr.run()
        scores = pr.scores()
        total = sum(scores)
        return [v / total for v in scores], pr.numberOfIterations(), {
            "sum_before_normalize": total,
        }
    if name == "wcc":
        cc = NK.components.ParallelConnectedComponents(g)
        t0 = time.perf_counter()
        cc.run()
        part = cc.getPartition()
        return list(part.getVector()), None, {
            "n_components": cc.numberOfComponents(),
        }
    if name == "cdlp":
        iters = max_iter if max_iter is not None else CDLP_MAX_ITER
        plp = NK.community.PLP(g, maxIterations=iters)
        t0 = time.perf_counter()
        plp.run()
        part = plp.getPartition()
        return list(part.getVector()), plp.numberOfIterations(), {
            "n_communities": part.numberOfSubsets(),
        }
    raise ValueError(f"unknown algorithm {name!r}; expected one of {ALGORITHMS}")


OUTPUT_COLUMN = {"pagerank": "pagerank", "wcc": "component", "cdlp": "community"}


def write_output(nodes, values, name: str, output_dir: str):
    """Write id + value parquet (single file) into output_dir."""
    pa = PA
    ids = nodes.column("id")
    if ids.type != pa.int64():
        ids = ids.cast(pa.int64())
    if name == "pagerank":
        arr = pa.array(values, type=pa.float64())
    else:
        arr = pa.array(values, type=pa.int64())
    table = pa.table({"id": ids.combine_chunks(), OUTPUT_COLUMN[name]: arr})
    PQ.write_table(table, f"{output_dir.rstrip('/')}/result.parquet")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    p.add_argument("--vertices", required=True, help="LDBC vertices parquet")
    p.add_argument("--edges", required=True, help="LDBC edges parquet")
    p.add_argument("--output", required=True, help="output directory for result.parquet")
    p.add_argument("--id-col", default="id")
    p.add_argument("--src-col", default="source")
    p.add_argument("--dst-col", default="target")
    p.add_argument("--max-iter", type=int, default=None,
                   help="iteration cap (default: 10 for pagerank/cdlp; ignored for wcc)")
    p.add_argument("--seed", type=int, default=42,
                   help="NetworKit RNG seed (PLP tie-breaking), mirrors graphframes --seed")
    p.add_argument("--threads", type=int, default=None,
                   help="NetworKit/OpenMP thread count (default: all cores)")
    p.add_argument("--report", default=None, help="write the JSON report to this file")
    args = p.parse_args(argv)

    NK.setSeed(args.seed, True)
    if args.threads:
        NK.setNumberOfThreads(args.threads)

    phases = {}
    wall0 = time.perf_counter()

    t0 = time.perf_counter()
    nodes, rels = _read_inputs(args.vertices, args.edges, args.id_col, args.src_col, args.dst_col)
    phases["read_parquet_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if args.algorithm == "pagerank":
        g = _directed_graph(nodes, rels)
        directed = True
    else:
        g = _undirected_graph(nodes, rels)
        directed = False
    phases["csr_and_graph_s"] = time.perf_counter() - t0

    # For the undirected (symmetrized) graph nk reports the CSR half-length,
    # i.e. the number of undirected adjacency entries / 2 == ~m.
    n_edges_alg = g.numberOfEdges()

    t0 = time.perf_counter()
    values, n_iters, extra = run_algorithm(args.algorithm, g, args.max_iter)
    phases["algorithm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    write_output(nodes, values, args.algorithm, args.output)
    phases["write_parquet_s"] = time.perf_counter() - t0
    phases["wall_in_process_s"] = time.perf_counter() - wall0

    report = {
        "algorithm": args.algorithm,
        "vertices": nodes.num_rows,
        "edges": rels.num_rows,
        "graph_edges_after_prepare": n_edges_alg,
        "directed": directed,
        "iterations": n_iters,
        "omp_threads": NK.getMaxNumberOfThreads(),
        "phases_s": phases,
        "import_s": IMPORT_TIME,
        **extra,
        "params": {"max_iter": args.max_iter if args.max_iter is not None
                   else (PAGERANK_MAX_ITER if args.algorithm == "pagerank" else CDLP_MAX_ITER),
                   "damp": PAGERANK_DAMP, "tol": PAGERANK_TOL, "seed": args.seed},
    }
    line = json.dumps(report)
    print(line)
    if args.report:
        with open(args.report, "w") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
