# graphframes-rs

An experimental single node out-of core graph algorithms.

## Usage

### CLI

#### Installation

```sh
cargo install --git https://github.com/SemyonSinchenko/graphframes-rs
```

This builds and installs the `graphframes` binary. The CLI is part of the default feature set, so no extra flags are required.

#### CLI

The binary takes an algorithm as a subcommand, reads a graph from a vertices file and an edges file, and writes the result as parquet to an output directory.

```sh
graphframes page-rank \
  --vertices vertices.parquet \
  --edges edges.parquet \
  --output file:///tmp/pagerank/ \
  --tol 0.01
```

Vertices must contain an Int64 `id` column; edges must contain Int64 `src` and `dst` columns. If your input uses different names, map them with `--id-col-name`, `--src-col-name`, and `--dst-col-name`. The input format defaults to parquet; `--format csv` and `--format json` are also available.

Available algorithms:

- `page-rank`: PageRank centrality;
- `wcc`: Weakly Connected Components;
- `mis`: Maximal Independent Set;
- `kcore`: K-Core decomposition;
- `hyperanf`: Approximate Neighbor Function;
- `shortest-path`: Multi-Source Shortest Path;
- `classical-lp`: Classical (Raghavan) Label Propagaion

Each has its own arguments — run `graphframes <algorithm> --help` for the full list. For what each algorithm computes, see [References](#references).

#### Tuning

This project is designed to be out-of-core and rely on DataFusion spilling during joins and manually offloading to disk intermediate stages. Three knobs trade off memory, parallelism, and disk usage. Each can be set on the command line **or** via an environment variable; the flag takes precedence over the env var, which takes precedence over the default.

**a) Memory pool** — `--max-memory` (env `GRAPHFRAMES_MAX_MEMORY`, default `4G`).

Size of the DataFusion's spill pool ([`FairSpillPool`](https://docs.rs/datafusion/latest/datafusion/execution/memory_pool/struct.FairSpillPool.html)). Operators that exceed their share of the pool spill intermediate data to disk instead of failing with an out-of-memory error. A larger value reduces spilling at the cost of RAM.

**b) Parallelism** — `--num-workers` (env `GRAPHFRAMES_NUM_WORKERS`, default `2`).

Maps to DataFusion's `target_partitions`: the number of partitions the engine processes concurrently.

**c) The trade-off.**

The spill pool is shared fairly among the operators running at the same time. Raising `--num-workers` runs more partitions concurrently, which tends to increase the number of concurrent memory consumers and so shrinks the fair share each one gets. More workers means more parallelism but more spilling (and slower work per partition); fewer workers means more memory per consumer but less parallelism. There is no universally correct value — measure on your own graph and hardware. Defaults are ~fine for something less than 500M of edges.

**d) Input / output.**

Input format is parquet by default; `--format csv` and `--format json` are supported and follow DataFusion's default schema inference and parsing. Output is always parquet, written to a `file://` directory; the directory is created if it does not exist.

**e) Checkpoint / spill directory** — `--checkpoint-dir` (env `GRAPHFRAMES_WORKDIR`, default `gf_workdir`).

The work directory holds two kinds of data: the algorithms' parquet checkpoints (written and re-read between iterations) and DataFusion's spill files. Both are high-throughput and latency-sensitive, and on large graphs the algorithms can stream large amounts of data through them. On my experiments during processing billion-scale graphs with limited memory (~5-6GB limit) the process read from disk more than 200GBs. Point this at the fastest local storage available, preferably a dedicated NVMe or SSD. Network filesystems and spinning disks will make spill and checkpoint latency dominate runtime. The directory is created if missing; relative paths resolve against the current directory. `--max-temp-file` (env `GRAPHFRAMES_MAX_TEMP_FILE`, default `200G`) bounds the total size of the spill subdirectory. For graphs of the size of few billions edges and vertices the default may be not enough.

### Libraray

TBD

## References

### Core

- **Pregel**:
  - _Malewicz, Grzegorz, et al. "Pregel: a system for large-scale graph processing." Proceedings of the 2010 ACM SIGMOD International Conference on Management of data. 2010._
  - _Gonzalez, Joseph E., et al. "{GraphX}: Graph processing in a distributed dataflow framework." 11th USENIX symposium on operating systems design and implementation (OSDI 14). 2014._

### Centrality Metrics

- **PageRank**: _Zadeh, R., et al. "Cme 323: Distributed algorithms and optimization, spring 2015." University Lecture (2015)._
- **K-Core**: _Mandal, Aritra, and Mohammad Al Hasan. "A distributed k-core decomposition algorithm on spark." 2017 IEEE International Conference on Big Data (Big Data). IEEE, 2017._
- **Approximate Neighbor Function**: _Boldi, Paolo, Marco Rosa, and Sebastiano Vigna. "HyperANF: Approximating the neighbourhood function of very large graphs on a budget." Proceedings of the 20th international conference on World Wide Web. 2011._

### Connectivity

- **Multi Source Shortest Paths**: _Malewicz, Grzegorz, et al. "Pregel: a system for large-scale graph processing." Proceedings of the 2010 ACM SIGMOD International Conference on Management of data. 2010._
- **Weakly Connected Components**: _Bögeholz, Harald, Michael Brand, and Radu-Alexandru Todor. "In-database connected component analysis." 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020._

### Community Detection

- **Classical Label Propagation**: _Raghavan, Usha Nandini, Réka Albert, and Soundar Kumara. "Near linear time algorithm to detect community structures in large-scale networks." Physical Review E—Statistical, Nonlinear, and Soft Matter Physics 76.3 (2007): 036106._

### Subgraphs

- **Maximal Independent Set**: _Ghaffari, Mohsen. "An improved distributed algorithm for maximal independent set." Proceedings of the twenty-seventh annual ACM-SIAM symposium on Discrete algorithms. Society for Industrial and Applied Mathematics, 2016._
