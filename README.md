# graphframes-rs

An experimental single node out-of core graph algorithms.

**!NOTE!**

*At the moment it is just a core. The existing `main.rs` should be fine for benchmarking / experimenting, but it is NOT A STABLE PUBLIC CONTRACT! I'm (the author) have an experience with distributed (out-of-core) graph algorithms but I have zero experience with building publuc surface APIs for DataFusion projects. So, there is not public API behind the existing low-level builder methods of the `GraphFrame` struct. I do not know yet what is the best way to make a DataFusion plugin, how should it work (Python? FFI? Arrow Flight?) as well how such an API should looks like. Should it even be a programmatic API with full access to methods or just a CLI/Server that supports DSL? If you are interesting in the project and have an idea (or experience with DataFusion-based public surface APIs) how the public surface should look like please, open an issue and share. I will appreciate. I'm (the author) interested in learning Rust, DataFusion and out-of-core graph analytics, so for me the public surface is not the top priority!*

**Supported algorithms**

1. Low-level `Pregel API`: *Malewicz, Grzegorz, et al. "Pregel: a system for large-scale graph processing." Proceedings of the 2010 ACM SIGMOD International Conference on Management of data. 2010.*
   a. Pregel-based PageRank: *Zadeh, R., et al. "Cme 323: Distributed algorithms and optimization, spring 2015." University Lecture (2015).*
   b. Pregel-based Multi Source Shortest Paths: *Malewicz, Grzegorz, et al. "Pregel: a system for large-scale graph processing." Proceedings of the 2010 ACM SIGMOD International Conference on Management of data. 2010.*
2. Weakly Connected Components: *Bögeholz, Harald, Michael Brand, and Radu-Alexandru Todor. "In-database connected component analysis." 2020 IEEE 36th International Conference on Data Engineering (ICDE). IEEE, 2020.*
