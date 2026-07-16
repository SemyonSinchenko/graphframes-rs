use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::compute::SortOptions;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::Statistics;
use datafusion::common::config::ConfigOptions;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl, PartitionedFile,
};
use datafusion::datasource::physical_plan::{FileGroup, FileScanConfig};
use datafusion::datasource::source::{DataSource, DataSourceExec, OpenArgs};
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::object_store::buffered::BufWriter;
use datafusion::object_store::path::Path;
use datafusion::parquet::arrow::AsyncArrowWriter;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionExprs;
use datafusion::physical_expr::{
    EquivalenceProperties, LexOrdering, Partitioning, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_plan::execution_plan::SchedulingType;
use datafusion::physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion::physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion::physical_plan::{DisplayFormatType, ExecutionPlan, ExecutionPlanProperties};
use datafusion::physical_plan::{repartition::RepartitionExec, sorts::sort::SortExec};
use datafusion::prelude::{DataFrame, SessionContext, col};
use futures::StreamExt;
use tokio::task::JoinSet;

/// Sort the data within partitions
async fn create_writer_plan(
    df: &DataFrame,
    key: &str,
    num_partitions: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Plan the *source* only. We must NOT go through `df.repartition(Hash).create_physical_plan()`:
    // the physical optimizer can drop a logical `Repartition(Hash)` (e.g. over a single-partition
    // source), leaving the wrong partition count and corrupting the co-partitioning contract.
    // Building `RepartitionExec` by hand keeps it out of the optimizer's reach.
    let base = df.clone().create_physical_plan().await?;

    let schema = base.schema();
    let partitioning = Partitioning::Hash(
        vec![Arc::new(Column::new_with_schema(key, &schema)?) as _],
        num_partitions,
    );
    let repartitioned = Arc::new(RepartitionExec::try_new(base, partitioning)?);

    let sort_options = SortOptions::new(false, true);
    let sort_col = Arc::new(Column::new_with_schema(key, &schema)?);
    let sorted: Arc<dyn ExecutionPlan> = Arc::new(
        SortExec::new(
            LexOrdering::new(vec![PhysicalSortExpr::new(sort_col, sort_options)]).unwrap(),
            repartitioned,
        )
        .with_preserve_partitioning(true),
    );

    assert_eq!(
        sorted.output_partitioning().partition_count(),
        num_partitions,
    );

    Ok(sorted)
}

/// Mostly copy-pasted from the datafusion code;
///
/// 1. Repartition to N where N is number of threads
/// 2. Sort within partitions
/// 3. Write each partition
pub(crate) async fn write_batches(
    df: &DataFrame,
    key: &str,
    ctx: &SessionContext,
    path: &str,
) -> Result<()> {
    let task_ctx = ctx.task_ctx();
    let parsed = ListingTableUrl::parse(path)?;
    let object_store_ulr = parsed.object_store();
    let store = ctx.runtime_env().object_store(&object_store_ulr)?;

    let num_partitions = ctx.state().config().target_partitions();

    let plan = create_writer_plan(&df, key, num_partitions).await?;
    let mut join_set = JoinSet::new();

    for i in 0..num_partitions {
        let plan: Arc<dyn ExecutionPlan> = Arc::clone(&plan);
        let filename = format!("{}/part-{i}.parquet", parsed.prefix());
        let file = Path::parse(filename)?;
        let storeref = Arc::clone(&store);

        let buf_writer = BufWriter::with_capacity(
            storeref,
            file.clone(),
            task_ctx
                .session_config()
                .options()
                .execution
                .objectstore_writer_buffer_size,
        );
        let mut stream = plan.execute(i, Arc::clone(&task_ctx))?;
        join_set.spawn(async move {
            let mut writer = AsyncArrowWriter::try_new(buf_writer, plan.schema(), None)?;
            while let Some(next_batch) = stream.next().await {
                let batch = next_batch?;
                writer.write(&batch).await?;
            }
            writer
                .close()
                .await
                .map_err(DataFusionError::from)
                .map(|_| ())
        });
    }

    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(res) => res?,
            Err(e) => {
                if e.is_panic() {
                    std::panic::resume_unwind(e.into_panic());
                } else {
                    unreachable!();
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Reader: a `TableProvider` that declares the data is `Partitioning::Hash([key], N)`
// and sorted by `key`, so the SMJ can elide BOTH the per-partition sort and the
// hash repartition. Sortedness comes from `file_sort_order` (flowing through the
// delegated `eq_properties`); the distribution is injected by overriding
// `DataSource::output_partitioning` on a thin wrapper around the parquet scan.
// ============================================================================
#[derive(Debug)]
struct HashPartitionedSource {
    inner: Arc<dyn DataSource>,
    partitioning: Partitioning,
}

impl HashPartitionedSource {
    fn wrap(inner: Arc<dyn DataSource>, partitioning: Partitioning) -> Arc<dyn DataSource> {
        Arc::new(Self {
            inner,
            partitioning,
        })
    }
}

impl DataSource for HashPartitionedSource {
    fn open(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.inner.open(partition, context)
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.inner.fmt_as(t, f)
    }

    fn output_partitioning(&self) -> Partitioning {
        self.partitioning.clone()
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        self.inner.eq_properties()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        self.inner.partition_statistics(partition)
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        self.inner
            .with_fetch(limit)
            .map(|s| Self::wrap(s, self.partitioning.clone()))
    }

    fn fetch(&self) -> Option<usize> {
        self.inner.fetch()
    }

    fn try_swapping_with_projection(
        &self,
        _projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        Ok(None)
    }

    // ---- delegate the execution-path defaults (parquet morsel API) ----

    fn open_with_args(&self, args: OpenArgs) -> Result<SendableRecordBatchStream> {
        self.inner.open_with_args(args)
    }

    fn scheduling_type(&self) -> SchedulingType {
        self.inner.scheduling_type()
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.inner.metrics()
    }

    fn create_sibling_state(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.inner.create_sibling_state()
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        self.inner.try_pushdown_filters(filters, config)
    }

    fn with_new_state(&self, state: Arc<dyn Any + Send + Sync>) -> Option<Arc<dyn DataSource>> {
        self.inner
            .with_new_state(state)
            .map(|s| Self::wrap(s, self.partitioning.clone()))
    }
}

/// Extract the numeric bucket index from a `part-{i}.parquet` filename so files can
/// be re-grouped in partition order. Files that don't match sort last.
fn bucket_index(pf: &PartitionedFile) -> usize {
    pf.object_meta
        .location
        .filename()
        .and_then(|n| {
            n.strip_prefix("part-")
                .and_then(|s| s.strip_suffix(".parquet"))
        })
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

/// A `TableProvider` over a directory written by [`write_batches`], declaring the
/// data is hash-partitioned by `key` into `num_partitions` buckets and sorted by `key`.
#[derive(Debug)]
pub(crate) struct HashPartitionedTable {
    inner: Arc<ListingTable>,
    key: String,
    num_partitions: usize,
}

impl HashPartitionedTable {
    /// Build a provider for `uri` (a parquet directory written by [`write_batches`]).
    ///
    /// `num_partitions` MUST equal the `N` used at write time and the session's
    /// `target_partitions`, otherwise the declared co-partitioning is a lie.
    pub(crate) async fn try_new(
        ctx: &SessionContext,
        uri: &str,
        key: &str,
        num_partitions: usize,
    ) -> Result<Self> {
        let table_path = ListingTableUrl::parse(uri)?;
        let options = ListingOptions::new(Arc::new(ParquetFormat::new()))
            .with_file_sort_order(vec![vec![col(key).sort(true, true)]]);
        let state = ctx.state();
        let schema = options.infer_schema(&state, &table_path).await?;
        let config = ListingTableConfig::new(table_path)
            .with_listing_options(options)
            .with_schema(schema);
        let inner = Arc::new(ListingTable::try_new(config)?);
        Ok(Self {
            inner,
            key: key.to_string(),
            num_partitions,
        })
    }

    /// Read the directory back as a `DataFrame` whose scan declares
    /// `Partitioning::Hash([key], num_partitions)` + sortedness on `key`.
    pub(crate) async fn read_dataframe(self, ctx: &SessionContext) -> Result<DataFrame> {
        ctx.read_table(Arc::new(self))
    }
}

#[async_trait::async_trait]
impl TableProvider for HashPartitionedTable {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let base = self.inner.scan(state, projection, filters, limit).await?;

        let inner_source = base
            .downcast_ref::<DataSourceExec>()
            .ok_or_else(|| {
                DataFusionError::Internal("expected DataSourceExec from ListingTable scan".into())
            })?
            .data_source()
            .clone();

        // ListingTable groups files by statistics (for tiny buckets it puts all N
        // files into ONE group). That collapses the partition count below N and --
        // worse -- defeats `validated_output_ordering`, which only keeps the declared
        // ordering when a partition has a single file (the files are sorted within a
        // bucket, not globally across buckets). Re-split so each bucket
        // `part-i.parquet` is its own partition, in index order.
        let mut config = inner_source
            .downcast_ref::<FileScanConfig>()
            .ok_or_else(|| {
                DataFusionError::Internal("expected FileScanConfig from ListingTable".into())
            })?
            .clone();

        let mut files: Vec<PartitionedFile> = config
            .file_groups
            .iter()
            .flat_map(|g| g.files().iter().cloned())
            .collect();
        files.sort_by_key(bucket_index);
        config.file_groups = files
            .into_iter()
            .map(|pf| FileGroup::new(vec![pf]))
            .collect();

        // Resolve the key against the scan's *actual* output schema (after any
        // projection ListingTable applied). If the key was projected out (e.g. a
        // filtered `count` that only needs the activity column), we can't -- and
        // don't need to -- declare the partitioning; return the re-split scan as-is.
        let schema = base.schema();
        let source: Arc<dyn DataSource> = match Column::new_with_schema(&self.key, &schema) {
            Ok(key_col) => {
                let partitioning = Partitioning::Hash(
                    vec![Arc::new(key_col) as Arc<dyn PhysicalExpr>],
                    self.num_partitions,
                );
                HashPartitionedSource::wrap(Arc::new(config), partitioning)
            }
            Err(_) => Arc::new(config),
        };
        Ok(Arc::new(DataSourceExec::new(source)))
    }
}

#[cfg(test)]
mod tests {
    use super::{HashPartitionedTable, write_batches};
    use crate::utils::{assert_dataframes_equal, collect_to_i64};
    use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::hash_utils::create_hashes;
    use datafusion::physical_plan::repartition::REPARTITION_RANDOM_STATE;
    use datafusion::prelude::*;
    use std::collections::HashSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::id;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Unique temp dir (PID + counter) so parallel `cargo test` runs never collide.
    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("gf_hashpart_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// RAII: removes the temp dir on drop so tests stay self-contained.
    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn ctx_with_n(n: usize) -> SessionContext {
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(n))
    }

    /// Deterministic (key,val): `dup` rows per distinct key in `[0, distinct)`, val scattered
    /// so it is not correlated with key order. Built on `ctx` because `write_batches` plans via
    /// the DataFrame's own `SessionContext` (the `dataframe!` macro uses a throwaway one).
    async fn make_df(ctx: &SessionContext, distinct: i64, dup: usize) -> DataFrame {
        let total = (distinct as usize) * dup;
        let mut key = Vec::with_capacity(total);
        let mut val = Vec::with_capacity(total);
        for k in 0..distinct {
            for j in 0..dup {
                key.push(k);
                val.push((k * 7919 + (j as i64) * 31) % (distinct * 3 + 1));
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("val", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(key)),
                Arc::new(Int64Array::from(val)),
            ],
        )
        .expect("build batch");
        ctx.read_batch(batch).expect("read_batch")
    }

    fn file_uri(p: &Path) -> String {
        format!("file://{}", p.to_string_lossy())
    }

    /// Sorted list of `part-*.parquet` in a directory.
    fn list_parts(dir: &Path) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = fs::read_dir(dir)
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("parquet"))
            .collect();
        v.sort();
        v
    }

    async fn read_keys(ctx: &SessionContext, file: &Path) -> Vec<i64> {
        let df = ctx
            .read_parquet(&file_uri(file), ParquetReadOptions::default())
            .await
            .expect("read_parquet");
        collect_to_i64(&df, 0).await.expect("collect keys")
    }

    async fn read_df(ctx: &SessionContext, file: &Path) -> DataFrame {
        ctx.read_parquet(&file_uri(file), ParquetReadOptions::default())
            .await
            .expect("read_parquet")
    }

    async fn write(ctx: &SessionContext, df: &DataFrame, base: &Path, label: &str) -> PathBuf {
        let out = base.join(label);
        fs::create_dir_all(&out).expect("mkdir");
        write_batches(df, "key", ctx, &file_uri(&out))
            .await
            .expect("write_batches");
        out
    }

    /// Like [`make_df`] but with a caller-chosen key column name (for join tests where
    /// the two sides use different key names, e.g. `id` and `fk`).
    async fn make_keyed_df(
        ctx: &SessionContext,
        key_col: &str,
        val_col: &str,
        distinct: i64,
        dup: usize,
    ) -> DataFrame {
        let total = (distinct as usize) * dup;
        let mut key = Vec::with_capacity(total);
        let mut val = Vec::with_capacity(total);
        for k in 0..distinct {
            for j in 0..dup {
                key.push(k);
                val.push((k * 7919 + (j as i64) * 31) % (distinct * 3 + 1));
            }
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new(key_col, DataType::Int64, false),
            Field::new(val_col, DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(key)),
                Arc::new(Int64Array::from(val)),
            ],
        )
        .expect("build batch");
        ctx.read_batch(batch).expect("read_batch")
    }

    fn assert_sorted_ascending(keys: &[i64]) {
        let mut prev = None;
        for k in keys {
            if let Some(p) = prev {
                assert!(k >= &p, "not sorted ascending: {k} < {p}");
            }
            prev = Some(*k);
        }
    }

    /// Writes exactly N files and the union of all files equals the input (no loss/dup).
    #[tokio::test]
    async fn writes_n_files_and_roundtrips() {
        let n = 4;
        let ctx = ctx_with_n(n);
        let guard = TempGuard(unique_temp_dir("roundtrip"));
        let df = make_df(&ctx, 100, 2).await;

        let out = write(&ctx, &df, &guard.0, "out").await;
        let parts = list_parts(&out);
        assert_eq!(parts.len(), n, "expected exactly N part files");

        let mut back = read_df(&ctx, &parts[0]).await;
        for p in &parts[1..] {
            back = back.union_by_name(read_df(&ctx, p).await).expect("union");
        }
        let sort = vec![col("key").sort(true, true), col("val").sort(true, true)];
        let input_sorted = df.clone().sort(sort.clone()).expect("sort input");
        let back_sorted = back.sort(sort).expect("sort back");
        assert_dataframes_equal(input_sorted, back_sorted)
            .await
            .expect("round-trip data must be equal");
    }

    /// Each produced file is sorted ascending by the key.
    #[tokio::test]
    async fn each_file_sorted_by_key() {
        let n = 4;
        let ctx = ctx_with_n(n);
        let guard = TempGuard(unique_temp_dir("sorted"));
        let df = make_df(&ctx, 80, 3).await;
        let out = write(&ctx, &df, &guard.0, "out").await;

        for p in list_parts(&out) {
            assert_sorted_ascending(&read_keys(&ctx, &p).await);
        }
    }

    /// SMJ-correctness property (hash-agnostic): a given key value never appears in more
    /// than one file, and the union of per-file keys equals the input keys.
    #[tokio::test]
    async fn distinct_keys_are_partition_disjoint() {
        let n = 4;
        let ctx = ctx_with_n(n);
        let guard = TempGuard(unique_temp_dir("disjoint"));
        let df = make_df(&ctx, 120, 2).await;
        let out = write(&ctx, &df, &guard.0, "out").await;

        let mut seen: HashSet<i64> = HashSet::new();
        let mut sum_sizes = 0usize;
        for p in list_parts(&out) {
            let set: HashSet<i64> = read_keys(&ctx, &p).await.into_iter().collect();
            sum_sizes += set.len();
            for k in &set {
                assert!(seen.insert(*k), "key {k} appeared in more than one file");
            }
        }

        let input_keys: HashSet<i64> =
            collect_to_i64(&df.clone().select(vec![col("key")]).expect("select"), 0)
                .await
                .expect("collect input keys")
                .into_iter()
                .collect();
        assert_eq!(seen, input_keys, "union of file keys != input keys");
        assert_eq!(seen.len(), sum_sizes, "keys not disjoint across files");
    }

    /// Declaration invariant: every key in `part-i` satisfies `hash(key) % N == i`, using the
    /// *exact* hasher `RepartitionExec` routes by. N=3 (non-power-of-two) exercises the
    /// reciprocal-mod path.
    #[tokio::test]
    async fn hash_bucket_invariant_holds() {
        let n = 3;
        let ctx = ctx_with_n(n);
        let guard = TempGuard(unique_temp_dir("bucket"));
        let df = make_df(&ctx, 90, 2).await;
        let out = write(&ctx, &df, &guard.0, "out").await;
        let parts = list_parts(&out);
        assert_eq!(parts.len(), n);

        let rs = REPARTITION_RANDOM_STATE.random_state();
        for (i, p) in parts.iter().enumerate() {
            let keys = read_keys(&ctx, p).await;
            if keys.is_empty() {
                continue;
            }
            let arrays = vec![Arc::new(Int64Array::from(keys)) as ArrayRef];
            let mut buf = vec![0u64; arrays[0].len()];
            create_hashes(&arrays, rs, &mut buf).expect("create_hashes");
            for h in &buf {
                assert_eq!(
                    (*h % n as u64) as usize,
                    i,
                    "key in part-{i} has hash%{n} != {i}"
                );
            }
        }
    }

    /// N=1 degenerate case: one file, sorted, full set preserved.
    #[tokio::test]
    async fn single_partition() {
        let n = 1;
        let ctx = ctx_with_n(n);
        let guard = TempGuard(unique_temp_dir("single"));
        let df = make_df(&ctx, 50, 2).await;
        let out = write(&ctx, &df, &guard.0, "out").await;

        let parts = list_parts(&out);
        assert_eq!(parts.len(), 1);
        let keys = read_keys(&ctx, &parts[0]).await;
        assert_sorted_ascending(&keys);
        assert_eq!(keys.len(), df.clone().count().await.unwrap());
    }

    /// Empty input must still produce N (zero-row) files and not panic.
    #[tokio::test]
    async fn empty_input_produces_n_empty_files() {
        let n = 4;
        let ctx = ctx_with_n(n);
        let guard = TempGuard(unique_temp_dir("empty"));
        let df = make_df(&ctx, 0, 2).await; // 0 distinct keys -> 0 rows
        let out = write(&ctx, &df, &guard.0, "out").await;

        let parts = list_parts(&out);
        assert_eq!(parts.len(), n, "all N buckets must exist, even when empty");
        for p in &parts {
            assert_eq!(read_keys(&ctx, p).await.len(), 0);
        }
    }

    /// The headline property: two tables written co-partitioned by their join keys
    /// and read back via [`HashPartitionedTable`] feed a `SortMergeJoin` with NEITHER
    /// a `SortExec` NOR a `RepartitionExec(Hash)` above the inputs.
    #[tokio::test]
    async fn reader_skips_sort_and_repartition_for_smj() {
        let n = 4;
        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(n)
                .set_bool("datafusion.optimizer.prefer_hash_join", false),
        );
        let guard = TempGuard(unique_temp_dir("smj"));

        let left_df = make_keyed_df(&ctx, "id", "lv", 60, 2).await;
        let right_df = make_keyed_df(&ctx, "fk", "rv", 60, 2).await;

        let left_dir = guard.0.join("left");
        let right_dir = guard.0.join("right");
        fs::create_dir_all(&left_dir).unwrap();
        fs::create_dir_all(&right_dir).unwrap();
        write_batches(&left_df, "id", &ctx, &file_uri(&left_dir))
            .await
            .expect("write left");
        write_batches(&right_df, "fk", &ctx, &file_uri(&right_dir))
            .await
            .expect("write right");

        let left = HashPartitionedTable::try_new(&ctx, &file_uri(&left_dir), "id", n)
            .await
            .expect("left table")
            .read_dataframe(&ctx)
            .await
            .expect("read left");
        let right = HashPartitionedTable::try_new(&ctx, &file_uri(&right_dir), "fk", n)
            .await
            .expect("right table")
            .read_dataframe(&ctx)
            .await
            .expect("read right");

        let joined = left
            .join(right, JoinType::Inner, &["id"], &["fk"], None)
            .expect("join");
        let plan = joined.create_physical_plan().await.expect("physical plan");
        let rendered = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(true)
            .to_string();

        println!("SMJ PLAN:\n{rendered}");

        assert!(
            rendered.contains("SortMergeJoinExec"),
            "expected a SortMergeJoin; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("RepartitionExec"),
            "expected NO RepartitionExec (co-partitioning not recognised); got:\n{rendered}"
        );
        assert!(
            !rendered.contains("SortExec"),
            "expected NO SortExec (sortedness not recognised); got:\n{rendered}"
        );
    }
}
