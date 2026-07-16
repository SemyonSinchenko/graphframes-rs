use datafusion::common::plan_err;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::path::Path;
use datafusion::prelude::{DataFrame, ParquetReadOptions, SessionContext};
use futures::{StreamExt, TryStreamExt};
use std::collections::VecDeque;
use url::Url;

#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub store_url: ObjectStoreUrl,
    pub dir: Path,
}

impl CheckpointConfig {
    pub(crate) fn default_local_fs() -> Self {
        let url = ObjectStoreUrl::local_filesystem();
        // This is just a default
        let dir = Path::from("_graphframes_checkpoints");

        Self {
            store_url: url,
            dir: dir,
        }
    }

    /// Helper method: to avoid collision I should restrict overlapping
    /// of checkpoint_dir and output if they are on the same store.
    pub(crate) fn validate_output(&self, output: &str) -> Result<()> {
        let output_url = Url::parse(output).map_err(|e| {
            DataFusionError::Configuration(format!("invalid output URL '{output}': {e}"))
        })?;
        let same_store = self.store_url.as_str().trim_end_matches('/')
            == format!(
                "{}://{}/",
                output_url.scheme(),
                output_url.host_str().unwrap_or("")
            )
            .trim_end_matches('/');

        if !same_store {
            return Ok(());
        }

        let output_path = Path::from_url_path(output_url.path())?;
        if output_path == self.dir
            || self.dir.prefix_matches(&output_path)
            || output_path.prefix_matches(&self.dir)
        {
            let chk_dir_string = self.dir.to_string();
            plan_err!(
                "output dir and checkpoint dir must not overlap: output={output_path}, checkpoint={chk_dir_string}"
            )
        } else {
            Ok(())
        }
    }
}

pub(crate) struct ParquetCheckpointer {
    store_url: ObjectStoreUrl,
    base: Path,
    stored: VecDeque<Path>,
}

impl ParquetCheckpointer {
    pub(crate) fn new(store_url: ObjectStoreUrl, base: Path) -> Self {
        Self {
            store_url,
            base,
            stored: VecDeque::new(),
        }
    }

    /// Write `df` to parquet and read it straight back. No sorting, no declared
    /// partitioning -- use this when the SMJ savings don't justify the write-time
    /// sort + repartition cost.
    pub(crate) async fn push(
        &mut self,
        ctx: &SessionContext,
        postfix: &str,
        df: DataFrame,
    ) -> Result<DataFrame> {
        let dir = self.base.clone().join(postfix);
        let uri = format!("{}{}/", self.store_url.as_str(), dir);

        df.clone()
            .write_parquet(&uri, DataFrameWriteOptions::new(), None)
            .await?;

        // Empty dataframe is a valid case; only register/read back if files exist.
        let store = ctx.runtime_env().object_store(&self.store_url)?;
        let wrote_any = store.list(Some(&dir)).next().await.is_some();

        if wrote_any {
            self.stored.push_back(dir);
            ctx.read_parquet(&uri, ParquetReadOptions::default()).await
        } else {
            Ok(df.clone())
        }
    }

    /// Write `df` already hash-partitioned by `key` into `target_partitions` files
    /// (each sorted by `key`) and read it back declaring `Partitioning::Hash([key], N)`
    /// + sortedness, so a downstream `SortMergeJoin` on `key` skips BOTH the
    /// per-partition sort and the hash repartition.
    ///
    /// The session's `target_partitions` must stay equal to this `N` for the
    /// lifetime of the returned `DataFrame`, otherwise the optimizer inserts a
    /// repartition to raise/lower parallelism.
    pub(crate) async fn push_pre_sorted(
        &mut self,
        ctx: &SessionContext,
        postfix: &str,
        df: DataFrame,
        key: &str,
    ) -> Result<DataFrame> {
        let dir = self.base.clone().join(postfix);
        let uri = format!("{}{}/", self.store_url.as_str(), dir);
        let num_partitions = ctx.state().config().target_partitions();

        super::hash_partitioned::write_batches(&df, key, ctx, &uri).await?;

        self.stored.push_back(dir);
        let table =
            super::hash_partitioned::HashPartitionedTable::try_new(ctx, &uri, key, num_partitions)
                .await?;
        table.read_dataframe(ctx).await
    }

    pub(crate) async fn evict(&mut self, ctx: &SessionContext, n: usize) -> Result<()> {
        let store = ctx.runtime_env().object_store(&self.store_url)?;
        let to_remove = n.min(self.stored.len());
        for _ in 0..to_remove {
            let dir = self.stored.pop_front().unwrap();
            let paths = store.list(Some(&dir)).map_ok(|m| m.location).boxed();
            store.delete_stream(paths).try_collect::<Vec<_>>().await?;
        }
        Ok(())
    }

    pub(crate) async fn evict_all_but_latest_n(
        &mut self,
        ctx: &SessionContext,
        n: usize,
    ) -> Result<()> {
        self.evict(ctx, self.stored.len().saturating_sub(n)).await
    }

    pub(crate) async fn remove_last(&mut self, ctx: &SessionContext, n: usize) -> Result<()> {
        let store = ctx.runtime_env().object_store(&self.store_url)?;
        let to_remove = n.min(self.stored.len());
        for _ in 0..to_remove {
            let dir = self.stored.pop_back().unwrap();
            let paths = store.list(Some(&dir)).map_ok(|m| m.location).boxed();
            store.delete_stream(paths).try_collect::<Vec<_>>().await?;
        }
        Ok(())
    }

    pub(crate) async fn purge(&mut self, ctx: &SessionContext) -> Result<()> {
        self.evict(ctx, self.stored.len()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::assert_dataframes_equal;
    use datafusion::arrow::array::{Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::displayable;
    use datafusion::prelude::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use url::Url;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Returns a unique directory under `std::env::temp_dir()` for this test run, creating it.
    /// Combining the PID with a process-wide counter guarantees uniqueness across parallel
    /// `cargo test` invocations and concurrent tests.
    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("graphframes_ckptr_test_{}_{n}_{label}", id()));
        fs::create_dir_all(&dir).expect("failed to create unique temp dir");
        dir
    }

    /// RAII guard that recursively removes the temp directory when dropped, so tests stay
    /// self-contained without depending on the `tempfile` crate.
    struct TempGuard(PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Builds a `SessionContext`, a `ParquetCheckpointer` rooted at a unique temp directory,
    /// and a `TempGuard` that cleans up on drop. The temp dir is converted to a `file://` `Url`
    /// exactly as described in the task.
    fn setup(label: &str) -> Result<(SessionContext, ParquetCheckpointer, TempGuard)> {
        let dir = unique_temp_dir(label);
        let _file_url =
            Url::from_file_path(&dir).expect("temp dir must be convertible to file:// URL");

        // DataFusion's default `SessionContext` registers a `LocalFileSystem` at `file:///`.
        let store_url = ObjectStoreUrl::parse("file://")?;
        let base = Path::from_filesystem_path(&dir)
            .expect("temp dir must be convertible to object_store path");

        let ctx = SessionContext::new();
        let ckptr = ParquetCheckpointer::new(store_url, base);
        Ok((ctx, ckptr, TempGuard(dir)))
    }

    /// A small, deterministic DataFrame used across tests.
    fn dummy_df(_ctx: &SessionContext) -> Result<DataFrame> {
        Ok(dataframe!(
            "id" => vec![1i64, 2, 3],
            "name" => vec!["a", "b", "c"],
        )?)
    }

    /// Returns the absolute filesystem path for a given checkpoint postfix under `base`.
    fn ckpt_path(base: &std::path::Path, postfix: &str) -> PathBuf {
        base.join(postfix)
    }

    /// Returns true if a checkpoint directory exists on disk AND contains at least one
    /// (non-hidden) data file. `object_store::LocalFileSystem::delete` removes the part files
    /// but leaves the now-empty directory behind, so `Path::exists()` alone is not a reliable
    /// signal that checkpoint data is present.
    fn ckpt_data_present(base: &std::path::Path, postfix: &str) -> bool {
        fs::read_dir(ckpt_path(base, postfix))
            .map(|it| {
                it.filter_map(core::result::Result::ok)
                    .any(|e| e.file_name() != ".DS_Store")
            })
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_push_roundtrip_preserves_data() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("push_roundtrip")?;
        let base = _guard.0.clone();

        let original = dummy_df(&ctx)?;
        let restored = ckptr.push(&ctx, "c1", original.clone()).await?;

        assert_dataframes_equal(original, restored).await?;
        assert!(
            ckpt_path(&base, "c1").exists(),
            "checkpoint dir should exist"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_push_tracks_multiple_checkpoints() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("push_multiple")?;
        let base = _guard.0.clone();

        for postfix in ["c1", "c2", "c3"] {
            let df = dummy_df(&ctx)?;
            ckptr.push(&ctx, postfix, df).await?;
        }

        for postfix in ["c1", "c2", "c3"] {
            assert!(
                ckpt_path(&base, postfix).exists(),
                "checkpoint {postfix} should exist on disk"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_evict_removes_oldest_only() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("evict_oldest")?;
        let base = _guard.0.clone();

        for postfix in ["c1", "c2", "c3"] {
            let df = dummy_df(&ctx)?;
            ckptr.push(&ctx, postfix, df).await?;
        }

        ckptr.evict(&ctx, 1).await?;

        assert!(
            !ckpt_data_present(&base, "c1"),
            "oldest checkpoint should have been evicted"
        );
        assert!(
            ckpt_data_present(&base, "c2"),
            "non-evicted checkpoint should remain"
        );
        assert!(
            ckpt_data_present(&base, "c3"),
            "non-evicted checkpoint should remain"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_evict_more_than_stored_is_clamped() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("evict_clamp")?;
        let base = _guard.0.clone();

        for postfix in ["c1", "c2"] {
            let df = dummy_df(&ctx)?;
            ckptr.push(&ctx, postfix, df).await?;
        }

        // Requesting 10 evictions with only 2 stored must not panic and should remove all.
        ckptr.evict(&ctx, 10).await?;

        assert!(!ckpt_data_present(&base, "c1"));
        assert!(!ckpt_data_present(&base, "c2"));
        Ok(())
    }

    #[tokio::test]
    async fn test_evict_on_empty_is_noop() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("evict_empty")?;
        // Fresh checkpointer has nothing stored; evict should be a clean no-op.
        ckptr.evict(&ctx, 3).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_purge_removes_all_checkpoints() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("purge_all")?;
        let base = _guard.0.clone();

        for postfix in ["c1", "c2", "c3"] {
            let df = dummy_df(&ctx)?;
            ckptr.push(&ctx, postfix, df).await?;
        }

        ckptr.purge(&ctx).await?;

        for postfix in ["c1", "c2", "c3"] {
            assert!(
                !ckpt_data_present(&base, postfix),
                "checkpoint {postfix} should have been purged"
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_purge_on_empty_is_noop() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("purge_empty")?;
        ckptr.purge(&ctx).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_push_after_evict_continues_working() -> Result<()> {
        let (ctx, mut ckptr, _guard) = setup("push_after_evict")?;
        let base = _guard.0.clone();

        let df = dummy_df(&ctx)?;
        ckptr.push(&ctx, "c1", df).await?;
        let df = dummy_df(&ctx)?;
        ckptr.push(&ctx, "c2", df).await?;

        ckptr.evict(&ctx, 1).await?;
        assert!(!ckpt_data_present(&base, "c1"));
        assert!(ckpt_data_present(&base, "c2"));

        // Pushing after a partial evict must still work and yield a usable DataFrame.
        let original = dummy_df(&ctx)?;
        let restored = ckptr.push(&ctx, "c3", original.clone()).await?;
        assert_dataframes_equal(original, restored).await?;
        assert!(ckpt_data_present(&base, "c3"));

        // The internal VecDeque state must remain consistent: purge should clean up everything
        // that is currently tracked (c2, c3 — c1 was already evicted).
        ckptr.purge(&ctx).await?;
        assert!(!ckpt_data_present(&base, "c2"));
        assert!(!ckpt_data_present(&base, "c3"));
        Ok(())
    }

    /// Build a small `(key_col, "v")` Int64 DataFrame bound to `ctx`, so the writer
    /// plans/reads on the same `target_partitions` as the test context.
    async fn keyed_df(
        ctx: &SessionContext,
        key_col: &str,
        val_col: &str,
        keys: &[i64],
        vals: &[i64],
    ) -> DataFrame {
        let schema = Arc::new(Schema::new(vec![
            Field::new(key_col, DataType::Int64, false),
            Field::new(val_col, DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(keys.to_vec())),
                Arc::new(Int64Array::from(vals.to_vec())),
            ],
        )
        .expect("build batch");
        ctx.read_batch(batch).expect("read_batch")
    }

    /// Two tables written via [`ParquetCheckpointer::push_pre_sorted`] and joined on their
    /// co-partitioned keys must feed a `SortMergeJoin` with NEITHER a `SortExec` NOR a
    /// `RepartitionExec` above the inputs.
    #[tokio::test]
    async fn test_push_pre_sorted_join_skips_sort_and_repartition() -> Result<()> {
        let n = 4;
        let dir = unique_temp_dir("presort_smj");
        let store_url = ObjectStoreUrl::parse("file://")?;
        let base = Path::from_filesystem_path(&dir).expect("temp dir -> object_store path");
        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .with_target_partitions(n)
                .set_bool("datafusion.optimizer.prefer_hash_join", false),
        );

        let keys: Vec<i64> = (1..=12).collect();
        let vals: Vec<i64> = keys.iter().map(|k| k * 10).collect();
        let left_df = keyed_df(&ctx, "id", "lv", &keys, &vals).await;
        let right_df = keyed_df(&ctx, "fk", "rv", &keys, &vec![1i64; keys.len()]).await;

        let mut left_ckptr = ParquetCheckpointer::new(store_url.clone(), base.clone());
        let mut right_ckptr = ParquetCheckpointer::new(store_url.clone(), base.clone());

        let left = left_ckptr
            .push_pre_sorted(&ctx, "left", left_df, "id")
            .await?;
        let right = right_ckptr
            .push_pre_sorted(&ctx, "right", right_df, "fk")
            .await?;

        let joined = left.join(right, JoinType::Inner, &["id"], &["fk"], None)?;
        let plan = joined.create_physical_plan().await?;
        let rendered = displayable(plan.as_ref()).indent(true).to_string();
        println!("PUSH_PRE_SORTED SMJ PLAN:\n{rendered}");

        assert!(
            rendered.contains("SortMergeJoinExec"),
            "expected SMJ; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("RepartitionExec"),
            "expected NO repartition (co-partitioning not recognised):\n{rendered}"
        );
        assert!(
            !rendered.contains("SortExec"),
            "expected NO sort (sortedness not recognised):\n{rendered}"
        );

        let _ = TempGuard(dir); // cleanup
        Ok(())
    }
}
