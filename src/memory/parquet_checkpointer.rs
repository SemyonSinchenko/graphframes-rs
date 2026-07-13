use datafusion::common::plan_err;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::SortExpr;
use datafusion::object_store::path::Path;
use datafusion::prelude::{DataFrame, ParquetReadOptions, SessionContext, col};
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

    pub(crate) async fn push(
        &mut self,
        ctx: &SessionContext,
        postfix: &str,
        df: DataFrame,
        sort_by: Option<String>,
    ) -> Result<DataFrame> {
        let dir = self.base.clone().join(postfix);
        let uri = format!("{}{}/", self.store_url.as_str(), dir);

        let options = match sort_by.clone() {
            None => DataFrameWriteOptions::new(),
            Some(col_name) => DataFrameWriteOptions::new().with_sort_by(vec![SortExpr::new(
                col(col_name),
                true,
                true,
            )]),
        };

        df.clone().write_parquet(&uri, options, None).await?;

        // Empty dataframe is a valid case here,
        // so we should check that there are files
        // before an attempt to read back.
        let store = ctx.runtime_env().object_store(&self.store_url)?;
        let wrote_any = store.list(Some(&dir)).next().await.is_some();

        // This check is potentially dangereous:
        // on slow remote object stores list-after-write
        // may be empty and this may break the performance.
        //
        // It should not be a problem with modern implementations,
        // so I'm not going to fix it right now.
        if wrote_any {
            self.stored.push_back(dir);
            let options = match sort_by.clone() {
                None => ParquetReadOptions::default(),
                Some(col_name) => ParquetReadOptions::new()
                    .file_sort_order(vec![vec![SortExpr::new(col(col_name), true, true)]]),
            };
            ctx.read_parquet(&uri, options).await
        } else {
            Ok(df.clone())
        }
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
    use datafusion::prelude::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::id;
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
        let restored = ckptr.push(&ctx, "c1", original.clone(), None).await?;

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
            ckptr.push(&ctx, postfix, df, None).await?;
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
            ckptr.push(&ctx, postfix, df, None).await?;
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
            ckptr.push(&ctx, postfix, df, None).await?;
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
            ckptr.push(&ctx, postfix, df, None).await?;
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
        ckptr.push(&ctx, "c1", df, None).await?;
        let df = dummy_df(&ctx)?;
        ckptr.push(&ctx, "c2", df, None).await?;

        ckptr.evict(&ctx, 1).await?;
        assert!(!ckpt_data_present(&base, "c1"));
        assert!(ckpt_data_present(&base, "c2"));

        // Pushing after a partial evict must still work and yield a usable DataFrame.
        let original = dummy_df(&ctx)?;
        let restored = ckptr.push(&ctx, "c3", original.clone(), None).await?;
        assert_dataframes_equal(original, restored).await?;
        assert!(ckpt_data_present(&base, "c3"));

        // The internal VecDeque state must remain consistent: purge should clean up everything
        // that is currently tracked (c2, c3 — c1 was already evicted).
        ckptr.purge(&ctx).await?;
        assert!(!ckpt_data_present(&base, "c2"));
        assert!(!ckpt_data_present(&base, "c3"));
        Ok(())
    }
}
