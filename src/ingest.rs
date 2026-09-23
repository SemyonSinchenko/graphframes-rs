use datafusion::arrow::array::{Array, ArrayRef, Int64Builder, RecordBatch};
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::object_store::buffered::BufWriter;
use datafusion::parquet::arrow::AsyncArrowWriter;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::{
    arrow::datatypes::{DataType, Field, Schema},
    common::{JoinType, runtime::JoinSet},
    datasource::listing::ListingTableUrl,
    error::{DataFusionError, Result},
    object_store::path::Path,
    physical_plan::ExecutionPlanProperties,
    prelude::*,
};
use futures::StreamExt;
use std::sync::Arc;
use url::Url;

use crate::{EDGE_DST, EDGE_SRC, VERTEX_ID, memory::CheckpointConfig};

pub const ORIGIN_ID: &str = "origin_id";

pub struct IngestResult {
    /// Path of the written vertices dataset (`<output>/vertices`).
    pub vertices: String,
    /// Path of the written edges dataset (`<output>/edges`).
    pub edges: String,
    /// Number of vertices written.
    pub num_vertices: usize,
    /// Number of edges written.
    pub num_edges: usize,
}

pub async fn from_string_ids(
    ctx: &SessionContext,
    vertices: DataFrame,
    edges: DataFrame,
    output: &str,
    checkpoint_config: Option<CheckpointConfig>,
) -> Result<IngestResult> {
    // `validate_output` compares URL strings, so a plain local path must be
    // normalized to a `file://` URL first.
    let output_uri = to_output_uri(output)?;
    let resolved_config = checkpoint_config.unwrap_or_else(|| CheckpointConfig::default_local_fs());
    resolved_config.validate_output(&output_uri)?;

    if vertices
        .schema()
        .has_column_with_unqualified_name(ORIGIN_ID)
    {
        return Err(DataFusionError::Plan(
            "origin_id column is reserved".to_string(),
        ));
    }

    fn is_str(dt: &DataType) -> bool {
        dt.equals_datatype(&DataType::Utf8) || dt.equals_datatype(&DataType::Utf8View)
    }

    let orig_id_col = vertices.schema().field_with_unqualified_name(VERTEX_ID)?;
    if !is_str(orig_id_col.data_type()) {
        return Err(DataFusionError::Plan(format!(
            "expected id column data type Utf8 but got {}",
            orig_id_col.data_type()
        )));
    }

    let src_id = edges.schema().field_with_unqualified_name(EDGE_SRC)?;
    let dst_id = edges.schema().field_with_unqualified_name(EDGE_DST)?;

    if !(is_str(src_id.data_type()) && is_str(dst_id.data_type())) {
        return Err(DataFusionError::Plan(format!(
            "expected src, dst column data type Utf8 but got src={} and dst={}",
            src_id.data_type(),
            dst_id.data_type(),
        )));
    }

    // `write_edges` joins on this name; a collision would make it ambiguous.
    if edges.schema().has_column_with_unqualified_name(ORIGIN_ID) {
        return Err(DataFusionError::Plan(
            "origin_id is a reserved column name in edges".to_string(),
        ));
    }

    let vertices_path = format!("{output}/vertices");
    let edges_path = format!("{output}/edges");

    write_vertices(&vertices, ctx, &vertices_path).await?;
    let num_vertices = ctx
        .read_parquet(&format!("{vertices_path}/"), ParquetReadOptions::new())
        .await?
        .count()
        .await?;

    // The inner joins below drop edges whose endpoints are missing from the
    // vertex table; the row-count comparison turns that silent loss into an
    // error.
    let num_edges = edges.clone().count().await?;
    let written_edges = if num_edges == 0 {
        ensure_local_dir(&edges_path)?;
        0
    } else {
        write_edges(ctx, &edges, &vertices_path, &edges_path).await?;
        ctx.read_parquet(&format!("{edges_path}/"), ParquetReadOptions::new())
            .await?
            .count()
            .await?
    };

    if written_edges != num_edges {
        return Err(DataFusionError::Plan(format!(
            "dangling edge endpoints: {num_edges} input edges, but only {written_edges}              have both endpoints present in the vertex table"
        )));
    }

    Ok(IngestResult {
        vertices: vertices_path,
        edges: edges_path,
        num_vertices,
        num_edges: written_edges,
    })
}

async fn write_vertices(df: &DataFrame, ctx: &SessionContext, path: &str) -> Result<()> {
    ensure_local_dir(path)?;

    let frozen_plan = df.clone().create_physical_plan().await?;
    let partitions = frozen_plan.output_partitioning().partition_count();
    // 31 bits of the stamped id encode the partition index; bit 63 must stay
    // clear so the generated Int64 ids are non-negative.
    if partitions >= (1 << 30) {
        return Err(DataFusionError::Plan(format!(
            "cannot stamp ids: {partitions} partitions exceed the 31-bit id budget; \
             reduce target_partitions (--num-workers)"
        )));
    }

    let task_ctx = ctx.task_ctx();
    let parsed = ListingTableUrl::parse(path)?;
    let object_store_ulr = parsed.object_store();
    let store = ctx.runtime_env().object_store(&object_store_ulr)?;

    let mut join_set = JoinSet::new();

    let id_col_index = frozen_plan.schema().index_of(VERTEX_ID)?;
    let new_schema = {
        let old_fields = frozen_plan.schema().fields.clone();
        let mut new_fields: Vec<Arc<Field>> = Vec::new();

        for i in 0..old_fields.len() {
            let f = if i == id_col_index {
                Arc::new(Field::new(VERTEX_ID, DataType::Int64, false))
            } else {
                old_fields[i].clone()
            };

            new_fields.push(f);
        }
        new_fields.push(Arc::new(Field::new(
            ORIGIN_ID,
            frozen_plan.schema().field(id_col_index).data_type().clone(),
            false,
        )));

        Arc::new(Schema::new(new_fields))
    };
    let num_cols = frozen_plan.schema().fields.len();

    for i in 0..partitions {
        let plan: Arc<dyn ExecutionPlan> = Arc::clone(&frozen_plan);
        let task_schema = Arc::clone(&new_schema);
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
            let mut writer = AsyncArrowWriter::try_new(buf_writer, task_schema.clone(), None)?;
            let mut j = 0usize;
            while let Some(next_batch) = stream.next().await {
                let batch = next_batch?;
                let id_col = batch.column(id_col_index);
                let mut id_builder = Int64Builder::with_capacity(id_col.len());
                for _ in 0..id_col.len() {
                    id_builder.append_value(((i as i64) << 33) + j as i64);
                    j += 1;
                }

                let new_id = id_builder.finish();
                let mut columns: Vec<Arc<dyn Array>> = (0..num_cols)
                    .map(|jj| {
                        if jj == id_col_index {
                            Arc::new(new_id.clone()) as ArrayRef
                        } else {
                            batch.column(jj).clone()
                        }
                    })
                    .collect();
                // we already have +1 field in the schema
                columns.push(batch.column(id_col_index).clone());

                let new_batch = RecordBatch::try_new(task_schema.clone(), columns)?;

                writer.write(&new_batch).await?;
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

const REMAP_SRC: &str = "__ingest_src_id";
const REMAP_DST: &str = "__ingest_dst_id";

/// Remaps the string `src`/`dst` of `edges` to the stamped Int64 ids by two
/// inner joins against the vertex dataset written by [`write_vertices`], and
/// writes the result as a parquet dataset at `path`.
async fn write_edges(
    ctx: &SessionContext,
    edges: &DataFrame,
    vertices_path: &str,
    path: &str,
) -> Result<()> {
    ensure_local_dir(path)?;

    let map = ctx
        .read_parquet(&format!("{vertices_path}/"), ParquetReadOptions::new())
        .await?
        .select(vec![col(ORIGIN_ID), col(VERTEX_ID)])?;

    let rest: Vec<String> = edges
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .filter(|n| n.as_str() != EDGE_SRC && n.as_str() != EDGE_DST)
        .collect();

    // join 1: src -> id
    let map_src = map
        .clone()
        .select(vec![col(ORIGIN_ID), col(VERTEX_ID).alias(REMAP_SRC)])?;
    let with_src = edges
        .clone()
        .join_on(map_src, JoinType::Inner, [col(EDGE_SRC).eq(col(ORIGIN_ID))])?
        .select(
            std::iter::once(col(REMAP_SRC).alias(EDGE_SRC))
                .chain(std::iter::once(col(EDGE_DST)))
                .chain(rest.iter().map(|n| col(n.as_str())))
                .collect::<Vec<Expr>>(),
        )?;

    // join 2: dst -> id
    let map_dst = map.select(vec![col(ORIGIN_ID), col(VERTEX_ID).alias(REMAP_DST)])?;
    let remapped = with_src
        .join_on(map_dst, JoinType::Inner, [col(EDGE_DST).eq(col(ORIGIN_ID))])?
        .select(
            std::iter::once(col(EDGE_SRC))
                .chain(std::iter::once(col(REMAP_DST).alias(EDGE_DST)))
                .chain(rest.iter().map(|n| col(n.as_str())))
                .collect::<Vec<Expr>>(),
        )?;

    remapped
        .write_parquet(&format!("{path}/"), DataFrameWriteOptions::new(), None)
        .await?;

    Ok(())
}

/// Creates the directory `path` refers to. Accepts plain local paths and
/// `file://` URLs; any other scheme is rejected (ingest writes to the local
/// filesystem only).
fn ensure_local_dir(path: &str) -> Result<()> {
    let local = if let Ok(url) = Url::parse(path) {
        if url.scheme() == "file" {
            url.to_file_path()
                .map_err(|_| DataFusionError::Plan(format!("invalid file URL '{path}'")))?
        } else {
            return Err(DataFusionError::Plan(format!(
                "ingest writes only to local paths or file:// URLs, got '{path}'"
            )));
        }
    } else {
        std::path::PathBuf::from(path)
    };
    std::fs::create_dir_all(local).map_err(DataFusionError::IoError)
}

/// Normalizes `output` to a URL so [`CheckpointConfig::validate_output`] can
/// compare it against the checkpoint dir. Plain paths become `file://` URLs.
fn to_output_uri(output: &str) -> Result<String> {
    if Url::parse(output).is_ok() {
        return Ok(output.to_string());
    }
    let path = std::path::Path::new(output);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(DataFusionError::IoError)?
            .join(path)
    };
    Url::from_file_path(path)
        .map(|u| u.to_string())
        .map_err(|_| DataFusionError::Plan(format!("cannot convert '{output}' to a file URL")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Float64Array, Int64Array, StringArray, StringViewArray};
    use datafusion::arrow::compute::cast;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::collections::{HashMap, HashSet};

    fn tmpdir(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!("gf_ingest_{tag}_{}", uuid::Uuid::new_v4()))
            .to_str()
            .unwrap()
            .to_string()
    }

    fn vertex_df(ctx: &SessionContext, ids: &[&str]) -> Result<DataFrame> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(VERTEX_ID, DataType::Utf8, false),
            Field::new("score", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids.to_vec())),
                Arc::new(Float64Array::from(vec![1.0; ids.len()])),
            ],
        )?;
        ctx.read_batches(vec![batch])
    }

    fn edge_df(ctx: &SessionContext, pairs: &[(&str, &str)]) -> Result<DataFrame> {
        let schema = Arc::new(Schema::new(vec![
            Field::new(EDGE_SRC, DataType::Utf8, false),
            Field::new(EDGE_DST, DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(
                    pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
                )),
            ],
        )?;
        ctx.read_batches(vec![batch])
    }

    /// Reads a string-ish column back as `Vec<Option<String>>`, whatever string
    /// view type parquet round-tripped it into.
    fn string_values(col: &ArrayRef) -> Vec<Option<String>> {
        let utf8 = cast(col, &DataType::Utf8).unwrap();
        utf8.as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .iter()
            .map(|v| v.map(|s| s.to_string()))
            .collect()
    }

    fn int_values(col: &ArrayRef) -> Vec<i64> {
        col.as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .map(|v| v.unwrap())
            .collect()
    }

    #[tokio::test]
    async fn test_write_vertices_roundtrip() -> Result<()> {
        let ctx = SessionContext::new();
        let df = vertex_df(&ctx, &["a", "b", "c", "d"])?;
        let dir = tmpdir("roundtrip");

        write_vertices(&df, &ctx, &dir).await?;

        let read = ctx
            .read_parquet(&format!("{dir}/"), ParquetReadOptions::new())
            .await?;
        let schema = read.schema();
        let id_field = schema.field_with_unqualified_name(VERTEX_ID)?;
        assert_eq!(id_field.data_type(), &DataType::Int64);
        assert!(!id_field.is_nullable());
        assert!(schema.has_column_with_unqualified_name("score"));
        assert!(schema.has_column_with_unqualified_name(ORIGIN_ID));

        let mut ids = Vec::new();
        let mut origins = Vec::new();
        for b in read.collect().await? {
            ids.extend(int_values(b.column_by_name(VERTEX_ID).unwrap()));
            origins.extend(string_values(b.column_by_name(ORIGIN_ID).unwrap()));
        }
        assert_eq!(
            origins,
            vec![
                Some("a".to_string()),
                Some("b".to_string()),
                Some("c".to_string()),
                Some("d".to_string())
            ]
        );
        let unique: HashSet<i64> = ids.into_iter().collect();
        assert_eq!(unique.len(), 4);

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[tokio::test]
    async fn test_write_vertices_multi_partition_unique_ids() -> Result<()> {
        let ctx = SessionContext::new();
        let dir = tmpdir("multipart");
        let in_dir = format!("{dir}/in");
        std::fs::create_dir_all(&in_dir)?;

        for (k, id) in ["a", "b", "c", "d", "e", "f", "g", "h"].iter().enumerate() {
            let one = vertex_df(&ctx, &[id])?;
            one.write_parquet(
                &format!("{in_dir}/p{k}.parquet"),
                DataFrameWriteOptions::new(),
                None,
            )
            .await?;
        }

        let df = ctx
            .read_parquet(&format!("{in_dir}/"), ParquetReadOptions::new())
            .await?;
        let out = format!("{dir}/out");
        write_vertices(&df, &ctx, &out).await?;

        let read = ctx
            .read_parquet(&format!("{out}/"), ParquetReadOptions::new())
            .await?;
        let mut ids = HashSet::new();
        let mut rows = 0usize;
        for b in read.collect().await? {
            ids.extend(int_values(b.column_by_name(VERTEX_ID).unwrap()));
            rows += b.num_rows();
        }
        assert_eq!(rows, 8);
        assert_eq!(ids.len(), 8, "ids must be unique across partitions");

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[tokio::test]
    async fn test_write_vertices_utf8_view_ids() -> Result<()> {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![Field::new(
            VERTEX_ID,
            DataType::Utf8View,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringViewArray::from(vec!["a", "b"]))],
        )?;
        let df = ctx.read_batches(vec![batch])?;
        let dir = tmpdir("view");

        write_vertices(&df, &ctx, &dir).await?;

        let read = ctx
            .read_parquet(&format!("{dir}/"), ParquetReadOptions::new())
            .await?;
        let origin_type = read
            .schema()
            .field_with_unqualified_name(ORIGIN_ID)?
            .data_type()
            .clone();
        assert!(
            matches!(origin_type, DataType::Utf8 | DataType::Utf8View),
            "origin_id must stay a string type, got {origin_type}"
        );
        let rows: usize = read.collect().await?.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2);

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    #[tokio::test]
    async fn test_from_string_ids_end_to_end() -> Result<()> {
        let ctx = SessionContext::new();
        let v = vertex_df(&ctx, &["a", "b", "c", "d"])?;
        let e = edge_df(&ctx, &[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])?;
        let out = tmpdir("e2e");

        let res = from_string_ids(&ctx, v, e, &out, None).await?;
        assert_eq!(res.num_vertices, 4);
        assert_eq!(res.num_edges, 4);

        let vertices_path = res.vertices.clone();
        let vdf = ctx
            .read_parquet(&format!("{vertices_path}/"), ParquetReadOptions::new())
            .await?;
        let mut id_to_origin: HashMap<i64, String> = HashMap::new();
        for b in vdf.collect().await? {
            let os = string_values(b.column_by_name(ORIGIN_ID).unwrap());
            let is = int_values(b.column_by_name(VERTEX_ID).unwrap());
            for k in 0..b.num_rows() {
                id_to_origin.insert(is[k], os[k].clone().unwrap());
            }
        }

        let edges_path = res.edges.clone();
        let edf = ctx
            .read_parquet(&format!("{edges_path}/"), ParquetReadOptions::new())
            .await?;
        let mut got = HashSet::new();
        for b in edf.collect().await? {
            let srcs = int_values(b.column_by_name(EDGE_SRC).unwrap());
            let dsts = int_values(b.column_by_name(EDGE_DST).unwrap());
            for k in 0..b.num_rows() {
                got.insert((
                    id_to_origin[&srcs[k]].clone(),
                    id_to_origin[&dsts[k]].clone(),
                ));
            }
        }

        let expected: HashSet<(String, String)> = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
            .into_iter()
            .map(|(s, d)| (s.to_string(), d.to_string()))
            .collect();
        assert_eq!(got, expected);

        std::fs::remove_dir_all(&out).ok();
        Ok(())
    }

    #[tokio::test]
    async fn test_from_string_ids_rejects_dangling_edges() -> Result<()> {
        let ctx = SessionContext::new();
        let v = vertex_df(&ctx, &["a", "b"])?;
        let e = edge_df(&ctx, &[("a", "zzz")])?;
        let out = tmpdir("dangling");

        let res = from_string_ids(&ctx, v, e, &out, None).await;
        assert!(res.is_err(), "dangling endpoints must fail the ingest");

        std::fs::remove_dir_all(&out).ok();
        Ok(())
    }

    #[tokio::test]
    async fn test_from_string_ids_rejects_reserved_origin_id() -> Result<()> {
        let ctx = SessionContext::new();
        let v = vertex_df(&ctx, &["a"])?;
        let schema = Arc::new(Schema::new(vec![
            Field::new(EDGE_SRC, DataType::Utf8, false),
            Field::new(EDGE_DST, DataType::Utf8, false),
            Field::new(ORIGIN_ID, DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a"])),
                Arc::new(StringArray::from(vec!["a"])),
                Arc::new(StringArray::from(vec!["x"])),
            ],
        )?;
        let e = ctx.read_batches(vec![batch])?;

        assert!(
            from_string_ids(&ctx, v, e, &tmpdir("reserved"), None)
                .await
                .is_err()
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_from_string_ids_rejects_int_ids() -> Result<()> {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![Field::new(
            VERTEX_ID,
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2]))])?;
        let v = ctx.read_batches(vec![batch])?;
        let e = edge_df(&ctx, &[("a", "b")])?;

        assert!(
            from_string_ids(&ctx, v, e, &tmpdir("types"), None)
                .await
                .is_err()
        );
        Ok(())
    }
}
