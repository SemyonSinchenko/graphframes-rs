use crate::GraphFrame;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::error::Result;
use datafusion::prelude::{CsvReadOptions, DataFrame, SessionContext};
use std::collections::HashMap;
use std::io::Result as ioResult;
use std::{env, fs};

/// # Arguments
///
/// * `benchmark_run`: true for benchmark runs to read data from bench/data dir, false for tests to read data from testing/data.
fn _get_dataset_base_path(benchmark_run: bool) -> Result<String> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let dir_name = if benchmark_run { "benches" } else { "testing" };
    let base_path = format!("{}/{}/data/ldbc", manifest_dir, dir_name);
    Ok(base_path)
}

/// Creates Schema for Graphframe edges Dataframe based on is_3d flag
/// # Arguments
/// * `is_weighted`: Boolean value that defines if edges has weights field or not.
fn _create_edge_schema(is_weighted: bool) -> Schema {
    let mut edge_fields = vec![
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
    ];

    if is_weighted {
        edge_fields.push(Field::new("weights", DataType::Float64, false))
    }

    Schema::new(edge_fields)
}

/// Creates a GraphFrame from an LDBC-style dataset.
///
/// The dataset is expected to be located in the `benches/data/` directory.
///
/// # Arguments
///
/// * `dataset`: The name of the dataset directory (e.g., "test-pr-directed").
/// * `benchmark_run`: true for benchmark runs to read data from bench/data dir, false for tests to read data from testing/data.
/// * `is_weighted`: Boolean value that defines if edges has weights field or not.
pub async fn create_ldbc_test_graph(
    dataset: &str,
    benchmark_run: bool,
    is_weighted: bool,
) -> Result<GraphFrame> {
    let ctx = SessionContext::new();

    let edge_schema = _create_edge_schema(is_weighted);
    let vertices_schema = Schema::new(vec![Field::new("id", DataType::Int64, false)]);

    let ds_base_path = _get_dataset_base_path(benchmark_run)?;

    let edges_path = format!("{}/{}/{}.e.csv", ds_base_path, dataset, dataset);
    let vertices_path = format!("{}/{}/{}.v.csv", ds_base_path, dataset, dataset);

    println!("{}", edges_path);
    println!("{}", vertices_path);

    let edges = ctx
        .read_csv(
            &edges_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&edge_schema),
        )
        .await?;

    let vertices = ctx
        .read_csv(
            &vertices_path,
            CsvReadOptions::new()
                .delimiter(b' ')
                .has_header(false)
                .schema(&vertices_schema),
        )
        .await?;
    println!("read {} vertices", vertices.clone().count().await?);
    println!("read {} edges", edges.clone().count().await?);
    Ok(GraphFrame { vertices, edges })
}

// Reads the ldbc dataset properties file and converts it into a HashMap
#[allow(dead_code)]
pub fn parse_ldbc_properties_file(
    dataset: &str,
    benchmark_run: bool,
) -> ioResult<HashMap<String, String>> {
    let prop_fp = format!(
        "{}/{}/{}.properties",
        _get_dataset_base_path(benchmark_run)?,
        dataset,
        dataset
    );
    let content = fs::read_to_string(prop_fp)?;
    let mut properties_map: HashMap<String, String> = HashMap::new();

    for line in content.lines() {
        let trimmed_line = line.trim();

        if trimmed_line.is_empty() || trimmed_line.starts_with("#") {
            continue;
        }
        if let Some((key, value)) = trimmed_line.split_once("=") {
            properties_map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    Ok(properties_map)
}

pub(crate) async fn assert_dataframes_equal(a: DataFrame, b: DataFrame) -> Result<()> {
    use datafusion::arrow::array::{Array, ArrayRef};
    use datafusion::arrow::compute::cast;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::SchemaRef;
    use datafusion::arrow::record_batch::RecordBatch;

    async fn collect_and_concat(df: DataFrame) -> Result<RecordBatch> {
        use datafusion::arrow::compute::concat_batches;
        let schema: SchemaRef = df.schema().to_owned().into();
        let batches = df.collect().await?;
        Ok(concat_batches(&schema, &batches)?)
    }

    let ba = collect_and_concat(a).await?;
    let bb = collect_and_concat(b).await?;
    assert_eq!(ba.num_rows(), bb.num_rows(), "row count mismatch");
    assert_eq!(ba.num_columns(), bb.num_columns(), "column count mismatch");

    let schema_a = ba.schema();
    let schema_b = bb.schema();

    let is_string_like =
        |t: &DataType| matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View);

    for col_idx in 0..ba.num_columns() {
        let field_a = schema_a.field(col_idx);
        let field_b = schema_b.field(col_idx);
        assert_eq!(
            field_a.name(),
            field_b.name(),
            "column {col_idx} name mismatch"
        );

        let ca = ba.column(col_idx);
        let cb = bb.column(col_idx);

        // Normalise string-like arrays to Utf8View so the reader's physical type choice
        // (Utf8 vs Utf8View) doesn't defeat value comparison.
        let (na, nb): (ArrayRef, ArrayRef) = if is_string_like(ca.data_type())
            && is_string_like(cb.data_type())
            && ca.data_type() != cb.data_type()
        {
            let target = DataType::Utf8View;
            (cast(ca.as_ref(), &target)?, cast(cb.as_ref(), &target)?)
        } else {
            (ca.clone(), cb.clone())
        };

        assert_eq!(
            na.data_type(),
            nb.data_type(),
            "column {} normalised data type mismatch",
            field_a.name()
        );
        assert_eq!(
            na.as_ref(),
            nb.as_ref(),
            "column {} contents differ",
            field_a.name()
        );
    }
    Ok(())
}
