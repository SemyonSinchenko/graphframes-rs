use datafusion::arrow::datatypes::DataType;
use datafusion::common::ExprSchema;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::object_store::path::Path;
use datafusion::prelude::*;
use uuid::Uuid;

use crate::utils::scoped_ctx;
use crate::{
    EDGE_DST, EDGE_SRC, GraphFrame, memory::CheckpointConfig, ml::KMeansResult, utils::symmetrize,
};
use crate::{EDGE_WEIGHT, GraphFramesConfig};

#[derive(Debug, Copy, Eq, PartialEq, Hash, Clone)]
pub enum InitStrategy {
    Random,
    DegreeBased,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum WeightsStrategy {
    None,
    PPMI,
}

#[derive(Debug)]
pub struct PICBuilder {
    graph: GraphFrame,
    max_iterations: usize,
    init_strategy: InitStrategy,
    weight_col: Option<String>,
    k: Vec<usize>,
    checkpoint_config: CheckpointConfig,
    seed: u64,
    // TODO: multi-trajectories are postponed;
    // should add: mode -- either trajctory or last-t
    // should add: num trajectories;
}

impl PICBuilder {
    pub fn new(graph: GraphFrame) -> Self {
        Self {
            graph: graph,
            max_iterations: 100usize, // mirroring SparkML' default
            init_strategy: InitStrategy::DegreeBased,
            weight_col: None,
            k: vec![2], // mirroring SparkML' default
            checkpoint_config: CheckpointConfig::default_local_fs(),
            seed: 42u64,
        }
    }

    pub fn set_max_iterations(mut self, v: usize) -> Self {
        self.max_iterations = v;
        self
    }

    pub fn set_edge_weight_col(mut self, c: &str) -> Self {
        self.weight_col = Some(c.to_string());
        self
    }

    pub fn set_init_strategy(mut self, v: InitStrategy) -> Self {
        self.init_strategy = v;
        self
    }

    pub fn set_multiple_k(mut self, kk: Vec<usize>) -> Self {
        self.k = kk;
        self
    }

    pub fn set_k(mut self, k: usize) -> Self {
        self.k = vec![k];
        self
    }

    /// Set the object store URL
    pub fn with_checkpoint_store(mut self, store_url: ObjectStoreUrl) -> Self {
        self.checkpoint_config.store_url = store_url;
        self
    }

    /// Set the checkpoint directory
    pub fn set_checkpoint_dir(mut self, dir: Path) -> Self {
        self.checkpoint_config.dir = dir;
        self
    }

    pub async fn run(self, ctx: &SessionContext, output: &str) -> Result<KMeansResult> {
        let gf_config = ctx
            .state()
            .config()
            .options()
            .extensions
            .get::<GraphFramesConfig>()
            .cloned()
            .unwrap_or_default();

        let ctx = &scoped_ctx(ctx, gf_config.prefer_smj);
        self.checkpoint_config.validate_output(output)?;

        let run_id = Uuid::new_v4().to_string();
        log::info!("start pregel with ID {run_id}");

        // weight columns should:
        // a) exists
        // b) be an f32 data type
        let w_col = match self.weight_col {
            Some(c) => {
                if !self
                    .graph
                    .edges
                    .schema()
                    .has_column_with_unqualified_name(&c)
                {
                    return Err(DataFusionError::Plan(format!(
                        "column {} does not exist in edges",
                        c
                    )));
                }

                let resolved = self.graph.edges.schema().field_with_unqualified_name(&c)?;

                if resolved.data_type() != &DataType::Float32 {
                    return Err(DataFusionError::Plan(format!(
                        "weight column {} has data type {} file expected float32",
                        c,
                        resolved.data_type()
                    )));
                }

                col(c).alias(EDGE_WEIGHT)
            }
            None => lit(1.0f32).alias(EDGE_WEIGHT),
        };

        let symmetrized_edges = symmetrize(
            &self
                .graph
                .edges
                .clone()
                .select(vec![col(EDGE_SRC), col(EDGE_DST), w_col])?,
            false,
        );

        Ok(KMeansResult {
            num_iterations: 1,
            d: 1,
            runs: vec![],
        })
    }
}
