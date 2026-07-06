//! Maximal Independent Set
//!
//! Implementation inspired by the Spark GraphFrames Scala code distributed
//! under Apache License 2.0, ported to this project's out-of-core style.
//!
//! Algorithm: "An Improved Distributed Algorithm for Maximal Independent Set",
//! Mohsen Ghaffari
//!
//! https://arxiv.org/abs/1506.05093

use datafusion::error::Result;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::functions_aggregate;
use datafusion::functions_aggregate::expr_fn::bool_or;
use datafusion::prelude::*;
use datafusion::{object_store::path::Path, prelude::SessionContext};
use uuid::Uuid;

use crate::memory::ParquetCheckpointer;
use crate::utils::symmetrize;
use crate::{EDGE_DST, EDGE_SRC, VERTEX_ID};
use crate::{
    GraphFrame,
    memory::CheckpointConfig,
    utils::{GraphFramesConfig, scoped_ctx},
};

const PROB_COL: &str = "__mis_prob";
const DEG_COL: &str = "__effective_degree";
const IS_NOMINATED: &str = "__mis_is_nominated";
const HAS_NOMINATED_NBR: &str = "__has_nominated_nbr";

#[derive(Debug, Clone)]
pub struct MISBuilder<'a> {
    graph: &'a GraphFrame,
    /// Storage options
    checkpoint_config: CheckpointConfig,
}

impl<'a> MISBuilder<'a> {
    pub fn new(graph: &'a GraphFrame) -> Self {
        MISBuilder {
            graph,
            checkpoint_config: CheckpointConfig::default_local_fs(),
        }
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

    /// Run the MIS
    ///
    /// Each run is non determenistic.
    /// Tracking issue: https://github.com/apache/datafusion/issues/17686
    pub async fn run(self, ctx: &SessionContext, output: &str) -> Result<usize> {
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
        log::info!("start MIS with run-id {run_id}");

        let ckpt_base = self.checkpoint_config.dir.clone().join(run_id.clone());
        let store_url = self.checkpoint_config.store_url.clone();

        // a) edges checkpointer b) vertices left checkpointer c) current MIS checkpointer
        let mut edges_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("edges"));
        let mut vertex_ckptr =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("vertices"));
        let mut mis_checkpointer =
            ParquetCheckpointer::new(store_url.clone(), ckpt_base.clone().join("mis"));

        let mut current_mis = mis_checkpointer
            .push(
                &ctx,
                "0",
                self.graph
                    .vertices
                    .clone()
                    .select(vec![col(VERTEX_ID), lit(false).alias("mis")])?,
                None,
            )
            .await?;

        // initial p=1/2
        let mut vertices_left = vertex_ckptr
            .push(
                &ctx,
                "0",
                self.graph
                    .vertices
                    .clone()
                    .select(vec![col(VERTEX_ID), lit(0.5f64).alias(PROB_COL)])?,
                None,
            )
            .await?;

        // symmetrize and dedup edges
        //
        // de-dub is unavoidable here,
        // otherwise effictive degree will
        // be computed worng
        let mut edges = edges_ckptr
            .push(
                &ctx,
                "initial",
                symmetrize(&self.graph.edges.clone(), true)?,
                None,
            )
            .await?;

        let mut iteration = 0usize;
        let mut converged = false;

        while !converged {
            // we must contract edges here
            // but on the first iteration there is no actualy contraction
            // so we skipt it;
            edges = {
                let inner = edges.clone().join(
                    vertices_left.clone(),
                    JoinType::Inner,
                    &[EDGE_DST],
                    &[VERTEX_ID],
                    None,
                )?;
                if iteration == 0 {
                    inner
                } else {
                    // after the first iteration an amount of vertices is much less than initial
                    // it should make sense to update edges to increase the perf
                    let inner_e = edges_ckptr
                        .push(&ctx, &iteration.to_string(), inner, None)
                        .await?;
                    edges_ckptr.evict_all_but_latest_n(&ctx, 1).await?;
                    inner_e
                }
            };

            let effective_degrees = edges.clone().aggregate(
                vec![col(EDGE_SRC)],
                vec![functions_aggregate::sum::sum(col(PROB_COL)).alias(DEG_COL)],
            )?;

            let probs_to_join_mis = vertex_ckptr
                .push(
                    &ctx,
                    &format!("probs_{}", iteration),
                    vertices_left
                        .clone()
                        .join(
                            effective_degrees,
                            JoinType::Inner,
                            &[VERTEX_ID],
                            &[EDGE_SRC],
                            None,
                        )?
                        .with_column(
                            PROB_COL,
                            when(col(DEG_COL).gt_eq(lit(2.0)), col(PROB_COL).div(lit(2.0)))
                                .when(
                                    lit(2.0).mul(col(PROB_COL)).lt_eq(lit(0.5)),
                                    lit(2.0).mul(col(PROB_COL)),
                                )
                                .otherwise(lit(0.5))?,
                        )?
                        .with_column(IS_NOMINATED, random().lt_eq(col(PROB_COL)))?
                        .select(vec![col(VERTEX_ID), col(PROB_COL), col(IS_NOMINATED)])?,
                    None,
                )
                .await?;

            let isolated_vertices = vertices_left
                .clone()
                .join(
                    probs_to_join_mis.clone(),
                    JoinType::LeftAnti,
                    &[VERTEX_ID],
                    &[VERTEX_ID],
                    None,
                )?
                .select(vec![col(VERTEX_ID), lit(true).alias(IS_NOMINATED)])?;

            // if v is marked but no nbrs of v are marked v is joining MIS
            // and is removed along with all it's neighbors
            let joined_mis_nbrs_cond = edges
                .clone()
                .join(
                    probs_to_join_mis.clone(),
                    JoinType::Inner,
                    &[EDGE_DST],
                    &[VERTEX_ID],
                    None,
                )?
                .aggregate(
                    vec![col(EDGE_SRC)],
                    vec![bool_or(col(IS_NOMINATED)).alias(HAS_NOMINATED_NBR)],
                )?
                .join(
                    probs_to_join_mis.clone(),
                    JoinType::Inner,
                    &[EDGE_SRC],
                    &[VERTEX_ID],
                    None,
                )?;

            let joined_mis = joined_mis_nbrs_cond
                .filter(not(col(HAS_NOMINATED_NBR)).and(col(IS_NOMINATED)))?
                .select(vec![col(VERTEX_ID), lit(true).alias(IS_NOMINATED)])?;

            let updated_mis = current_mis
                .clone()
                .join(
                    isolated_vertices,
                    JoinType::Left,
                    &[VERTEX_ID],
                    &[VERTEX_ID],
                    None,
                )?
                .select(vec![
                    col(VERTEX_ID),
                    col("mis").or(col(IS_NOMINATED)).alias("mis"),
                ])?
                .join(
                    joined_mis.clone(),
                    JoinType::Left,
                    &[VERTEX_ID],
                    &[VERTEX_ID],
                    None,
                )?
                .select(vec![
                    col(VERTEX_ID),
                    col("mis").or(col(IS_NOMINATED)).alias("mis"),
                ])?;

            current_mis = mis_checkpointer
                .push(&ctx, &format!("mis_{}", iteration), updated_mis, None)
                .await?;

            let neighbors_of_mis = edges
                .clone()
                .join(
                    joined_mis.clone(),
                    JoinType::Inner,
                    &[EDGE_DST],
                    &[VERTEX_ID],
                    None,
                )?
                .select(vec![col(EDGE_SRC)])?;

            vertices_left = vertex_ckptr
                .push(
                    &ctx,
                    &format!("vertices-{}", iteration),
                    probs_to_join_mis
                        .join(
                            joined_mis,
                            JoinType::LeftAnti,
                            &[VERTEX_ID],
                            &[VERTEX_ID],
                            None,
                        )?
                        .join(
                            neighbors_of_mis,
                            JoinType::LeftAnti,
                            &[VERTEX_ID],
                            &[VERTEX_ID],
                            None,
                        )?,
                    None,
                )
                .await?;

            let cnt_v_left = vertices_left.clone().count().await?;
            log::info!("iteration {iteration} done, {cnt_v_left} are participating");

            converged = cnt_v_left == 0;
            iteration += 1;
        }

        Ok(iteration)
    }
}
