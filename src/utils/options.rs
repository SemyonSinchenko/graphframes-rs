use datafusion::common::extensions_options;
use datafusion::config::ConfigExtension;
use datafusion::prelude::SessionContext;

extensions_options! {
    pub struct GraphFramesConfig {
        pub prefer_smj: bool, default = false
        pub checkpoint_dir: String, default = "_gf_checkpoints".to_owned()
    }
}

impl ConfigExtension for GraphFramesConfig {
    const PREFIX: &'static str = "graphframes";
}

pub fn scoped_ctx(ctx: &SessionContext, prefer_smj: bool) -> SessionContext {
    let config = ctx
        .copied_config()
        .set_bool("datafusion.optimizer.prefer_hash_join", !prefer_smj);
    SessionContext::new_with_config_rt(config, ctx.runtime_env())
}
