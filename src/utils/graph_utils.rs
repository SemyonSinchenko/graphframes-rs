use crate::{EDGE_DST, EDGE_SRC};
use datafusion::prelude::*;
use datafusion::{error::Result, prelude::DataFrame};

/// Prepares the edge set: drops self-loops, symmetrizes
/// (adds the reverse of every edge), and deduplicates (optionally). The result is the
/// undirected simple graph the algorithm operates on.
pub(crate) fn symmetrize(edges: &DataFrame, do_distinct: bool) -> Result<DataFrame> {
    // some algorithms to the aggregation over edges and are tolerant
    // to duplicates;
    //
    // that is the only reason this function is
    // a) packgage-private
    // b) has second argument
    let no_loops = edges.clone().filter(col(EDGE_SRC).not_eq(col(EDGE_DST)))?;
    let reversed = no_loops.clone().select(vec![
        col(EDGE_DST).alias(EDGE_SRC),
        col(EDGE_SRC).alias(EDGE_DST),
    ])?;

    let res = if do_distinct {
        no_loops.union(reversed)?.distinct()?
    } else {
        no_loops.union(reversed)?
    };

    Ok(res)
}
