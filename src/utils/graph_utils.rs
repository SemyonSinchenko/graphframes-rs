use crate::{EDGE_DST, EDGE_SRC};
use datafusion::prelude::*;
use datafusion::{error::Result, prelude::DataFrame};

/// Prepares the edge set: drops self-loops, symmetrizes
/// (adds the reverse of every edge), and deduplicates (optionally). The result is the
/// undirected simple graph the algorithm operates on.
pub(crate) fn symmetrize(
    edges: &DataFrame,
    do_distinct: bool,
    attr_cols: Option<Vec<String>>,
) -> Result<DataFrame> {
    // some algorithms to the aggregation over edges and are tolerant
    // to duplicates;
    //
    // that is the only reason this function is
    // a) packgage-private
    // b) has second argument
    let (forward_cols, backward_cols) = match attr_cols {
        Some(cols) => {
            let mut fcc = vec![col(EDGE_SRC), col(EDGE_DST)];
            let mut bcc = vec![col(EDGE_DST).alias(EDGE_SRC), col(EDGE_SRC).alias(EDGE_DST)];

            let attrs: Vec<Expr> = cols.iter().map(|c| col(c)).collect();

            fcc.extend(attrs.clone());
            bcc.extend(attrs.clone());

            (fcc, bcc)
        }
        None => (
            vec![col(EDGE_SRC), col(EDGE_DST)],
            vec![col(EDGE_DST).alias(EDGE_SRC), col(EDGE_SRC).alias(EDGE_DST)],
        ),
    };
    let no_loops = edges
        .clone()
        .filter(col(EDGE_SRC).not_eq(col(EDGE_DST)))?
        .select(forward_cols)?;
    let reversed = no_loops.clone().select(backward_cols)?;

    let res = if do_distinct {
        no_loops.union(reversed)?.distinct()?
    } else {
        no_loops.union(reversed)?
    };

    Ok(res)
}
