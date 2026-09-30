# No-progress advantage cap and RMS preservation

Enable with `algorithm.treehca.no_progress_advantage_cap=True`. After TreeHCA
credit assignment, successful root-to-leaf paths are scanned for consecutive
non-terminal nodes whose `info_gain` is strictly below
`no_progress_info_gain_threshold` (default `0.05`). Runs of at least
`no_progress_turns_threshold` nodes (default `3`) have their advantages capped
at zero. Terminal nodes neither contribute to runs nor have their advantages
modified, including during redistribution.

For each connected tree, let `R` be the sum of squared positive advantages
removed by the cap, and `P` the sum of squared positive advantages on eligible
remaining nodes. Eligible nodes are uncapped, non-terminal nodes with positive
advantages. Both sums use valid response tokens, including any duplicated
training rows. Every eligible positive token is multiplied by
`min(no_progress_rms_max_multiplier, sqrt(1 + R / P))`.

`algorithm.treehca.no_progress_rms_max_multiplier` defaults to `3.0` and must
be finite and at least `1`. Set it to `1` to retain the cap without rescaling.
With no eligible positive mass (`P == 0`), no rescaling occurs. Negative and
zero advantages are unchanged by redistribution. Returns are unchanged.

Trees are identified by the highest materialized ancestor through
`parent_node_uid`; separate roots sharing the sentinel `"root"` or prompt group
do not exchange norm. Within a tree, eligible nodes on sibling branches can
receive norm. The calculation runs before actor minibatching; it does not
renormalize each optimizer minibatch separately.

When the multiplier is not limited and recipients exist, this preserves the
tree's masked advantage RMS (up to rounding). It does not guarantee the exact
parameter-gradient norm. It uses token weighting, matching the default
`token-mean` policy loss; other loss aggregation modes can weight nodes differently.

Metrics under `treehca/no_progress/` report multiplier mean/max, rescaled and
limited tree counts, trees without recipients, and removed/restored/unrestored
positive squared norm. Unrestored norm records the shortfall from the multiplier
limit or lack of eligible recipients.
