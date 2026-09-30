# No-progress advantage cap and RMS preservation

Enable with `algorithm.treehca.no_progress_advantage_cap=True`. After TreeHCA
credit assignment, paths ending in success or allocation pruning are scanned for
consecutive nodes whose `info_gain` is less than or equal to
`no_progress_info_gain_threshold` (default `0.05`). Runs of at least
`no_progress_turns_threshold` nodes (default `3`) have their advantages capped
at zero. Leaves with `termination_reason == "pruned"` count toward the run
length and can have their advantages capped, along with qualifying ancestors.
The path need not contain a successful outcome. Successful terminals remain
excluded from runs and capping. Paths ending in failure, the turn limit, or
another terminal reason do not initiate a scan. Shared ancestors can still be
capped through another successful or pruned path.

All terminal nodes, including uncapped pruned leaves, remain excluded from
receiving redistributed norm. Norm removed from a capped pruned leaf is
redistributed to eligible non-terminal nodes in the same tree.

The comparison is inclusive (`info_gain <= no_progress_info_gain_threshold`),
with no tolerance or rounding applied by this check. At threshold `0`, zero
and negative information gains qualify; any stored positive information gain,
however small, breaks the run and is not capped. Nodes must still belong to a
run of sufficient length to be capped.

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

Metrics under `treehca/no_progress/` report successful/pruned rollout counts,
multiplier mean/max, rescaled and
limited tree counts, trees without recipients, and removed/restored/unrestored
positive squared norm. Unrestored norm records the shortfall from the multiplier
limit or lack of eligible recipients.
