# Premature-leaf propagation

TreeHCA can stop a live rollout when the branching allocator gives its leaf no
expansions. The rollout logger records this as `termination_reason="pruned"`.
This differs from `termination_reason="turn_limit"`: the latter completed the
configured interaction budget and remains an ordinary terminal sample.

## Configuration

The filter is disabled by default. Enable it with:

```yaml
algorithm:
  treehca:
    filter_premature_leaves: true
```

The option controls both the tree reward backup and TreeHCA's SNIS/Q credit
backup, so they cannot silently use different leaf populations. Omitting the
option, or setting it to `false`, preserves the original propagation behavior.

## Backup rule

A pruned leaf keeps its own score and can propagate within a branch that has no
full-score outcome. Propagation stops at the first ancestor whose subtree also
contains a leaf with `termination_reason="success"`. Since every ancestor above
that point contains the same successful descendant, the pruned leaf cannot
affect any higher node either. Turn-limit, environment-terminal, and failed
leaves are not filtered.

`success` is the full-score marker instead of a numeric threshold. Reward
scales are environment-specific—for example, WebShop maps full credit to a
training reward of 10—while the rollout termination marker has one meaning
across environments.

The rule is applied in both TreeHCA backup layers:

1. `TreeStructureRewardManager` stops the pruned leaf before updating the
   protected ancestor's reward, trajectory count, mean depth, or tool-call
   count. This prevents average-mode denominators as well as reward numerators
   from retaining the censored rollout. The behavior requires both
   `algorithm.adv_estimator=treehca` and
   `algorithm.treehca.filter_premature_leaves=true`; IGRPO is unchanged.
2. The SNIS and Q/hindsight credit estimators maintain an auxiliary backup that
   excludes pruned leaves. Protected ancestors use that backup. For SNIS,
   subtree-size weights and the optional group baseline are recomputed from the
   remaining leaves. For Q credit, empty all-pruned child subtrees are omitted
   from the parent mean. TD residuals use the same eligible child set at that
   boundary: an excluded all-pruned child gets zero, and retained child
   residuals use their pruned-free Q values. This keeps the retained sibling
   residuals zero-sum instead of assigning the excluded branch an unmatched
   negative residual. Below the boundary, ordinary Q residuals are preserved.

The leaf's own training row is deliberately retained, as are nodes below the
first protected ancestor. This preserves useful local supervision while
preventing an incomplete rollout from diluting a known full-score path.

The trainer reports whether filtering is enabled, plus the number of pruned
leaves, full-score leaves, and protected ancestors under
`treehca/premature_leaf_filter/*`. Counts are zero when the option is disabled.
