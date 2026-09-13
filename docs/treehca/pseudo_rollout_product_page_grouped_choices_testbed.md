# Grouped-choice pseudo-rollout probability testbed

[`pseudo_rollout_product_page_grouped_choices_testbed.py`](../../treehca/pseudo_rollout_product_page_grouped_choices_testbed.py)
checks whether the probabilities produced by the group-specific pseudo-rollout
scorer agree with empirical choices made by the ordinary product-page policy.
It is the option-group counterpart to the existing full-action testbed.

## Experiment

The testbed uses only synthetic WebShop goals (`human_goals=False`). It samples
goal-bearing catalog products, renders their real no-history product-page
prompts, and constructs one pseudo-rollout for every parsed option group. The
exact synthetic `goal_options` mapping is passed to
`prepare_product_page_grouped_choice_rollouts`, so every fuzzy-equivalent
correct option is retained in `correct_options`.

For every multi-option group, the program performs two measurements:

1. It calls `score_product_page_grouped_choice_rollouts` to get the constrained
   next-token pseudo distribution over that group's labels. Before the label,
   it appends this artificial assistant response prefix, with the actual group
   name substituted:

   ```text
   <think>The best choice for the {group_name} group corresponds to the label:
   ```

2. It samples ordinary complete responses from the original product-page
   prompt and applies the production WebShop projection. The empirical option
   distribution for a group is conditional on the configured empirical
   denominator. By default, that denominator contains only responses selecting
   one of that group's options.

All groups on a page reuse the same ordinary completion batch. Singleton groups
are excluded because both their pseudo and conditional empirical distributions
are necessarily 1.0 and would add no evidence about calibration.

The conditioning matters. A completion that selects an option from another
group, a navigation control, `buy now`, or an inadmissible action does not enter
the current group's default empirical denominator. The JSON report records the
same-group count as `recognized_group_option_completions` and the actual
denominator as `empirical_option_probability_denominator`; groups with a zero
denominator retain their pseudo probabilities but have null empirical
probabilities and are excluded by the analyzer.

With `--include-valid-non-option-actions-in-empirical-denominator`, a valid
non-option action such as `click[buy now]`, `click[description]`, or
`click[back to search]` enters every option group's denominator. It does not
enter an option numerator. An option belonging to another group is still
excluded from the current group's denominator, as is an inadmissible projected
action. Consequently, the displayed empirical option probabilities can sum to
less than one in this mode; the missing mass represents included valid
non-option actions.

The artificial prefix is tokenized without special tokens and appended after
the pseudo prompt's chat-template assistant boundary. The constrained label is
the next token. The testbed represents it as a `GroupResponsePrefix` containing
text, token IDs, and a source marker, then passes its token IDs through the
scorer's existing `assistant_response_prefix_token_ids` interface. A future
LLM-generated-thinking mode can supply sampled token IDs through the same
interface instead of changing the scorer.

## Running the testbed

Run from the repository root in the `webshop` environment:

```bash
conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_product_page_grouped_choices_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --samples-per-page 256 \
  --tensor-parallel-size 2 \
  --output outputs/pseudo_rollout_grouped_choice_calibration.json
```

To include valid actions such as `buy now` in every group's denominator while
still excluding options from other groups, add:

```bash
--include-valid-non-option-actions-in-empirical-denominator
```

The testbed writes the full JSON report and, by default, a Markdown file beside
it. Important controls are:

- `--samples-per-page`: ordinary responses sampled for each source page;
- `--bootstrap-replicates`: null samples used by the per-group multinomial
  goodness-of-fit calculation;
- `--high-probability-threshold`: the strict threshold used by the report's
  decisive-option summary, defaulting to `0.6`;
- `--include-valid-non-option-actions-in-empirical-denominator`: changes the
  empirical denominator as described above; without it, behavior remains
  conditional on selecting an option from the current group;
- `--page-batch-size`, `--max-new-tokens`, `--max-model-len`,
  `--gpu-memory-utilization`, and `--tensor-parallel-size`: inference and memory
  controls analogous to the full-action testbed.

As in the existing scorer testbed, the program selects vLLM V0 before importing
vLLM because the allowed-label logprob calculation requires the masked V0
behavior.

## Aggregate correct-option analysis

Run the analysis script after inference:

```bash
conda run --no-capture-output -n webshop \
  python -m treehca.analyze_pseudo_rollout_product_page_grouped_choices_results \
  outputs/pseudo_rollout_grouped_choice_calibration.json
```

The analysis defaults to the threshold stored in the report. It can be changed
without rerunning inference using `--high-probability-threshold`, which must be
strictly between 0.5 and 1.0.

Before applying the threshold, the analyzer sums the probabilities of every
fuzzy-correct option in a group. Each group therefore contributes one pseudo
correct probability and one conditional empirical correct probability. This is
important when several displayed options satisfy WebShop's fuzzy matcher: they
can collectively be high probability even when none is individually above the
threshold.

The headline comparison is deliberately bidirectional:

- **pseudo correct mass high → empirical correct mass high** is the fraction of
  groups whose aggregate pseudo correct probability is strictly above the
  threshold and whose aggregate empirical correct probability is also above
  it;
- **empirical correct mass high → pseudo correct mass high** is the reverse
  fraction;
- **bidirectional decisive agreement** uses the union of groups whose aggregate
  correct probability is high in either distribution and counts an agreement
  only when it is high in both.

Groups below the threshold in both distributions do not enter these headline
rates. Consequently, cases whose probability mass does not favor the correct
set do not manufacture agreement through many uninteresting low/low option
pairs. The analysis still retains those group aggregates in its scatter plot in
muted gray for context. It also reports the mean absolute pseudo/empirical gap
among decisive group aggregates and lists every one-direction-only mismatch.

The output directory contains:

- `grouped_correct_probability_scatter.*`, with threshold lines and decisive
  aggregate matches/mismatches highlighted;
- `high_probability_directional_agreement.*`, comparing the two directional
  confirmation rates;
- `analysis_summary.json` and `analysis_summary.md`.

## Interpreting the result

The empirical denominator should be inspected for every mismatch. A group that
the ordinary policy rarely acts on can have a noisy conditional frequency even
when `samples_per_page` is large. Each option row includes a Wilson 95% interval
to make that uncertainty visible.

`is_correct`, `correct_options`, `pseudo_correct_probability`, and
`empirical_correct_probability_conditional` are included for downstream reward
analysis. Individual option rows remain available for diagnosis, but all
high-probability checks use the summed correct-option fields.

This testbed measures whether the group intervention preserves the ordinary
policy's relative preference after conditioning on that group. It does not
measure how often the policy chooses one group rather than another, and it does
not test a multi-step trajectory after earlier options have been selected.
