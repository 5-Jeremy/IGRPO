# WebShop pseudo probabilities during TreeHCA training

TreeHCA now selects its pseudo scorer with
`algorithm.treehca.pseudo_scorer`. The default, `auto`, selects WebShop scoring
when `env.env_name` contains `webshop` (case insensitive), and the existing
`compute_answer_block_avg_log_prob` otherwise. Explicit values are `webshop`
and `answer_block`. Other algorithms always retain the existing dispatch.

The training implementation is in
[`training_webshop_scoring.py`](../../treehca/training_webshop_scoring.py).
`TrainingWebshopTurnSuccessScorer` is a separate specialization of the existing
`WebshopTurnSuccessScorer`; it shares the tested native reward aggregation and
two-stage pruning logic without modifying the standalone scorer or its vLLM
backend. `WebshopInfoGainScorer.compute(batch, policy_version=...)` provides the
training `DataProto` interface.

## Which turn supplies the scoring prompt?

A node's probability is based on the state **after its action**, using the
prompt that would be supplied to its child for the next action. It does not
use the prompt that generated the node's own action. The child does not need
to be generated or expanded for this scoring to happen.

For example, let A be a child of the synthetic root:

| Field or operation | Context |
| --- | --- |
| A's generation input | Initial search page |
| A's generated action | `search[red shoes]` |
| A's scoring prompt | Search-results page returned by that search, with the search action in history when enabled |
| A's `avg_ans_log_probs` | Log success probability estimated from those search results |
| A child's generation input | The same search-results prompt body |
| A's saved `page_type` | Initial-page type (`""`), because this field describes A's generation input |

The prompt is constructed and passed through the following code path:

1. In [`rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py),
   `vanilla_multi_turn_loop_with_tree_structure` executes
   `node_management.envs.step(text_actions)` and assigns the returned
   `next_obs` to `node_management.obs`.
2. That call uses `WebshopEnvironmentManager.step` in
   [`env_manager.py`](../../agent_system/environments/env_manager.py).
   After executing the action, it stores the previous observation/action in
   memory and calls `build_text_obs` with the resulting observation and
   available actions. Thus `next_obs["text"]` already contains complete
   next-turn prompt bodies, including the configured history (subject to the
   existing long-prompt fallback).
3. The collector passes `next_obs["text"]` to
   `TreeHCAWebshopEnvironmentManager.scoring_payloads` in
   [`training_webshop_env.py`](../../treehca/training_webshop_env.py).
   Each worker forwards its prompt to `WebshopSnapshotSource.capture` in
   [`webshop_probability_snapshot.py`](../../treehca/webshop_probability_snapshot.py).
   The snapshot stores that text as `snapshot.prompt` and reads `page_type`
   from the environment's current, post-action URL. `previous_page_type` is
   separate transition metadata; it does not select an earlier prompt.
4. `WebshopInfoGainScorer.compute` reads
   `non_tensor_batch["webshop_scoring_payload"]` and passes its snapshots to
   `TrainingWebshopTurnSuccessScorer.score`. For uncached search-results
   probes, the inherited implementation in
   [`webshop_turn_success.py`](../../treehca/webshop_turn_success.py) passes
   `job.snapshot.prompt` to `prepare_results_page_answer_probe` in
   [`pseudo_rollout_results_page.py`](../../treehca/pseudo_rollout_results_page.py).
   The scoring text comes from this snapshot, not from decoding the batch's
   tokenized `prompts` tensor.
5. The collector copies the resulting `avg_ans_log_probs` into the current
   node's record. If the node is expanded, the next iteration preprocesses
   the same `node_management.obs` for its child's generation input.

The saved node `page_type` describes the **pre-action** page, whereas the
scoring snapshot describes the **post-action** page. These can differ without
an off-by-one error. Actual environment termination bypasses page probes and
uses the observed purchase outcome, as described below. Pruning or reaching
the rollout step limit does not shift the scoring context back to the input
page; deferred page scores still follow the rules below.

The synthetic root is an exception to this prompt-based scoring path:
currently its score is inferred from its children's log scores. No
pseudo-rollout is run from the initial no-history search prompt to score the
root. See "Deferred tree scores" below.

## Batch contract

The adapter accepts the observation batch described in
[`info_gain_batch.md`](info_gain_batch.md), including duplicated or reordered
rows. These additional aligned NumPy arrays are required:

| `non_tensor_batch` field | Meaning |
| --- | --- |
| `webshop_scoring_payload` | Immutable turn snapshot plus the native products/prices needed for this turn; `None` for terminal or inactive slots |
| `webshop_active` | Slot was active at the start of the step |
| `webshop_terminated` | Actual environment `done`, before applying the rollout step limit |
| `webshop_won` | Observed full-reward purchase outcome |

It returns the same `DataProto`, adding:

- `batch["avg_ans_log_probs"]`: log success probability, retaining the old
  field name for compatibility. This is **not** a mean token log probability.
  Zero probability is `-inf`; an unscored observation is `NaN`.
- `batch["webshop_scored"]`: Boolean mask identifying actual scores.
- `non_tensor_batch["webshop_success_probability"]`: floats or `None`.
- `non_tensor_batch["webshop_skipped_reason"]`: reason for an unscored row.

Initial search pages and item subpages return zero-probability placeholders
with reasons `deferred_initial_search_page` and `deferred_item_sub_page`.
The rollout resolves these using the tree rules below. Other unsupported pages
still retain `None` and **raise an error when active**. Inactive slots are
ignored. Actual terminated rollouts receive `1.0` for a full-reward purchase
and `0.0` otherwise; native WebShop's automatically reset session is never
scored as the purchased state. Hitting the rollout step limit does not invent
an observed purchase outcome.

In probability-difference mode the rollout stores probabilities as before.
In log-difference mode, log scores are bounded below by
`log(algorithm.treehca.prob_floor)` before computing differences, avoiding
`-inf - -inf` and undefined branch weights. The recorded WebShop probability
still retains exact zero. Both modes stop on active `None` scores.

The WebShop path does not pad/reorder observation slots unnecessarily: it
pads the generated teacher-forcing requests to the actor worker-group size.
The answer-block path still uses its original `adjust_batch` flow.

## Deferred tree scores

[`webshop_deferred_scores.py`](../../treehca/webshop_deferred_scores.py) maintains
a separate virtual root for each prompt-group `uid`. Roots are not inserted
into the training node list. Their initial placeholder is probability zero,
or `avg_ans_log_probs = -inf`, until their children are available.

- Each root's log score is the arithmetic mean of its children's log scores.
  In probability space this is a geometric mean, not an arithmetic mean.
- Every deferred initial-search node uses its own group's root log score,
  including search pages reached again later in the trajectory.
- A deferred item subpage uses its first child's log score. The first child is
  the first active rollout slot for that parent when the next level arrives;
  later siblings do not replace it. A subpage that never gets a child retains
  its zero-probability placeholder.
- First-child dependencies remain linked: if that child is also deferred,
  its eventual score propagates to its deferred ancestors.

A root child can itself depend on the root mean through an initial-search
page. In that case, average the children whose scores do not depend on the
root, and assign that mean to all root-dependent children. This satisfies
`root = mean(root children)` without a circular calculation. If every child
depends on the root, retain the placeholder. A childless item subpage is still
an independent placeholder and participates in the mean as `-inf`.

Each iteration first registers the new active nodes, resolves all deferred
log scores and root means, and updates cumulative scores in the parent lookup.
It then updates historical parents' gains before computing the current
children's gains. Saved dictionaries in `total_batch_list` are updated in
place, including already-terminal siblings whose parent baseline changed.
This keeps training records consistent with the values used for current
branching. Past branch allocations are not replayed.

WebShop node records retain `avg_ans_log_probs` and
`webshop_root_avg_ans_log_probs` in `non_tensor_batch`, as well as the effective
`webshop_success_probability`, `info_gain_sum`, and `info_gain`. Deferred reason
strings are retained to identify the source of an inferred score. Raw log
scores are averaged before applying the log-difference floor described above.
All deferred handling is confined to the WebShop scoring path; other
environments retain their existing zero root baseline and gain computation.

## Inference

[`training_pseudo_probes.py`](../../treehca/training_pseudo_probes.py) calls
`actor_rollout_wg.compute_log_prob` using the current training actor:

- Results probes sum log probabilities for tokens overlapping the action text
  after `click[`, then exponentiate. A token that merges part of `click[` with
  the product identifier still counts.
- Each product option group gets its own prompt listing exact option names and
  a synthetic `none` choice. The assistant response is
  `<think> The best choice for the {group_name} group is {choice_name}`.
  Teacher forcing sums log probabilities for tokens overlapping the choice
  name. The preceding text provides context but does not enter that sum,
  unless a token merges it with part of the choice. The raw joint probabilities of
  rewarding choice combinations are summed for native reward aggregation,
  with the weighting controlled by `webshop_include_partial_scores` below.
- `algorithm.treehca.webshop_prune_unsuccessful_choices` defaults to `true`.
  It scores only option names that occur in at least one full-reward purchase
  combination, using the native option-coverage plan. Set it to `false` to
  score every displayed option and `none`; choices outside full-reward
  combinations still contribute zero to the final probability in the default
  mode. With partial scores enabled, pruning instead keeps choices belonging
  to at least one **positive-score** combination.
- Identical teacher-forcing requests are deduplicated per stage. Up to
  `algorithm.treehca.webshop_probe_batch_size` unique rows (default 32) are
  submitted per call, followed by any required worker-divisibility padding.
- Requests are left padded with explicit attention and position tensors. When
  a request would exceed the context limit, training rebuilds the scorer-only
  prompt after dropping the oldest captured observation/action pair, repeating
  until it fits. Retained entries and the current observation are never cut at
  token boundaries; all history may be removed if necessary. If the request
  still overflows with no history, it is an error. The policy rollout prompt
  and standalone scorer are unchanged.
- Training's `data.apply_chat_template_kwargs` also applies to pseudo prompts.

FSDP and Megatron workers honor `treehca_probe_temperature=1.0` only when
explicitly supplied by these probe requests. All other log-probability calls
keep their configured temperature. No new model, vLLM engine, generation RPC,
or vLLM V0/prefix-cache configuration is needed. This initial implementation
performs one forward row per complete option-name response.

For a page with a `color` group containing `red` and `blue`, one pseudo
rollout has this shape (the task and observation precede this excerpt):

```text
Your available options for color are:
[
red
blue
none (do not select any option in this group)
].

Now choose one option for the "color" group that best satisfies the user's needs. The "none" choice means never clicking an option in this group. Think about what is the best choice inside <think>...</think> before giving your answer.
```

The `red` probe appends `<think> The best choice for the color group is red` after the assistant boundary.
If only `red` can occur in a full-reward combination, the default setting
scores just that response. With pruning disabled, the `blue` and `none` probes
use the same prompt with their respective exact names. A second option group
receives a separate prompt and set of probes.

The context limit is `actor_rollout_ref.rollout.max_model_len`, falling back
to `data.max_prompt_length + data.max_response_length` when unset. Pruning uses
`algorithm.treehca.webshop_path_probability_threshold` (default `1e-3`).

## Optional partial-score weighting

Enable this training-only mode with the existing Hydra configuration key:

```bash
algorithm.treehca.webshop_include_partial_scores=true
```

The default is `false`, preserving perfect-purchase scoring. Both
`WebshopInfoGainScorer` and `TrainingWebshopTurnSuccessScorer` also accept the
Boolean keyword `include_partial_scores`. The rollout collector forwards the
configuration to the adapter, which forwards it to each catalog scorer.
Non-Boolean values raise `ValueError`.

With this setting enabled, the compatibility field
`webshop_success_probability` contains **native-score-weighted pseudo mass**,
and `avg_ans_log_probs` contains its logarithm. It is no longer specifically
the probability of a perfect purchase. Actual environment-terminal rows still
use the observed binary `webshop_won` outcome, and training purchase rewards
and terminal penalties retain their behavior described below. This option
changes the nonterminal page estimates.

For a product with unresolved groups `G`, let `p_g(a)` be the raw
teacher-forced probability for choice `a` in group `g`, and let `R(c)` be
WebShop's native purchase reward for the combined choices `c`. The estimate is

```text
S(product) = sum over combinations c [R(c) * product over g in G p_g(c_g)]
S(results) = sum over retained visible products i [p_entry(i) * S(i)]
```

`R(c)` includes product type, attribute matches, native price eligibility,
and option matches; it is not a fraction invented by the scorer. A product
whose best configuration earns a positive partial score is eligible for
entry probes. A product whose maximum native score is zero is excluded.
The existing entry-probability threshold is applied to `p_entry(i)` before
option probes; it is not applied to reward-weighted mass or option choices.
Final estimates retain the existing clamp to `[0, 1]`.

Writing `M_i = max_c R_i(c)`, a retained product's contribution can also be
expressed as `p_entry(i) * M_i * sum_c [(R_i(c)/M_i) * product_g p_g(c_g)]`.
The code directly sums `R_i(c)` instead of dividing and multiplying by `M_i`;
this applies the maximum-score weight exactly once. Before option factors,
its entry contribution is `p_entry(i) * M_i`.

The implementation in
[`webshop_option_success.py`](../../treehca/webshop_option_success.py) is specific
about combinations:

1. `build_native_option_score_plan` uses native `get_option_reward` fuzzy
   matching to map each displayed option to a bit mask of requested targets
   it satisfies. These are target matches across groups, following native
   reward semantics; overlapping matches are counted only once. Each mutable
   group also has `none`, whose mask preserves that group's current selection
   (zero for a fresh entry).
2. It builds reachable masks with one concrete selection witness per mask,
   and evaluates native `get_reward` on those witnesses to populate
   `scores_by_mask` and `max_score`. Native reward depends on the union of
   matched targets, so configurations sharing a mask share a score. This
   avoids enumerating the Cartesian option grid; the state space is bounded
   by `2 ** number_of_requested_targets`.
3. Already selected values matching a target are held fixed only when they
   allow a maximum-score completion. A matching selection that blocks the
   best achievable score remains mutable. Groups whose choices cannot change
   reward are omitted. If every reachable configuration has the same score,
   the plan returns that score directly without option probes, including
   products with no options or no requested option targets.
4. When `webshop_prune_unsuccessful_choices=true`, the plan's
   `successful_actions()` keeps any choice occurring in a positive-reward
   completion. Wrong choices and `none` often remain because product/price/
   attribute credit, or another group's correct choice, gives their
   combination positive reward. Pruning disabled scores all choices in the
   required groups. Zero-reward combinations contribute zero either way.
5. `NativeOptionScorePlan.aggregate_success_mass` starts with mass one on
   the fixed coverage mask. For each required group it multiplies existing
   mass by each probed choice's raw probability, merges equal union masks
   with `math.fsum`, then sums `mass(mask) * scores_by_mask[mask]`.
   Group choices are treated as independent, as in the original estimator.
   No normalization or `1 - P(correct)` substitution occurs. An unprobed
   choice has zero mass; missing answer mass does not receive partial credit.
6. The inherited search aggregation multiplies entry probability by this
   product estimate. Fresh-entry and results caches store the weighted value
   for the scorer's fixed mode and are still cleared on policy-version changes.
   Configured pages retain selection-specific plans and do not overwrite the
   fresh-entry score.

For an independent worked example, suppose a backpack request specifies two
attributes, a price limit, and three option targets: oval shape, large size,
and 30-liter capacity. A candidate meets one attribute and the price limit,
has native type multiplier one, and offers all three target options. The
native denominator is `2 attributes + 3 option targets + 1 price = 6`.
Its reward is `(2 + number_of_matched_option_targets) / 6`, so its maximum is
`5/6` despite all option targets being achievable.

Assume these probe probabilities, chosen to sum to one per group for this
arithmetic example:

| Group | Target choice | Other choice | `none` |
| --- | ---: | ---: | ---: |
| Shape | oval: 0.60 | round: 0.25 | 0.15 |
| Size | large: 0.50 | small: 0.30 | 0.20 |
| Capacity | 30 liters: 0.40 | 20 liters: 0.40 | 0.20 |

The nine distinct labels create 27 configurations. Merging the other choice
and `none` for display gives the following eight coverage states; the code
adds their separately probed masses when they share a mask.

| Matched targets | Joint mass | Native score |
| --- | ---: | ---: |
| None | 0.12 | 2/6 |
| Shape only | 0.18 | 3/6 |
| Size only | 0.12 | 3/6 |
| Capacity only | 0.08 | 3/6 |
| Shape and size | 0.18 | 4/6 |
| Shape and capacity | 0.12 | 4/6 |
| Size and capacity | 0.08 | 4/6 |
| All three | 0.12 | 5/6 |

Thus the product estimate is
`0.12*(2/6) + 0.38*(3/6) + 0.38*(4/6) + 0.12*(5/6) = 7/12`.
If its entry probability is `0.24`, the pre-option weighted contribution is
`0.24 * (5/6) = 0.20`. The score relative to its maximum, after accounting for
options, is `(7/12)/(5/6) = 0.70`, giving final results-page contribution
`0.20 * 0.70 = 0.14`, equivalently `0.24 * (7/12)`. The maximum score is not
multiplied a second time. In actual probing, group masses need not sum to
one; the same sum of products uses the measured raw values.

## Native state and branching

Only TreeHCA training uses the specialized worker and manager in
[`training_webshop_env.py`](../../treehca/training_webshop_env.py). Validation
and other algorithms retain the existing WebShop classes.

Workers capture native page type, ASIN, selected options, full reward goal,
and bounded prompt history. The driver receives only relevant products and
their actual native prices, not an environment or search index. Catalog
identities include products, prices, and rendering settings: workers with
different seeded prices cannot share cached probabilities. Hypothetical
product entries use the existing isolated native renderer.

The specialized manager also supplies the previously missing WebShop tree
fork and terminal-node metrics. A fork copies the live session, browser,
history, task, and prices into the freed slot. Mutable state is isolated, and
prices are restored to that worker's original table on the next rollout reset.

A fresh adapter is created for each tree rollout collection, during which
actor weights are fixed. Its caches therefore cannot survive a policy update.
The adapter API also requires `policy_version`; changing it clears probability
caches while retaining model-independent plans.

## Terminal penalties and padding-independent GRPO

TreeHCA records WebShop's native `task_score` on every rollout row so full
success can be distinguished from partial purchases. The worker's ordinary
training reward remains binary: a perfect purchase is 10 and other outcomes are
0. If an actual environment-terminal row has native `task_score == 1.0` but
its training score is nevertheless zero, the trainer restores the full
training reward of 10.0. Invalid full-success terminals use a fixed 0.5
format/language penalty, so they receive 9.5; valid full-success terminals
remain at 10.0.

The restoration and stronger penalty do not apply to positive partial
purchases: they remain 0 when valid and -0.1 when invalid. They also do not
apply to nonterminal turns, allocation-pruned leaves, turn-limit leaves, or
zero-score failures. Full-success rows that already carry the normal reward do
not need restoration, but invalid ones still receive the fixed 0.5 penalty.
The trainer reports the restoration count as
`episode/full_success_terminal_scores_restored`, and saved TreeHCA rollouts
expose the source value as `webshop_task_score`.

TreeHCA's `q_hindsight` estimator combines a GRPO term with its Q auxiliary
term. `adjust_batch` may append copies of random rows solely to make the batch
divisible by worker micro-batch sizes. Those rows still train and receive the
same advantage as their logical `node_uid`, but only the first row for each
logical node contributes to the GRPO group mean and standard deviation. Thus
hardware padding no longer changes the TreeHCA baseline. The generic GRPO
estimator retains its previous behavior unless a logical-row deduplication key
is supplied.

## Verification

Run in the existing `webshop` environment with `PYTHONPATH=.`:

```bash
conda run --no-capture-output -n webshop env PYTHONPATH=. python -m pytest \
  tests/treehca/test_training_webshop_scoring.py \
  tests/treehca/test_webshop_deferred_scores.py \
  tests/treehca/test_webshop_turn_success.py \
  tests/treehca/test_pseudo_rollout_results_page.py \
  tests/treehca/test_pseudo_rollout_product_page.py \
  tests/treehca/test_webshop_success_probability.py -q
```

These tests exercise native catalog/rendering, controlled actor responses,
probability aggregation, padding/reordering, cache invalidation, episode-state
transfer, deferred ancestor/root updates, and collector dispatch. They do not
run distributed GPU training.
