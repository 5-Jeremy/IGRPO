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
  full-reward choice combinations are summed for native reward aggregation.
- `algorithm.treehca.webshop_prune_unsuccessful_choices` defaults to `true`.
  It scores only option names that occur in at least one full-reward purchase
  combination, using the native option-coverage plan. Set it to `false` to
  score every displayed option and `none`; choices outside full-reward
  combinations still contribute zero to the final probability.
- Identical teacher-forcing requests are deduplicated per stage. Up to
  `algorithm.treehca.webshop_probe_batch_size` unique rows (default 32) are
  submitted per call, followed by any required worker-divisibility padding.
- Requests are left padded without truncation, with explicit attention and
  position tensors. Context overflow and missing/misaligned scores are errors.
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
