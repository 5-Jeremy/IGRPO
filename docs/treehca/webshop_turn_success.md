# Batched WebShop turn success probabilities

The reusable API in [`webshop_turn_success.py`](../../treehca/webshop_turn_success.py)
scores observations before the next policy action. It is not connected to the
training rollout loop. The existing `webshop_success_probability.py` estimator
and existing testbed entry points retain their behavior.

TreeHCA training uses a separate specialization and actor-worker inference
adapter, described in [`training_webshop_scoring.md`](training_webshop_scoring.md).

## Usage

```python
from treehca.pseudo_rollout_batch import MixedPseudoRolloutScorer
from treehca.webshop_probability_snapshot import (
    WebshopSnapshotSource,
    snapshot_webshop_rollouts,
)
from treehca.webshop_turn_success import WebshopTurnSuccessScorer

# Reuse these objects across turns. server is the already-loaded native server;
# rollout states expose env, manager, and the current raw prompt body.
source = WebshopSnapshotSource(server)
probes = MixedPseudoRolloutScorer(engine, tokenizer, batch_size=32)
scorer = WebshopTurnSuccessScorer(source, probes, path_probability_threshold=1e-3)
snapshots = snapshot_webshop_rollouts(
    source,
    rollout_states,
    previous_page_types=previous_page_types,
    terminated=done_flags,
)
scores = scorer.score(snapshots, policy_version=checkpoint_or_training_step)
probabilities = [score.probability for score in scores]
```

For a single slot in a multi-rollout prompt manager, use
`source.capture(env, manager, prompt, rollout_index=i, ...)`. The batch helper
above accepts separate single-rollout managers. Both extraction APIs read only
the environment; hypothetical product rendering never steps, resets, or clones
live environments. Native WebShop must already be importable, as it is when
constructing the native environment. Run Python with `PYTHONPATH=.` in conda's
`webshop` environment.

Results preserve input order and length. Initial search observations and product
detail subpages (Description, Features, Attributes, Reviews) return zero with
`deferred_initial_search_page` or `deferred_item_sub_page`, respectively. These
zeros are placeholders; training resolves them from the rollout tree as described
in [`training_webshop_scoring.md`](training_webshop_scoring.md). Other unsupported
states can return `probability=None` and a `skipped_reason`. A zero on a scored
results/product page means no retained full-reward path. Pass termination flags from the actual step
result: WebShop automatically resets its internal session after a purchase.

## Snapshot contract

[`WebshopTurnSnapshot`](../../treehca/webshop_probability_snapshot.py) is an
immutable dataclass containing ordinary serializable values:

- Full shopping instruction and a canonical JSON copy of the reward goal.
- Current raw prompt body and native page type.
- Effective search terms, results page number, and ordered visible ASINs.
- Current product ASIN and selected options, captured from the native session.
- Immediately preceding page type and termination/randomized-search flags.
- The bounded prompt history, completed-step count, and history-window size.
- An opaque catalog-source key.

The catalog, prices, search index, browser, and environment are not copied into
each snapshot. Snapshots remain valid after their live rollout advances. Text
observations do not expose selected options or reliably identify the current
ASIN, so decoded rollout prompts alone cannot supply a complete snapshot.
Capture from production `text` mode without the native environment's additional
embedded observation/action history; the manager supplies prompt history.

Use one source for environments sharing the same catalog and price-table
objects, including native episode clones. Create a new source/scorer when
catalog data, prices, or rendering settings change. Do not submit snapshots
to another source. The source uses native HTML rendering and production prompt
templates to construct a hypothetical product entry, retaining the correct
bounded history and the 13,000-character history fallback.

## Scoring and pruning

The new `_estimate_search_results_probabilities` is a trimmed current-page
aggregation, coordinated by `WebshopTurnSuccessScorer.score`:

1. Resolve cached and constant probabilities. Prepare action probes for
   full-reward-capable products displayed on each current results page and
   group probes for direct product observations. Score these together.
2. Discard results-to-product branches with probability below the threshold,
   including branches whose product score is already cached. Expand retained
   branches, reusing direct-product scores and fresh-entry cache entries;
   score remaining product groups together.
3. Sum each retained entry probability times its conditional product success
   probability, bounding the result to `[0, 1]`.

There is no pagination traversal, next-page probe, greedy fallback, entry-only
metric, or pruning between option groups. Equality to the threshold is kept;
zero-probability branches require no expansion even with threshold zero. Final
nonzero terminal mass is kept if the product factor lowers it below threshold.
All uncertain groups on a retained product are scored concurrently.

At most **32 individual pseudo prompts** are submitted per inference call,
across both kinds of probes. Larger input snapshot batches are accepted.
Identical tokenized requests are deduplicated within each stage, then their
scores are returned to every owner. Results probes share the tokenization of
their results prompt. One tokenizer-safe label catalog is reused across groups,
stages, and calls. Constant/cached conditional scores require no product HTML
rendering. Model-independent option plans are also reused by full goal, product,
and selected-option state.

## Native product success

Every product group uses the unchanged default product-testbed assistant cue:

```text
<think>The best choice for the {group_name} group corresponds to the label:
```

The next-token distribution is constrained and normalized over each option's
bare/space-prefixed labels, including the original `none` choice and wording.
Results probes instead measure the unnormalized joint probability of the action
tokens within an appended `<answer>...</answer>` response, just as the existing
results scorer does. No sampled thinking is generated.

[`webshop_option_success.py`](../../treehca/webshop_option_success.py) calls
native option matching to encode which goal requirements each choice covers.
It obtains a native full-reward witness to check product type, attributes, price,
and option reachability. Since native non-option reward is invariant under option
selection, full reward then requires complete goal-option coverage. A dynamic
program sums probability mass by coverage rather than enumerating all option
combinations. This matches native full-reward aggregation under independent
group distributions and the following maintained-choice assumption:

- A selected value that matches a native goal target is held fixed if a
  full-reward completion remains possible with it and previously held choices.
  Catalog group order resolves coupled cases deterministically. A partial
  match that blocks full reward still receives a correction probe.
- Groups that cannot change success for any reachable choices of the other
  groups are omitted. Merely being omittable in *some* successful combination
  is insufficient: interchangeable ways of meeting a requirement still need
  their respective probabilities.
- An impossible product returns zero. A product already guaranteed full reward
  by the maintained selections returns one, including eligible products with
  no groups or no required options.

`none` retains the testbed's omission semantics. In a mutable group it contributes
no goal-option coverage; it is not reinterpreted as a corrective click. Preserved
groups never generate a `none` probe. Native aggregation does not estimate the
probability of clicking Buy Now, leaving the product, or correlated group choices.

## Cache boundaries

Every lookup first selects a cache by the full shopping instruction **and full
reward goal**, not by catalog category or search terms alone.

- Results keys additionally contain exact effective search terms, page number,
  ordered visible products, and current observation text. History is deliberately
  ignored. Native `<r>` searches never read or write the persistent results cache.
  For another randomized search implementation, set `randomized_search=True`
  when capturing. Randomized snapshots can share identical requests within a
  call, but their results probabilities are not retained between calls.
- Product probabilities are cached/retrieved only for an unconfigured product
  reached **directly from a results page**, including hypothetical entries in
  stage two. Within the same shopping goal, different searches can share that
  product score. Configured pages, invalid-action repeats, and returns from
  detail pages do not read or write this fresh-entry cache.
- `policy_version` is required on every score call. Changing it clears both
  probability caches. Change it whenever weights, adapters, or scoring settings
  change. `clear_probability_cache()` also explicitly clears them. The scorer's
  configuration and source are otherwise fixed for its lifetime.

The API is synchronous; do not overlap calls on the same scorer or mutate its
shared catalog/model during a call. Native group plans and tokenizer metadata
are independent of model weights and survive probability-cache invalidation.

## Inference requirements and verification

Use vLLM V0 (`VLLM_USE_V1=0` before loading it), with prefix caching disabled for
results prompt logprobs. Set engine `max_logprobs` to at least twice the largest
number of group choices (real options plus `none`) that may be scored. Supply
an engine with the same tokenizer as the scorer. Optional `lora_request` applies
to the entire scorer. Context overflow or missing/misaligned scores raise errors;
they are not converted into probability zero or partial group products.

The focused tests exercise native rendering, transitions, selected options,
and reward aggregation with a cached Qwen tokenizer and controlled engine
outputs. They verify mixed request settings/decoding and the 32-row cap without
loading model weights. They do not constitute a real GPU inference test.

```bash
conda run --no-capture-output -n webshop env PYTHONPATH=. python -m pytest \
  tests/treehca/test_webshop_turn_success.py \
  tests/treehca/test_pseudo_rollout_results_page.py \
  tests/treehca/test_pseudo_rollout_product_page.py \
  tests/treehca/test_webshop_success_probability.py -q
```
