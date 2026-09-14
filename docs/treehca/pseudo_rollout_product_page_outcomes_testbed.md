# Product-page continuation outcome testbed

[`pseudo_rollout_product_page_outcomes_testbed.py`](../../treehca/pseudo_rollout_product_page_outcomes_testbed.py)
compares `score_product_page_grouped_choice_rollouts` with the final outcomes of
real WebShop continuations. Each query has one fixed, initially unconfigured
product page. Independent environments continue from it for **at most 13
additional actions**. Their purchases receive the native environment's reward.

`--count_partial_reward` changes the headline comparison to purchases with
**positive native reward**, including both full and partial credit. Expected raw
rewards are reported in both modes, using empirical trajectories and the native
rewards of independently weighted pseudo option combinations.

The existing [grouped-choice testbed](pseudo_rollout_product_page_grouped_choices_testbed.md)
measures the distribution of actions on one turn. This testbed instead measures
whether the agent eventually buys, and whether that purchase earns full reward.
It does not change the scorer, WebShop reward rules, or any trainer.

## Running it

Run from the repository root. `PYTHONPATH=.` is required because the conda runner
does not add this checkout's `treehca` package to the import path.

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_product_page_outcomes_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --samples-per-page 256 \
  --max-steps 13 \
  --output pseudo_prob_test_results/product_page_outcomes.json
```

Use `--model /path/to/merged/checkpoint` to evaluate a trained policy. Pass
`--tokenizer` only if its tokenizer resides elsewhere. The inference path uses
vLLM V0, configured before importing vLLM, because the existing constrained-label
scorer requires its masked logprob behavior. Native WebShop requires the same
catalog, attributes, spaCy model, Java/Pyserini, and search index as training.
No external shopping website or real transaction is involved.

To count both full and partial rewards, add `--count_partial_reward` (the alias
`--count-partial-reward` is also accepted):

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_product_page_outcomes_testbed \
  --count_partial_reward --num-pages 20 --samples-per-page 256 \
  --output pseudo_prob_test_results/product_page_partial_rewards.json
```

A model-free check constructs every start and verifies a successful native
purchase for each query:

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_product_page_outcomes_testbed \
  --validate-only --num-pages 20 \
  --output /tmp/product_page_starts.json
```

This mode loads neither a tokenizer nor model weights. Its report is explicitly
marked `mode: validate_only`; it contains setup prompts and oracle receipts,
not empirical policy estimates or pseudo probabilities.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--num-pages` | 20 | Distinct sampled products, each paired with one generated query |
| `--samples-per-page` | 256 | Independent continuations of exactly the same starting state |
| `--count_partial_reward` | off | Replace headline full-credit rates with positive-reward rates and integrate pseudo mass over positive-reward combinations |
| `--max-steps` | 13 | Additional actions; positive values up to 13 are allowed |
| `--history-length` | 2 | Production memory window; must be positive |
| `--results-size` | 10 | Constructed result set size, from 2 to 10 |
| `--seed` | 0 | Catalog prices, goal construction/order, sampling, setup, and inference seeds |
| `--max-prompt-length` | 4096 | Ordinary prompt limit with training's `truncation=error` |
| `--max-new-tokens` | 512 | Token budget for each ordinary response |
| `--temperature`, `--top-p`, `--top-k` | 1, 1, -1 | Ordinary policy sampling settings |
| `--rollout-batch-size` | 64 | Maximum live prompts in a vLLM call |
| `--num-products` | 1000 | Native catalog/search-index size |
| `--max-model-len` | model default | Optional vLLM context limit |
| `--store-trajectory-prompts` | off | Save every ordinary continuation prompt as well as its actions and observations |

`--catalog` and `--attributes` override the default 1,000-product files. Use a
matching native index size with `--num-products`. vLLM also accepts the usual
`--tensor-parallel-size`, `--gpu-memory-utilization`, `--dtype`, and
`--trust-remote-code` options exposed by this CLI. There is no generated-thinking
conditioning mode in this testbed: the pseudo measurement uses the existing
grouped testbed's deterministic assistant cue.

## 1. Sample the queries and matching products

`create_server` constructs a real `SimServer` from this checkout. The server
loads products and prices once, generates synthetic goals (`human_goals=False`),
and shuffles those goals with the configured seed. The scorer and all native
continuations use these same product records, goals, and prices.

`sample_diverse_product_goals` is reused from the grouped-choice testbed:

1. Retain products with at least one option group containing two or more values,
   and goals with a nonempty `goal_options` mapping.
2. Exclude products with missing titles or repeated option fragments that
   violate the established parser contract.
3. Balance the sample across action-count strata, prefer category diversity,
   and sample one generated instruction for each selected product, without
   repeating products.
4. Preserve the **exact** selected goal, including attributes, price bound, and
   option mapping. Ambiguous identical instructions with conflicting option
   mappings raise an error.

This follows the other testbeds' product-stratified sampling, rather than
sampling instructions uniformly. All generated synthetic goals are candidates;
this is not restricted to the training manager's goal-index split, and it does
not use the goal weights for selection. The report records the full goals and
the number available. Selection tasks are dataset instructions, not searches
generated by the evaluated policy.

## 2. Establish a real product-page state and its history

`construct_start_episode` gives the selected goal a private native server session
and instantiates `WebAgentTextEnv(observation_mode="text")`. It resets to that
goal's search page, then constructs a result set:

- Include the goal's own product, guaranteeing that it is visible.
- Seed-shuffle other catalog products, then prefer those with the same catalog
  query and category as distractors.
- Fill the remaining result slots and shuffle the target's position.
- Render the set through WebShop's actual `map_action_to_html("search", ...)`
  template. Set the native session's keywords, page number, search counter,
  browser URL, and HTML consistently with that observation.

The setup search text is the product's catalog `query` (falling back to its
title). This result set is constructed, **not claimed to be a Lucene ranking**.
There is no model decision in the setup. The native search index is loaded by
the server but is not queried to choose these results.

The code then actually calls `env.step("click[<target asin>]")`. The environment
resolves the visible product link and opens the product page, records the
visited ASIN, and begins with an empty selected-options dictionary. The testbed
checks all of these conditions.

The production `WebshopEnvironmentManager` and `SimpleMemory` build the agent
prompt. The memory contains the pre-action observations paired with their
projected actions, exactly as the training manager stores them:

| Training step | Observation used for the action | Action |
| --- | --- | --- |
| 1, constructed | Initial search page | `search[<catalog query>]` |
| 2, executed natively | Constructed search results containing the target | `click[<target asin>]` |
| 3, first sampled action | Target product page with no options selected | Sampled by the policy |

Thus 13 continuation actions correspond to training steps **3 through 15**.
The setup steps and oracle check do not consume the continuation budget. If the
production formatter's 13,000-character fallback removes the starting history,
the testbed raises an error instead of accepting a start that violates this
experiment. A smaller `--results-size` or `--history-length 1` can shorten setup
history while retaining the product-selection step.

## 3. Prove that this product can satisfy its query

`validate_full_reward` clones the start, clicks the goal option values, and
clicks `buy now`. The actual environment must report a purchase with raw reward
**exactly 1.0**. Otherwise the run stops and reports the native reward and chosen
options; it does not change the reward function, repair a goal, discard the
failure silently, or substitute fuzzy option correctness for native reward.

This verifies the product, attributes, price, and option combination together.
The oracle's receipt and reward components are saved for each page. Its actions
and outcome never enter model history or empirical statistics.

## 4. Score the starting page's grouped pseudo prompts

The same history-bearing prompt used for the first ordinary action is parsed
with `extract_product_page_contexts`. The exact goal option mapping is passed to
`prepare_product_page_grouped_choice_rollouts`.

For each displayed option group, that helper preserves the task, product-page
observation, and history. It replaces the admissible-action list with labeled
choices for this group and replaces the final instructions with the group's
choice instructions. Correct-option metadata is used to evaluate the output;
it is not an extra answer hint added to the prompt.

After the tokenizer's assistant generation boundary, the testbed appends:

```text
<think>The best choice for the {group_name} group corresponds to the label:
```

`score_product_page_grouped_choice_rollouts` takes one constrained next-token
step. The existing scorer selects tokenizer-safe labels, permits both the bare
label token and its leading-space variant, combines their mass, and normalizes
over allowed labels. The probe uses temperature 1, as implemented by the scorer.
Changing ordinary rollout sampling settings does not change the pseudo probe.

For group `g`, sum all fuzzy-correct labels' probabilities:

```text
q_g = sum of pseudo probabilities of correct options in group g
q_all = product of q_g over every displayed option group
```

Unlike the next-action grouped testbed, **singleton groups are retained**. Their
pseudo correct mass is normally 1, while a real agent can still fail to select
them before buying. Every starting group must produce a score; context overflow
or missing scores stop the run instead of computing a product over an
incomplete set of groups. No later page is pseudo-scored during a continuation.

Without `--count_partial_reward`, `q_all` is the headline
**independence-based estimate of choosing all correct options**.
For example, masses 0.8, 0.9, and 1 imply 0.72. It is not a learned joint
probability, and the probes do not model action ordering, revisions, stopping,
subpage visits, or whether a purchase occurs. In partial-credit mode, the
headline pseudo probability instead includes all positive-reward combinations,
as described in section 6. The same group probes support both modes.

## 5. Run independent native continuations

Each sample receives a cloned native `WebAgentTextEnv` and browser, an isolated
server session and goal, and its own production memory. The catalog and search
resources can be shared; selected options, visited products, action counters,
HTML/URL state, and memories cannot. Session IDs and initial observations are
kept identical across clones. Derived BeautifulSoup clickable caches are
reparsed by the environment rather than deep-copied.

For each live sample, `VllmPolicy`:

1. Formats a fresh, single user message containing the production observation
   prompt and its bounded history. It applies the model's chat template with
   `add_generation_prompt=True`.
2. Uses training's `tokenize_and_postprocess_data`, no extra special tokens,
   left padding, and `truncation="error"`; padding is removed for vLLM input.
3. Samples one ordinary full response, with no action/label constraint.
   Defaults match the WebShop training recipe's 512 response tokens and the
   training rollout configuration's temperature 1, top-p 1, and top-k -1.
   Penalties and min-p retain neutral values.
4. Decodes with `skip_special_tokens=True` and calls the production
   `webshop_projection`. Its projected action is sent to WebShop even if the
   response-format validity flag is false, matching training behavior.
5. Executes one environment step unless the exit rule below applies. Invalid
   actions are native no-ops and still consume a step. Actual option clicks,
   replacement of an earlier selection, detail-page visits, and purchases all
   use the environment itself.
6. Stores the pre-action observation and projected action in `SimpleMemory`,
   then rebuilds the next prompt through the production formatter. It retains
   the formatter's normal long-prompt history fallback during continuations.

The standard training prompt contains observations and projected actions; this
is not an accumulating chat of assistant responses. Full sampled responses and
reasoning blocks are not stored; projected actions, including the projection's
short suffix fallback for malformed responses, are stored.
`--store-trajectory-prompts` stores the actual prompts supplied for each
continuation step.

The per-step seed is:

```text
seed + 10000 + page_index * samples_per_page * max_steps
             + sample_index * max_steps + continuation_step_zero_based
```

Assigning seeds to trajectories keeps early termination and batch boundaries
from shifting the remaining trajectories' assigned random streams. Bitwise
reproducibility still depends on the model, vLLM, and hardware. Processing is
batched by live step within each page; finished samples are never sampled again.

### Exit and horizon rules

The testbed interprets projected actions using WebShop's own `parse_action`,
clickables, control constants, and current page name. It intercepts these
transitions **before** execution, counts the sampled action in trajectory length,
and records `executed: false`:

| Projected action / state | Termination reason |
| --- | --- |
| Nonempty `search[...]`, even without a visible search bar | `search_action` |
| Admissible `click[back to search]` | `back_to_search` |
| Admissible `click[< prev]` on the main product page | `back_to_results` |

WebShop accepts a nonempty search action even on a product page, so this is
checked independently of advertised actions. `click[< prev]` **on a detail
subpage** returns to the product page and is executed normally. An inadmissible
click is not an exit merely because its text resembles a navigation action.

A purchase ends with `termination_reason: purchase`. A live trajectory after its
13th sampled action ends with `step_limit`. A purchase on action 13 counts as a
purchase; action 14 is never sampled. An unexpected native terminal state is
recorded as `environment_done`. Infrastructure errors, token-budget errors, and
sampler alignment errors raise exceptions instead of being counted as agent
failures.

### Capture the receipt before auto-reset loses it

`WebAgentTextEnv.step` returns raw reward and immediately resets after a
purchase. `ProductEpisode.advance` retains a reference to the completed session
before stepping and reads its purchase counter, ASIN, final selected options,
and reward components afterward. It does not inspect the freshly reset session
as though it were the purchase state.

Full success uses `purchased and raw_reward == 1.0`, matching the worker's
success criterion. Training maps this to reward 10 and other outcomes to 0;
the report retains the native fractional reward for diagnosing partial
purchases. The oracle always requires full reward. The optional partial-credit
reporting mode counts positive native rewards without changing environment
transitions, the oracle, or the training worker's sparse reward mapping.

## 6. Compare final outcomes with the pseudo estimate

For one page, let `N` be all continuations, `B` all purchases, and `F` full-reward
purchases. The report includes:

| Field | Definition |
| --- | --- |
| `no_purchase_rate` | `(N - B) / N`, including exits and step-limit cases |
| `partial_reward_purchase_rate` | `(B - F) / N`, including zero-reward purchases |
| `full_reward_purchase_rate` | `F / N` |
| `full_reward_given_purchase_rate` | `F / B`, or null if there are no purchases |
| `pseudo_all_correct_probability` | `q_all` from the starting page |
| `pseudo_minus_full_reward_purchase_rate` | `q_all - F/N` |
| `pseudo_minus_full_reward_given_purchase_rate` | `q_all - F/B`, or null |

The no-purchase, partial-purchase, and full-purchase rates partition all samples
and sum to one. The report includes
counts, termination-reason counts, Wilson 95% intervals for the four empirical
rates, and histograms of final option dictionaries paired with purchase status
and native reward. No unrecognized/invalid responses are dropped from the
empirical denominator.

Both comparisons are deliberate. `F/N` measures actual success from this page,
including failure to buy; `F/B` isolates the observed purchases but has selection
bias toward trajectories that chose to buy. Neither is automatically equal to
the independent group probe. WebShop's reward also matches option values across
the purchased set using its fuzzy rules, whereas the grouped scorer labels
correctness within named groups. These distinctions can explain discrepancies
and are not reasons to relabel native receipts.

### Counting partial rewards

With `--count_partial_reward`, a counted outcome is a purchase with `R > 0`,
where `R` is the actual environment reward in `[0, 1]`. A positive partial reward
has `0 < R < 1`; a full reward has `R = 1`. A zero-reward purchase does **not**
count as partial credit. Non-purchases remain in the all-rollout denominator
with reward zero, even if the options selected before exit could have earned
reward in a hypothetical purchase.

Let `P` be the number of positive-reward purchases. The Markdown columns
**Full reward / all** and **Full reward / purchases** become **Full or partial
reward / all** (`P/N`) and **Full or partial reward / purchases** (`P/B`). The
pseudo headline becomes **Pseudo positive reward**. The JSON fields that track
the selected mode are:

| Field | Default | With `--count_partial_reward` |
| --- | --- | --- |
| `reward_event` | `full_native_reward` | `positive_native_reward` |
| `reward_purchase_count` | `F` | `P` |
| `reward_purchase_rate` | `F/N` | `P/N` |
| `reward_given_purchase_rate` | `F/B` | `P/B` |
| `pseudo_reward_probability` | `q_all` | `q_positive` defined below |
| `pseudo_minus_reward_purchase_rate` | `q_all - F/N` | `q_positive - P/N` |
| `pseudo_minus_reward_given_purchase_rate` | `q_all - F/B` | `q_positive - P/B` |

The empirical rates have Wilson 95% intervals in the corresponding
`*_wilson_95` fields. Purchase-conditioned rates and gaps are null when `B=0`.
Both the full-credit and positive-credit diagnostic rates remain in JSON,
regardless of the flag. In particular, `pseudo_all_correct_probability` keeps its
original meaning; `pseudo_reward_probability` is the mode-dependent headline.

The new counts `positive_reward_purchase_count`,
`positive_partial_reward_purchase_count`, and `zero_reward_purchase_count`
distinguish positive credit from zero credit. For compatibility, the old field
`partial_reward_purchase_count` still means all purchases below full reward,
including zero-reward purchases, as in the original report.

### Native reward distribution from the pseudo probabilities

`compute_pseudo_rewards` enumerates the Cartesian product of the option groups:
one displayed option per group, including singleton groups. For combination
`c = (o_1, ..., o_G)`, its independent pseudo probability is:

```text
w(c) = product over groups g of p_g(o_g)
R(c) = native WebShop get_reward(product, exact_goal, price, options=c)
q_positive = sum over combinations c with R(c) > 0 of w(c)
q_native_full = sum over combinations c with R(c) = 1 of w(c)
```

The native function is the same one invoked by `SimServer.done`, with the same
fixed product, goal, price, and lowercased option selections as an actual
purchase. It includes type, attributes, price, and the environment's set-wise
fuzzy option matching. This is not an approximation based on the fraction of
groups labeled correct. Native full-credit probability can differ from the
original product of group-correct masses; both are retained for diagnosis.

Every group and option must be present exactly once, and every group probability
distribution must be finite, nonnegative, and normalized. Rewards are aggregated
into `page.pseudo_rewards.reward_distribution`, a list of native reward values,
probability masses, and counts of contributing combinations. A final mass
normalization removes floating-point roundoff. `combination_count`,
`full_reward_probability`, and `positive_reward_probability` are also stored.
Enumeration is exact and requires `product_g |options_g|` native reward calls
per page, in addition to model inference. It does not step or mutate the rollout
environments. This cost grows with the number of option combinations.

There is no synthetic "do not select this group" choice, exit choice, or purchase
choice added to the pseudo distribution. The pseudo experiment assumes one
selection per group followed by purchase. The empirical trajectories may omit
groups, revise selections, or never buy; these differences remain part of the
comparison.

Because setup verifies a product that satisfies the query, its attribute and
price credit can make **every** option combination earn positive reward. Then
`q_positive=1`, even if many combinations receive less than full credit.
Expected reward below captures that remaining difference in quality. Counting
partial credit is a binary event (`R>0`), not weighting an event by its reward.

### Expected rewards

Expected **raw** rewards are calculated in both modes, always including
fractional credit. They are on WebShop's native 0–1 scale, not the training
worker's sparse 0-or-10 reward scale. For trajectory `i`, define `r_i` as its
actual purchase reward, or zero if it never purchases:

```text
empirical_expected_reward_all = sum_i r_i / N
empirical_expected_reward_given_purchase = sum_i r_i / B
pseudo_expected_reward = sum_c w(c) * R(c)
```

The empirical means equivalently sum each observed reward multiplied by its
empirical frequency, with denominators `N` and `B` respectively. The pseudo mean
weights every combination's native reward by its pseudo probability, including
zero and partial rewards. It is also stored as `page.pseudo_rewards.expected_reward`.
The report includes `pseudo_minus_expected_reward_all` and
`pseudo_minus_expected_reward_given_purchase`, and a second Markdown table
displays all three expected rewards and the two signed gaps. With no purchases,
the all-rollout empirical mean is zero and the purchase-conditioned mean and
gap are null; with no trajectories, both empirical means are null. Wilson
intervals describe the binary rates, not the expected-reward estimates.

For example, suppose four trajectories produce rewards `1`, `0.5`, `0` on
purchases and `0` on an exit. Then `F/N=1/4`, `P/N=2/4`, `F/B=1/3`, `P/B=2/3`,
and the empirical expected rewards are `1.5/4=0.375` and `1.5/3=0.5`. If two
pseudo groups assign probabilities `(0.6, 0.4)` and `(0.7, 0.3)`, their four
combination masses are `0.42`, `0.18`, `0.28`, `0.12`. For illustrative native
rewards `1`, `0.5`, `0.25`, `0` on these combinations, `q_positive=0.88` and
`E_pseudo[R]=0.42 + 0.18*0.5 + 0.28*0.25 = 0.58`. The expected-reward gaps are
`0.58-0.375=0.205` and `0.58-0.5=0.08`.

The pseudo expectation is conditional on the hypothetical purchase of a complete
option combination. Its comparison with the all-rollout empirical mean includes
the effect of non-purchases; comparison with the purchase-conditioned empirical
mean removes those zeros but remains subject to purchase-selection bias. The
testbed does not infer a purchase probability from option probes or multiply the
pseudo estimate by the observed purchase rate.

The summary pools outcome counts and empirical reward means over all
trajectories, reports the page-mean pseudo all-correct probability, selected-mode
probability, and expected reward, and averages absolute gaps **per page**.
The probability gaps follow the selected full/partial-credit mode. Expected-reward
gap averages are stored as `page_mean_absolute_expected_reward_gap_all` and
`page_mean_absolute_expected_reward_gap_given_purchase`. Pages with no purchases
are excluded only from the purchase-conditioned gap average. A single pseudo
gap is not computed against the pooled conditional rate because purchases weight
pages differently. Per-page intervals are the calibration uncertainty estimates;
pooled Wilson intervals are descriptive and do not account for sampling new
queries or variation between queries.

## Representative prompts

The checked-in [exact prompt examples](examples/product_page_outcomes_prompts.json)
contain complete ordinary and pseudo prompts, Qwen chat framing, the setup
observations, the next prompt after an option click, and a native oracle receipt.
They were generated from catalog seed 0, sampled page index 10, history length 2,
and results size 10, using product `B086VP8Q9T`. There are no fabricated model
probabilities in that artifact.

Here is the ordinary starting prompt. Only the long results observation is
abbreviated below; the JSON example preserves it in full:

```text
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: Find me artificial ingredients, non gmo, gluten free meat & seafood with quality ingredients with size: 3.5 ounce, and price lower than 40.00 dollars.
Prior to this step, you have already taken 2 step(s). Below are the most recent 2 observations and the corresponding actions you took: [Observation 1: ''Search'', Action 1: 'search[meat & seafood]']
[Observation 2: '<the complete ten-product results observation>', Action 2: 'click[b086vp8q9t]']
You are now at step 3 and your current observation is: 'Back to Search' [SEP] '< Prev' [SEP] 'size' [SEP] '3.5 ounce' [SEP] '10.5 ounce' [SEP] 'Hen of the Woods Bourbon Barrel Smoked Peppercorn Meat Rub - 3.5 ounce - Seasoning for Meat, Pork, Poultry or Seafood' [SEP] 'Price: $7.99' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Buy Now'.
Your admissible actions of the current situation are:
[
'click[back to search]',
'click[< prev]',
'click[description]',
'click[features]',
'click[buy now]',
'click[3.5 ounce]',
'click[10.5 ounce]',
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```

For the `size` pseudo probe, the introduction, task, history, and current
observation above are unchanged. Its action section and final instructions are:

```text
Your admissible actions of the current situation are:
[
A: 3.5 ounce
B: 10.5 ounce
].

Now you must select exactly one option for the "size" group.
You must give the letter (e.g. A, B, C, AB) corresponding to the option you want to select (NOT the name of the option). You should think about which option in the "size" group best satisfies the query, and finish your thought with "The best choice for the size group corresponds to the label:" followed by the label of your chosen option.
```

The Qwen chat template closes that user message and opens the assistant message;
the testbed appends the following unfinished assistant prefix and scores the
very next token:

```text
<think>The best choice for the size group corresponds to the label:
```

For ordinary continuations the artificial prefix is absent. If the projected
ordinary response is `click[3.5 ounce]`, the next prompt has completed-step count
3, current-step count 4, and the most recent two memory records are the result
selection and the option selection. The real session now holds
`{"size": "3.5 ounce"}`. Text-mode WebShop may display the same option text before
and after selection; the session and action history carry the selection. A
subsequent `click[buy now]` in this example earns raw reward 1.0. The complete
post-click prompt is also included in the example JSON.

## Outputs and verification

`--output` writes a schema-version-2 atomic JSON report and same-stem Markdown
probability and expected-reward comparison tables. The JSON contains
`configuration` (including `count_partial_reward`), `sampling`, `pages`, and, after all
pages complete, `summary`. Each page contains the exact goal and price, setup
observations/prompts, oracle receipt, group probe prompts/probabilities,
trajectories, and outcome statistics. Each trajectory includes its termination
reason, final options, raw reward, and step records with seeds, projected
actions, format-validity flags, execution flags, and native observations.

The starting and pseudo prompts are always saved. Complete continuation prompts
are optional because they repeat substantial text. A checkpoint is written
after each completed page. `status: running` and `pages_completed` identify a
partial inference report; `status: complete` means all requested work for the
reported mode finished. Checkpoints are diagnostic, not automatic resume files.

The focused test suite uses actual native environments for isolation, oracle
purchases, partial purchases, all exit paths, detail-page return behavior,
invalid response projection, horizon exhaustion, and purchase on action 13.
A complete orchestration test uses the cached Qwen tokenizer and controlled
logits/responses in place of GPU inference, exercising the real grouped scorer,
ordinary tokenization, environments, and report writing. Singleton inclusion,
seed preservation, denominators, serialization, and error behavior are covered.

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop python -m pytest \
  tests/treehca/test_pseudo_rollout_product_page_outcomes_testbed.py -q
```

The model-free 20-page default-seed validation has also been run successfully:
all 20 starts retained selection history and their oracle purchases earned
native reward 1.0. This verifies environment setup and reward reachability; it
is not a Monte Carlo calibration result.
