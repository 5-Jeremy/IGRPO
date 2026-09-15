# Search-results continuation outcome testbed

[`pseudo_rollout_search_results_outcomes_testbed.py`](../../treehca/pseudo_rollout_search_results_outcomes_testbed.py)
compares `estimate_search_results_success_probability` with independent native
WebShop continuations from the same search-results state. Each empirical
continuation receives at most **14 sampled actions**. Success is a completed
purchase whose native WebShop reward is exactly `1.0`.

## Running it

Run from the repository root in the `webshop` conda environment:

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_search_results_outcomes_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --samples-per-page 256 \
  --max-steps 14 \
  --path-probability-threshold 1e-6 \
  --output pseudo_prob_test_results/search_results_outcomes.json
```

Use `--model /path/to/merged/checkpoint` for a trained policy and pass
`--tokenizer` if its tokenizer is stored separately. The testbed uses vLLM V0:
results-page action probabilities require chosen-token prompt log probabilities,
and product-page grouped choices require constrained next-token log probabilities.
Prefix caching is disabled because vLLM V0 cannot combine it with prompt
log-probability scoring.

Use `--validate-only` to construct starts and execute their native forward-path
oracle purchases without loading a tokenizer or model:

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_search_results_outcomes_testbed \
  --validate-only --num-pages 20 \
  --output /tmp/search_results_starts.json
```

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--num-pages` | 20 | Independent search-results starting states |
| `--samples-per-page` | 256 | Native policy continuations cloned from each start |
| `--max-steps` | 14 | Sampled actions after the setup search; values from 1 through 14 are accepted |
| `--path-probability-threshold` | `1e-6` | Cumulative pseudo-path mass below which expansion stops |
| `--history-length` | 1 | Production prompt-memory window |
| `--seed` | 0 | Goal selection, environment, rollout, and inference seeds |
| `--max-prompt-length` | 4096 | Ordinary rollout prompt limit, with truncation treated as an error |
| `--max-new-tokens` | 512 | Maximum tokens in each sampled policy response |
| `--rollout-batch-size` | 64 | Maximum live continuation prompts in one vLLM call |
| `--num-products` | 1000 | Native catalog and matching search-index size |

The CLI also exposes `--temperature`, `--top-p`, `--top-k`,
`--tensor-parallel-size`, `--gpu-memory-utilization`, `--dtype`,
`--max-model-len`, `--trust-remote-code`, and
`--store-trajectory-prompts`.

## Starting states

The testbed reuses the product-diverse synthetic-goal sampler from the grouped
product testbed. For each selected goal, it creates a private native WebShop
session and performs one deterministic setup search using WebShop's `<q>`
catalog-query route. This returns every catalog product with the source
product's exact query and supports ordinary native forward pagination. The
source product is therefore guaranteed to occur on one of those pages; other
products can also be valid full-reward alternatives.

The first policy action sees training step 2. Its prompt contains the initial
search observation and setup search in the production memory format. No
product has been opened and no options have been selected. The report stores
the exact prompt, query, full matching-ASIN list, goal, and price.

Before inference, an oracle clone follows `next >` until it reaches the source
ASIN, selects the goal options, and purchases. Native reward must equal `1.0`.
The oracle path is recorded but never enters either the pseudo estimate or the
empirical samples.

## Pseudo estimate

`estimate_search_results_success_probability` receives a cloneable wrapper
around the exact start. It:

1. Scores correct product links and forward-page actions with
   `compute_results_page_answer_probability`.
2. Multiplies forward-pagination and product-selection probability along each
   retained path.
3. Scores product option groups with
   `score_product_page_grouped_choice_rollouts` and multiplies their total
   correct-choice masses.
4. Prunes partial paths below `--path-probability-threshold`, retaining any
   already completed terminal path, and uses the estimator's greedy nonzero
   fallback when pruning completes none.
5. Sums terminal masses and bounds the result to `[0, 1]`.

The pseudo calculation assumes purchase after the correct option choices; it
does not separately score `click[buy now]`. It also does not impose the
empirical 14-action horizon. These are the same interpretation and limitation
as the estimator itself and should be considered when reading calibration
gaps.

## Empirical continuations

Every sample clones the same native environment, browser, active session,
prompt manager, and memory. The ordinary policy uses the training chat template
and unconstrained generation settings. Its text is projected with the
production `webshop_projection` before execution.

The pseudo estimator excludes backtracking and replacement searches. To keep
the empirical event aligned, a rollout terminates as an out-of-scope failure
when it:

- issues any new `search[...]` action;
- clicks `Back to Search`;
- clicks `< Prev` on a results page; or
- clicks `< Prev` on a main product page, which would return to results.

Returning from Description, Features, Reviews, or Attributes to the same main
product page remains allowed. Forward pagination, product selection, option
selection, and purchases use the unmodified native environment. Invalid model
actions are executed with ordinary WebShop behavior and consume one action.

A purchase on action 14 counts; action 15 is never sampled. A non-purchase at
the horizon ends with `step_limit`.

## Outputs

The JSON report checkpoints after every completed starting page. For each page
it includes the setup, oracle, pseudo probability, all projected empirical
trajectories, termination counts, the empirical full-reward success rate and
Wilson 95% interval, and the signed pseudo-minus-empirical gap.

The Markdown companion shows the per-page comparison plus:

- mean pseudo success probability;
- pooled empirical success probability;
- mean absolute per-page gap; and
- root mean squared per-page gap.

All starts receive the same number of rollouts by construction, so the pooled
empirical rate gives every starting page equal weight.
