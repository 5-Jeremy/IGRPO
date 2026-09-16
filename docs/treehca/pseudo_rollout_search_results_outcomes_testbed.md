# Search-results continuation outcome testbed

[`pseudo_rollout_search_results_outcomes_testbed.py`](../../treehca/pseudo_rollout_search_results_outcomes_testbed.py)
compares pseudo and empirical probabilities of both entering a full-reward-capable
product page and completing a full-reward purchase from the same search-results
state. Each empirical
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
  --path-probability-threshold 1e-3 \
  --output pseudo_prob_test_results/search_results_outcomes.json
```

All visible GPUs are used automatically as independent data-parallel workers.
For example, this uses four one-GPU model replicas:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 PYTHONPATH=. \
  conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_search_results_outcomes_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --samples-per-page 256 \
  --tensor-parallel-size 1
```

Each worker owns a disjoint round-robin shard of starting pages and performs
both the pseudo scoring and empirical rollouts for those pages. Results are
merged back into the original page order. The number of active workers is the
smaller of the available GPU groups and `--num-pages`, so at least four pages
are needed to keep four replicas busy. Samples within one page are not divided
among replicas.

`--tensor-parallel-size N` assigns `N` GPUs to each model replica. Thus four
visible GPUs with `--tensor-parallel-size 2` create two data-parallel workers,
each spanning two GPUs. The number of visible GPUs must be divisible by the
tensor-parallel size. `--data-parallel-size` can cap the number of replicas;
when omitted, every available tensor-parallel group is used.

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
| `--path-probability-threshold` | `1e-3` | Cumulative pseudo-path mass below which expansion stops |
| `--history-length` | 1 | Production prompt-memory window |
| `--seed` | 0 | Goal selection, environment, rollout, and inference seeds |
| `--max-prompt-length` | 4096 | Ordinary rollout prompt limit, with truncation treated as an error |
| `--max-new-tokens` | 512 | Maximum tokens in each sampled policy response |
| `--rollout-batch-size` | 64 | Maximum live continuation prompts in one vLLM call |
| `--data-parallel-size` | all available groups | Independent model replicas; active replicas are also capped by `--num-pages` |
| `--tensor-parallel-size` | 1 | GPUs assigned to each model replica |
| `--num-products` | 1000 | Native catalog and matching search-index size |

The CLI also exposes `--temperature`, `--top-p`, `--top-k`,
`--gpu-memory-utilization`, `--dtype`, `--max-model-len`, `--trust-remote-code`, and
`--store-trajectory-prompts`.

## Starting states

The testbed filters synthetic goals by scoring their stated product and options
with WebShop's native reward function, then uses the product-diverse sampler
from the grouped product testbed. Some generated goals cannot earn full reward
even when their stated options are selected because native option matching is
fuzzy and includes option group names. For each selected goal, it creates a private native WebShop
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

`estimate_search_results_probabilities` receives a cloneable wrapper
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

For product entry, it separately sums the forward-pagination mass times each
full-reward-capable product-link probability. This stops at the product page:
option choices and purchase are not factors. Entry includes every eligible
forward path, even when its mass is below the purchase-scoring threshold. Both
pseudo metrics use the native full-reward capability check for product
eligibility. The original purchase-only estimator retains its threshold-pruned
scoring path.

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

An empirical product-entry event occurs on the first executed transition from
search results to the main page of any product that could earn native reward
`1.0` with some valid option choices. It counts even if the rollout subsequently
selects wrong options or never purchases. Opening a nonqualifying product does
not count. Each rollout contributes at most one entry event.

## Outputs

The JSON report records the validated starting states before model workers are
launched and writes all completed worker results after they are joined. For each
page it includes the setup, oracle, both pseudo probabilities, all projected
empirical trajectories, termination counts, empirical full-reward product-entry
and purchase-success rates with Wilson 95% intervals, and signed
pseudo-minus-empirical gaps for both events.

The Markdown companion shows the per-page comparison plus:

- mean pseudo success probability;
- pooled empirical success probability;
- mean absolute per-page gap; and
- root mean squared per-page gap.

The same pooled empirical rate, mean pseudo rate, mean absolute gap, and root
mean squared gap are also reported for full-reward product entry.

The Markdown report also counts starting pages where the cumulative pseudo
probability of entering a full-reward-capable product on later results pages
is strictly above 1%, and separately where the empirical rate is above 1%.
Each empirical rollout counts only if its first qualifying product entry
follows at least one forward results-page click. The JSON report stores both
later-page probabilities per start and the aggregate counts.

All starts receive the same number of rollouts by construction, so the pooled
empirical rate gives every starting page equal weight.

## Analyze a completed report

Run the companion analysis script after a Monte Carlo testbed run:

```bash
conda run -n webshop env PYTHONPATH=. python -m treehca.analyze_pseudo_rollout_search_results_outcomes_results \
  pseudo_prob_test_results/pseudo_rollout_search_results_outcomes_report.json
```

The script writes separate pseudo-versus-empirical scatter plots for full-reward
product entry and purchase success, plus a directional agreement chart for each
event. It also writes JSON and Markdown summaries beside the plots in a
`<report stem>_analysis` directory. By default, a rate is high when it is
strictly greater than `0.6`; `--high-probability-threshold` changes that cutoff.
The agreement denominator includes pages high in either distribution.
