# Pseudo-rollout probability testbed

[`pseudo_rollout_testbed.py`](../../treehca/pseudo_rollout_testbed.py) tests
whether the forced-choice probabilities produced by the pseudo-rollout prompt
agree with actions sampled from the ordinary WebShop agent prompt. It is a
standalone evaluation program. It does not start WebShop, mutate an environment,
or change training behavior.

## What the testbed evaluates

For each sampled catalog product, the program uses one model and one sampling
temperature in two inference paths:

1. Render the actual WebShop item page, obtain its production no-history agent
   prompt and admissible actions, convert it to the action-label pseudo prompt,
   and capture the forced-choice probability of every action with
   `score_product_page_pseudo_rollouts`.
2. Apply the same tokenizer chat template to the unmodified no-history agent
   prompt and ask vLLM for many full responses. Each response is processed by
   the production `webshop_projection`. The resulting action counts form the
   empirical distribution.

The comparison uses temperature `1.0`, `top_p=1`, `top_k=-1`, `min_p=0`, and
neutral repetition/frequency/presence penalties in both paths. These are the
normal training-rollout defaults in `ppo_trainer.yaml`. Restricting the testbed
to untruncated sampling is intentional: top-p or top-k would operate on two
different vocabularies in the ordinary and forced-choice prompts, so matching
their numeric settings would not define the same truncation.

The full ordinary response is allowed up to `--max-new-tokens` (512 by default)
so the model can emit the requested reasoning and `<action>` block. The pseudo
path still generates only one constrained label token.

## Catalog sampling and page construction

The source is the real 1,000-item catalog
`items_shuffle_1000.json`, loaded through WebShop's `load_products` with
`human_goals=False`, matching the repository's default training configuration.
The program then calls the production `get_goals` function. This reads base
synthetic instructions and attributes from `items_ins_v2_1000.json`, expands
product-option combinations, and adds the same sampled price constraints used
by the environment. The checked-in files currently produce 6,910 goals over
415 distinct products after WebShop's option expansion.

Only products that have at least one generated training goal, a nonempty title,
and unambiguous option fragments are candidates. The last condition rejects the
same duplicate-option pages documented in the
[catalog parser test](product_page_catalog_test.md) if they occur in the goal
set. Products without a generated training goal are intentionally not filled
with invented tasks.

Sampling is seeded and stratified by the exact number of admissible actions.
Each cycle draws no more than one product from each action-count stratum, and
within a stratum it prefers a category not already represented. Selection
inside those constraints is random. This deliberately covers pages with
different option-set sizes rather than producing a simple random sample
dominated by the most common page shape. After selecting a distinct product,
the program randomly chooses one of that product's generated training goals.

Selected pages are rendered with WebShop's real `map_action_to_html`, converted
to text by `WebAgentTextEnv`, and formatted by
`WebshopEnvironmentManager.build_text_obs`. No constructor, search index,
browser server, or trajectory is needed. History length is set to zero for both
the real and pseudo prompts, as permitted for this testbed. The shopping task is
the selected goal's exact generated `instruction_text`.

Every rendered page is checked by `parse_product_page_fields` before model
loading. A renderer/parser contract change therefore fails early instead of
contaminating the inference results.

## Running it

Run from the IGRPO repository root in the required environment. `--model` may
be a Hugging Face model name or a local merged checkpoint that ordinary vLLM
can load. The program automatically sets `VLLM_USE_V1=0` before importing
vLLM. Version 0.8.5's V1 sampler reports full-vocabulary logprobs before its
allowed-token mask, so V1 cannot guarantee that all action-label probabilities
are returned. V0 applies the constraint before calculating returned logprobs.
If the surrounding shell set `VLLM_USE_V1=1`, the testbed logs that it is
overriding it.

```bash
conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --samples-per-page 256 \
  --tensor-parallel-size 2 \
  --output outputs/pseudo_rollout_calibration.json \
  --markdown-output outputs/pseudo_rollout_calibration.md
```

Useful controls are:

- `--seed`: controls product, task, generation, and statistical-bootstrap
  randomization.
- `--num-pages`: number of distinct product pages to select.
- `--samples-per-page`: full ordinary-prompt responses generated for each
  page. Larger values reduce Monte Carlo noise. At least 200 is a reasonable
  first pass; pages with many actions benefit from more.
- `--bootstrap-replicates`: simulated null samples used for each goodness-of-fit
  p-value. The default is 10,000, giving minimum resolution about `0.0001`.
- `--page-batch-size`: number of distinct prompts in each ordinary-generation
  call. Each is still one vLLM request with `n=--samples-per-page`, which shares
  its prompt prefill across all samples. Lower this if output memory is tight.
- `--max-new-tokens`: ordinary response budget. Reducing it saves substantial
  inference but may increase truncated responses without closing action tags.
- `--max-model-len`, `--gpu-memory-utilization`, `--dtype`, and
  `--tensor-parallel-size`: passed to the standalone vLLM engine.
- `--tokenizer`: optional tokenizer override; it defaults to the model path.

The V0 engine enables prefix caching and is initialized with exactly the
`max_logprobs` required by the sampled maximum action count. This is independent
of the training configuration's `engine_kwargs.vllm.max_logprobs` setting.

The program performs real GPU inference and can be expensive. With the defaults,
20 pages and 256 samples produce 5,120 full agent responses in addition to 20
single-token pseudo probes.

### Conditioning on each sampled thought

The default testbed computes one pseudo distribution per product page before
sampling the ordinary agent responses. An alternate paired mode computes a
separate pseudo distribution for every Monte Carlo response, conditioned on
that response's complete sampled `<think>...</think>` token prefix:

```bash
conda run --no-capture-output -n webshop \
  python -m treehca.pseudo_rollout_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --samples-per-page 256 \
  --condition-pseudo-on-thinking \
  --pseudo-score-batch-size 1024 \
  --output outputs/pseudo_rollout_thinking_calibration.json
```

For a well-formed response, the exact generated token IDs through the first
closing `</think>` tag are appended to the pseudo prompt's assistant boundary.
The action itself is never included. A response without a complete thinking
block is still scored with an empty prefix and marked
`no_complete_thinking_block`; consequently, the mode always produces one probe
per Monte Carlo response.

The JSON adds `conditioned_pseudo_rollouts` to each page. Every record contains
the sample index, thinking-prefix length and status, projected action/label,
format validity, entropy, the complete label-probability distribution, the
probability assigned to the projected action, and a tie-aware conditional
top-action match flag. It intentionally omits the decoded chain-of-thought
text to keep the report size manageable. `--pseudo-score-batch-size` limits
how many one-token probes are submitted in each vLLM call; lowering it reduces
peak request/output memory.
Page and aggregate summaries include the paired top-action agreement and mean
probability assigned to the projected action.

The page-level `pseudo_probability` is the mean of the conditioned
distributions for completions that projected to a recognized action, matching
the empirical distribution's conditioning. `pseudo_probability_all_samples`
also reports the unconditional mean over every generated response. Statistical
bootstrap samples use each recognized response's own distribution, forming a
Poisson-multinomial null rather than treating all thoughts as identically
distributed.

This mode is substantially more expensive: the example above performs 5,120
additional one-token pseudo probes instead of 20. Pages are conservatively
checked against `max_model_len` using the full `--max-new-tokens` allowance, so
every generated thinking prefix is guaranteed to fit. Without
`--condition-pseudo-on-thinking`, inference and report behavior remain on the
original single-probe-per-page path.

## Output and interpretation

The JSON file is the complete machine-readable result. It contains the model
and sampling configuration, omitted overlength pages, an aggregate summary,
and for every evaluated product:

- ASIN, category, shopping task, action count, and both prompt lengths;
- pseudo probability and pseudo entropy;
- empirical count and conditional frequency for every action;
- a Wilson 95% interval for every empirical action frequency;
- formatting-valid, recognized-action, and invalid counts;
- up to five malformed or inadmissible response examples;
- total-variation distance, Jensen-Shannon divergence, and a multinomial
  goodness-of-fit result.

The Markdown file is a compact companion with aggregate and per-page results.
The JSON remains authoritative.

### Plotting a completed report

[`analyze_pseudo_rollout_results.py`](../../treehca/analyze_pseudo_rollout_results.py)
turns a completed JSON report into the two requested scatter plots and a small
analysis summary. It does not load a model or repeat inference.

```bash
conda run --no-capture-output -n webshop \
  python -m treehca.analyze_pseudo_rollout_results \
  outputs/pseudo_rollout_calibration.json
```

By default, results go into
`outputs/pseudo_rollout_calibration_analysis/`. Use `--output-dir` to choose a
different directory and `--image-format png|pdf|svg` to choose the plot format.
The directory contains:

- `probability_scatter_by_label.*`, with one point per page/action and color
  indicating the multiple-choice label;
- `entropy_scatter.*`, with one point per page and entropy measured in nats;
- `analysis_summary.json` and `analysis_summary.md`, including the top-action
  agreement fraction and page-level maximizers.

Monte Carlo counts can tie, particularly when the sample count is small. The
reported agreement is therefore tie-aware: a page matches if the set of labels
with maximum pseudo probability and the set with maximum empirical probability
overlap. Pages with no recognized empirical actions are listed and excluded
from both plots and the agreement denominator.

The pseudo prompt is a forced-choice question and assigns no probability to an
invalid response. Consequently, action-distribution metrics condition the
ordinary completions on the production projection yielding an admissible
action. This is also faithful to what the environment attempts: even a response
that fails the separate `<think>` format check can still project to and execute
an admissible action. The report therefore includes recognized actions from
format-invalid responses, records that overlap, and reports these two rates
separately:

- **recognized-action rate**: fraction of all completions whose projected value
  is one of the page's admissible actions;
- **production-format-valid rate**: fraction satisfying the existing
  `webshop_projection` format checks.

A conditional fit should not be treated as evidence that the pseudo prompt
models the complete policy when the recognized-action rate is low.

### Statistical comparison

Total-variation distance is directly interpretable as probability mass that
would need to move to make the distributions equal. Jensen-Shannon divergence
is a finite, symmetric information-distance effect size, reported in nats.
Smaller values indicate closer correspondence.

For each page with at least one recognized action, the testbed calculates the
multinomial likelihood-ratio (G) statistic. Its p-value comes from a parametric
bootstrap: repeatedly draw the same number of actions from that page's pseudo
distribution and compare the simulated statistic with the observed statistic.
This avoids relying on the asymptotic chi-squared approximation when a page has
many low-probability actions. A small p-value is evidence against the pseudo
distribution, but it is not an effect size and becomes increasingly sensitive
as `--samples-per-page` grows.

The summary reports both raw page-level rejections at 0.05 and the number that
remain after Benjamini-Hochberg false-discovery-rate correction across tested
pages. Prefer the corrected count when making an overall judgment. Inspect TV,
Jensen-Shannon, confidence intervals, and invalid rates before deciding whether
a statistically detectable difference is practically important.

## Reproducibility and limitations

The report records all randomization and inference settings. vLLM receives a
distinct deterministic seed for each page. Exact bitwise reproduction can
still depend on model, vLLM, CUDA, parallelism, and hardware versions.

This is a prompt-intervention validation, not an on-policy trajectory
evaluation. Each task targets the product whose page is shown, no options have
already been selected, and no navigation history is present. It
does not test whether probabilities remain faithful after a real sequence of
searches and clicks. It also does not explain a discrepancy: label priors,
action order, the removed reasoning requirement, and the changed response
format can all contribute. Those effects can be isolated in follow-up ablations
using the per-page records.
