# Search-results pseudo-probability testbed

[`pseudo_rollout_results_page_testbed.py`](../../treehca/pseudo_rollout_results_page_testbed.py)
generates genuine WebShop search-result pages and scores every displayed action
with `compute_results_page_answer_probability`. It does not sample agent
responses or perform environment rollouts.

For every selected synthetic goal, the testbed submits the product's catalog
query to WebShop's real Lucene search index. It renders one nonempty page from
the returned ranking through WebShop's HTML template. The agent prompt is then
built with history length one, so its history records both the initial search
page observation and the exact `search[query]` action that produced the current
results.

Run it from the repository root in the `webshop` conda environment:

```bash
python -m treehca.pseudo_rollout_results_page_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-pages 20 \
  --output pseudo_prob_test_results/results_page_report.json
```

The interface follows the product-page probability testbed: model/tokenizer,
catalog, seed, context length, tensor parallelism, dtype, GPU utilization, and
JSON/Markdown output paths are configurable on the command line. Use
`--help` for the complete interface.

## Scoring and correctness

Each admissible action is appended to the unchanged prompt as
`<answer>action</answer>`. vLLM chosen-token prompt log-probabilities are exposed
through the same `compute_log_prob` interface used by the training worker. The
testbed calls `compute_results_page_answer_probability`, which sums the log
probabilities of tokens overlapping the action text and exponentiates in
float64. These are raw joint sequence probabilities; they are not normalized
over the displayed actions and generally do not sum to one.

The testbed disables vLLM V0 automatic prefix caching. V0 does not support
combining cached prefixes with `prompt_logprobs`; repeated action probes for the
same page can otherwise fail inside vLLM's sampler with mismatched prompt-token
and log-probability indices.

A displayed product is marked correct if some selection of its available
options makes WebShop's production `get_reward` function return exactly `1.0`
for the active goal. This permits full-reward alternatives to the goal's source
ASIN.

The JSON report retains every prompt, displayed product, correct product, raw
action probability, and tied top action. Its summary and Markdown companion
include:

- aggregate and action-kind probability distributions;
- the percentage of eligible pages on which `click[next >]` is a top action;
- the percentage of eligible pages on which `click[< prev]` is a top action;
- the percentage of all pages on which either pagination action is top; and
- the average summed joint probability assigned to full-reward product actions,
  considering only pages that display at least one such product.
