# `info_gain_batch` in the tree rollout loop

This describes `info_gain_batch` at the pseudo-scoring call in
[`rollout_loop.py`](../../agent_system/multi_turn_rollout/rollout_loop.py).
The original answer-block path is described below. TreeHCA's WebShop path adds
native snapshot metadata and uses a separate adapter, as documented in
[`training_webshop_scoring.md`](training_webshop_scoring.md). It pads generated
probes internally instead of padding observation slots with `adjust_batch`.
It also retains raw `avg_ans_log_probs` in the saved WebShop node records so
initial-search and item-subpage placeholders can be resolved from tree children
before their information gains are calculated.

The batch is a `DataProto` made from the observations returned by the preceding
environment step. `preprocess_batch` processes every rollout slot, including inactive
slots. It uses `original_gen_batch_index` to find each slot's original prompt metadata.
The `input_ids` field is then renamed to `prompts`.

Before the call, its main fields are:

| Location | Field | Contents |
| --- | --- | --- |
| `batch` | `prompts` | Left-padded token IDs for the new observations, shape `[B', max_prompt_length]` |
| `batch` | `attention_mask` | Prompt attention masks, shape `[B', max_prompt_length]` |
| `batch` | `position_ids` | Prompt position IDs, normally `[B', max_prompt_length]` for text observations; multimodal observations use additional position channels |
| `non_tensor_batch` | `raw_prompt_ids`, `anchor_obs`, `index`, `data_source`, `ground_truth` | Per-slot NumPy arrays |
| `non_tensor_batch` | `raw_prompt`, `multi_modal_data`, `multi_modal_inputs` | Present when enabled or applicable |
| `meta_info` | — | Metadata carried over from `gen_batch` |

Here `B` is the number of rollout slots, `len(gen_batch.batch)`. `adjust_batch` may
append copies of rows so that `B'` is divisible by
`info_gain_compute_log_prob_micro_batch_size_per_gpu * world_size`. It then reorders
all rows to balance sequence lengths across workers, returning that permutation as
`reorder_index`.

## Relationship to active rollouts

Before `adjust_batch`, there is one prompt per rollout **slot**, so every active
rollout has one corresponding prompt. Inactive slots also have prompts because
`preprocess_batch` does not filter on `active_nodes`. After `adjust_batch`, copied
rows may add more prompts. Consequently, the batch passed to answer-block scoring is **not**
exactly one prompt per active rollout.

On the answer-block path, `compute_answer_block_avg_log_prob` adds `batch["avg_ans_log_probs"]`, a tensor of
shape `[B']`, and returns the same `DataProto`. The caller restores the original row
order with `torch.argsort(reorder_index)` and keeps the first `B` scores. It later
uses `active_masks` to calculate information gain only for slots that were active
at the start of the step.
