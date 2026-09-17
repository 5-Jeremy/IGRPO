# Training-style rollout and pseudo-probability testbed

[`webshop_turn_success_testbed.py`](../../treehca/webshop_turn_success_testbed.py)
runs N concurrent WebShop trajectories from the initial search-input page,
captures immutable snapshots before each action, and then evaluates all saved
snapshots with the [batched turn scorer](webshop_turn_success.md). It writes
one JSONL row per active rollout turn. No training loop or trainer is modified.

## Run

From the repository root, using the existing `webshop` environment:

```bash
conda run --no-capture-output -n webshop env PYTHONPATH=. \
  python -m treehca.webshop_turn_success_testbed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-rollouts 64 \
  --group-size 8 \
  --output pseudo_prob_test_results/webshop_turn_success_rollouts.jsonl
```

For a trained model, set `--model` to its merged checkpoint directory and
optionally specify `--tokenizer`. Use `--tensor-parallel-size 2`, for example,
to put the single inference engine across two visible GPUs. Independent
data-parallel model replicas are not created by this testbed.

The same engine generates real actions and subsequently scores pseudo probes.
It uses vLLM V0 with prefix caching and chunked prefill disabled, and configures
the catalog-wide label-logprob budget automatically. Run it as a fresh Python
module so V0 is selected before vLLM is imported. Context overflow raises an
error rather than silently truncating observations or pseudo probes.

| Setting | Default | Meaning |
| --- | --- | --- |
| `--num-rollouts`, `-n` | 64 | Total concurrent trajectories |
| `--group-size` | 8 | Trajectories sharing one shopping goal; last group may be smaller |
| `--split` | `train` | Native goal indices 500 onward; `test` uses the first 500 |
| `--goal-indices` | sampled | Optional one native goal index per group, within the chosen split |
| `--seed` | 0 | Catalog, goal selection, environment, and inference seed |
| `--max-steps` | 15 | Maximum policy actions in each complete trajectory |
| `--history-length` | 2 | Production prompt history window |
| `--max-prompt-length` | 4096 | Ordinary policy prompt budget, with truncation treated as an error |
| `--max-new-tokens` | 512 | Ordinary policy response budget |
| `--rollout-batch-size` | 32 | Active policy prompts per inference call |
| `--pseudo-batch-size` | 32 | Pseudo prompts per inference call (1–32); lower values reduce prompt-logprob peak memory |
| `--temperature`, `--top-p`, `--top-k` | 1, 1, -1 | Training sampling defaults |
| `--invalid-action-penalty` | 0.1 | Per-turn training response-format penalty; 0 disables it |
| `--path-probability-threshold` | 0.001 | Strict pruning before results-to-product expansion |
| `--num-products` | 1000 | Native catalog/search-index size |

The pseudo inference batch limit is independent of N or `--rollout-batch-size`.
Other options select catalog and
attribute files, GPU memory utilization, dtype, model context length, and
tokenizer remote-code behavior. `--help` lists the complete interface.

## Training correspondence

The implementation uses the production `WebshopEnvironmentManager`,
`webshop_projection`, and `WebshopWorker.reset/step` methods. Its native sessions
begin on the search-input page: there are no forced setup searches, oracle
filters, restrictions on backtracking, or restrictions on detail-page visits.
Invalid actions still consume turns. Only purchases and the step limit end a
trajectory, and finished slots contribute no further JSONL rows.

N independent environments share one loaded catalog, price table, and search
index. Policy inference is batched across active trajectories; native transitions
execute locally rather than through Ray processes. Each local session has its
own Python/NumPy random state, so random searches and purchase auto-reset cannot
consume a neighboring session's random stream. The engine's RNG is preserved
while native environments are constructed.

Goal groups use training's `RandomState(seed).choice(..., replace=False)` rule
and the same split boundary. All sessions use one shared catalog/goal ordering
and price table, whereas production workers independently initialize servers.
Together with explicit per-rollout/per-turn model seeds and omission of inactive
policy calls, this means the run is not a bit-for-bit replay of distributed
training. The saved configuration and per-rollout goal metadata identify what
was actually evaluated.

Ordinary prompts use the production chat-template/tokenization path with left
padding and `truncation="error"`. The original generated response token IDs are
retained. JSONL `input` and `output` are decoded with `skip_special_tokens=True`,
matching `RayPPOTrainer._dump_generations`; visible `system`, `user`, and
`assistant` text therefore matches the training logs' style.

## Score semantics

The testbed calls the actual `EpisodeRewardManager` with length normalization
disabled, then calls the trainer's `apply_invalid_action_penalty`. The resulting
`score` is the sum of `token_level_scores`, exactly the quantity dumped by the
training logger:

```text
score(turn) = final episode reward - invalid_action_penalty * (not is_action_valid)
```

Native reward exactly 1 on purchase gives episode reward **10**. Partial or zero
native reward, and non-purchase timeouts, give **0**. Every turn of that trajectory
receives its final episode reward, then its own validity penalty. Consequently:

- A valid early turn in a successful trajectory has score 10.
- An invalid turn in that trajectory has score approximately 9.9.
- Valid turns in unsuccessful trajectories have score 0; invalid ones have
  approximately -0.1.

Values retain training's float32 arithmetic, including representations such as
`-0.10000000149011612`. Validity comes from the production response parser:
missing required tags or Chinese characters can mark a response invalid even
when its extracted action executes successfully. Conversely, a properly formatted
but unavailable environment action need not incur this format penalty. `score`
is neither native fractional reward nor an advantage or KL-adjusted reward.

## JSONL and metadata

Rows are sorted by zero-based `rollout_index`, then zero-based `traj_step`, as
in the trajectory-major collection before training's later batch rearrangements.
The training-global `step` field is omitted. Each row contains:

| Field | Meaning |
| --- | --- |
| `input`, `output`, `score` | Training-style decoded prompt, generated response, and score |
| `rollout_index` | Stable integer identifying one of the N trajectories |
| `pseudo_probability` | Success probability of the observation **before this row's action**, or null |
| `pseudo_skip_reason` | Null for scored observations; reason for an unsupported page |
| `uid`, `traj_uid`, `traj_step` | Training-style goal-group UUID, trajectory UUID, and turn index |
| `data_source` | `text`, matching the prepared training dataset |
| `is_action_valid` | Production response-format validity |
| `rewards` | Immediate training environment reward for this action |
| `episode_rewards`, `episode_lengths`, `tool_callings` | Training-style trajectory totals; WebShop tool-call count is 0 |

A product purchase row can have a pseudo-probability: its saved observation is
the product page before purchasing, not the auto-reset search page afterward.
Search-input and product-detail observations retain their rollout records with
null pseudo-probabilities. Configured product snapshots retain their selected
options even after the underlying environment advances or resets.

All real rollouts finish before any pseudo inference begins. The complete
snapshot set is submitted in collection order to the scorer, permitting sharing
between observed fresh product entries and hypothetical product branches.
The scorer's query, selected-option, randomized-search, and model-version cache
boundaries remain in force.

`<output stem>.metadata.json` contains configuration, run status, goal and UUID
mapping for each rollout, terminal outcomes, episode lengths/rewards, and counts
of recorded/scored turns. The per-episode `task_score` retains the last native
fractional reward separately from the training score.

The JSONL is first saved after real rollout scoring with
`pseudo_skip_reason="pending"`. A successful pseudo pass atomically replaces
that file with completed annotations. If pseudo scoring raises an error, the
empirical trajectories remain available, and metadata says `status="failed"`
with the error. Pending annotations must not be interpreted as unsupported pages.

## Validation

```bash
conda run --no-capture-output -n webshop env PYTHONPATH=. python -m pytest \
  tests/treehca/test_webshop_turn_success_testbed.py \
  tests/treehca/test_webshop_turn_success.py -q
```

Tests use real native transitions/rewards, a cached Qwen tokenizer, production
reward functions, and controlled model outputs. They cover successful and partial
purchases, invalid but executable purchases, timeouts, early termination, goal
grouping, snapshot timing, training tokenization, mixed pseudo inference, JSONL
serialization, and preservation of trajectories on a pseudo-scoring failure.
They do not load model weights or verify a real GPU inference run.
