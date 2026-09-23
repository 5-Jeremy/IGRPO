# WebShop rollout visualizer

From the repository root, run:

```bash
PYTHONPATH=. conda run --no-capture-output -n webshop python -m treehca.visualizer \
  --log-dir runs/ICLR/treehca-webshop/rollouts/20260918T044934.925035877-1763789
```

Open <http://127.0.0.1:8050>. For a remote server, forward the port with
`ssh -L 8050:127.0.0.1:8050 your-host`. `--host` and `--port` override the
local bind address and port. Dependencies are Dash and dash-cytoscape, listed in
`treehca/visualizer/requirements.txt`; install them only with environment-owner
approval. No trainer, Torch, or WebShop environment is imported.

The viewer scans immediate `*.jsonl` children of the supplied directory and
orders filenames naturally (2 before 10). Select a file, then a tree. **Refresh
files** discovers new logs and reloads changed files without restarting. Files
load lazily, with at most three parsed files retained in memory. An incomplete
JSON line produces an error with its line number; refresh after writing finishes.
An empty directory can be opened while waiting for training to produce logs.

The layout and styles are adapted from `example_tree_visualizer`: fixed depth
rows, rounded boxes, directed edges, pan/zoom, **Fit graph**, **Fit subtree**, a tall scrollable
details pane, and full-width bottom tabs. Labels show saved `<action>` text when
available. **Fit subtree** fits the selected node and all of its descendants in
the graph pane without removing other nodes; **Fit graph** restores the whole
tree. Colors distinguish
the synthetic root, internal nodes, and leaves, or any numeric saved field.
Metric colors normalize finite values within the selected tree; missing and
nonfinite values are gray. Leaves describe graph structure, not environment
termination, which is not present in the example logs.

Select **Page type** to color nodes categorically by their saved `page_type`
(the page before the action). Colors stay consistent across trees and files;
the legend lists categories present in the selected tree. The empty-string page
type is labeled **Initial search ("")**, separately from missing data and the
synthetic root. As with numeric colors, duplicate nodes use their first record.

Click a node to inspect saved fields, including unknown future fields, nested
objects, arrays, and full numeric precision. Node/parent/child UIDs, ancestry
paths, and JSONL line numbers are omitted from the details pane. Duplicate saved
records remain individually accessible through Previous/Next record buttons,
shown only for nodes with multiple records. Graph labels and
colors use the first saved record.

The details pane also hides the token arrays `advantages`, `values`, and
`webshop_session_id`; scalar `advantage` and `value` remain visible. Double-click
a field name to pin it at the top, in pinning order; double-click again to unpin.
Pinned names have a star and a blue background. Keyboard users can focus a field
name and press Enter or Space to toggle its pin. Pins survive node, tree, and file changes
within the open page; reloading or closing the page clears them. A pinned field
absent from a record displays “Not saved for this node.” The color menu excludes
`sampled_expansion_count`, `branch_logit`, `webshop_session_id`, and
`webshop_task_id`.

**Saved text & context** splits the selected node's decoded input into
`shopping_task`, `history`, `current_observation`, and `available_actions`, in
four separate scrollable panels. It uses the prompt-body boundaries and shared
outer-context parser described in [prompt terminology](prompt_terminology.md).
The chat wrapper and response instructions are not shown. Absent history is
explicitly labeled; malformed inputs produce a parsing message instead of guessed
sections. A separate searchable pane shows output, any of the four sections,
or the post-action observation. Literal search supports case sensitivity,
Previous/Next, Enter, and Jump to end. Saved markup is displayed as plain text.
Tree/file statistics are derived from records in that file, not training-wide
metrics. The unchanged raw records remain available in the JSONL files.

## Reconstruction and limitations

Each JSONL line must be an object containing `node_path`, ordered from the current
node to its oldest ancestor. Tree algorithms end paths with the `"root"` sentinel;
GiGPO paths end at the first action, with separate trajectories displayed side by
side in their saved `uid` group. Single-node paths are valid. The path reconstructs
edges even if records arrive out of order. When present, a synthetic root joins initial branches; ancestors lacking their
own saved rows are shown as placeholders. Repeated IDs within a path and
conflicting ancestry are rejected instead of producing misleading edges.

The example logs contain `input`, `output`, `score`, `node_path`,
`avg_ans_log_probs`, `info_gain`, `info_gain_sum`, and `page_type`. They do **not**
contain the tree/group ID. The viewer groups first-level branches with identical
initial `input` strings. It labels this grouping as inferred: separate episodes
with identical initial prompts cannot be distinguished from these logs alone.
If the initial record is missing or its input is ambiguous, that initial branch
is shown separately. If records include `tree_uid` (preferred) or `uid`, the
viewer uses that saved ID instead. Save IDs consistently across all node rows.

## Additional TreeHCA logging

New TreeHCA rollouts save the following fields, without schema-version or run/step
metadata. Existing logs cannot recover fields that were never saved.

| Fields | Meaning |
| --- | --- |
| `uid` | Exact tree/group identity, including across repeated prompts. |
| `node_uid`, `parent_node_uid`, `node_path` | Explicit links and node-to-root ancestry. |
| `traj_step` | Zero-based rollout turn counter, distinct from graph depth. |
| `is_terminal`, `deactivate`, `environment_done` | Final training terminal flag, pruning flag, and environment termination before pruning. |
| `termination_reason` | `success`, `environment_failure`, `environment_terminal`, `turn_limit`, `pruned`, or null for continuing nodes. |
| `expansion_probability` | Probability used by the expansion sampler, within the active prompt group. |
| `sampled_expansion_count`, `expansion_count` | Sampled allocation and executed allocation. On the last turn, allocation is still sampled by the existing algorithm but executed count is zero. Counts include the continuing source slot. |
| `branch_score`, `branch_logit` | `(info_gain_sum + info_gain) / 2` at the sampling decision and that score times gamma. These are snapshots, even if deferred scores are revised later. |
| `post_action_observation` | Returned environment observation (`anchor`), before prompt wrapping; includes the terminal observation. Falls back to text for environments without an anchor. |
| `webshop_task_id`, `webshop_session_id` | Original reset task index and session before the action, preserved through forks. Native purchase auto-reset therefore cannot substitute a new session ID. |
| `values`, `advantages` | Training tensor values for response positions selected by `response_mask`, in token order, excluding masked positions. |
| `value`, `advantage` | Means of those unmasked token values, for compact display and metric coloring. |
| `is_action_valid`, `rewards` | Saved action-validity flag and final shaped/aggregated reward in the training batch (see below). |

TreeHCA normally has no critic, so `values` and `value` are null rather than
invented estimates. Empty masked responses also have null means. Nonfinite
additional numbers are represented by strings (`"-Infinity"`, `"Infinity"`,
`"NaN"`) for valid JSON. Terminal failure means the environment finished without
full success, including partial-credit purchases; it does not claim a process
exception occurred. Exceptions that abort a rollout cannot produce a completed
node log. When an environment supplies no success flag, the reason is the less
specific `environment_terminal`. Actual environment termination takes precedence
over the turn limit, which takes precedence over pruning.

GiGPO also saves these shared diagnostics: group and trajectory IDs, node links
and paths, turn, terminal flag/reason, environment termination, pre-action page
type, post-action observation, WebShop task/session IDs, action validity, rewards,
and masked training values/advantages. It additionally saves episode rewards,
episode lengths, and tool-call counts. Node IDs use `traj_uid:traj_step`; initial
nodes have a null parent, without a synthetic root. This also applies to GiGPO's
dynamic sampling through the shared collection loop.

GiGPO does not compute TreeHCA's answer-probability or information-gain estimates;
`avg_ans_log_probs`, `info_gain`, and `info_gain_sum` are null when unavailable.
Pruning and expansion fields are omitted for GiGPO. Serialization reads the final
training batch after reordering, so values and advantages align with saved nodes.
Rollout decisions and reward computation are unchanged. Older GiGPO files that
only saved input/output/score lack trajectory identity and cannot be reconstructed;
use newly generated logs.

## `rewards` versus `score`

`rewards` is serialized from the final `non_tensor_batch["rewards"]`, after reward
computation. It is not an untouched copy of the immediate environment reward:

1. The WebShop worker converts a terminal full-credit purchase to **10**, and all
   other outcomes to **0**. The original WebShop partial-credit score is a
   separate `task_score` in environment info.
2. Rollout gathering replaces a pruned node's reward with `info_gain_sum * 0.5`
   while `global_steps <= stable_steps`, and `info_gain_sum` afterward. With
   `stable_method='threshold'`, the later value becomes 1 if `info_gain_sum >= 0.5`
   and 0 otherwise.
3. The tree reward manager updates internal-node rewards from terminal descendants:
   maximum in `reward_mode='max'` (the current WebShop launcher), or average in
   `reward_mode='avg'`. Terminal nodes retain their own rewards.
4. That reward is placed at the response's last valid token. The exported `score`
   is the sum of `token_level_scores` for the response.

The training entry point sets `normalize_by_length=False`, so **`score` equals
the logged `rewards`**, up to float precision. If length normalization were
enabled, the score would instead be `rewards / subtree_traj_depths`. The score is
computed before any KL reward penalty; it is not an advantage or the native
WebShop partial-credit score.

Source: [WebShop worker](../../agent_system/environments/env_package/webshop/envs.py),
[rollout gathering](../../agent_system/multi_turn_rollout/rollout_loop.py),
[tree reward manager](../../agent_system/reward_manager/tree_structure.py), and
[training export](../../verl/trainer/ppo/ray_trainer.py).

CPU checks:

```bash
PYTHONPATH=. conda run -n webshop python -m pytest tests/treehca/test_visualizer.py -q
```
