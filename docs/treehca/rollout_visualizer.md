# WebShop rollout tree visualizer

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
rows, rounded boxes, directed edges, pan/zoom, **Fit graph**, a tall scrollable
details pane, and full-width bottom tabs. Labels show saved `<action>` text when
available; full IDs and all values appear in the details pane. Colors distinguish
the synthetic root, internal nodes, and leaves, or any numeric saved field.
Metric colors normalize finite values within the selected tree; missing and
nonfinite values are gray. Leaves describe graph structure, not environment
termination, which is not present in the example logs.

Click a node to inspect **all saved fields**, including unknown future fields,
nested objects, arrays, exact input/output text, and full numeric precision.
**Saved text & context** provides the initial input alongside the selected node's
input, output, or raw JSON record. Literal search supports case sensitivity,
Previous/Next, Enter, and Jump to end. Saved markup is displayed as plain text.
Tree/file statistics are derived from the records in that file; they are not
training-wide diagnostics. Duplicate node records are retained and selectable by
JSONL line number; graph labels and colors use the first saved record.

## Reconstruction and limitations

Each JSONL line must be an object containing `node_path`, ordered from the current
node to the `"root"` sentinel. The path reconstructs edges even if records arrive
out of order. A synthetic root joins initial branches; ancestors lacking their
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

## Useful additional logging

No logging or training behavior is changed by this viewer. Recommended additions:

- **`uid`/`tree_uid` on every node:** the most important addition, enabling exact
  tree grouping even with repeated prompts or missing initial records.
- **`node_uid`, `parent_node_uid`, `traj_step`:** explicit structure and saved turn
  counter, independently checkable against `node_path` and graph depth.
- **`is_terminal`, `deactivate`, and a termination reason:** distinguish successful
  completion, pruning, environment failure, and reaching a turn limit.
- **Expansion count/probability and branch score:** explain why a node was selected
  or pruned. Values and advantages would help compare branching with training.
- **Post-action observation and WebShop task/session ID:** show the immediate
  result of a leaf action and identify the task. A leaf's next input never exists,
  so its final observation cannot be recovered from the current logs.
- **Schema version and run/step metadata:** make future formats unambiguous.

CPU checks:

```bash
PYTHONPATH=. conda run -n webshop python -m pytest tests/treehca/test_visualizer.py -q
```
