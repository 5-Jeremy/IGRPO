# Rollout Tree Structure in IGRPO

IGRPO maintains rollout trees implicitly as a fixed-size pool of parallel
rollout slots. It does not construct a conventional `Tree` object containing
node objects and explicit child lists. Instead, completed rollout nodes are
stored as flat records, and tree edges are represented by UUID-based
`node_uid` and `parent_node_uid` fields.

## Live frontier state

The live rollout frontier is managed by
[`TrajectoryNodeStateManagement`](../../igrpo/core_igrpo.py). Its main fields
are:

- `envs`: the parallel environment states.
- `obs`: the current observations for every rollout slot.
- `node_uid`: the UUID of the current node represented by each slot. It is
  initialized to the sentinel value `"root"`.
- `active_nodes`: a Boolean NumPy mask indicating which slots are currently
  part of the live frontier.
- `uid_batch`: the tree/group UUID. All rollout copies belonging to the same
  original prompt share this value.
- `original_gen_batch_index`: a mapping from each slot to the original prompt
  used to construct its model input.
- `tool_callings`: the accumulated number of tool calls for each slot.

Most of these fields are NumPy arrays with one entry per rollout slot. The
environment manager and observation collection hold the corresponding mutable
execution state.

## Materialized tree representation

Completed nodes are appended to `total_batch_list`, a flat `List[Dict]`, in
[`vanilla_multi_turn_loop_with_tree_structure`](../../agent_system/multi_turn_rollout/rollout_loop.py).
Each node dictionary contains model and environment data together with tree
metadata, including:

- `node_uid`
- `parent_node_uid`
- `uid`, the tree/group identifier
- `traj_step`
- `is_terminal`
- `deactivate`
- reward and information-gain values
- generated action text and model tensors

Consequently, `total_batch_list` and the `(node_uid, parent_node_uid)` pairs
form an edge-list representation of a forest. The initial root is not stored as
a normal node; it is identified by the `"root"` sentinel.

The rollout loop also maintains a `node_uid2info_gain_sum` dictionary. This
maps a materialized node UUID to its cumulative information-gain value, allowing
a child to compute its incremental information gain from its parent's value.

## Creating node identifiers

At every rollout step, the current value of `node_management.node_uid` is
written to the new batch records as `parent_node_uid`. The call to
`update_node_uid()` then generates a fresh UUID for every slot and stores those
values as the new records' `node_uid` values:

```python
batch.non_tensor_batch["parent_node_uid"] = node_management.node_uid
batch.non_tensor_batch["node_uid"] = node_management.update_node_uid()
```

For the first level, every generated node has `parent_node_uid == "root"`.
Afterward, a slot's newly generated UUID becomes the parent UUID for whatever
that slot generates in the next iteration.

## Selecting nodes for expansion

After generating and scoring a level, IGRPO allocates the next-level rollout
budget among active nodes:

1. Terminal environments are deactivated.
2. `compute_expand_prob()` computes a softmax over the active nodes in each
   prompt group. Its logits are the configured `gamma` multiplied by the
   node's information value.
3. `get_expand_num()` samples an integer expansion count for each active node.
   The total budget depends on `expand_mode` (`full`, `mid`, or `low`), and each
   node is limited by `max_traj_to_expand_per_node`.
4. A node assigned zero expansions is marked as deactivated and terminal.
5. A node assigned `k > 0` expansions supplies `k` outgoing rollouts at the
   next level.

An expansion count includes the existing source slot. Therefore, a count of
one requires no copying, while a count of `k` requires `k - 1` additional
slots.

## Adding a branch

A branch is added by cloning a selected live node into an inactive rollout
slot. The operation is implemented by `TrajectoryNodeStateManagement.expand()`
and `fork_from()`:

1. `expand()` iterates over active source slots and their allocated expansion
   counts.
2. The source slot itself represents the first continuation.
3. For every additional continuation, `expand()` finds an inactive destination
   slot and calls `fork_from(destination, source)`.
4. `fork_from()` verifies that the source is active and the destination is
   inactive.
5. It forks the underlying environment state and copies or assigns the current
   observation, tree/group UUID, current node UUID, tool-call count, and
   original-prompt mapping into the destination slot.
6. The destination slot is marked active.

The environment manager is responsible for making the environment itself
forkable. For the search environment, this includes deep-copying chat history
and copying the ground truth, turn count, terminal flag, and other environment
metadata. The rollout memory is also deep-copied by the environment manager.

Immediately after the fork, the source and destination slots carry the same
current `node_uid`. When the next model-generation step runs, this shared UUID
is recorded as the parent of every resulting child, while each child receives
a new UUID:

```text
Before expansion:             C

Expansion count of three:   [C] [C] [C]
                              |   |   |
Next generation:             D1  D2  D3

Recorded edges: C -> D1, C -> D2, C -> D3
```

The generated node is converted to a standalone dictionary and appended to
`total_batch_list` before `expand()` mutates or reuses any live slot. This
ordering preserves the historical node record while allowing the fixed-size
slot pool to be recycled.

## Consuming the tree

[`gather_rollout_data_tree_structure`](../../agent_system/multi_turn_rollout/rollout_loop.py)
supports two representations for training:

- With `reward_mode == "full"`, it builds complete root-to-leaf trajectories.
  Starting at every terminal node, it repeatedly follows `parent_node_uid`
  through a `node_uid -> index` lookup. Shared prefixes are duplicated into the
  resulting trajectory batch.
- With other reward modes, the flat node list is preserved and consumed by
  [`TreeStructureRewardManager`](../../agent_system/reward_manager/tree_structure.py).
  The reward manager builds a `node_uid -> list[index]` reverse lookup and walks
  from terminal nodes toward their ancestors, aggregating terminal rewards,
  depths, and tool-call counts into each subtree. A list of indices is used
  because later batch duplication can turn the materialized representation
  into a DAG with repeated copies of the same logical node.

In short, the durable tree is the flat collection of UUID-linked node records,
while `TrajectoryNodeStateManagement` maintains only the live frontier needed
to generate and fork the next level.
