# Copyright 2026 verl-agent contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stable turn identities and JSONL serialization for rollout records."""

import json
import uuid

ROLLOUT_METADATA_FIELDS = ("id", "previous_step_ids", "trajectory_id", "trajectory_length")
ROLLOUT_OPTIONAL_FIELDS = (
    "igpo_step_reward", "igpo_turn_advantage", "node_uid", "parent_node_uid",
    "episode_success", "decision_model_probabilities", "reasoning_steps", "reasoning_trace",
    "reasoning_step_advantages", "reasoning_alignment_status",
    "reasoning_step_entropies", "answer_entropy", "entropy_alignment_status",
    "gigpo_episode_advantage", "gigpo_turn_advantage", "gigpo_weighted_turn_advantage",
)


def add_trajectory_metadata(steps, trajectory_id):
    """Annotate active turns in trajectory order, before batch padding or reordering."""
    previous_step_ids = []
    for data in steps:
        if not data["active_masks"]:
            continue
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        data["previous_step_ids"] = previous_step_ids.copy()
        data["trajectory_id"] = trajectory_id
        data["trajectory_length"] = len(previous_step_ids) + 1
        previous_step_ids.append(data["id"])


def add_tree_metadata(nodes):
    """Attach ancestry in generation order, before any full-path expansion."""
    ancestors = {}
    for node in nodes:
        parent = node["parent_node_uid"]
        previous = [] if parent == "root" else ancestors[parent]
        node_id = node["node_uid"]
        node.update(id=node_id, previous_step_ids=previous, trajectory_length=len(previous) + 1,
                    trajectory_id=node_id if node["is_terminal"] else None)
        ancestors[node_id] = [*previous, node_id]


def dump_generations(filename, inputs, outputs, scores, reward_extra_infos_dict, rollout_metadata=None, *, step=None):
    """Write each turn once, omitting training padding copies with the same ID.

    Multi-turn callers must supply metadata attached by the collector. Without
    metadata, each generation is treated as an independent single-turn trajectory.
    """
    n = len(inputs)
    base_data = {"input": inputs, "output": outputs, "score": scores}
    if step is not None:
        base_data["step"] = [step] * n
    for key, values in reward_extra_infos_dict.items():
        if key not in (*base_data, *ROLLOUT_METADATA_FIELDS, *ROLLOUT_OPTIONAL_FIELDS, "group_id", "turn_reward", "step") and len(values) == n:
            base_data[key] = values

    if rollout_metadata is None:
        base_data.update(
            {
                "id": [str(uuid.uuid4()) for _ in range(n)],
                "previous_step_ids": [[] for _ in range(n)],
                "trajectory_id": [str(uuid.uuid4()) for _ in range(n)],
                "trajectory_length": [1] * n,
            }
        )
    else:
        for key in ROLLOUT_METADATA_FIELDS:
            values = rollout_metadata[key]
            # DataProto stores non-tensor columns as NumPy object arrays.
            base_data[key] = values.tolist() if hasattr(values, "tolist") else values
        # The collector's uid identifies the sampled training group, even when
        # the same problem is sampled more than once. Preserve its row alignment.
        if "uid" in rollout_metadata:
            values = rollout_metadata["uid"]
            base_data["group_id"] = values.tolist() if hasattr(values, "tolist") else values
        if "turn_reward" in rollout_metadata:
            base_data["turn_reward"] = list(rollout_metadata["turn_reward"])
        elif "rewards" in rollout_metadata:
            base_data["turn_reward"] = [float(value) for value in rollout_metadata["rewards"]]
        for key in ROLLOUT_OPTIONAL_FIELDS:
            if key in rollout_metadata:
                values = rollout_metadata[key]
                base_data[key] = values.tolist() if hasattr(values, "tolist") else values

    for key, values in base_data.items():
        if len(values) != n:
            raise ValueError(f"Rollout column {key!r} has {len(values)} entries; expected {n}")

    seen_ids = set()
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(n):
            entry = {key: values[i] for key, values in base_data.items()}
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
