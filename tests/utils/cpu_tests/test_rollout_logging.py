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

"""Exercise trajectory links through collection, padding, reordering, and JSONL."""

import json
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import torch

from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.decision_model_requests import attach_decision_model_probabilities
from agent_system.multi_turn_rollout.utils import adjust_batch
from verl.utils.rollout_logging import add_trajectory_metadata, dump_generations


def collect_trajectories(lengths, success=None):
    trajectories = []
    for trajectory_index, length in enumerate(lengths):
        steps = []
        for turn_index in range(max(lengths)):
            steps.append(
                {
                    "input_ids": torch.tensor([trajectory_index, turn_index]),
                    "uid": "shared-prompt-group",
                    "traj_uid": f"trajectory-{trajectory_index}",
                    "active_masks": turn_index < length,
                    "rewards": np.float32(turn_index - 1),
                }
            )
        trajectories.append(steps)
    trajectories[0][0]["id"] = "existing-turn-id"
    collector = TrajectoryCollector(config=None, tokenizer=None)
    return collector.gather_rollout_data(
        total_batch_list=trajectories,
        episode_rewards=np.zeros(len(lengths)),
        episode_lengths=np.array(lengths),
        success={} if success is None else {"success_rate": np.array(success)},
        traj_uid=np.array([f"trajectory-{i}" for i in range(len(lengths))]),
        tool_callings=np.zeros(len(lengths)),
    )


def test_collection_preserves_trajectory_success():
    batch = collect_trajectories((3, 1), success=(True, False))
    assert batch.non_tensor_batch["episode_success"].tolist() == [True, True, True, False]
    assert batch.non_tensor_batch["success_rate"].tolist() == [0.5] * 4


def test_decision_probabilities_reach_visualizer_after_reordering(tmp_path):
    from rollout_visualizer.data import read_document

    batch = collect_trajectories((3, 1))
    tensors_before = {key: value.clone() for key, value in batch.batch.items()}
    first = {"required to achieve goal": 0.8, "has no effect on goal": 0.15,
             "counterproductive towards goal": 0.05}
    second = {"required to achieve goal": 0.1, "has no effect on goal": 0.2,
              "counterproductive towards goal": 0.7}
    other_trajectory = {"required to achieve goal": 0.3, "has no effect on goal": 0.4,
                        "counterproductive towards goal": 0.3}
    ids = batch.non_tensor_batch["id"]
    responses = {ids[0]: {"step_1": first, "step_2": second}, ids[1]: {"step_1": second},
                 ids[3]: {"step_1": other_trajectory}}
    attach_decision_model_probabilities(batch, responses)
    for key, expected in tensors_before.items():
        torch.testing.assert_close(batch.batch[key], expected)
    values = [responses[ids[0]], responses[ids[1]], None, responses[ids[3]]]
    assert batch.non_tensor_batch["decision_model_probabilities"].tolist() == values
    expected = dict(zip(ids, values, strict=True))
    batch = batch.select_idxs([3, 2, 0, 1, 0])  # Reorder and duplicate one padding row.
    filename = tmp_path / "1.jsonl"
    dump_generations(filename, ["problem"] * 5, ["<action>Lima, Peru</action>"] * 5, [0.0] * 5,
                     {"decision_model_probabilities": [{"wrong": 1}] * 5}, batch.non_tensor_batch)
    records = [json.loads(line) for line in filename.read_text().splitlines()]
    assert len(records) == 4
    assert {r["id"]: r["decision_model_probabilities"] for r in records} == expected
    document = read_document(filename)
    displayed = [r for group in document["groups"] for node in group["nodes"] for r in node["records"]]
    assert {r["id"]: r["decision_model_probabilities"] for r in displayed} == expected


def test_missing_decision_responses_are_null_and_old_logs_remain_compatible(tmp_path):
    batch = collect_trajectories((1,))
    filename = tmp_path / "1.jsonl"
    dump_generations(filename, ["problem"], ["output"], [0.0], {}, batch.non_tensor_batch)
    assert "decision_model_probabilities" not in json.loads(filename.read_text())
    attach_decision_model_probabilities(batch, {})
    dump_generations(filename, ["problem"], ["output"], [0.0], {}, batch.non_tensor_batch)
    assert json.loads(filename.read_text())["decision_model_probabilities"] is None


def test_reasoning_steps_stay_with_saved_nodes_after_reordering(tmp_path):
    from agent_system.reasoning_steps import attach_reasoning_steps
    from rollout_visualizer.data import read_document

    batch = collect_trajectories((3, 1))
    outputs = [
        "<think>First. But wait, second.</think><action>Lima, Peru</action>",
        "<think>Has thinking but invalid action.</think><action>bad</action>",
        "<action>Lima, Peru</action>",
        "Generated only. Alternatively, reconsider.</think><action>Lima, Peru</action>",
    ]
    batch.batch["responses"] = torch.arange(4).unsqueeze(1)
    batch.batch["prompts"] = torch.arange(4).unsqueeze(1)
    batch.non_tensor_batch["is_action_valid"] = np.array([True, False, True, True])
    tensors_before = {key: value.clone() for key, value in batch.batch.items()}
    tokenizer = SimpleNamespace(
        batch_decode=lambda rows, **kwargs: [outputs[int(row[0])] for row in rows],
        decode=lambda row, **kwargs: "Prompt\n<think>\n" if int(row[0]) == 3 else "Prompt",
    )
    attach_reasoning_steps(batch, tokenizer)
    formatted = ["[1] First.\n\n[2] But wait, second.\n\n<action>Lima, Peru</action>", None, None,
                 "[1] Generated only.\n\n[2] Alternatively, reconsider.\n\n<action>Lima, Peru</action>"]
    assert batch.non_tensor_batch["reasoning_steps"].tolist() == formatted
    for key, value in tensors_before.items():
        torch.testing.assert_close(batch.batch[key], value)
    expected = dict(zip(batch.non_tensor_batch["id"], formatted, strict=True))
    expected_traces = dict(zip(batch.non_tensor_batch["id"], batch.non_tensor_batch["reasoning_trace"], strict=True))
    batch = batch.select_idxs([3, 0, 2, 1, 0])
    filename = tmp_path / "1.jsonl"
    dump_generations(filename, ["problem"] * 5, tokenizer.batch_decode(batch.batch["responses"]),
                     [0.0] * 5, {"reasoning_steps": ["incorrect"] * 5}, batch.non_tensor_batch)
    doc = read_document(filename)
    records = [record for group in doc["groups"] for node in group["nodes"] for record in node["records"]]
    assert len(records) == 4
    assert {record["id"]: record["reasoning_steps"] for record in records} == expected
    assert {record["id"]: record["reasoning_trace"] for record in records} == expected_traces


@pytest.mark.parametrize("lengths", [(3, 1), (1, 1)])
def test_trajectory_links_survive_padding_and_reordering(tmp_path, lengths):
    batch = collect_trajectories(lengths)
    # Pad to a multiple of three, then reverse the batch to model token balancing.
    config = SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=3),
            actor=SimpleNamespace(ppo_micro_batch_size_per_gpu=3, use_kl_loss=False),
        ),
        algorithm=SimpleNamespace(use_kl_in_reward=False),
    )
    batch = adjust_batch(config, batch)
    batch = batch.select_idxs(list(reversed(range(len(batch)))))
    inputs = [f"problem-{row[0]}-turn-{row[1]}" for row in batch.batch["input_ids"].tolist()]
    filename = tmp_path / "1.jsonl"
    dump_generations(
        filename,
        inputs,
        ["action"] * len(batch),
        [0.0] * len(batch),
        {"valid_action": [True] * len(batch), "unmatched": []},
        batch.non_tensor_batch,
    )
    records = [json.loads(line) for line in filename.read_text().splitlines()]
    by_id = {record["id"]: record for record in records}
    assert len(records) == len(by_id) == sum(lengths)
    assert "existing-turn-id" in by_id

    for record in records:
        trajectory_index = int(record["trajectory_id"].split("-")[-1])
        turn_index = int(record["input"].split("-")[-1])
        assert turn_index < lengths[trajectory_index]  # Inactive filler is excluded.
        assert record["trajectory_length"] == turn_index + 1
        assert record["turn_reward"] == turn_index - 1
        history = [by_id[step_id] for step_id in record["previous_step_ids"]]
        assert [prior["input"] for prior in history] == [f"problem-{trajectory_index}-turn-{i}" for i in range(turn_index)]
        assert all(prior["trajectory_id"] == record["trajectory_id"] for prior in history)
        assert record["valid_action"] is True
        assert record["group_id"] == "shared-prompt-group"
        assert "step" not in record and "unmatched" not in record
        if record["id"] != "existing-turn-id":
            UUID(record["id"])


def test_repeated_annotation_preserves_ids_and_history_snapshots():
    steps = [{"active_masks": True} for _ in range(3)]
    add_trajectory_metadata(steps, "trajectory")
    original = [step.copy() for step in steps]
    add_trajectory_metadata(steps, "trajectory")
    assert steps == original
    assert steps[0]["previous_step_ids"] == []
    assert steps[-1]["previous_step_ids"] == [step["id"] for step in steps[:-1]]


def test_repeated_problem_groups_survive_logging_and_visualization(tmp_path):
    from rollout_visualizer.data import read_document

    batch = collect_trajectories((2, 2, 2, 2))
    batch.non_tensor_batch['uid'] = np.array(['first'] * 4 + ['second'] * 4, dtype=object)
    expected = dict(zip(batch.non_tensor_batch['id'], batch.non_tensor_batch['uid'], strict=True))
    batch = batch.select_idxs([7, 2, 0, 6, 4, 3, 5, 1, 0])
    filename = tmp_path / '1.jsonl'
    dump_generations(filename, ['identical problem'] * 9, ['<action>look</action>'] * 9,
                     [0.0] * 9, {'group_id': ['incorrect'] * 9}, batch.non_tensor_batch)
    records = [json.loads(line) for line in filename.read_text().splitlines()]
    assert {r['id']: r['group_id'] for r in records} == expected
    groups = read_document(filename)['groups']
    assert len(groups) == 2
    assert [g['record_count'] for g in groups] == [4, 4]
    for group in groups:
        saved = [r for n in group['nodes'] for r in n['records']]
        assert len({r['group_id'] for r in saved}) == 1
        assert len({r['trajectory_id'] for r in saved}) == 2


def test_single_turn_fallback_and_reserved_reward_keys(tmp_path):
    filename = tmp_path / "1.jsonl"
    dump_generations(
        filename,
        ["世界", "same prompt"],
        ["action", "action"],
        [0.0, 1.0],
        {"id": ["bad"] * 2, "trajectory_length": [99] * 2, "step": [42] * 2, "group_id": ['bad'] * 2},
    )
    records = [json.loads(line) for line in filename.read_text().splitlines()]
    assert records[0]["input"] == "世界"
    assert len({record["id"] for record in records}) == 2
    assert len({record["trajectory_id"] for record in records}) == 2
    for record in records:
        UUID(record["id"])
        UUID(record["trajectory_id"])
        assert record["previous_step_ids"] == []
        assert record["trajectory_length"] == 1
        assert "step" not in record
        assert "group_id" not in record


def test_inconsistent_columns_fail_before_overwriting_file(tmp_path):
    filename = tmp_path / "1.jsonl"
    filename.write_text("existing dump\n")
    with pytest.raises(ValueError, match="'output'.*expected 1"):
        dump_generations(filename, ["prompt"], [], [0.0], {})
    assert filename.read_text() == "existing dump\n"
