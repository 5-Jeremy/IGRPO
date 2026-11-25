"""Keep GiGPO components tied to rollout nodes without changing policy signals."""

import json

import numpy as np
import pytest
import torch

from gigpo.core_gigpo import compute_gigpo_outcome_advantage
from rollout_visualizer.data import read_document
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.utils.rollout_logging import dump_generations


def make_batch():
    return DataProto.from_dict(tensors={
        "prompts": torch.ones((4, 2), dtype=torch.long),
        "responses": torch.ones((4, 3), dtype=torch.long),
        "response_mask": torch.tensor([[1, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]]),
        "token_level_rewards": torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
        "step_rewards": torch.tensor([1., 0., 0.5, 0.]),
    }, non_tensors={
        "uid": np.array(["group"] * 4),
        "traj_uid": np.array(["a", "b", "c", "d"]),
        "anchor_obs": np.array(["shared", "shared", "unique", "shared"]),
        "id": np.array(["a", "b", "c", "d"]),
        "previous_step_ids": np.empty((4, 0), dtype=object),
        "trajectory_id": np.array(["a", "b", "c", "d"]),
        "trajectory_length": np.ones(4, dtype=int),
    })


def test_optional_components_preserve_existing_gigpo_return_contract():
    batch = make_batch()
    kwargs = dict(
        token_level_rewards=batch.batch["token_level_rewards"],
        step_rewards=batch.batch["step_rewards"], response_mask=batch.batch["response_mask"],
        anchor_obs=batch.non_tensor_batch["anchor_obs"], index=batch.non_tensor_batch["uid"],
        traj_index=batch.non_tensor_batch["traj_uid"], step_advantage_w=0.4,
    )
    original = compute_gigpo_outcome_advantage(**kwargs)
    assert len(original) == 2
    advantages, returns, components = compute_gigpo_outcome_advantage(**kwargs, return_components=True)
    torch.testing.assert_close(advantages, original[0])
    torch.testing.assert_close(returns, original[1])
    torch.testing.assert_close(advantages, components["episode"] + 0.4 * components["turn"])


@pytest.mark.parametrize("weight", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
def test_gigpo_components_reach_visualizer_after_reordering_and_padding(tmp_path, weight, mode):
    batch = compute_advantage(make_batch(), "gigpo", step_advantage_w=weight, gigpo_mode=mode)
    expected_turn = torch.tensor([2 / 3, -1 / 3, 0.])
    expected_episode = torch.tensor([0.75, -0.25, -0.25])
    if mode == "mean_std_norm":
        expected_turn /= torch.tensor([1., 0., 0.]).std() + 1e-6
        expected_episode /= torch.tensor([1., 0., 0., 0.]).std() + 1e-6
    for field, expected in [
        ("gigpo_turn_advantage", expected_turn),
        ("gigpo_weighted_turn_advantage", weight * expected_turn),
        ("gigpo_episode_advantage", expected_episode),
    ]:
        values = batch.non_tensor_batch[field]
        torch.testing.assert_close(torch.tensor(values[:3].tolist()), expected)
        assert values[3] is None  # No trainable response tokens, not a measured zero.
    expected_combined = torch.cat((expected_episode + weight * expected_turn, torch.zeros(1)))
    torch.testing.assert_close(batch.batch["advantages"], expected_combined[:, None] * batch.batch["response_mask"])
    torch.testing.assert_close(batch.batch["returns"], batch.batch["advantages"])
    fields = ["gigpo_turn_advantage", "gigpo_weighted_turn_advantage", "gigpo_episode_advantage"]
    expected_by_id = {node_id: {field: batch.non_tensor_batch[field][i] for field in fields}
                      for i, node_id in enumerate(batch.non_tensor_batch["id"])}
    # Later reasoning hooks must not change these captured GiGPO components.
    batch.batch["advantages"] = batch.batch["advantages"] * -2
    batch = batch.select_idxs([3, 1, 2, 0, 1])
    path = tmp_path / "1.jsonl"
    dump_generations(path, ["problem"] * 5, ["<action>Lima</action>"] * 5, [0.] * 5,
                     {field: [99.] * 5 for field in fields}, batch.non_tensor_batch)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 4  # Padding copy is omitted, including its diagnostics.
    for record in records:
        assert {field: record[field] for field in fields} == expected_by_id[record["id"]]
    document = read_document(path)
    displayed = [record for group in document["groups"] for node in group["nodes"] for record in node["records"]]
    assert {record["id"]: {field: record[field] for field in fields} for record in displayed} == expected_by_id
