"""TreeHCA pruned leaves use the environment's full-success reward scale."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agent_system.environments.env_manager import SearchEnvironmentManager, WebshopEnvironmentManager
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector


def pruned_row(gt_prob):
    return {
        "node_uid": "leaf",
        "parent_node_uid": "root",
        "uid": "group",
        "prompts": torch.ones(1, dtype=torch.long),
        "responses": torch.ones(1, dtype=torch.long),
        "input_ids": torch.ones(2, dtype=torch.long),
        "attention_mask": torch.ones(2, dtype=torch.long),
        "rewards": np.float32(0),
        "info_gain_sum": np.float32(gt_prob),
        "traj_step": np.int32(0),
        "deactivate": True,
        "is_terminal": True,
        "current_tool_callings": np.float32(0),
        "data_source": "test",
        "termination_reason": "pruned",
    }


def collector(prob_diff_mode, estimator="treehca"):
    config = OmegaConf.create({
        "env": {"max_steps": 3},
        "algorithm": {"adv_estimator": estimator, "igrpo": {
            "prob_diff_mode": prob_diff_mode, "reward_mode": "max",
            "stable_steps": 10, "stable_method": "threshold",
        }},
    })
    return TrajectoryCollector(config, SimpleNamespace())


@pytest.mark.parametrize("manager,full_reward", [(SearchEnvironmentManager, 1.0), (WebshopEnvironmentManager, 10.0)])
@pytest.mark.parametrize("prob_diff_mode", [True, False])
@pytest.mark.parametrize("global_steps", [0, 100])
def test_pruned_leaf_reward_uses_environment_scale_and_true_probability(manager, full_reward, prob_diff_mode, global_steps):
    probability = 0.25
    stored_value = probability if prob_diff_mode else math.log(probability)
    row = pruned_row(stored_value)
    batch = collector(prob_diff_mode).gather_rollout_data_tree_structure(
        [row], {"success_rate": np.asarray([0.0])}, global_steps,
        full_success_reward=manager.full_success_reward,
    )
    assert batch.non_tensor_batch["rewards"][0] == pytest.approx(full_reward * probability)


def test_igrpo_pruned_leaf_keeps_existing_reward_rule():
    row = pruned_row(0.25)
    batch = collector(True, estimator="igrpo").gather_rollout_data_tree_structure(
        [row], {"success_rate": np.asarray([0.0])}, global_steps=0,
    )
    assert batch.non_tensor_batch["rewards"][0] == pytest.approx(0.125)


def test_successful_terminal_supplies_scale_when_manager_does_not():
    pruned = pruned_row(math.log(0.25))
    success = {**pruned, "node_uid": "success", "deactivate": False,
               "termination_reason": "success", "rewards": np.float32(7.0)}
    batch = collector(False).gather_rollout_data_tree_structure(
        [pruned, success], {"success_rate": np.asarray([1.0])}, global_steps=0,
    )
    assert batch.non_tensor_batch["rewards"][0] == pytest.approx(1.75)


def test_unknown_success_scale_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="full-success reward"):
        collector(True).gather_rollout_data_tree_structure(
            [pruned_row(0.25)], {"success_rate": np.asarray([0.0])}, global_steps=0,
        )
