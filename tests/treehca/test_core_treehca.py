import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from treehca.core_treehca import cap_no_progress_advantages, compute_treehca_outcome_advantage, compute_treehca_q_outcome_advantage
from verl import DataProto
from verl.utils.advantage_estimator import AdvantageEstimator


def _cap(advantages, node_uid, parent_node_uid, info_gain, successful_terminal, is_terminal=None, **kwargs):
    tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
    mask = torch.ones_like(tensor)
    if is_terminal is None:
        is_terminal = successful_terminal
    return cap_no_progress_advantages(
        tensor,
        mask,
        np.asarray(node_uid, dtype=object),
        np.asarray(parent_node_uid, dtype=object),
        np.asarray(is_terminal, dtype=bool),
        np.asarray(info_gain, dtype=np.float32),
        np.asarray(successful_terminal, dtype=bool),
        **kwargs,
    )


def test_caps_complete_low_progress_runs_only_on_successful_rollouts():
    modified, metrics = _cap(
        advantages=[2, 3, -4, 5, 9, 6, 7, 8],
        node_uid=["a", "b", "c", "d", "terminal", "e", "f", "g"],
        parent_node_uid=["root", "a", "b", "c", "d", "a", "e", "f"],
        info_gain=[0.2, 0.01, 0.02, 0.03, 0.01, 0.01, 0.02, 0.03],
        successful_terminal=[False, False, False, False, True, False, False, False],
        rms_max_multiplier=1.0,
    )

    # b-c-d qualifies; c's negative advantage is preserved. The successful
    # terminal and the unsuccessful e-f-g branch are untouched.
    assert modified.squeeze(-1).tolist() == pytest.approx([2, 0, -4, 0, 9, 6, 7, 8])
    assert metrics["treehca/no_progress/successful_rollout_count"] == 1
    assert metrics["treehca/no_progress/qualifying_run_count"] == 1
    assert metrics["treehca/no_progress/capped_node_count"] == 3


def test_threshold_equality_breaks_a_run_and_short_runs_are_not_capped():
    modified, metrics = _cap(
        advantages=[1, 2, 3, 4, 5],
        node_uid=["a", "b", "c", "d", "e"],
        parent_node_uid=["root", "a", "b", "c", "d"],
        info_gain=[0.01, 0.02, 0.05, 0.03, 0.04],
        successful_terminal=[False, False, False, False, True],
    )

    assert modified.squeeze(-1).tolist() == pytest.approx([1, 2, 3, 4, 5])
    assert metrics["treehca/no_progress/qualifying_run_count"] == 0


def test_terminal_node_does_not_complete_a_run():
    modified, metrics = _cap(
        advantages=[1, 2, 3],
        node_uid=["a", "b", "terminal"],
        parent_node_uid=["root", "a", "b"],
        info_gain=[0.01, 0.02, 0.03],
        successful_terminal=[False, False, True],
    )

    assert modified.squeeze(-1).tolist() == pytest.approx([1, 2, 3])
    assert metrics["treehca/no_progress/qualifying_run_count"] == 0
    assert metrics["treehca/no_progress/capped_node_count"] == 0


@pytest.mark.parametrize("max_multiplier", [1.0, 1.2, 3.0])
def test_masked_rms_redistributes_only_to_positive_nonterminal_nodes_in_same_tree(max_multiplier):
    # Two trees share the external root sentinel. The second has different removed mass.
    nodes = np.asarray(["a", "b", "s", "negative", "zero", "x", "y", "t"], dtype=object)
    parents = np.asarray(["root", "a", "b", "a", "a", "root", "x", "y"], dtype=object)
    terminal = np.asarray([False, False, True, False, False, False, False, True])
    gains = np.asarray([0.2, 0.01, 0, 0.2, 0.2, 0.2, 0.01, 0])
    advantages = torch.tensor([[2., 2., 999.], [3., 999., 999.], [10., 10., 999.], [-4., -4., 999.], [0., 0., 999.], [4., 999., 999.], [1., 1., 999.], [20., 999., 999.]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0]])
    original = advantages.clone()
    modified, metrics = cap_no_progress_advantages(
        advantages, mask, nodes, parents, terminal, gains, terminal,
        turns_threshold=1, rms_max_multiplier=max_multiplier,
    )
    scale_a = min(max_multiplier, np.sqrt(17 / 8))
    scale_x = min(max_multiplier, np.sqrt(18 / 16))
    assert modified[0, :2].tolist() == pytest.approx([2 * scale_a] * 2)
    assert modified[5, 0].item() == pytest.approx(4 * scale_x)
    assert torch.equal(modified[[2, 3, 4, 7]], original[[2, 3, 4, 7]])
    assert torch.count_nonzero(modified[[1, 6]]) == 0
    assert modified[0, 2] == 999  # Padding neither contributes nor receives redistributed norm.
    assert torch.equal(advantages, original)
    assert metrics["treehca/no_progress/removed_positive_squared_norm"] == 11
    restored = 8 * (scale_a ** 2 - 1) + 16 * (scale_x ** 2 - 1)
    assert metrics["treehca/no_progress/restored_positive_squared_norm"] == pytest.approx(restored)
    assert metrics["treehca/no_progress/unrestored_positive_squared_norm"] == pytest.approx(11 - restored, abs=1e-12)
    if max_multiplier == 3:
        for rows in ([0, 1, 2, 3, 4], [5, 6, 7]):
            torch.testing.assert_close((modified[rows].square() * mask[rows]).sum(), (original[rows].square() * mask[rows]).sum())


def test_rms_handles_duplicate_nodes_and_reordered_rows():
    # Both copies of the removed node contribute to the actual training-token norm.
    advantages, metrics = _cap(
        advantages=[3, 5, 2, 3, 2],
        node_uid=["b", "s", "a", "b", "a"],
        parent_node_uid=["a", "b", "root", "a", "root"],
        info_gain=[0.01, 0, 0.2, 0.01, 0.2],
        successful_terminal=[False, True, False, False, False],
        turns_threshold=1,
    )
    scale = np.sqrt((8 + 18) / 8)
    assert advantages.squeeze(-1).tolist() == pytest.approx([0, 5, 2 * scale, 0, 2 * scale])
    assert metrics["treehca/no_progress/removed_positive_squared_norm"] == 18
    assert metrics["treehca/no_progress/rms_rescaled_tree_count"] == 1


@pytest.mark.parametrize("recipient", [0., -2., 1e-30])
def test_rms_no_positive_recipients_or_tiny_remaining_norm_is_safe(recipient):
    modified, metrics = _cap(
        advantages=[recipient, 3, 4], node_uid=["a", "b", "s"],
        parent_node_uid=["root", "a", "b"], info_gain=[0.2, 0.01, 0],
        successful_terminal=[False, False, True], turns_threshold=1,
    )
    assert torch.isfinite(modified).all()
    assert modified[0, 0].item() == pytest.approx(recipient * 3 if recipient > 0 else recipient, abs=1e-35)
    assert modified[1:, 0].tolist() == [0, 4]
    assert metrics["treehca/no_progress/rms_no_recipient_tree_count"] == (recipient <= 0)
    assert metrics["treehca/no_progress/rms_limited_tree_count"] == (recipient > 0)


@pytest.mark.parametrize("value", [0.9, -1, True, "3", None, float("nan"), float("inf")])
def test_rejects_invalid_rms_multiplier(value):
    with pytest.raises(ValueError, match="no_progress_rms_max_multiplier"):
        _cap([1], ["a"], ["root"], [0], [True], rms_max_multiplier=value)


def test_caps_all_rows_for_a_duplicated_logical_node_and_preserves_padding():
    advantages = torch.tensor([[4.0, 99.0], [5.0, 99.0], [6.0, 99.0], [7.0, 99.0]])
    mask = torch.tensor([[1.0, 0.0]] * 4)
    modified, metrics = cap_no_progress_advantages(
        advantages,
        mask,
        np.asarray(["a", "b", "c", "b"], dtype=object),
        np.asarray(["root", "a", "b", "a"], dtype=object),
        np.asarray([False, False, True, False]),
        np.asarray([0.01, 0.02, 0.03, 0.02]),
        np.asarray([False, False, True, False]),
        turns_threshold=2,
    )

    assert modified.tolist() == [[0, 0], [0, 0], [6, 99], [0, 0]]
    assert metrics["treehca/no_progress/capped_node_count"] == 2
    assert metrics["treehca/no_progress/capped_row_count"] == 3


@pytest.mark.parametrize("credit", ["snis", "q_hindsight"])
@pytest.mark.parametrize("max_multiplier", [1.25, 3.0])
def test_trainer_applies_cap_after_tree_credit_and_leaves_returns_unchanged(credit, max_multiplier):
    from verl.trainer.ppo.ray_trainer import compute_advantage

    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 2.0]]),
            "response_mask": torch.ones(4, 2),
        },
        non_tensors={
            "uid": np.asarray(["group"] * 4, dtype=object),
            "node_uid": np.asarray(["a", "b", "c", "d"], dtype=object),
            "parent_node_uid": np.asarray(["root", "a", "b", "c"], dtype=object),
            "is_terminal": np.asarray([False, False, False, True]),
            "info_gain_sum": np.asarray([0.2, 0.3, 0.4, 0.5]),
            "info_gain": np.asarray([0.2, 0.02, 0.03, 0.01]),
            "traj_step": np.asarray([0, 1, 2, 3]),
            "termination_reason": np.asarray([None, None, None, "success"], dtype=object),
        },
    )

    credit_kwargs = dict(treehca_credit=credit, treehca_leaf_baseline="none", treehca_grpo_weight=0.0, treehca_q_weight=1.0)
    baseline = compute_advantage(data, AdvantageEstimator.TREEHCA, **credit_kwargs)
    before = baseline.batch["advantages"].clone()
    before_returns = baseline.batch["returns"].clone()
    result = compute_advantage(
        data,
        AdvantageEstimator.TREEHCA,
        **credit_kwargs,
        treehca_no_progress_advantage_cap=True,
        treehca_no_progress_turns_threshold=2,
        treehca_no_progress_rms_max_multiplier=max_multiplier,
    )

    scale = min(max_multiplier, torch.sqrt(before[:3].square().sum() / before[0].square().sum()).item())
    torch.testing.assert_close(result.batch["advantages"][0], before[0] * scale)
    assert torch.count_nonzero(result.batch["advantages"][1:3]) == 0
    torch.testing.assert_close(result.batch["advantages"][3], before[3])
    torch.testing.assert_close(result.batch["returns"], before_returns)
    assert result.meta_info["treehca_metrics"]["treehca/no_progress/capped_node_count"] == 2


@pytest.mark.parametrize(
    ("turns_threshold", "info_gain_threshold"),
    [(0, 0.05), (1.5, 0.05), (True, 0.05), (3, -0.01), (3, float("nan")), (3, float("inf"))],
)
def test_rejects_invalid_thresholds(turns_threshold, info_gain_threshold):
    with pytest.raises(ValueError):
        _cap(
            advantages=[1],
            node_uid=["a"],
            parent_node_uid=["root"],
            info_gain=[0.01],
            successful_terminal=[True],
            turns_threshold=turns_threshold,
            info_gain_threshold=info_gain_threshold,
        )


def _mixed_leaf_tree():
    """A pruned branch, a full-score leaf, and a turn-limit leaf."""
    return {
        "token_level_rewards": torch.tensor([[0.0], [0.0], [2.0], [10.0], [4.0]]),
        "response_mask": torch.ones(5, 1),
        "uid": np.asarray(["group"] * 5, dtype=object),
        "node_uid": np.asarray(["root", "branch", "pruned", "success", "limited"], dtype=object),
        "parent_node_uid": np.asarray(["outside", "root", "branch", "root", "root"], dtype=object),
        "is_terminal": np.asarray([False, False, True, True, True]),
        "termination_reason": np.asarray([None, None, "pruned", "success", "turn_limit"], dtype=object),
        "info_gain_sum": np.ones(5),
        "traj_step": np.asarray([0, 1, 2, 1, 1]),
    }


def test_premature_leaf_filter_defaults_off():
    advantages, _, metrics = compute_treehca_outcome_advantage(
        **_mixed_leaf_tree(),
        leaf_baseline="none",
    )

    # With the option omitted, all three root branches propagate as before.
    assert advantages.squeeze(-1).tolist() == pytest.approx([16.0 / 3.0, 2.0, 2.0, 10.0, 4.0])
    assert metrics["treehca/premature_leaf_filter/enabled"] == 0
    assert metrics["treehca/premature_leaf_filter/pruned_leaf_count"] == 0

    q_advantages, _, q_metrics = compute_treehca_q_outcome_advantage(
        **_mixed_leaf_tree(),
        grpo_weight=0.0,
        q_weight=1.0,
        aux_mode="td",
    )
    assert q_advantages.squeeze(-1).tolist() == pytest.approx(
        [0.0, -10.0 / 3.0, 0.0, 14.0 / 3.0, -4.0 / 3.0]
    )
    assert q_metrics["treehca/premature_leaf_filter/enabled"] == 0


def test_snis_stops_pruned_leaf_at_successful_ancestor_but_keeps_turn_limit_leaf():
    advantages, returns, metrics = compute_treehca_outcome_advantage(
        **_mixed_leaf_tree(),
        leaf_baseline="none",
        filter_premature_leaves=True,
    )

    # The pruned leaf still trains itself and its private branch. At root its
    # branch is omitted, while the turn-limit score remains: (10 + 4) / 2 = 7.
    expected = [[7.0], [2.0], [2.0], [10.0], [4.0]]
    assert advantages.tolist() == expected
    assert returns.tolist() == expected
    assert metrics["treehca/premature_leaf_filter/enabled"] == 1
    assert metrics["treehca/premature_leaf_filter/pruned_leaf_count"] == 1
    assert metrics["treehca/premature_leaf_filter/full_score_leaf_count"] == 1
    assert metrics["treehca/premature_leaf_filter/protected_ancestor_count"] == 1


def test_q_backup_stops_pruned_leaf_at_successful_ancestor_but_keeps_turn_limit_leaf():
    advantages, _, _ = compute_treehca_q_outcome_advantage(
        **_mixed_leaf_tree(),
        grpo_weight=0.0,
        q_weight=1.0,
        aux_mode="td",
        filter_premature_leaves=True,
    )

    # Q(root)=7 after the same filtering. The all-pruned branch gets zero on
    # the protected crossing instead of an uncompensated 2 - 7 residual. The
    # two eligible child residuals remain zero-sum: (10 - 7) + (4 - 7) = 0.
    values = advantages.squeeze(-1).tolist()
    assert values == pytest.approx([0.0, 0.0, 0.0, 3.0, -3.0])
    assert sum(values[i] for i in (1, 3, 4)) == pytest.approx(0.0)


def test_q_backup_keeps_local_td_signal_below_filtered_crossing():
    tree = {
        "token_level_rewards": torch.tensor([[0.0], [0.0], [2.0], [4.0], [10.0], [4.0]]),
        "response_mask": torch.ones(6, 1),
        "uid": np.asarray(["group"] * 6, dtype=object),
        "node_uid": np.asarray(["root", "branch", "pruned-a", "pruned-b", "success", "limited"], dtype=object),
        "parent_node_uid": np.asarray(["outside", "root", "branch", "branch", "root", "root"], dtype=object),
        "is_terminal": np.asarray([False, False, True, True, True, True]),
        "termination_reason": np.asarray([None, None, "pruned", "pruned", "success", "turn_limit"], dtype=object),
        "info_gain_sum": np.ones(6),
        "traj_step": np.asarray([0, 1, 2, 2, 1, 1]),
    }

    advantages, _, _ = compute_treehca_q_outcome_advantage(
        **tree,
        grpo_weight=0.0,
        q_weight=1.0,
        aux_mode="td",
        filter_premature_leaves=True,
    )

    # The all-pruned branch stops at root, but its internal Q mean remains 3,
    # preserving the useful local residuals 2 - 3 and 4 - 3.
    assert advantages.squeeze(-1).tolist() == pytest.approx([0.0, 0.0, -1.0, 1.0, 3.0, -3.0])


def test_pruned_leaf_propagates_normally_without_a_full_score_descendant():
    inputs = _mixed_leaf_tree()
    keep = np.asarray([0, 2, 4])
    inputs = {
        key: value[torch.as_tensor(keep)] if isinstance(value, torch.Tensor) else value[keep]
        for key, value in inputs.items()
    }
    inputs["parent_node_uid"] = np.asarray(["outside", "root", "root"], dtype=object)
    inputs["traj_step"] = np.asarray([0, 1, 1])

    advantages, _, metrics = compute_treehca_outcome_advantage(
        **inputs,
        leaf_baseline="none",
        filter_premature_leaves=True,
    )

    assert advantages.squeeze(-1).tolist() == pytest.approx([3.0, 2.0, 4.0])
    assert metrics["treehca/premature_leaf_filter/protected_ancestor_count"] == 0


def test_q_grpo_baseline_ignores_adjust_batch_copies():
    rewards = torch.tensor([[0.0], [2.0], [4.0], [4.0]])
    advantages, _, _ = compute_treehca_q_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=torch.ones_like(rewards),
        uid=np.asarray(["group"] * 4, dtype=object),
        node_uid=np.asarray(["a", "b", "c", "c"], dtype=object),
        parent_node_uid=np.asarray(["outside"] * 4, dtype=object),
        is_terminal=np.ones(4, dtype=bool),
        termination_reason=np.asarray(["turn_limit"] * 4, dtype=object),
        info_gain_sum=np.ones(4),
        traj_step=np.zeros(4, dtype=int),
        grpo_weight=1.0,
        q_weight=0.0,
        aux_mode="td",
    )

    # The logical rewards are [0, 2, 4], whose sample mean/std are 2/2.
    # The copied c row receives the same result without reweighting the baseline.
    assert advantages.squeeze(-1).tolist() == pytest.approx([-1.0, 0.0, 1.0, 1.0], abs=1e-5)


def test_invalid_penalty_restores_and_more_strongly_penalizes_full_success_terminals():
    from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty

    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(7, 1, dtype=torch.long),
            "responses": torch.ones(7, 2, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]] * 7),
            "token_level_scores": torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [10.0, 0.0],
                ]
            ),
        },
        non_tensors={
            "is_action_valid": np.asarray([True, False, False, False, True, False, False]),
            "environment_done": np.asarray([True, True, False, True, True, True, True]),
            "webshop_task_score": np.asarray([0.4, 0.4, 0.4, 0.0, 1.0, 1.0, 1.0], dtype=object),
        },
    )

    result, metrics = apply_invalid_action_penalty(data, 0.1)

    assert result.batch["token_level_scores"][:, 0].tolist() == pytest.approx(
        [0.0, -0.1, -0.1, -0.1, 10.0, 9.5, 9.5]
    )
    assert metrics["episode/full_success_terminal_scores_restored"] == 2


def test_tree_reward_manager_excludes_pruned_leaf_from_protected_ancestor_average():
    from types import SimpleNamespace

    from agent_system.reward_manager.tree_structure import TreeStructureRewardManager

    tree = _mixed_leaf_tree()
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(5, 1, dtype=torch.long),
            "responses": torch.ones(5, 1, dtype=torch.long),
            "attention_mask": torch.ones(5, 2, dtype=torch.long),
        },
        non_tensors={
            "node_uid": tree["node_uid"],
            "parent_node_uid": tree["parent_node_uid"],
            "is_terminal": tree["is_terminal"],
            "termination_reason": tree["termination_reason"],
            "traj_step": tree["traj_step"],
            "current_tool_callings": np.zeros(5, dtype=np.float32),
            "rewards": np.asarray([0.0, 0.0, 2.0, 10.0, 4.0], dtype=np.float32),
            "data_source": np.asarray(["test"] * 5, dtype=object),
        },
    )
    config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "treehca",
                "igrpo": {"reward_mode": "avg"},
                "treehca": {"filter_premature_leaves": True},
            }
        }
    )
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "")

    rewards = TreeStructureRewardManager(tokenizer, 0, config)(batch)

    assert rewards.squeeze(-1).tolist() == pytest.approx([7.0, 2.0, 2.0, 10.0, 4.0])
    assert batch.non_tensor_batch["subtree_traj_num"].tolist() == [2, 1, 1, 1, 1]


def test_tree_reward_manager_premature_leaf_filter_defaults_off():
    from types import SimpleNamespace

    from agent_system.reward_manager.tree_structure import TreeStructureRewardManager

    tree = _mixed_leaf_tree()
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(5, 1, dtype=torch.long),
            "responses": torch.ones(5, 1, dtype=torch.long),
            "attention_mask": torch.ones(5, 2, dtype=torch.long),
        },
        non_tensors={
            "node_uid": tree["node_uid"],
            "parent_node_uid": tree["parent_node_uid"],
            "is_terminal": tree["is_terminal"],
            "termination_reason": tree["termination_reason"],
            "traj_step": tree["traj_step"],
            "current_tool_callings": np.zeros(5, dtype=np.float32),
            "rewards": np.asarray([0.0, 0.0, 2.0, 10.0, 4.0], dtype=np.float32),
            "data_source": np.asarray(["test"] * 5, dtype=object),
        },
    )
    config = OmegaConf.create(
        {"algorithm": {"adv_estimator": "treehca", "igrpo": {"reward_mode": "avg"}}}
    )
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "")

    rewards = TreeStructureRewardManager(tokenizer, 0, config)(batch)

    assert rewards.squeeze(-1).tolist() == pytest.approx([16.0 / 3.0, 2.0, 2.0, 10.0, 4.0])
    assert batch.non_tensor_batch["subtree_traj_num"].tolist() == [3, 1, 1, 1, 1]


def test_q_components_reach_rollout_fields_before_weighting(tmp_path):
    from types import SimpleNamespace

    from treehca.rollout_records import build_rollout_fields
    from treehca.visualizer.data import load_file
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage

    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]]),
            "response_mask": torch.tensor([[1.0, 0.0]] * 3),
        },
        non_tensors={
            "uid": np.asarray(["group"] * 3, dtype=object),
            "node_uid": np.asarray(["a", "b", "c"], dtype=object),
            "parent_node_uid": np.asarray(["root", "a", "a"], dtype=object),
            "is_terminal": np.asarray([False, True, True]),
            "info_gain_sum": np.asarray([0.5, 0.5, 0.5]),
            "traj_step": np.asarray([0, 1, 1]),
        },
    )
    result = compute_advantage(
        data,
        AdvantageEstimator.TREEHCA,
        treehca_credit="q_hindsight",
        treehca_aux_mode="td",
        treehca_grpo_weight=0.25,
        treehca_q_weight=0.5,
        treehca_norm_adv_by_std=False,
    )
    grpo = result.batch["grpo_advantages"]
    auxiliary = result.batch["auxiliary_advantages"]
    assert auxiliary[:, 0].tolist() == pytest.approx([0.0, -1.0, 1.0])
    assert torch.allclose(result.batch["advantages"], 0.25 * grpo + 0.5 * auxiliary)
    fields = build_rollout_fields(result.non_tensor_batch, result.batch)
    assert fields["grpo_advantage"] == pytest.approx(grpo[:, 0].tolist())
    assert fields["auxiliary_advantage"] == pytest.approx([0.0, -1.0, 1.0])
    assert fields["advantage"] == pytest.approx(result.batch["advantages"][:, 0].tolist())
    assert fields["grpo_advantages"] == [[value] for value in fields["grpo_advantage"]]
    assert fields["auxiliary_advantages"] == [[value] for value in fields["auxiliary_advantage"]]

    RayPPOTrainer._dump_generations(
        SimpleNamespace(global_steps=1), ["p"] * 3, ["o"] * 3,
        [0.0, 2.0, 4.0], {}, str(tmp_path), rollout_fields=fields,
    )
    tree = load_file(tmp_path / "1.jsonl").trees[0]
    assert tree.nodes["b"].records[0]["auxiliary_advantage"] == -1.0
    assert tree.nodes["b"].records[0]["advantage"] == fields["advantage"][1]
