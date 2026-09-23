import numpy as np
import pytest
import torch

from treehca.core_treehca import cap_no_progress_advantages
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


def test_trainer_applies_cap_after_tree_credit_and_leaves_returns_unchanged():
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
            "info_gain": np.asarray([0.01, 0.02, 0.03, 0.01]),
            "traj_step": np.asarray([0, 1, 2, 3]),
            "termination_reason": np.asarray([None, None, None, "success"], dtype=object),
        },
    )

    result = compute_advantage(
        data,
        AdvantageEstimator.TREEHCA,
        treehca_leaf_baseline="none",
        treehca_no_progress_advantage_cap=True,
    )

    assert result.batch["advantages"].tolist() == [[0, 0], [0, 0], [0, 0], [2, 2]]
    assert result.batch["returns"].tolist() == [[2, 2], [2, 2], [2, 2], [2, 2]]
    assert result.meta_info["treehca_metrics"]["treehca/no_progress/capped_node_count"] == 3


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
