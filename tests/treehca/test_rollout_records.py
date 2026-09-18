import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from treehca.rollout_records import build_treehca_rollout_fields, capture_branch_logging
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_treehca_rollout_fields_follow_parent_links_after_reordering(tmp_path):
    batch = {
        "node_uid": np.array(["child_b", "root_a", "child_a", "root_b"]),
        "parent_node_uid": np.array(["root_b", "root", "root_a", "root"]),
        "avg_ans_log_probs": np.array([-0.4, -np.inf, -0.3, -0.2]),
        "info_gain": np.array([0.4, 0.1, 0.3, 0.2]),
        "info_gain_sum": np.array([0.8, 0.2, 0.6, 0.4]),
        "page_type": np.array(["item_page", "", "search_results", ""], dtype=object),
    }
    fields = build_treehca_rollout_fields(batch)
    assert fields["node_path"] == [
        ["child_b", "root_b", "root"],
        ["root_a", "root"],
        ["child_a", "root_a", "root"],
        ["root_b", "root"],
    ]

    trainer = SimpleNamespace(global_steps=7)
    RayPPOTrainer._dump_generations(
        trainer,
        inputs=["a", "b", "c", "d"],
        outputs=["A", "B", "C", "D"],
        scores=[1, 2, 3, 4],
        reward_extra_infos_dict={},
        dump_path=str(tmp_path),
        rollout_fields=fields,
    )
    rows = [json.loads(line) for line in (tmp_path / "7.jsonl").read_text().splitlines()]
    assert [row["node_path"] for row in rows] == fields["node_path"]
    assert [row["avg_ans_log_probs"] for row in rows] == [-0.4, "-Infinity", -0.3, -0.2]
    assert '"avg_ans_log_probs": "-Infinity"' in (tmp_path / "7.jsonl").read_text()
    assert [row["info_gain"] for row in rows] == pytest.approx([0.4, 0.1, 0.3, 0.2])
    assert [row["info_gain_sum"] for row in rows] == pytest.approx([0.8, 0.2, 0.6, 0.4])
    assert [row["page_type"] for row in rows] == ["item_page", "", "search_results", ""]
    assert all("step" not in row for row in rows)


def test_other_algorithm_dump_keeps_step(tmp_path):
    trainer = SimpleNamespace(global_steps=3)
    RayPPOTrainer._dump_generations(trainer, ["prompt"], ["answer"], [1], {}, str(tmp_path))
    row = json.loads((tmp_path / "3.jsonl").read_text())
    assert row["step"] == 3


@pytest.mark.parametrize("last", [False, True])
def test_branch_diagnostics_capture_actual_decision_and_terminal_observation(last):
    probabilities = np.array([0, 0, 0.3, 0.7])
    counts = np.array([0, 0, 0, 4])
    scores = np.array([1, 0.2, 0.4, 0.8])
    observations = ["purchase success", "partial-credit purchase", "results", "product"]
    fields = capture_branch_logging(
        next_obs={"anchor": observations, "text": ["wrapped"] * 4},
        infos=[{"won": True, "webshop_task_id": 7, "webshop_session_id": "original"}, {"won": False}, {}, {}],
        dones=np.array([True, True, False, False]),
        is_last_step=last,
        expand_prob=probabilities,
        expand_num=counts,
        branch_score=scores,
        gamma=2,
    )
    assert fields["termination_reason"].tolist() == ["success", "environment_failure", "turn_limit" if last else "pruned", "turn_limit" if last else None]
    assert fields["expansion_count"].tolist() == ([0] * 4 if last else [0, 0, 0, 4])
    assert fields["sampled_expansion_count"].tolist() == [0, 0, 0, 4]
    assert fields["branch_logit"].tolist() == pytest.approx([2, 0.4, 0.8, 1.6])
    assert fields["post_action_observation"].tolist() == observations
    probabilities[:] = 0
    counts[:] = 0
    scores[:] = 0
    assert fields["expansion_probability"].tolist() == [0, 0, 0.3, 0.7]
    assert fields["branch_score"].tolist() == [1, 0.2, 0.4, 0.8]
    assert fields["webshop_session_id"][0] == "original"


def test_logged_training_values_mask_padding_and_keep_absence_explicit():
    batch = {
        "node_uid": np.array(["b", "a"]),
        "parent_node_uid": np.array(["a", "root"]),
        "uid": np.array(["group", "group"]),
        "traj_step": np.array([1, 0]),
        "avg_ans_log_probs": np.array([-0.4, -0.2]),
        "info_gain": np.array([0.2, 0.1]),
        "info_gain_sum": np.array([0.3, 0.1]),
        "page_type": np.array(["item_page", ""]),
        "is_terminal": np.array([True, False]),
        "deactivate": np.array([False, False]),
        "termination_reason": np.array(["success", None], dtype=object),
    }
    tensors = {"response_mask": torch.tensor([[1, 0, 1], [0, 1, 0]]), "advantages": torch.tensor([[2.0, 999.0, 4.0], [999.0, -1.0, 999.0]])}
    fields = build_treehca_rollout_fields(batch, tensors)
    assert fields["uid"] == ["group", "group"]
    assert fields["traj_step"] == [1, 0]
    assert fields["is_terminal"] == [True, False]
    assert fields["advantages"] == [[2.0, 4.0], [-1.0]]
    assert fields["advantage"] == [3.0, -1.0]
    assert fields["value"] == fields["values"] == [None, None]
    tensors["values"] = torch.tensor([[0.25, 999.0, 0.75], [999.0, 1.0, 999.0]])
    assert build_treehca_rollout_fields(batch, tensors)["value"] == [0.5, 1.0]
    json.dumps(fields, allow_nan=False)
