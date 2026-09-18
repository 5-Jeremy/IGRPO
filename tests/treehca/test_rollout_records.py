import json
from types import SimpleNamespace

import numpy as np
import pytest

from treehca.rollout_records import build_treehca_rollout_fields
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
