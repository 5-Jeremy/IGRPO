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
        infos=[{"won": True, "task_score": 1.0, "webshop_task_id": 7, "webshop_session_id": "original"}, {"won": False, "task_score": 0.4}, {}, {}],
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
    assert fields["webshop_task_score"].tolist() == [1.0, 0.4, None, None]


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


def test_gigpo_fields_preserve_reordered_independent_trajectories(tmp_path):
    from treehca.rollout_records import build_rollout_fields, capture_step_logging
    from treehca.visualizer.data import load_file, graph_elements

    batch = {
        "uid": ["group"] * 3,
        "traj_uid": ["one", "two", "one"],
        "traj_step": [1, 0, 0],
        "page_type": ["item_page", "", ""],
        **capture_step_logging(next_obs={"anchor": ["purchased", "results", "results"]},
            infos=[{"won": True}, {}, {}], dones=[True, False, False], is_last_step=False),
    }
    fields = build_rollout_fields(batch, {"response_mask": torch.tensor([[1, 0]] * 3), "advantages": torch.tensor([[3., 999.], [2., 999.], [1., 999.]])})
    assert fields["node_path"] == [["one:1", "one:0"], ["two:0"], ["one:0"]]
    assert fields["parent_node_uid"] == ["one:0", None, None]
    assert fields["termination_reason"] == ["success", None, None]
    assert fields["advantage"] == [3., 2., 1.]
    assert fields["info_gain"] == [None] * 3
    assert "expansion_count" not in fields
    RayPPOTrainer._dump_generations(SimpleNamespace(global_steps=1), ["p"] * 3, ["a"] * 3, [1, 0, 0], {}, str(tmp_path), rollout_fields=fields)
    tree = load_file(tmp_path / "1.jsonl").trees[0]
    assert tree.statistics["saved_nodes"] == 3
    assert tree.nodes["one:1"].records[0]["advantage"] == 3.
    assert len(graph_elements(tree)[0]) == 4


def test_chain_terminal_reason_and_observation_fallback():
    from treehca.rollout_records import capture_step_logging
    fields = capture_step_logging(next_obs={"text": ["a", "b", "c"]},
        infos=[{"won": False}, {}, {}], dones=[True, True, False], is_last_step=True)
    assert fields["termination_reason"].tolist() == ["environment_failure", "environment_terminal", "turn_limit"]
    assert fields["post_action_observation"].tolist() == ["a", "b", "c"]


def test_gigpo_collection_captures_page_before_action_and_filters_finished_slots():
    from omegaconf import OmegaConf
    from verl import DataProto
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from treehca.rollout_records import build_rollout_fields

    config = OmegaConf.create({"algorithm": {"adv_estimator": "gigpo"}, "env": {"max_steps": 2, "rollout": {"n": 2}}})
    collector = TrajectoryCollector(config, SimpleNamespace(batch_decode=lambda *a, **kw: ["action"] * 2))
    collector.preprocess_batch = lambda **kw: DataProto.from_dict(
        tensors={key: torch.ones(2, 1, dtype=torch.long) for key in ("input_ids", "attention_mask", "position_ids")},
        non_tensors={"raw_prompt_ids": np.asarray([[1], [1]])})

    class Actor:
        world_size = 1
        def generate_sequences(self, data):
            return DataProto.from_dict(tensors={key: torch.ones(2, 1, dtype=torch.long) for key in ("responses", "input_ids")})

    class Environment:
        turn = 0
        def reset(self, **kwargs):
            return {"text": ["initial"] * 2}, [{"page_type": ""}] * 2
        def step(self, actions):
            self.turn += 1
            return ({"text": ["wrapped"] * 2, "anchor": ["terminal", f"obs-{self.turn}"]},
                    np.array([10., 0.]), np.array([True, False]),
                    [{"won": True, "page_type": "done"}, {"page_type": "search_results", "webshop_task_id": 42}])
        def success_evaluator(self, **kwargs):
            return {"success_rate": np.array([1., 0.])}

    rows, rewards, lengths, success, trajectories, calls = collector.vanilla_multi_turn_loop(
        DataProto.from_dict(tensors={"prompts": torch.ones(2, 1)}), Actor(), Environment())
    batch = collector.gather_rollout_data(rows, rewards, lengths, success, trajectories, calls)
    fields = build_rollout_fields(batch.non_tensor_batch)
    assert fields["traj_step"] == [0, 0, 1]
    assert fields["page_type"] == ["", "", "search_results"]
    assert fields["termination_reason"] == ["success", None, "turn_limit"]
    assert fields["environment_done"] == [True, False, False]
    assert fields["is_terminal"] == [True, False, True]
    assert fields["post_action_observation"] == ["terminal", "obs-1", "obs-2"]
    assert fields["webshop_task_id"] == [None, 42, 42]


def test_base_webshop_worker_logs_session_before_purchase_reset():
    from agent_system.environments.env_package.webshop.envs import WebshopWorker

    class Environment:
        session = "original"
        page_type = "done"
        @property
        def unwrapped(self):
            return self
        def reset(self, session):
            self.session = "original"
            return "initial", {}
        def step(self, action):
            self.session = "auto-reset"
            return "terminal", 1., True, {}
        def get_available_actions(self):
            return []

    worker = object.__new__(WebshopWorker)
    worker.env = Environment()
    worker.reset(42)
    obs, reward, done, info = worker.step("buy")
    assert (obs, reward, done) == ("terminal", 10., True)
    assert info["webshop_session_id"] == "original"
    assert info["webshop_task_id"] == 42
