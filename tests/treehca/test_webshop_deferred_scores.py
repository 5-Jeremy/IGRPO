"""Deferred state values and retroactive edge gains for the WebShop forest."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from treehca.webshop_deferred_scores import INITIAL_SEARCH, ITEM_SUB_PAGE, WebshopDeferredScores
from verl import DataProto


def level(tracker, lookup, rows, active=None):
    """Rows: node, parent, group, probability placeholder/score, reason."""
    count = len(rows)
    data = {key: np.asarray([row[index] for row in rows], dtype=object) for index, key in enumerate(("node_uid", "parent_node_uid", "uid"))}
    data.update(
        avg_ans_log_probs=np.asarray([math.log(row[3]) if row[3] else -math.inf for row in rows]),
        webshop_success_probability=np.asarray([row[3] for row in rows], dtype=object),
        webshop_skipped_reason=np.asarray([row[4] for row in rows], dtype=object),
        info_gain_sum=np.zeros(count),
        info_gain=np.zeros(count),
    )
    batch = DataProto.from_dict(tensors={"input_ids": torch.ones(count, 1)}, non_tensors=data)
    active = np.ones(count, dtype=bool) if active is None else np.asarray(active)
    tracker.update(batch, active, lookup)
    records = [{key: value[index] for key, value in batch.non_tensor_batch.items()} for index in np.flatnonzero(active)]
    tracker.register_rows(records)
    return records


@pytest.mark.parametrize("prob_diff_mode", [True, False])
def test_roots_mean_logs_per_group_and_initial_pages_reuse_root(prob_diff_mode):
    tracker = WebshopDeferredScores(["g", "h"], prob_diff_mode=prob_diff_mode, prob_floor=1e-6)
    assert tracker.roots["g"]["avg_ans_log_probs"] == -math.inf
    lookup = {}
    rows = level(tracker, lookup, [("a", "root", "g", 0.25, None), ("b", "root", "g", 0.81, None), ("x", "root", "h", 0.04, None)])
    assert tracker.roots["g"]["avg_ans_log_probs"] == pytest.approx((math.log(0.25) + math.log(0.81)) / 2)
    assert tracker.roots["g"]["webshop_success_probability"] == pytest.approx(0.45)
    assert tracker.roots["h"]["webshop_success_probability"] == pytest.approx(0.04)
    transform = (lambda p: p) if prob_diff_mode else math.log
    assert rows[0]["info_gain"] == pytest.approx(transform(0.25) - transform(0.45))
    (initial,) = level(tracker, lookup, [("c", "a", "g", 0, INITIAL_SEARCH)])
    assert initial["avg_ans_log_probs"] == pytest.approx(math.log(0.45))
    assert initial["info_gain"] == pytest.approx(transform(0.45) - transform(0.25))


@pytest.mark.parametrize("prob_diff_mode", [True, False])
def test_first_child_updates_saved_parent_sibling_and_lookup_before_child_gain(prob_diff_mode):
    tracker = WebshopDeferredScores(["g"], prob_diff_mode=prob_diff_mode, prob_floor=1e-6)
    lookup = {}
    parent, sibling = level(tracker, lookup, [("a", "root", "g", 0, ITEM_SUB_PAGE), ("b", "root", "g", 0.25, None)])
    children = level(tracker, lookup, [("first", "a", "g", 0.81, None), ("second", "a", "g", 0.04, None)])
    transform = (lambda p: p) if prob_diff_mode else math.log
    assert parent["avg_ans_log_probs"] == pytest.approx(math.log(0.81))
    assert parent["info_gain_sum"] == lookup["a"] == pytest.approx(transform(0.81))
    assert parent["info_gain"] == pytest.approx(transform(0.81) - transform(0.45))
    assert sibling["info_gain"] == pytest.approx(transform(0.25) - transform(0.45))
    assert children[0]["info_gain"] == 0
    assert children[1]["info_gain"] == pytest.approx(transform(0.04) - transform(0.81))


def test_chained_deferrals_propagate_to_ancestors_and_initial_search_rows():
    tracker = WebshopDeferredScores(["g"], prob_diff_mode=True, prob_floor=1e-6)
    lookup = {}
    ancestor, other = level(tracker, lookup, [("a", "root", "g", 0, ITEM_SUB_PAGE), ("b", "root", "g", 0.25, None)])
    middle, initial = level(tracker, lookup, [("c", "a", "g", 0, ITEM_SUB_PAGE), ("d", "b", "g", 0, INITIAL_SEARCH)])
    (child,) = level(tracker, lookup, [("e", "c", "g", 0.64, None)])
    assert ancestor["info_gain_sum"] == middle["info_gain_sum"] == pytest.approx(0.64)
    assert ancestor["info_gain"] == pytest.approx(0.24)
    assert middle["info_gain"] == child["info_gain"] == 0
    assert initial["webshop_success_probability"] == pytest.approx(0.4)
    assert initial["info_gain"] == pytest.approx(0.15)
    assert other["info_gain"] == pytest.approx(-0.15)


def test_childless_subpage_keeps_placeholder_and_inactive_slots_are_ignored():
    tracker = WebshopDeferredScores(["g"], prob_diff_mode=False, prob_floor=1e-6)
    lookup = {}
    (row,) = level(tracker, lookup, [("a", "root", "g", 0, ITEM_SUB_PAGE), ("inactive", "missing", "g", 1, None)], active=[True, False])
    assert row["avg_ans_log_probs"] == -math.inf
    assert row["info_gain_sum"] == pytest.approx(math.log(1e-6))
    assert row["info_gain"] == 0
    assert "inactive" not in lookup


def test_root_dependent_children_satisfy_mean_and_all_deferred_keeps_placeholder():
    tracker = WebshopDeferredScores(["g", "h"], prob_diff_mode=True, prob_floor=1e-6)
    rows = level(tracker, {}, [("a", "root", "g", 0, INITIAL_SEARCH), ("b", "root", "g", 0.25, None), ("c", "root", "h", 0, INITIAL_SEARCH)])
    assert rows[0]["avg_ans_log_probs"] == rows[1]["avg_ans_log_probs"] == pytest.approx(math.log(0.25))
    assert rows[0]["info_gain"] == rows[1]["info_gain"] == 0
    assert rows[2]["avg_ans_log_probs"] == tracker.roots["h"]["avg_ans_log_probs"] == -math.inf


def test_first_child_zero_is_preserved_even_when_another_child_succeeds():
    tracker = WebshopDeferredScores(["g"], prob_diff_mode=True, prob_floor=1e-6)
    lookup = {}
    (parent,) = level(tracker, lookup, [("a", "root", "g", 0, ITEM_SUB_PAGE)])
    children = level(tracker, lookup, [("first", "a", "g", 0, None), ("second", "a", "g", 1, None)])
    assert parent["avg_ans_log_probs"] == -math.inf
    assert [child["info_gain"] for child in children] == [0, 1]


@pytest.mark.parametrize("reward_mode", ["avg", "max"])
@pytest.mark.parametrize("prob_diff_mode", [True, False])
def test_deferred_records_survive_reward_collation_and_training_metrics(reward_mode, prob_diff_mode):
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from agent_system.reward_manager.tree_structure import TreeStructureRewardManager
    from verl.trainer.ppo.metric_utils import compute_data_metrics

    tracker = WebshopDeferredScores(["g"], prob_diff_mode=prob_diff_mode, prob_floor=1e-6)
    lookup = {}
    parent, pruned = level(tracker, lookup, [("parent", "root", "g", 0, ITEM_SUB_PAGE), ("pruned", "root", "g", 0.25, None)])
    (child,) = level(tracker, lookup, [("child", "parent", "g", 0.81, None)])
    rows = [parent, pruned, child]
    for index, row in enumerate(rows):
        row.update(
            prompts=torch.ones(1, dtype=torch.long),
            responses=torch.ones(1, dtype=torch.long),
            input_ids=torch.ones(2, dtype=torch.long),
            attention_mask=torch.ones(2, dtype=torch.long),
            advantages=torch.ones(1),
            returns=torch.ones(1),
            rewards=np.float32(0),
            traj_step=np.int32(index == 2),
            deactivate=index == 1,
            is_terminal=index != 0,
            current_tool_callings=np.float32(0),
            data_source="webshop",
        )
    config = OmegaConf.create({"env": {"max_steps": 3}, "algorithm": {"igrpo": {"stable_steps": 150, "stable_method": "normal", "reward_mode": reward_mode}}})
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "")
    collector = TrajectoryCollector(config, tokenizer)
    batch = collector.gather_rollout_data_tree_structure(rows, {"success_rate": np.asarray([0.0])}, global_steps=0)
    reward = TreeStructureRewardManager(tokenizer, 0, config)(batch)
    batch.batch["token_level_scores"] = reward
    batch.batch["token_level_rewards"] = reward
    metrics = compute_data_metrics(batch, use_critic=False)
    expected_reward = 0.125 if prob_diff_mode else math.log(0.25) * 0.5
    assert metrics["episode/reward/mean"] == pytest.approx(expected_reward / 2)
    assert metrics["episode/reward/max"] == pytest.approx(max(0, expected_reward))
    assert metrics["episode/reward/min"] == pytest.approx(min(0, expected_reward))


@pytest.mark.parametrize("prob_diff_mode", [True, False])
def test_collector_updates_archived_parent_before_using_child_gains(monkeypatch, prob_diff_mode):
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from igrpo.core_igrpo import TrajectoryNodeStateManagement
    from treehca.training_webshop_scoring import WebshopInfoGainScorer

    config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "treehca", "treehca": {"prob_floor": 1e-6}, "igrpo": {"prob_diff_mode": prob_diff_mode, "gamma": 1, "max_traj_to_expand_per_node": 2, "expand_mode": "full"}},
            "env": {"env_name": "Webshop", "max_steps": 2, "rollout": {"n": 2}},
            "data": {"max_prompt_length": 32, "max_response_length": 8},
            "actor_rollout_ref": {"rollout": {}},
        }
    )
    collector = TrajectoryCollector(config, SimpleNamespace(batch_decode=lambda *args, **kwargs: ["action"] * 2))
    collector.preprocess_batch = lambda **kwargs: DataProto.from_dict(
        tensors={key: torch.ones(2, 1, dtype=torch.long) for key in ("input_ids", "attention_mask", "position_ids")},
        non_tensors={"raw_prompt_ids": np.asarray([[1], [1]])},
    )

    class Actor:
        world_size = 2

        def generate_sequences(self, data):
            return DataProto.from_dict(tensors={"responses": torch.ones(2, 1, dtype=torch.long), "prompts": torch.ones(2, 1, dtype=torch.long), "input_ids": torch.ones(2, 2, dtype=torch.long)})

    class Environment:
        step_index = 0

        def reset(self, **kwargs):
            return {"text": ["prompt"] * 2}, [{}, {}]

        def step(self, actions):
            self.step_index += 1
            return {"text": ["prompt"] * 2}, np.zeros(2), np.zeros(2, dtype=bool), [{"won": False}] * 2

        def scoring_page_types(self):
            return ["" if self.step_index == 0 else "search_results"] * 2

        def scoring_payloads(self, *args):
            return np.asarray([None, None], dtype=object)

        def fork_from(self, destination, source):
            assert (destination, source) == (1, 0)

        def success_evaluator(self, **kwargs):
            return {}

    env = Environment()

    def compute(scorer, batch, **kwargs):
        probabilities = [0, 0.25] if env.step_index == 1 else [0.81, 0.04]
        batch.batch["avg_ans_log_probs"] = torch.tensor([math.log(p) if p else -math.inf for p in probabilities], dtype=torch.float64)
        batch.batch["webshop_scored"] = torch.ones(2, dtype=torch.bool)
        batch.non_tensor_batch["webshop_success_probability"] = np.asarray(probabilities, dtype=object)
        batch.non_tensor_batch["webshop_skipped_reason"] = np.asarray([ITEM_SUB_PAGE, None] if env.step_index == 1 else [None, None], dtype=object)
        return batch

    monkeypatch.setattr(WebshopInfoGainScorer, "compute", compute)
    monkeypatch.setattr(TrajectoryNodeStateManagement, "get_expand_num", lambda *args, **kwargs: np.asarray([2, 0]))
    branch_values = []
    original = TrajectoryNodeStateManagement.compute_expand_prob

    def expansion(manager, val, gamma):
        branch_values.append(val.copy())
        return original(manager, val, gamma)

    monkeypatch.setattr(TrajectoryNodeStateManagement, "compute_expand_prob", expansion)
    rows, _ = collector.vanilla_multi_turn_loop_with_tree_structure(DataProto.from_dict(tensors={"prompts": torch.ones(2, 1)}), Actor(), env)
    parent, sibling, first, second = rows
    transform = (lambda p: p) if prob_diff_mode else math.log
    assert parent["info_gain_sum"] == pytest.approx(transform(0.81))
    assert parent["info_gain"] == pytest.approx(transform(0.81) - transform(0.45))
    assert sibling["info_gain"] == pytest.approx(transform(0.25) - transform(0.45))
    assert first["info_gain"] == 0
    assert second["info_gain"] == pytest.approx(transform(0.04) - transform(0.81))
    assert [row["page_type"] for row in rows] == ["", "", "search_results", "search_results"]
    assert first["avg_ans_log_probs"] == pytest.approx(math.log(0.81))
    assert branch_values[1] == pytest.approx([transform(0.81) / 2, (2 * transform(0.04) - transform(0.81)) / 2])
