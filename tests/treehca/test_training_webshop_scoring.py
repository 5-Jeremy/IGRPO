"""Training transport and teacher-forcing contracts without model weights."""

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from treehca.product_page_parser import ProductOptionGroup, extract_product_page_contexts
from treehca.pseudo_rollout_product_page import ProductOptionGroupPseudoRollout
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import sample_diverse_product_goals
from treehca.pseudo_rollout_product_page_outcomes_testbed import _DEFAULT_ATTRIBUTES, _DEFAULT_CATALOG, construct_start_episode, create_server
from treehca.pseudo_rollout_results_page import ResultsPageAnswerProbe
from treehca.training_pseudo_probes import TrainingPseudoProbeScorer
from treehca.training_webshop_env import TreeHCAWebshopWorker
from treehca.training_webshop_scoring import TrainingWebshopTurnSuccessScorer, WebshopInfoGainScorer, resolve_treehca_scorer
from treehca.webshop_option_success import NativeOptionSuccessPlan, OptionCoverageGroup
from treehca.webshop_probability_snapshot import WebshopTurnSnapshot
from treehca.webshop_turn_success import _ProductJob
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor


class FakeActor:
    world_size = 4

    def __init__(self):
        self.calls = []

    def compute_log_prob(self, data):
        assert len(data) % self.world_size == 0
        assert data.meta_info["treehca_probe_temperature"] == 1.0
        self.calls.append(data)
        # An unnormalized label weight proportional to its integer token ID.
        logs = torch.log((data.batch["responses"].double() + 1) / 1_000_000)
        return DataProto.from_dict(tensors={"old_log_probs": logs}, meta_info={"temperature": 1.0})


def test_teacher_forcing_mixed_rows_normalization_padding_and_deduplication():
    actor = FakeActor()
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda *a, **kw: [8, 9])
    scorer = TrainingPseudoProbeScorer(tokenizer, actor, max_model_len=20, batch_size=3)
    group = ProductOptionGroup(name="color", values=("red", "blue"), correct_options=())
    product = ProductOptionGroupPseudoRollout("unused", (1, 2), ("A", "B"), ("click[red]", "click[blue]"), ((10, 11), (12, 13)), 0, group)
    results = ResultsPageAnswerProbe((1, 2, 3), (20, 21, 22, 23), (1, 2))
    scores = scorer.score([results, product, results])
    assert scores[0] == pytest.approx((22 / 1e6) * (23 / 1e6))
    assert scores[0] == scores[2]
    assert scores[1].action_probabilities == pytest.approx({"click[red]": 23 / 50, "click[blue]": 27 / 50})
    assert len(actor.calls) == 2  # Five unique rows, chunked at three.
    assert scorer.forward_pass_time_seconds > 0
    first = actor.calls[0].batch
    assert first["input_ids"].shape[1] == 7
    assert first["attention_mask"][1].tolist() == [0, 0, 1, 1, 1, 1, 1]
    assert first["position_ids"][1, -5:].tolist() == list(range(5))


def test_overflow_is_an_error_before_actor_call():
    actor = FakeActor()
    scorer = TrainingPseudoProbeScorer(SimpleNamespace(), actor, max_model_len=2)
    with pytest.raises(ValueError, match="overflowing"):
        scorer.score([ResultsPageAnswerProbe((1, 2), (3,), (0,))])
    assert not actor.calls


def test_unsuccessful_choice_pruning_defaults_on_and_requires_boolean():
    scorer = WebshopInfoGainScorer(SimpleNamespace(), FakeActor(), max_model_len=64)
    assert scorer.prune_unsuccessful_choices is True
    assert WebshopInfoGainScorer(SimpleNamespace(), FakeActor(), max_model_len=64, prune_unsuccessful_choices=False).prune_unsuccessful_choices is False
    with pytest.raises(ValueError, match="prune_unsuccessful_choices"):
        WebshopInfoGainScorer(SimpleNamespace(), FakeActor(), max_model_len=64, prune_unsuccessful_choices="false")


@pytest.mark.parametrize(
    "algorithm,environment,choice,expected",
    [
        ("treehca", "Webshop", "auto", "webshop"),
        ("treehca", "search", "auto", "answer_block"),
        ("treehca", "Webshop", "answer_block", "answer_block"),
        ("treehca", "custom", "webshop", "webshop"),
        ("igrpo", "Webshop", "webshop", "answer_block"),
        ("gigpo", "Webshop", "invalid", "answer_block"),
    ],
)
def test_dispatch_preserves_existing_algorithms(algorithm, environment, choice, expected):
    config = OmegaConf.create({"algorithm": {"adv_estimator": algorithm, "treehca": {"pseudo_scorer": choice}}, "env": {"env_name": environment}})
    assert resolve_treehca_scorer(config) == expected


def info_batch(payloads, *, active=None, terminated=None, won=None):
    count = len(payloads)
    objects = np.empty(count, dtype=object)
    objects[:] = payloads
    return DataProto.from_dict(
        tensors={"prompts": torch.ones(count, 3, dtype=torch.long)},
        non_tensors={
            "webshop_scoring_payload": objects,
            "webshop_active": np.asarray(active if active is not None else [True] * count),
            "webshop_terminated": np.asarray(terminated if terminated is not None else [False] * count),
            "webshop_won": np.asarray(won if won is not None else [False] * count),
        },
    )


def test_terminal_outcomes_and_inactive_slots_need_no_snapshot_or_inference():
    actor = FakeActor()
    scorer = WebshopInfoGainScorer(SimpleNamespace(), actor, max_model_len=4096)
    batch = info_batch([None] * 3, active=[True, True, False], terminated=[True, True, False], won=[True, False, False])
    assert scorer.compute(batch, policy_version=0) is batch
    assert batch.non_tensor_batch["webshop_success_probability"].tolist() == [1.0, 0.0, None]
    assert batch.batch["avg_ans_log_probs"][:2].tolist() == [0.0, -math.inf]
    assert torch.isnan(batch.batch["avg_ans_log_probs"][2])
    scorer.require_active_scores(batch)
    assert not actor.calls
    assert scorer.metrics()["scorer/cache_reuses"] == 0
    assert scorer.metrics()["scorer/forward_pass_time_seconds"] == 0
    assert scorer.metrics()["scorer/scoring_time_seconds"] > 0
    assert not any("probability_min" in key for key in scorer.metrics())


@pytest.fixture(scope="module")
def native_training():
    server = create_server(_DEFAULT_CATALOG, _DEFAULT_ATTRIBUTES, 0, 1000)
    selected = sample_diverse_product_goals(server.all_products, server.goals, 1, 0)[0]
    episode = construct_start_episode(server, selected, seed=0)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    return episode, tokenizer


def worker_for(episode):
    worker = object.__new__(TreeHCAWebshopWorker)
    worker.env = episode.env
    worker._initial_prices = episode.env.server.product_prices
    return worker


def payload_for(worker, episode, previous="search_results"):
    manager = episode.manager
    return worker.scoring_payload(episode.prompt, manager.tasks[0], manager.memory[0], len(manager.memory[0]), manager.config.env.history_length, previous)


def test_native_payload_order_duplicates_cache_and_policy_invalidation(native_training):
    episode, tokenizer = native_training
    worker = worker_for(episode)
    payload = payload_for(worker, episode)
    actor = FakeActor()
    scorer = WebshopInfoGainScorer(tokenizer, actor, max_model_len=32768)
    batch = info_batch([payload, None], terminated=[False, True], won=[False, True])
    batch, _ = pad_dataproto_to_divisor(batch, 4)
    order = torch.tensor([3, 0, 2, 1])
    batch.reorder(order)
    scorer.compute(batch, policy_version=0)
    values = batch.non_tensor_batch["webshop_success_probability"]
    assert values[0] == values[3] == 1
    assert 0 < values[1] < 1 and values[1] == values[2]
    calls = len(actor.calls)
    first_metrics = scorer.metrics()
    assert first_metrics["scorer/item_page_probability_min"] == pytest.approx(values[1])
    assert first_metrics["scorer/item_page_probability_max"] == pytest.approx(values[1])
    assert first_metrics["scorer/forward_pass_time_seconds"] > 0
    scorer.compute(batch, policy_version=0)
    assert len(actor.calls) == calls
    assert scorer.metrics()["scorer/cache_reuses"] > first_metrics["scorer/cache_reuses"]
    assert scorer.metrics()["scorer/forward_pass_time_seconds"] == first_metrics["scorer/forward_pass_time_seconds"]
    scorer.compute(batch, policy_version=1)
    assert len(actor.calls) > calls
    assert torch.exp(batch.batch["avg_ans_log_probs"]).tolist() == pytest.approx(values.tolist())


def test_training_product_probe_uses_exact_names_and_joint_answer_tokens(native_training):
    episode, tokenizer = native_training
    snapshot = payload_for(worker_for(episode), episode)["snapshot"]
    parts = extract_product_page_contexts([snapshot.prompt])[0]
    from treehca.product_page_parser import parse_product_page_fields

    group = parse_product_page_fields(parts.current_observation, parts.admissible_actions).option_groups[0]
    actions = tuple(f"click[{name}]" for name in group.values) + ("none",)
    plan = NativeOptionSuccessPlan((OptionCoverageGroup(group.name, actions, (1,) + (0,) * (len(actions) - 1)),), 0, 1)
    job = _ProductJob(snapshot, plan)
    scorer = object.__new__(TrainingWebshopTurnSuccessScorer)
    scorer.probe_scorer = SimpleNamespace(tokenizer=tokenizer)
    scorer.prune_unsuccessful_choices = False
    probes, owners = [], []
    scorer._append_product_probes([job], probes, owners)
    assert owners == [(job, group.name)]
    probe = probes[0]
    assert probe.option_names == (*group.values, "none")
    assert probe.actions == actions
    rendered_prompt = tokenizer.decode(probe.answer_probes[0].prompt_token_ids)
    assert f"Your available options for {group.name} are:" in rendered_prompt
    assert all(f"\n{name}\n" in rendered_prompt for name in group.values)
    assert "letter" not in rendered_prompt
    for name, answer in zip(probe.option_names, probe.answer_probes):
        response = tokenizer.decode(answer.response_token_ids)
        assert response == f"<think> The best choice for the {group.name} group is {name}"
        assert answer.answer_token_indices
        scored_text = tokenizer.decode([answer.response_token_ids[index] for index in answer.answer_token_indices])
        assert scored_text in {name, f" {name}"}
    actor = FakeActor()
    scores = TrainingPseudoProbeScorer(tokenizer, actor, max_model_len=32768).score(probes)[0]
    raw = [math.prod((token + 1) / 1_000_000 for token in (answer.response_token_ids[index] for index in answer.answer_token_indices)) for answer in probe.answer_probes]
    assert scores.action_probabilities == pytest.approx(dict(zip(actions, raw)))
    assert plan.aggregate_success_mass({group.name: scores.action_probabilities}) == pytest.approx(raw[0])

    scorer.prune_unsuccessful_choices = True
    pruned, pruned_owners = [], []
    scorer._append_product_probes([job], pruned, pruned_owners)
    assert pruned_owners == owners
    assert pruned[0].actions == (actions[0],)
    assert pruned[0].option_names == (group.values[0],)
    pruned_scores = TrainingPseudoProbeScorer(tokenizer, FakeActor(), max_model_len=32768).score(pruned)[0]
    assert plan.aggregate_success_mass({group.name: pruned_scores.action_probabilities}) == pytest.approx(raw[0])


def test_successful_actions_include_cross_group_completions():
    plan = NativeOptionSuccessPlan(
        (
            OptionCoverageGroup("size", ("click[small]", "click[large]", "none"), (1, 0, 0)),
            OptionCoverageGroup("color", ("click[red]", "click[blue]", "none"), (2, 0, 0)),
        ),
        0,
        3,
    )
    assert plan.successful_actions() == {"size": ("click[small]",), "color": ("click[red]",)}
    assert plan.aggregate_success_mass({"size": {"click[small]": 0.3}, "color": {"click[red]": 0.4}}) == pytest.approx(0.12)


@pytest.mark.parametrize("page", ["index", "unknown"])
def test_unsupported_pages_remain_none_and_fail_training(native_training, page):
    episode, tokenizer = native_training
    payload = payload_for(worker_for(episode), episode)
    payload["snapshot"] = replace(payload["snapshot"], page_type=page)
    scorer = WebshopInfoGainScorer(tokenizer, FakeActor(), max_model_len=32768)
    batch = scorer.compute(info_batch([payload]), policy_version=0)
    assert batch.non_tensor_batch["webshop_success_probability"][0] is None
    with pytest.raises(ValueError, match=f"unsupported_page:{page}"):
        scorer.require_active_scores(batch)


@pytest.mark.parametrize("page,reason", [("", "deferred_initial_search_page"), ("item_sub_page", "deferred_item_sub_page")])
def test_deferred_pages_have_zero_probability_placeholders(native_training, page, reason):
    episode, tokenizer = native_training
    payload = payload_for(worker_for(episode), episode)
    payload["snapshot"] = replace(payload["snapshot"], page_type=page)
    scorer = WebshopInfoGainScorer(tokenizer, FakeActor(), max_model_len=32768)
    batch = scorer.compute(info_batch([payload]), policy_version=0)
    assert batch.non_tensor_batch["webshop_success_probability"][0] == 0
    assert batch.non_tensor_batch["webshop_skipped_reason"][0] == reason
    assert batch.batch["avg_ans_log_probs"][0] == -math.inf
    scorer.require_active_scores(batch)


def test_native_episode_transfer_isolated_and_preserves_scoring_state(native_training):
    episode, _ = native_training
    source_episode, target_episode = episode.clone(), episode.clone()
    source, target = worker_for(source_episode), worker_for(target_episode)
    source._rollout_task_id = 123
    state = source.export_episode()
    target.import_episode(state)
    assert target._rollout_task_id == 123
    assert payload_for(source, source_episode)["snapshot"] == payload_for(target, target_episode)["snapshot"]
    target.env.server.user_sessions[target.env.session]["options"]["test"] = "changed"
    target.env.server.product_prices["test"] = 123
    assert "test" not in source.env.server.user_sessions[source.env.session]["options"]
    assert "test" not in source.env.server.product_prices
    target.reset(0)
    assert target.env.server.product_prices is target._initial_prices
    assert target._rollout_task_id == 0


def test_worker_logging_keeps_pre_purchase_session(monkeypatch):
    from agent_system.environments.env_package.webshop.envs import WebshopWorker

    worker = object.__new__(TreeHCAWebshopWorker)
    worker.env = SimpleNamespace(unwrapped=SimpleNamespace(session="original-session"))
    worker._rollout_task_id = 42

    def purchase(self, action):
        self.env.unwrapped.session = "autoreset-session"
        return "terminal observation", 10.0, True, {"won": True}

    monkeypatch.setattr(WebshopWorker, "step", purchase)
    observation, reward, done, info = worker.step("click[buy now]")
    assert observation == "terminal observation" and reward == 10 and done
    assert info["webshop_session_id"] == "original-session"
    assert info["webshop_task_id"] == 42


def test_native_results_expand_and_share_fresh_product_cache(native_training):
    episode, tokenizer = native_training
    results_episode = episode.clone()
    asin = episode.env.server.user_sessions[episode.env.session]["asin"]
    results_episode.advance("click[< prev]")
    while f"click[{asin.lower()}]" not in extract_product_page_contexts([results_episode.prompt])[0].admissible_actions:
        results_episode.advance("click[next >]")
    actor = FakeActor()
    scorer = WebshopInfoGainScorer(tokenizer, actor, max_model_len=32768, path_probability_threshold=0)
    payload = payload_for(worker_for(results_episode), results_episode, "item_page")
    result = scorer.compute(info_batch([payload]), policy_version=0)
    assert result.non_tensor_batch["webshop_success_probability"][0] > 0
    probability = result.non_tensor_batch["webshop_success_probability"][0]
    assert scorer.metrics()["scorer/search_results_probability_min"] == pytest.approx(probability)
    assert scorer.metrics()["scorer/search_results_probability_max"] == pytest.approx(probability)
    calls = len(actor.calls)
    direct = payload_for(worker_for(episode), episode)
    assert direct["snapshot"].catalog_key == payload["snapshot"].catalog_key
    scorer.compute(info_batch([direct]), policy_version=0)
    assert len(actor.calls) == calls


@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize("prob_diff_mode", [True, False])
@pytest.mark.parametrize("environment", ["Webshop", "search"])
def test_rollout_loop_dispatch_terminal_values_and_missing_score_error(terminated, prob_diff_mode, environment, monkeypatch):
    from agent_system.multi_turn_rollout import rollout_loop
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    legacy_calls = []

    def legacy_score(batch, **kwargs):
        legacy_calls.append(kwargs)
        batch.batch["avg_ans_log_probs"] = torch.tensor([math.log(0.25), math.log(0.81)], dtype=torch.float64)
        return batch

    monkeypatch.setattr(rollout_loop, "compute_answer_block_avg_log_prob", legacy_score)
    monkeypatch.setattr(rollout_loop, "adjust_batch", lambda data, **kwargs: (data, torch.arange(len(data))))
    config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "treehca", "treehca": {"prob_floor": 1e-6}, "igrpo": {"prob_diff_mode": prob_diff_mode, "gamma": 1, "max_traj_to_expand_per_node": 2, "expand_mode": "full"}},
            "env": {"env_name": environment, "max_steps": 1, "rollout": {"n": 2}},
            "data": {"max_prompt_length": 32, "max_response_length": 8},
            "actor_rollout_ref": {"rollout": {}},
        }
    )
    tokenizer = SimpleNamespace(batch_decode=lambda *a, **kw: ["action"] * 2)
    collector = TrajectoryCollector(config, tokenizer)

    def preprocess(**kwargs):
        return DataProto.from_dict(
            tensors={key: torch.ones(2, 1, dtype=torch.long) for key in ("input_ids", "attention_mask", "position_ids")},
            non_tensors={"raw_prompt_ids": np.asarray([[1], [1]])},
        )

    collector.preprocess_batch = preprocess

    class Actor(FakeActor):
        world_size = 2

        def generate_sequences(self, data):
            return DataProto.from_dict(
                tensors={
                    "responses": torch.ones(2, 1, dtype=torch.long),
                    "prompts": torch.ones(2, 1, dtype=torch.long),
                    "input_ids": torch.ones(2, 2, dtype=torch.long),
                }
            )

    class Environment:
        def reset(self, **kwargs):
            return {"text": ["prompt"] * 2}, [{}, {}]

        def scoring_page_types(self):
            assert environment == "Webshop"
            return ["index"] * 2

        def step(self, actions):
            return {"text": ["prompt"] * 2}, np.asarray([10, 0]), np.asarray([terminated] * 2), [{"won": True}, {"won": False}]

        def scoring_payloads(self, *args):
            if terminated:
                return np.asarray([None, None], dtype=object)
            payload = {"snapshot": WebshopTurnSnapshot("test", "{}", "", "", "index"), "products": {}, "prices": {}, "max_choices": 0, "show_attrs": False}
            return np.asarray([payload, payload], dtype=object)

        def success_evaluator(self, **kwargs):
            return {"success_rate": np.asarray([1, 0])}

    gen_batch = DataProto.from_dict(tensors={"prompts": torch.ones(2, 1, dtype=torch.long)})
    if environment == "search":
        rows, _ = collector.vanilla_multi_turn_loop_with_tree_structure(gen_batch, Actor(), Environment())
        expected = [0.25, 0.81] if prob_diff_mode else [math.log(0.25), math.log(0.81)]
        assert [row["info_gain_sum"] for row in rows] == pytest.approx(expected)
        assert [row["info_gain"] for row in rows] == pytest.approx(expected)  # Original zero root baseline.
        assert len(legacy_calls) == 1 and legacy_calls[0]["think"] is False
        assert all("webshop_root_avg_ans_log_probs" not in row for row in rows)
    elif terminated:
        rows, _ = collector.vanilla_multi_turn_loop_with_tree_structure(gen_batch, Actor(), Environment())
        expected = [1.0, 0.0] if prob_diff_mode else [0.0, math.log(1e-6)]
        assert [row["info_gain_sum"] for row in rows] == pytest.approx(expected)
        assert [row["webshop_success_probability"] for row in rows] == [1.0, 0.0]
    else:
        with pytest.raises(ValueError, match="unsupported_page:index"):
            collector.vanilla_multi_turn_loop_with_tree_structure(gen_batch, Actor(), Environment())
