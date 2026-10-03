"""Training transport and teacher-forcing contracts without model weights."""

import math
import copy
import itertools
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from treehca.product_page_parser import ProductOptionGroup, extract_product_page_contexts
from treehca.pseudo_rollout_product_page import ActionChoiceScore, ProductOptionGroupPseudoRollout, ProductPagePseudoRolloutScores
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import sample_diverse_product_goals
from treehca.pseudo_rollout_product_page_outcomes_testbed import _DEFAULT_ATTRIBUTES, _DEFAULT_CATALOG, construct_start_episode, create_server
from treehca.pseudo_rollout_results_page import ResultsPageAnswerProbe, prepare_results_page_answer_probe
from treehca.training_pseudo_probes import TrainingPseudoProbeScorer
from treehca.training_webshop_env import TreeHCAWebshopWorker
from treehca.training_webshop_scoring import TrainingWebshopTurnSuccessScorer, WebshopInfoGainScorer, _render_scoring_prompt, resolve_treehca_scorer
from treehca.webshop_option_success import NativeOptionSuccessPlan, OptionCoverageGroup, build_native_option_score_plan
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


def test_product_probe_drops_oldest_history_until_it_fits(native_training):
    episode, tokenizer = native_training
    snapshot = payload_for(worker_for(episode), episode)["snapshot"]
    parts = extract_product_page_contexts([snapshot.prompt])[0]
    from treehca.product_page_parser import parse_product_page_fields

    group = parse_product_page_fields(parts.current_observation, parts.admissible_actions).option_groups[0]
    actions = tuple(f"click[{name}]" for name in group.values) + ("none",)
    plan = NativeOptionSuccessPlan((OptionCoverageGroup(group.name, actions, (1,) + (0,) * (len(actions) - 1)),), 0, 1)
    history = (("oldest " * 4000, "search[old]"), ("newest observation", "click[new]"))
    snapshot = replace(snapshot, completed_steps=2, history_limit=2)
    snapshot = _render_scoring_prompt(snapshot, history)

    scorer = object.__new__(TrainingWebshopTurnSuccessScorer)
    scorer.probe_scorer = TrainingPseudoProbeScorer(tokenizer, FakeActor(), max_model_len=32768)
    scorer.prune_unsuccessful_choices = False
    job = _ProductJob(snapshot, plan)
    one_history = _render_scoring_prompt(snapshot, history[1:])
    one_history_probes = scorer._prepare_product_probes(job, one_history)
    limit = max(len(answer.prompt_token_ids) + len(answer.response_token_ids) for probe, _ in one_history_probes for answer in probe.answer_probes)
    full_probes = scorer._prepare_product_probes(job, snapshot)
    assert max(len(answer.prompt_token_ids) + len(answer.response_token_ids) for probe, _ in full_probes for answer in probe.answer_probes) > limit

    scorer.probe_scorer.max_model_len = limit
    probes, owners = [], []
    scorer._append_product_probes([job], probes, owners)
    assert job.snapshot.history == history[1:]
    assert "oldest" not in job.snapshot.prompt
    assert "newest observation" in job.snapshot.prompt
    scorer.probe_scorer.score(probes)


def test_results_probe_drops_oldest_history_until_it_fits(native_training):
    episode, tokenizer = native_training
    results_episode = episode.clone()
    results_episode.advance("click[< prev]")
    snapshot = payload_for(worker_for(results_episode), results_episode, "item_page")["snapshot"]
    history = (("oldest " * 4000, "search[old]"), ("newest observation", "click[new]"))
    snapshot = replace(snapshot, completed_steps=2, history_limit=2)
    snapshot = _render_scoring_prompt(snapshot, history)
    one_history = _render_scoring_prompt(snapshot, history[1:])

    def largest_request(candidate):
        candidate_parts = extract_product_page_contexts([candidate.prompt])[0]
        visible = {asin.lower() for asin in candidate.visible_asins}
        actions = [action for action in candidate_parts.admissible_actions if action.startswith("click[") and action[6:-1].lower() in visible]
        probes = [prepare_results_page_answer_probe(candidate.prompt, action, tokenizer) for action in actions]
        return max(len(probe.prompt_token_ids) + len(probe.response_token_ids) for probe in probes)

    limit = largest_request(one_history)
    assert largest_request(snapshot) > limit
    scorer = object.__new__(TrainingWebshopTurnSuccessScorer)
    scorer.probe_scorer = TrainingPseudoProbeScorer(tokenizer, FakeActor(), max_model_len=limit)
    fitted = scorer._fit_results_snapshot(snapshot)
    assert fitted.history == history[1:]
    assert "oldest" not in fitted.prompt
    assert "newest observation" in fitted.prompt


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


def test_partial_score_setting_defaults_off_and_requires_boolean():
    assert WebshopInfoGainScorer(SimpleNamespace(), FakeActor(), max_model_len=64).include_partial_scores is False
    assert WebshopInfoGainScorer(SimpleNamespace(), FakeActor(), max_model_len=64, include_partial_scores=True).include_partial_scores is True
    with pytest.raises(ValueError, match="include_partial_scores"):
        WebshopInfoGainScorer(SimpleNamespace(), FakeActor(), max_model_len=64, include_partial_scores="true")
    with pytest.raises(ValueError, match="include_partial_scores"):
        TrainingWebshopTurnSuccessScorer(SimpleNamespace(), SimpleNamespace(), include_partial_scores=1)


@pytest.fixture
def partial_product(native_training):
    episode, _ = native_training
    asin = episode.env.server.user_sessions[episode.env.session]["asin"]
    item = copy.deepcopy(episode.env.server.product_item_dict[asin])
    item.update(name="travel backpack", Title="Travel Backpack", query="backpack", product_category="Bags › Backpacks", Attributes=["waterproof"], BulletPoints=[], Description="", options={"shape": ["oval", "round"], "size": ["large", "small"]})
    goal = copy.deepcopy(episode.env.server.user_sessions[episode.env.session]["goal"])
    goal.update(name="travel backpack", query="backpack", product_category="Bags › Backpacks", attributes=["waterproof", "durable"], price_upper=100, goal_options={"shape": "oval", "size": "large"})
    return item, goal


def test_partial_option_scores_include_wrong_choices_and_none_without_normalizing(partial_product):
    item, goal = partial_product
    plan = build_native_option_score_plan(item, goal, 50, {})
    assert plan.max_score == pytest.approx(0.8)
    assert plan.constant_probability is None
    assert plan.successful_actions() == {group.name: group.actions for group in plan.groups}
    distributions = {"shape": {"click[oval]": 0.3, "click[round]": 0.1, "none": 0.1}, "size": {"click[large]": 0.2, "click[small]": 0.15, "none": 0.15}}
    # Joint mass is 0.25, not 1. Each of the nine combinations has native
    # score 0.8, 0.6, or 0.4. Missing raw answer mass contributes nothing.
    assert plan.aggregate_success_mass(distributions) == pytest.approx(0.15)
    normalized = {name: {action: probability / 0.5 for action, probability in choices.items()} for name, choices in distributions.items()}
    assert plan.aggregate(normalized) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="unknown choice"):
        plan.aggregate_success_mass({**distributions, "shape": {"click[triangle]": 0.1}})
    with pytest.raises(ValueError, match="finite"):
        plan.aggregate_success_mass({**distributions, "shape": {"none": math.nan}})


@pytest.mark.parametrize("variant", ["ordinary", "unreachable", "expensive", "zero_type", "overlap", "selected", "blocking", "no_options", "no_targets"])
def test_partial_plan_matches_native_cartesian_reward_oracle(partial_product, variant):
    from web_agent_site.engine.goal import get_reward

    item, goal = partial_product
    selected, price = {}, 50
    if variant == "unreachable":
        goal["goal_options"]["size"] = "giant"
    elif variant == "expensive":
        price = 200
    elif variant == "zero_type":
        item["name"] = "spatula"
    elif variant == "overlap":
        item["options"] = {"first": ["shape oval size large", "shape oval"], "second": ["shape oval", "size small"]}
    elif variant == "selected":
        selected = {"shape": "oval", "size": "small"}
    elif variant == "blocking":
        item["options"] = {"configuration": ["shape oval", "shape oval size large"]}
        selected = {"configuration": "shape oval"}
    elif variant == "no_options":
        item["options"] = {}
    elif variant == "no_targets":
        goal["goal_options"] = {}
    plan = build_native_option_score_plan(item, goal, price, selected)
    distributions = {group.name: dict(zip(group.actions, [0.7 / len(group.actions)] * len(group.actions))) for group in plan.groups}
    mutable = set(distributions)
    fixed = {name: value for name, value in selected.items() if name not in mutable}
    scores, weighted = [], []
    for choices in itertools.product(*(tuple(distributions[group.name].items()) for group in plan.groups)):
        options = dict(fixed)
        for group, (action, _) in zip(plan.groups, choices):
            if action == "none":
                if group.name in selected:
                    options[group.name] = selected[group.name]
            else:
                options[group.name] = action[6:-1].lower()
        reward = float(get_reward(item, goal, price, options))
        scores.append(reward)
        weighted.append(reward * math.prod(probability for _, probability in choices))
    assert plan.max_score == pytest.approx(max(scores))
    assert plan.aggregate_success_mass(distributions) == pytest.approx(math.fsum(weighted))


def test_partial_pruning_keeps_only_positive_reward_completions(partial_product):
    item, goal = partial_product
    item["Attributes"] = []
    plan = build_native_option_score_plan(item, goal, 200, {})
    # Neither target gives zero, but a wrong shape can accompany the right
    # size, and vice versa. Those wrong choices must not be pruned.
    assert plan.successful_actions() == {group.name: group.actions for group in plan.groups}
    item["options"].pop("size")
    plan = build_native_option_score_plan(item, goal, 200, {})
    assert plan.successful_actions() == {"shape": ("click[oval]",)}
    assert plan.aggregate_success_mass({"shape": {"click[oval]": 0.3}}) == pytest.approx(0.06)


@pytest.mark.parametrize("include_partial,prune,threshold", [(False, True, 0), (True, True, 0), (True, False, 0), (True, True, 0.21)])
def test_partial_results_weight_products_once_and_share_cache(partial_product, native_training, monkeypatch, include_partial, prune, threshold):
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE_NO_HIS

    item, goal = partial_product
    _, tokenizer = native_training
    partial = dict(item, asin="B000PART01")
    full = dict(item, asin="B000FULL01", Attributes=["waterproof", "durable"])
    constant = dict(item, asin="B000CONST1", options={})
    zero = dict(item, asin="B000ZERO01", name="spatula")
    products = {product["asin"]: product for product in (partial, full, constant, zero)}
    goal["instruction_text"] = "Find a durable waterproof travel backpack, oval and large."
    actions = tuple(f"click[{asin.lower()}]" for asin in products)
    prompt = WEBSHOP_TEMPLATE_NO_HIS.format(task_description=goal["instruction_text"], current_observation="Search results", available_actions="\n".join(f"'{action}'," for action in actions))
    snapshot = WebshopTurnSnapshot("partial-catalog", json.dumps(goal), goal["instruction_text"], prompt, "search_results", visible_asins=tuple(products))
    payload = dict(snapshot=snapshot, products=products, prices={asin: 50 for asin in products}, show_attrs=False)
    scorer = WebshopInfoGainScorer(tokenizer, FakeActor(), max_model_len=32768, path_probability_threshold=threshold, include_partial_scores=include_partial, prune_unsuccessful_choices=prune)
    seen = []
    entries = {"B000PART01": 0.25, "B000FULL01": 0.2, "B000CONST1": 0.1}
    probabilities = {"click[oval]": 0.3, "click[round]": 0.1, "click[large]": 0.2, "click[small]": 0.15, "none": 0.1}

    def score(probes):
        seen.extend(probes)
        scores = []
        for probe in probes:
            if isinstance(probe, ResultsPageAnswerProbe):
                response = tokenizer.decode(probe.response_token_ids).upper()
                scores.append(next(probability for asin, probability in entries.items() if asin in response))
            else:
                # Size's none probability differs from shape's.
                size = "click[large]" in probe.actions
                choices = tuple(ActionChoiceScore(name, action, 0.15 if size and action == "none" else probabilities[action], 0, 0, 0) for name, action in zip(probe.option_names, probe.actions))
                scores.append(ProductPagePseudoRolloutScores(choices))
        return scores

    monkeypatch.setattr(scorer.probes, "score", score)
    batch = scorer.compute(info_batch([payload, payload]), policy_version=0)
    expected = 0.25 * 0.15 + 0.2 * 0.2 + 0.1 * 0.4 if include_partial else 0.2 * 0.3 * 0.2
    if threshold:
        # Threshold the entry, not the maximum-score-weighted entry (0.20)
        # or its final contribution (0.0375). Other branches stay excluded.
        expected = 0.25 * 0.15
    assert batch.non_tensor_batch["webshop_success_probability"].tolist() == pytest.approx([expected, expected])
    assert torch.exp(batch.batch["avg_ans_log_probs"]).tolist() == pytest.approx([expected, expected])
    assert scorer.scorers[snapshot.catalog_key].include_partial_scores == include_partial
    if include_partial:
        assert sum(isinstance(probe, ResultsPageAnswerProbe) for probe in seen) == 3
        assert all(len(probe.actions) == 3 for probe in seen if not isinstance(probe, ResultsPageAnswerProbe))
    before = len(seen)
    source = scorer.scorers[snapshot.catalog_key].source
    direct = dict(payload, snapshot=source.product_entry(snapshot, partial["asin"] if include_partial else full["asin"]))
    direct_batch = scorer.compute(info_batch([direct]), policy_version=0)
    assert direct_batch.non_tensor_batch["webshop_success_probability"][0] == pytest.approx(0.15 if include_partial else 0.06)
    scorer.compute(info_batch([payload]), policy_version=0)
    assert len(seen) == before
    scorer.compute(info_batch([payload]), policy_version=1)
    assert len(seen) > before
