"""Batched turn scoring with native rendering/rewards and controlled inference."""

import copy
import json
import math
from dataclasses import FrozenInstanceError, replace
from itertools import product
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_batch import MixedPseudoRolloutScorer
from treehca.pseudo_rollout_product_page import prepare_product_page_grouped_choice_rollouts
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import build_artificial_group_response_prefixes, sample_diverse_product_goals
from treehca.pseudo_rollout_product_page_outcomes_testbed import _DEFAULT_ATTRIBUTES, _DEFAULT_CATALOG, construct_start_episode, create_server
from treehca.pseudo_rollout_results_page import ResultsPageAnswerProbe, prepare_results_page_answer_probe
from treehca.webshop_option_success import build_native_option_success_plan
from treehca.webshop_probability_snapshot import WebshopSnapshotSource, snapshot_webshop_rollouts
from treehca.webshop_turn_success import WebshopTurnSuccessScorer


@pytest.fixture(scope="module")
def native():
    server = create_server(_DEFAULT_CATALOG, _DEFAULT_ATTRIBUTES, 0, 1000)
    selected = sample_diverse_product_goals(server.all_products, server.goals, 1, 0)[0]
    start = construct_start_episode(server, selected, seed=0)
    source = WebshopSnapshotSource(server)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    return server, selected, start, source, tokenizer


def results_start(native):
    """Return to actual native search results, then find the target page."""
    _, selected, start, _, _ = native
    episode = start.clone()
    episode.advance("click[< prev]")
    while f"click[{selected.product['asin'].lower()}]" not in extract_product_page_contexts([episode.prompt])[0].admissible_actions:
        episode.advance("click[next >]")
    return episode


class ControlledEngine:
    """Exercise the real mixed decoder with uniform labels and chosen-token logs."""

    def __init__(self, tokenizer, results_probability=0.4):
        self.tokenizer = tokenizer
        self.results_probability = results_probability
        self.calls = []
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32768, max_logprobs=2048), cache_config=SimpleNamespace(enable_prefix_caching=False))

    def generate(self, *, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params))
        outputs = []
        for prompt, params in zip(prompts, sampling_params):
            tokens = prompt["prompt_token_ids"]
            if params.allowed_token_ids:
                values = {token: -math.log(len(params.allowed_token_ids)) for token in params.allowed_token_ids}
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[values])]))
            else:
                text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                action = text.rsplit("<answer>", 1)[1].split("</answer>")[0]
                response = self.tokenizer("<answer>" + action + "</answer>", add_special_tokens=False, return_offsets_mapping=True)
                count = sum(start != end and max(start, 8) < min(end, 8 + len(action)) for start, end in response["offset_mapping"])
                value = -math.inf if self.results_probability == 0 else math.log(self.results_probability) / count
                outputs.append(SimpleNamespace(prompt_logprobs=[{token: value} for token in tokens]))
        return outputs


def scorer_for(native, *, threshold=1e-3, probability=0.4):
    *_, source, tokenizer = native
    engine = ControlledEngine(tokenizer, probability)
    scorer = WebshopTurnSuccessScorer(source, MixedPseudoRolloutScorer(engine, tokenizer), path_probability_threshold=threshold)
    return scorer, engine


def capture(source, episode, previous="search_results"):
    return source.capture(episode.env, episode.manager, episode.prompt, previous_page_type=previous)


def test_snapshot_is_immutable_serializable_and_renders_exact_native_child(native):
    _, selected, _, source, _ = native
    episode = results_start(native)
    snapshot = capture(source, episode, "item_page")
    before = copy.deepcopy(episode.env.server.user_sessions)
    child = source.product_entry(snapshot, selected.product["asin"])
    assert episode.env.server.user_sessions == before
    actual = episode.clone()
    actual.advance(f"click[{selected.product['asin'].lower()}]")
    assert child.prompt == actual.prompt
    assert child.fresh_product_entry and not child.selected_options
    assert len(snapshot.history) <= snapshot.history_limit
    with pytest.raises(FrozenInstanceError):
        snapshot.asin = "changed"
    json.dumps(snapshot.__dict__)
    actual.advance(f"click[{next(iter(selected.goal['goal_options'].values()))}]")
    assert not child.selected_options
    assert snapshot_webshop_rollouts(source, [episode], previous_page_types=["item_page"]) == (snapshot,)


def test_two_stages_mix_probes_and_reuse_direct_fresh_products(native):
    _, selected, _, source, tokenizer = native
    episode = results_start(native)
    results = capture(source, episode, "item_page")
    direct = source.product_entry(results, selected.product["asin"])
    scorer, engine = scorer_for(native)
    scores = scorer.score([results, direct, direct], policy_version="v1")
    assert scores[1] == scores[2] and 0 < scores[1].probability < 1
    first_params = engine.calls[0][1]
    assert any(p.prompt_logprobs == 0 for p in first_params)
    assert any(p.allowed_token_ids for p in first_params)
    product_prompts = [tokenizer.decode(prompt["prompt_token_ids"]) for prompts, params in engine.calls for prompt, p in zip(prompts, params) if p.allowed_token_ids]
    assert len(product_prompts) == len(set(product_prompts))
    assert all(text.endswith("corresponds to the label:") for text in product_prompts)
    assert all(len(prompts) <= 32 for prompts, _ in engine.calls)
    before = len(engine.calls)
    assert scorer.score([direct, results], policy_version="v1") == [scores[1], scores[0]]
    assert len(engine.calls) == before
    scorer.score([direct], policy_version="v2")
    assert len(engine.calls) > before


def test_search_discovered_products_are_cached_for_direct_entries(native):
    _, selected, _, source, _ = native
    results = capture(source, results_start(native), "item_page")
    direct = source.product_entry(results, selected.product["asin"])
    scorer, engine = scorer_for(native)
    scorer.score([results], policy_version=0)
    assert all(not p.allowed_token_ids for p in engine.calls[0][1])
    assert any(p.allowed_token_ids for _, params in engine.calls[1:] for p in params)
    before = len(engine.calls)
    scorer.score([direct], policy_version=0)
    assert len(engine.calls) == before


def test_randomized_search_is_not_cached_but_product_scores_are(native, monkeypatch):
    _, _, _, source, _ = native
    results = replace(capture(source, results_start(native), "item_page"), randomized_search=True)
    scorer, engine = scorer_for(native)
    first = scorer.score([results], policy_version=0)
    before = len(engine.calls)
    monkeypatch.setattr(source, "product_entry", lambda *_: pytest.fail("Cached products should not be rendered again"))
    assert scorer.score([results], policy_version=0) == first
    assert len(engine.calls) > before
    assert all(not p.allowed_token_ids for _, params in engine.calls[before:] for p in params)
    assert not scorer._results_cache


def test_cache_checks_full_goal_and_does_not_reuse_intermediate_products(native):
    _, _, start, source, _ = native
    snapshot = capture(source, start)
    scorer, engine = scorer_for(native)
    scorer.score([snapshot], policy_version=0)
    before = len(engine.calls)
    scorer.score([replace(snapshot, previous_page_type="item_sub_page")], policy_version=0)
    assert len(engine.calls) > before
    before = len(engine.calls)
    changed_goal = json.loads(snapshot.goal_json)
    changed_goal["price_upper"] *= 2
    # Same instruction and product, but a different reward goal must miss.
    changed = replace(snapshot, goal_json=json.dumps(changed_goal, sort_keys=True))
    scorer.score([changed], policy_version=0)
    assert len(engine.calls) > before


def test_strict_pruning_no_fallback_even_with_cached_product(native):
    _, selected, start, source, _ = native
    results = capture(source, results_start(native), "item_page")
    scorer, engine = scorer_for(native, threshold=0.5)
    scorer.score([capture(source, start)], policy_version=0)
    before = len(engine.calls)
    (score,) = scorer.score([results], policy_version=0)
    assert score.probability == 0
    assert all(not p.allowed_token_ids for _, params in engine.calls[before:] for p in params)
    assert selected.product["asin"] in scorer._product_cache[results.query_key]


def test_threshold_equality_and_current_page_only(native):
    _, _, _, source, _ = native
    results = capture(source, results_start(native), "item_page")
    # Probability one avoids roundoff when checking equality to the threshold.
    scorer, engine = scorer_for(native, threshold=1, probability=1)
    (score,) = scorer.score([results], policy_version=0)
    assert score.probability > 0
    tokenizer = native[-1]
    actions = [tokenizer.decode(prompt["prompt_token_ids"]).rsplit("<answer>", 1)[1] for prompts, params in engine.calls for prompt, p in zip(prompts, params) if not p.allowed_token_ids]
    assert actions and all("next >" not in action and "< prev" not in action for action in actions)
    assert all(any(asin.lower() in action for asin in results.visible_asins) for action in actions)


def test_selected_correct_options_skip_groups_wrong_ones_are_rescored(native):
    server, selected, start, source, _ = native
    episode = start.clone()
    scorer, engine = scorer_for(native)
    initial = capture(source, episode)
    scorer.score([initial], policy_version=0)
    for group, value in selected.goal["goal_options"].items():
        episode.advance(f"click[{value}]")
    before = len(engine.calls)
    (score,) = scorer.score([capture(source, episode, "item_page")], policy_version=0)
    assert score.probability == 1 and len(engine.calls) == before
    from web_agent_site.engine.goal import get_option_reward

    group, correct = next(iter(selected.goal["goal_options"].items()))
    wrong = next(value for value in server.product_item_dict[selected.product["asin"]]["options"][group] if get_option_reward((value,), tuple(selected.goal["goal_options"].items()))[1] == 0)
    episode.advance(f"click[{wrong}]")
    scorer.score([capture(source, episode, "item_page")], policy_version=0)
    assert len(engine.calls) > before
    new_prompts = [native[-1].decode(prompt["prompt_token_ids"]) for prompts, _ in engine.calls[before:] for prompt in prompts]
    assert new_prompts and all(f"available options for {group}" in prompt for prompt in new_prompts)
    assert all("never clicking any option in this group" in prompt for prompt in new_prompts)


def test_skipped_states_and_query_mismatch(native):
    _, _, start, source, _ = native
    snapshot = capture(source, start)
    scorer, engine = scorer_for(native)
    states = [replace(snapshot, page_type=page) for page in ["", "item_sub_page", "done"]]
    states.append(replace(snapshot, terminated=True))
    scores = scorer.score(states, policy_version=0)
    assert [(score.probability, score.skipped_reason) for score in scores[:2]] == [(0.0, "deferred_initial_search_page"), (0.0, "deferred_item_sub_page")]
    assert all(score.probability is None and score.skipped_reason for score in scores[2:])
    assert not engine.calls
    with pytest.raises(ValueError, match="same shopping query"):
        scorer.score([replace(snapshot, shopping_task="a different instruction")], policy_version=0)
    with pytest.raises(ValueError, match="different catalog"):
        scorer.score([replace(snapshot, catalog_key="other")], policy_version=0)


def test_mixed_batch_limit_alignment_and_default_prefix(native):
    _, _, start, source, tokenizer = native
    snapshot = capture(source, start)
    parts = extract_product_page_contexts([snapshot.prompt])
    groups = prepare_product_page_grouped_choice_rollouts(parts, tokenizer, [{}], annotate_correctness=False)
    engine = ControlledEngine(tokenizer)
    mixed = MixedPseudoRolloutScorer(engine, tokenizer)
    expected = build_artificial_group_response_prefixes(groups, tokenizer)
    assert [mixed._product_prefix(group.option_group.name) for group in groups] == [prefix.token_ids for prefix in expected]
    # Distinct prefixes force 70 actual inference rows; repeat requests are deduplicated.
    probes = [replace(groups[0], prompt_token_ids=groups[0].prompt_token_ids + tuple(tokenizer.encode(f" {index}"))) for index in range(70)]
    scores = mixed.score([*probes, probes[0]])
    assert [len(prompts) for prompts, _ in engine.calls] == [32, 32, 6]
    assert len(scores) == 71 and scores[0] == scores[-1]
    assert all(math.isclose(sum(score.action_probabilities.values()), 1) for score in scores)


def test_mixed_results_decoder_matches_scalar_action_span(native):
    *_, tokenizer = native
    episode = results_start(native)
    action = next(action for action in extract_product_page_contexts([episode.prompt])[0].admissible_actions if action.startswith("click[b"))
    probe = prepare_results_page_answer_probe(episode.prompt, action, tokenizer)
    engine = ControlledEngine(tokenizer, results_probability=0.123)
    assert MixedPseudoRolloutScorer(engine, tokenizer).score([probe]) == pytest.approx([0.123])
    assert isinstance(probe, ResultsPageAnswerProbe)
    with pytest.raises(ValueError, match="exceeding"):
        MixedPseudoRolloutScorer(engine, tokenizer, max_model_len=10).score([probe])


def test_native_aggregation_matches_exhaustive_native_rewards(native):
    server, selected, _, _, _ = native
    from web_agent_site.engine.goal import get_reward

    item, goal = selected.product, selected.goal
    plan = build_native_option_success_plan(item, goal, server.product_prices[item["asin"]], {})
    distributions = {group.name: {action: 1 / len(group.actions) for action in group.actions} for group in plan.groups}
    expected = 0
    for choices in product(*(tuple(values) + (None,) for values in item["options"].values())):
        options = {name: value for name, value in zip(item["options"], choices) if value is not None}
        if get_reward(item, goal, price=server.product_prices[item["asin"]], options=options) == 1:
            expected += math.prod(1 / (len(values) + 1) for values in item["options"].values())
    assert plan.aggregate(distributions) == pytest.approx(expected)


def test_native_cross_group_alternatives_need_both_distributions(native):
    server, selected, _, _, _ = native
    from web_agent_site.engine.goal import get_reward

    item = copy.deepcopy(selected.product)
    goal = copy.deepcopy(selected.goal)
    goal["goal_options"] = {"color": "red"}
    item["options"] = {"first": ["color red"], "second": ["red color"]}
    price = server.product_prices[item["asin"]]
    assert get_reward(item, goal, price=price, options={"first": "color red"}) == 1
    plan = build_native_option_success_plan(item, goal, price, {})
    assert {group.name for group in plan.groups} == {"first", "second"}
    distributions = {group.name: {action: 0.5 for action in group.actions} for group in plan.groups}
    assert plan.aggregate(distributions) == pytest.approx(0.75)
    assert build_native_option_success_plan(item, goal, price, {"first": "color red"}).constant_probability == 1


def test_native_plan_uses_one_action_for_repeated_catalog_value(native):
    server, selected, _, _, _ = native
    item = copy.deepcopy(selected.product)
    goal = copy.deepcopy(selected.goal)
    item["options"] = {"color": ["red", "red", "blue"]}
    goal["goal_options"] = {"color": "red"}
    plan = build_native_option_success_plan(item, goal, server.product_prices[item["asin"]], {})
    assert plan.groups[0].actions == ("click[red]", "click[blue]", "none")


def test_native_trivial_and_impossible_products_do_not_require_groups(native):
    server, selected, _, _, _ = native
    goal = copy.deepcopy(selected.goal)
    goal["goal_options"] = {}
    price = server.product_prices[selected.product["asin"]]
    assert build_native_option_success_plan(selected.product, goal, price, {}).constant_probability == 1
    no_options = dict(selected.product, options={})
    assert build_native_option_success_plan(no_options, goal, price, {}).constant_probability == 1
    assert build_native_option_success_plan(no_options, selected.goal, price, {}).constant_probability == 0
    goal["price_upper"] = price / 2
    assert build_native_option_success_plan(selected.product, goal, price, {}).constant_probability == 0


def test_native_group_irrelevance_checks_joint_coverage(native):
    server, selected, _, _, _ = native
    item, goal = copy.deepcopy(selected.product), copy.deepcopy(selected.goal)
    goal["goal_options"] = {"shape": "oval", "size": "large"}
    # Avoid color normalization, which discards non-color words in a value.
    # Only first can satisfy size, and doing so also satisfies shape.
    item["options"] = {"first": ["shape oval size large"], "redundant": ["shape oval"]}
    plan = build_native_option_success_plan(item, goal, server.product_prices[item["asin"]], {})
    assert [group.name for group in plan.groups] == ["first"]
    assert plan.aggregate({"first": {"click[shape oval size large]": 0.3, "none": 0.7}}) == pytest.approx(0.3)


def test_partially_matching_selection_that_blocks_success_is_not_frozen(native):
    server, selected, _, _, _ = native
    item, goal = copy.deepcopy(selected.product), copy.deepcopy(selected.goal)
    goal["goal_options"] = {"shape": "oval", "size": "large"}
    item["options"] = {"configuration": ["shape oval", "shape oval size large"]}
    plan = build_native_option_success_plan(item, goal, server.product_prices[item["asin"]], {"configuration": "shape oval"})
    assert [group.name for group in plan.groups] == ["configuration"]
    assert plan.aggregate({"configuration": {"click[shape oval]": 0.2, "click[shape oval size large]": 0.6, "none": 0.2}}) == pytest.approx(0.6)


def test_results_cache_separates_search_terms_and_page_numbers(native):
    _, _, _, source, _ = native
    snapshot = capture(source, results_start(native), "item_page")
    scorer, engine = scorer_for(native)
    scorer.score([snapshot], policy_version=0)
    before = len(engine.calls)
    for changed in [replace(snapshot, search_terms=("other",)), replace(snapshot, results_page=snapshot.results_page + 1)]:
        scorer.score([changed], policy_version=0)
        assert len(engine.calls) > before
        assert all(not params.allowed_token_ids for _, row in engine.calls[before:] for params in row)
        before = len(engine.calls)


def test_source_detects_native_random_search_and_captures_termination(native):
    _, _, start, source, _ = native
    episode = start.clone()
    episode.advance("search[<r>]")
    snapshot = capture(source, episode, "item_page")
    assert snapshot.randomized_search and snapshot.search_terms == ("<r>",)
    assert snapshot.selected_options == () and snapshot.asin is None
    (terminated,) = snapshot_webshop_rollouts(source, [episode], previous_page_types=["item_page"], terminated=[True])
    assert terminated.terminated


def test_empty_batch_and_missing_model_scores(native):
    _, _, start, source, tokenizer = native
    scorer, engine = scorer_for(native)
    assert scorer.score([], policy_version=0) == [] and not engine.calls
    parts = extract_product_page_contexts([start.prompt])
    probes = prepare_product_page_grouped_choice_rollouts(parts, tokenizer, [{}], annotate_correctness=False)
    broken = SimpleNamespace(llm_engine=engine.llm_engine, generate=lambda **_: [])
    with pytest.raises(ValueError, match="misaligned"):
        MixedPseudoRolloutScorer(broken, tokenizer).score(probes)
    bad = copy.copy(engine)
    bad.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32768, max_logprobs=1))
    with pytest.raises(ValueError, match="max_logprobs"):
        MixedPseudoRolloutScorer(bad, tokenizer).score(probes)
