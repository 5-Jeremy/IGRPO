"""Native-environment continuation tests; model inference is replaced by scripts."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from treehca import pseudo_rollout_product_page_outcomes_testbed as testbed
from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_product_page import ActionChoiceScore, ProductOptionGroupPseudoRolloutScores, prepare_product_page_grouped_choice_rollouts
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import sample_diverse_product_goals


@pytest.fixture(scope="module")
def native_start():
    server = testbed.create_server(testbed._DEFAULT_CATALOG, testbed._DEFAULT_ATTRIBUTES, 0, 1000)
    selected = sample_diverse_product_goals(server.all_products, server.goals, 1, 0)[0]
    start = testbed.construct_start_episode(server, selected, seed=0)
    return start, selected


def response(action):
    return f"<think>Choose the next action.</think><action>{action}</action>"


def test_start_is_native_product_click_with_training_history_and_empty_options(native_start):
    start, selected = native_start
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    assert isinstance(start.env, WebAgentTextEnv)
    parts = extract_product_page_contexts([start.prompt])[0]
    assert parts.completed_steps == 2
    assert parts.current_step == 3
    assert parts.history_length == 2
    assert start.setup["search_action"] in parts.history_block
    assert start.setup["selection_action"] in parts.history_block
    assert selected.product["asin"] in start.setup["results_asins"]
    assert len(start.setup["results_asins"]) == 10
    session = start.env.server.user_sessions[start.env.session]
    assert session["options"] == {}
    assert session["asins"] == {selected.product["asin"]}
    assert session["actions"]["search"] == session["actions"]["asin"] == 1


def test_native_oracle_and_clone_isolation_survive_purchase_autoreset(native_start):
    start, selected = native_start
    first, second = start.clone(), start.clone()
    original_prompt = start.prompt
    assert first.env.server is not second.env.server
    assert first.env.server.product_item_dict is second.env.server.product_item_dict
    assert first.env.browser.server is first.env.server
    for value in selected.goal["goal_options"].values():
        first.advance(f"click[{value}]")
    receipt = first.advance("click[buy now]")
    assert receipt["purchased"] and receipt["full_reward"]
    assert receipt["raw_reward"] == 1.0
    assert receipt["training_reward"] == 10.0
    assert receipt["selected_options"] == selected.goal["goal_options"]
    assert receipt["purchased_asin"] == selected.product["asin"]
    assert first.env.server.get_page_name(first.env.browser.current_url) == ""
    assert second.env.server.user_sessions[second.env.session]["options"] == {}
    assert len(start.manager.memory[0]) == len(second.manager.memory[0]) == 2
    assert start.prompt == second.prompt == original_prompt
    assert testbed.validate_full_reward(start, selected.goal["goal_options"])["full_reward"]


def test_empirical_outcomes_include_exits_partial_purchases_invalid_actions_and_step_cap(native_start):
    start, selected = native_start
    correct_actions = [f"click[{value}]" for value in selected.goal["goal_options"].values()] + ["click[buy now]"]
    scripts = [correct_actions, ["click[buy now]"], ["click[back to search]"], ["click[< prev]"], ["search[new query]"], ["nonsense"] * 13]
    seen_prompts, seen_seeds = [], []

    def sample(prompts, seeds):
        seen_prompts.extend(prompts)
        seen_seeds.extend(seeds)
        return [response(scripts[seed // 13][seed % 13]) for seed in seeds]

    trajectories = testbed.collect_rollouts(start, sample, samples_per_page=6, max_steps=13, store_prompts=True)
    assert [record["termination_reason"] for record in trajectories] == ["purchase", "purchase", "back_to_search", "back_to_results", "search_action", "step_limit"]
    assert trajectories[0]["full_reward"]
    assert trajectories[1]["purchased"] and not trajectories[1]["full_reward"]
    assert len(trajectories[-1]["steps"]) == 13
    assert trajectories[-1]["steps"][-1]["training_step"] == 15
    assert trajectories[2]["steps"][0]["executed"] is False
    assert all(prompt == start.prompt for prompt in seen_prompts[:6])
    assert len(seen_seeds) == len(set(seen_seeds))
    assert "Observation 3" in trajectories[0]["steps"][1]["prompt"]
    stats = testbed.summarize_outcomes(trajectories, 0.75)
    assert stats["no_purchase_rate"] == pytest.approx(4 / 6)
    assert stats["full_reward_purchase_rate"] == pytest.approx(1 / 6)
    assert stats["full_reward_given_purchase_rate"] == 0.5
    assert stats["pseudo_minus_full_reward_given_purchase_rate"] == 0.25
    assert stats["partial_reward_purchase_count"] == 1
    assert stats["first_turn_back_to_results_count"] == 1
    assert stats["first_turn_other_search_exit_count"] == 2
    assert stats["first_turn_purchase_count"] == 1
    assert stats["first_turn_back_to_results_rate"] == pytest.approx(1 / 6)
    assert stats["first_turn_other_search_exit_rate"] == pytest.approx(2 / 6)
    assert stats["first_turn_purchase_rate"] == pytest.approx(1 / 6)
    markdown = testbed.format_markdown_summary({"mode": "monte_carlo", "configuration": {"model": "example"}, "pages": [{"asin": selected.product["asin"], "outcomes": stats}]})
    assert "Return to results on first turn" in markdown
    assert "Buy immediately" in markdown
    assert f"| {selected.product['asin']} | 1 (16.667%) | 2 (33.333%) | 1 (16.667%) |" in markdown
    assert sum(row["count"] for row in stats["final_option_sets"]) == 6
    assert len(start.manager.memory[0]) == 2


def test_prev_from_description_returns_to_product_and_does_not_terminate(native_start):
    start, selected = native_start
    script = ["click[description]", "click[< prev]", *[f"click[{value}]" for value in selected.goal["goal_options"].values()], "click[buy now]"]
    trajectories = testbed.collect_rollouts(start, lambda prompts, seeds: [response(script[seeds[0]])], samples_per_page=1)
    assert trajectories[0]["termination_reason"] == "purchase"
    assert trajectories[0]["full_reward"]
    assert trajectories[0]["steps"][1]["executed"]


def test_malformed_response_can_still_purchase_as_in_training(native_start):
    start, _ = native_start
    trajectories = testbed.collect_rollouts(start, lambda prompts, seeds: ["click[buy now]"], samples_per_page=1)
    assert trajectories[0]["purchased"]
    assert trajectories[0]["steps"][0]["format_valid"] is False
    assert trajectories[0]["steps"][0]["projected_action"] == "click[buy now]"


def test_buy_on_thirteenth_action_counts_and_fourteenth_is_never_sampled(native_start):
    start, _ = native_start
    calls = []

    def sample(prompts, seeds):
        calls.extend(seeds)
        return [response("click[buy now]" if seeds[0] == 12 else "nonsense")]

    trajectories = testbed.collect_rollouts(start, sample, samples_per_page=1)
    assert calls == list(range(13))
    assert trajectories[0]["purchased"]
    assert trajectories[0]["termination_reason"] == "purchase"


def test_empty_purchase_denominator_is_null_and_exit_options_are_retained(native_start):
    start, selected = native_start
    value = next(iter(selected.goal["goal_options"].values()))
    scripts = [f"click[{value}]", "click[back to search]"]
    trajectories = testbed.collect_rollouts(start, lambda prompts, seeds: [response(scripts[seeds[0]])], samples_per_page=1)
    assert value in trajectories[0]["selected_options"].values()
    stats = testbed.summarize_outcomes(trajectories, 0.8)
    assert stats["full_reward_given_purchase_rate"] is None
    assert stats["full_reward_given_purchase_wilson_95"] == [None, None]
    assert stats["pseudo_minus_full_reward_given_purchase_rate"] is None
    assert stats["no_purchase_rate"] == 1.0
    json.dumps(stats, allow_nan=False)


def test_sampler_alignment_error_is_not_counted_as_an_agent_failure(native_start):
    start, _ = native_start
    with pytest.raises(ValueError, match="wrong number"):
        testbed.collect_rollouts(start, lambda prompts, seeds: [], samples_per_page=1)


def test_vllm_policy_uses_training_tokenization_and_preserves_seeds_across_batches(monkeypatch):
    import torch

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return messages[0]["content"]

        def __call__(self, prompt, **kwargs):
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": torch.tensor([[int(prompt), 99]]), "attention_mask": torch.tensor([[1, 1]])}

        def batch_decode(self, ids, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return [str(row[0]) for row in ids]

    class Engine:
        calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[row.seed])]) for row in kwargs["sampling_params"]]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)))
    args = testbed.build_argument_parser().parse_args(["--rollout-batch-size", "2", "--max-new-tokens", "4", "--max-prompt-length", "8"])
    engine = Engine()
    policy = testbed.VllmPolicy(engine, Tokenizer(), args, 10)
    assert policy(["1", "2", "3"], [13, 55, 72]) == ["13", "55", "72"]
    assert len(engine.calls) == 2
    assert engine.calls[0]["prompts"] == [{"prompt_token_ids": [1, 99]}, {"prompt_token_ids": [2, 99]}]
    assert engine.calls[0]["sampling_params"][0].temperature == 1
    assert engine.calls[0]["sampling_params"][0].top_p == 1
    assert engine.calls[0]["sampling_params"][0].top_k == -1
    assert not hasattr(engine.calls[0]["sampling_params"][0], "allowed_token_ids")
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        testbed.VllmPolicy(engine, Tokenizer(), args, 5)(["1"], [1])


def test_argument_limits_and_defaults():
    parser = testbed.build_argument_parser()
    args = parser.parse_args([])
    testbed._validate_arguments(args)
    assert args.max_steps == 13 and args.history_length == 2
    assert args.max_prompt_length == 4096 and args.max_new_tokens == 512
    for arguments in (["--max-steps", "14"], ["--history-length", "0"], ["--results-size", "1"], ["--temperature", "0"], ["--top-p", "nan"]):
        with pytest.raises(ValueError):
            testbed._validate_arguments(parser.parse_args(arguments))


def test_report_serialization_and_markdown(tmp_path):
    stats = testbed.summarize_outcomes([{"purchased": False, "full_reward": False, "raw_reward": 0.0, "selected_options": {}, "termination_reason": "step_limit"}], 0.7)
    report = {"mode": "monte_carlo", "configuration": {"model": "example"}, "pages": [{"asin": "EXAMPLE", "outcomes": stats}]}
    output = tmp_path / "report.json"
    testbed.write_report(report, output)
    assert json.loads(output.read_text()) == report
    markdown = output.with_suffix(".md").read_text()
    assert "Full reward / purchases" in markdown and "70.000%" in markdown and "n/a" in markdown


@pytest.mark.parametrize("count_partial", [False, True])
def test_complete_testbed_with_native_environments_cached_tokenizer_and_fake_logits(native_start, monkeypatch, tmp_path, count_partial):
    import math

    from transformers import AutoTokenizer

    start, selected = native_start
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    actions = [f"click[{value}]" for value in selected.goal["goal_options"].values()] + ["click[buy now]"]

    class Engine:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192, max_logprobs=kwargs["max_logprobs"]))

        def generate(self, prompts, sampling_params, **kwargs):
            outputs = []
            for params in sampling_params:
                if hasattr(params, "allowed_token_ids"):
                    logprobs = {token: -math.log(len(params.allowed_token_ids)) for token in params.allowed_token_ids}
                    completion = SimpleNamespace(token_ids=[params.allowed_token_ids[0]], logprobs=[logprobs])
                else:
                    step = (params.seed - 10_000) % 13
                    completion = SimpleNamespace(token_ids=tokenizer.encode(response(actions[step]), add_special_tokens=False))
                outputs.append(SimpleNamespace(outputs=[completion]))
            return outputs

    monkeypatch.setattr(testbed, "create_server", lambda *args: start.env.server)
    monkeypatch.setattr(testbed, "_configure_vllm_engine", lambda: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs), LLM=Engine))
    flags = ["--count_partial_reward"] if count_partial else []
    args = testbed.build_argument_parser().parse_args(["--num-pages", "1", "--samples-per-page", "3", "--output", str(tmp_path / "report.json"), *flags])
    report = testbed.run_testbed(args)
    assert report["status"] == "complete" and report["pages_completed"] == 1
    page = report["pages"][0]
    assert page["outcomes"]["full_reward_purchase_rate"] == 1
    assert page["outcomes"]["no_purchase_count"] == 0
    assert page["outcomes"]["pseudo_all_correct_probability"] == pytest.approx(math.prod(group["correct_probability"] for group in page["groups"]))
    assert all("already taken 2 step(s)" in group["prompt"] for group in page["groups"])
    assert len(page["groups"]) == len(selected.product["options"])
    assert report["schema_version"] == 2
    assert page["outcomes"]["empirical_expected_reward_all"] == 1
    expected = page["pseudo_rewards"]["expected_reward" if count_partial else "full_reward_probability"]
    assert page["outcomes"]["pseudo_expected_reward"] == expected
    expected_probability = page["pseudo_rewards"]["positive_reward_probability" if count_partial else "full_reward_probability"]
    assert page["outcomes"]["pseudo_reward_probability"] == expected_probability
    assert report["summary"]["page_mean_pseudo_expected_reward"] == expected
    assert json.loads(args.output.read_text())["summary"]["full_reward_purchase_count"] == 3


def test_singleton_groups_are_included_in_start_scoring(native_start, monkeypatch):
    from transformers import AutoTokenizer

    start, original = native_start
    # The current small catalog has no mixed singleton/multi-value products.
    # Render a controlled catalog variant through the native environment.
    product, goal = copy.deepcopy(original.product), copy.deepcopy(original.goal)
    name = next(iter(product["options"]))
    product["options"][name] = [goal["goal_options"][name]]
    server = copy.copy(start.env.server)
    server.product_item_dict = dict(server.product_item_dict, **{product["asin"]: product})
    selected = testbed.SelectedProductGoal(product, goal)
    episode = testbed.construct_start_episode(server, selected, seed=0)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    pseudos = prepare_product_page_grouped_choice_rollouts(extract_product_page_contexts([episode.prompt]), tokenizer, [goal["goal_options"]])
    assert any(len(pseudo.option_group.values) == 1 and len(pseudo.actions) == 2 for pseudo in pseudos)

    def score(_engine, rows, **kwargs):
        return [
            ProductOptionGroupPseudoRolloutScores(
                choices=tuple(ActionChoiceScore(label, action, 1 / len(row.actions), 0, 0, 0) for label, action in zip(row.labels, row.actions)),
                source_page_index=row.source_page_index,
                option_group=row.option_group,
            )
            for row in rows
        ]

    monkeypatch.setattr(testbed, "score_product_page_grouped_choice_rollouts", score)
    groups = testbed.score_start_pages(object(), tokenizer, pseudos, model_limit=8192)
    assert len(groups) == len(product["options"])
    singleton = next(group for group in groups if len(group["options"]) == 2)
    assert singleton["options"][-1]["is_none"]
    assert singleton["correct_probability"] == 0.5


def test_partial_reward_rates_and_expected_rewards_use_distinct_denominators():
    trajectories = [{"purchased": bought, "full_reward": reward == 1, "raw_reward": reward, "selected_options": {}, "termination_reason": "purchase" if bought else "step_limit"} for bought, reward in [(True, 1.0), (True, 0.5), (True, 0.0), (False, 0.0)]]
    pseudo = {"positive_reward_probability": 0.8, "expected_reward": 0.6, "full_reward_probability": 0.4}
    full = testbed.summarize_outcomes(trajectories, 0.3, pseudo_rewards=pseudo)
    partial = testbed.summarize_outcomes(trajectories, 0.3, count_partial_reward=True, pseudo_rewards=pseudo)
    assert full["reward_purchase_rate"] == 0.25
    assert full["pseudo_reward_probability"] == 0.4
    assert full["empirical_expected_reward_all"] == 0.25
    assert full["empirical_expected_reward_given_purchase"] == pytest.approx(1 / 3)
    assert full["pseudo_expected_reward"] == 0.4
    assert full["pseudo_minus_expected_reward_all"] == pytest.approx(0.15)
    assert full["reward_scale"] == "training_binary_0_1"
    assert partial["reward_purchase_rate"] == 0.5
    assert partial["reward_given_purchase_rate"] == pytest.approx(2 / 3)
    assert partial["pseudo_reward_probability"] == 0.8
    assert partial["pseudo_minus_reward_purchase_rate"] == pytest.approx(0.3)
    assert partial["positive_partial_reward_purchase_count"] == 1
    assert partial["zero_reward_purchase_count"] == 1
    assert partial["no_purchase_rate"] == 0.25
    for stats in (partial,):
        assert stats["empirical_expected_reward_all"] == 0.375
        assert stats["empirical_expected_reward_given_purchase"] == 0.5
        assert stats["pseudo_expected_reward"] == 0.6
        assert stats["pseudo_minus_expected_reward_all"] == pytest.approx(0.225)
        assert stats["pseudo_minus_expected_reward_given_purchase"] == pytest.approx(0.1)
    empty = testbed.summarize_outcomes(trajectories[-1:], 0.3, count_partial_reward=True, pseudo_rewards=pseudo)
    assert empty["empirical_expected_reward_all"] == 0
    assert empty["reward_given_purchase_rate"] is None
    assert empty["empirical_expected_reward_given_purchase"] is None
    assert empty["pseudo_minus_expected_reward_given_purchase"] is None
    assert empty["reward_given_purchase_wilson_95"] == [None, None]


def test_pseudo_reward_integration_weights_combinations_by_probability(monkeypatch):
    from itertools import product

    rewards = {("red", "small"): 1.0, ("red", "large"): 0.5, ("blue", "small"): 0.25, ("blue", "large"): 0.0}
    groups = [
        {"group_name": "color", "options": [{"action": "click[red]", "probability": 0.6}, {"action": "click[blue]", "probability": 0.4}]},
        {"group_name": "size", "options": [{"action": "click[small]", "probability": 0.7}, {"action": "click[large]", "probability": 0.3}]},
    ]
    calls = []
    item = {"options": {"color": ["red", "blue"], "size": ["small", "large"]}}
    goal = {"goal_options": {"color": "red", "size": "small"}}
    server = SimpleNamespace(user_sessions={"session": {"asin": "ITEM", "goal": goal}}, product_item_dict={"ITEM": item}, product_prices={"ITEM": 10})
    start = SimpleNamespace(env=SimpleNamespace(server=server, session="session"))

    def reward(product_item, exact_goal, *, price, options):
        assert product_item is item and exact_goal is goal and price == 10
        calls.append(dict(options))
        return rewards[(options["color"], options["size"])]

    engine = SimpleNamespace(parse_action=lambda action: ("click", action[6:-1]))
    monkeypatch.setattr(testbed, "_webshop_modules", lambda: (engine, SimpleNamespace(get_reward=reward)))
    result = testbed.compute_pseudo_rewards(start, groups)
    assert result["combination_count"] == 4
    assert len(calls) == 4
    assert {(row["color"], row["size"]) for row in calls} == set(product(["red", "blue"], ["small", "large"]))
    assert result["full_reward_probability"] == pytest.approx(0.42)
    assert result["positive_reward_probability"] == pytest.approx(0.88)
    assert result["expected_reward"] == pytest.approx(0.42 + 0.18 * 0.5 + 0.28 * 0.25)
    assert sum(row["probability"] for row in result["reward_distribution"]) == pytest.approx(1)
    assert {row["raw_reward"] for row in result["reward_distribution"]} == {0, 0.25, 0.5, 1}
    with pytest.raises(ValueError, match="every product option group"):
        testbed.compute_pseudo_rewards(start, groups[:1])
    invalid = copy.deepcopy(groups)
    invalid[0]["options"][0]["probability"] = 0.9
    with pytest.raises(ValueError, match="sum to one"):
        testbed.compute_pseudo_rewards(start, invalid)


def test_pseudo_expected_reward_matches_native_option_click_receipts(native_start):
    from itertools import product

    start, selected = native_start
    groups, choices = [], []
    for name, values in selected.product["options"].items():
        correct = selected.goal["goal_options"][name]
        other = next(value for value in values if value != correct)
        choices.append([(name, correct, 0.75), (name, other, 0.25)])
        groups.append({"group_name": name, "options": [{"action": f"click[{value}]", "probability": 0.75 if value == correct else 0.25 if value == other else 0} for value in values]})
    expected_reward = positive_mass = full_mass = 0
    for combination in product(*choices):
        episode = start.clone()
        mass = 1
        for _, value, probability in combination:
            episode.advance(f"click[{value}]")
            mass *= probability
        receipt = episode.advance("click[buy now]")
        expected_reward += mass * receipt["raw_reward"]
        positive_mass += mass * (receipt["raw_reward"] > 0)
        full_mass += mass * receipt["full_reward"]
    result = testbed.compute_pseudo_rewards(start, groups)
    assert result["expected_reward"] == pytest.approx(expected_reward)
    assert result["positive_reward_probability"] == pytest.approx(positive_mass)
    assert result["full_reward_probability"] == pytest.approx(full_mass)
    assert start.env.server.user_sessions[start.env.session]["options"] == {}


def test_partial_reward_cli_aliases_and_markdown():
    parser = testbed.build_argument_parser()
    assert not parser.parse_args([]).count_partial_reward
    for flag in ("--count_partial_reward", "--count-partial-reward"):
        assert parser.parse_args([flag]).count_partial_reward
    trajectories = [{"purchased": True, "full_reward": False, "raw_reward": 0.5, "selected_options": {}, "termination_reason": "purchase"}]
    stats = testbed.summarize_outcomes(trajectories, 0.2, count_partial_reward=True, pseudo_rewards={"positive_reward_probability": 0.9, "expected_reward": 0.6, "full_reward_probability": 0.2})
    markdown = testbed.format_markdown_summary({"mode": "monte_carlo", "configuration": {"model": "example", "count_partial_reward": True}, "pages": [{"asin": "EXAMPLE", "outcomes": stats}]})
    assert "Full or partial reward / all" in markdown
    assert "Full or partial reward / purchases" in markdown
    assert "Pseudo positive reward" in markdown
    assert "Full reward / all" not in markdown
    assert "90.000%" in markdown and "100.000%" in markdown
    assert "0.500000" in markdown and "0.600000" in markdown and "0.100000" in markdown


def test_pseudo_none_omits_group_keys_and_retains_native_partial_rewards(native_start):
    start, selected = native_start
    groups = []
    for name, values in selected.product["options"].items():
        groups.append({"group_name": name, "options": [*[{"action": f"click[{value}]", "probability": 0.0} for value in values], {"action": "none", "probability": 1.0}]})
    immediate_purchase = start.clone().advance("click[buy now]")
    assert immediate_purchase["selected_options"] == {}
    result = testbed.compute_pseudo_rewards(start, groups)
    assert result["expected_reward"] == immediate_purchase["raw_reward"]
    assert result["full_reward_probability"] == 0
    assert result["positive_reward_probability"] == 1
    import math

    assert result["combination_count"] == math.prod(len(values) + 1 for values in selected.product["options"].values())
    full = testbed.summarize_outcomes([immediate_purchase | {"termination_reason": "purchase"}], 0, pseudo_rewards=result)
    partial = testbed.summarize_outcomes([immediate_purchase | {"termination_reason": "purchase"}], 0, pseudo_rewards=result, count_partial_reward=True)
    assert full["pseudo_expected_reward"] == full["empirical_expected_reward_all"] == 0
    assert partial["pseudo_expected_reward"] == partial["empirical_expected_reward_all"] == immediate_purchase["raw_reward"]


def test_none_in_unrequested_native_group_can_receive_full_reward(native_start):
    from treehca.product_page_parser import ProductOptionGroup
    from treehca.pseudo_rollout_product_page import _none_can_satisfy_goal

    start, selected = native_start
    episode = start.clone()
    goal = episode.env.server.user_sessions[episode.env.session]["goal"]
    omitted = next(iter(goal["goal_options"]))
    del goal["goal_options"][omitted]
    groups = []
    for name, values in selected.product["options"].items():
        options = [{"action": f"click[{value}]", "probability": float(name != omitted and value == goal["goal_options"][name])} for value in values]
        groups.append({"group_name": name, "options": [*options, {"action": "none", "probability": float(name == omitted)}]})
    for value in goal["goal_options"].values():
        episode.advance(f"click[{value}]")
    # Use an independent initial clone with the same modified goal for integration.
    initial = start.clone()
    initial.env.server.user_sessions[initial.env.session]["goal"] = copy.deepcopy(goal)
    result = testbed.compute_pseudo_rewards(initial, groups)
    receipt = episode.advance("click[buy now]")
    assert receipt["full_reward"] and omitted not in receipt["selected_options"]
    assert result["full_reward_probability"] == result["expected_reward"] == 1
    parsed_groups = [ProductOptionGroup(name, tuple(values)) for name, values in selected.product["options"].items()]
    assert _none_can_satisfy_goal(parsed_groups, omitted, goal["goal_options"])
