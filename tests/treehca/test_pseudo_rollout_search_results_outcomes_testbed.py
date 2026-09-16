import json
import sys
from types import SimpleNamespace

import pytest

from treehca import pseudo_rollout_search_results_outcomes_testbed as testbed
from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import sample_diverse_product_goals


def response(action):
    return f"<think>Choose.</think><action>{action}</action>"


@pytest.fixture(scope="module")
def native_results_start():
    server = testbed.create_server(testbed._DEFAULT_CATALOG, testbed._DEFAULT_ATTRIBUTES, 0, 1000)
    selected = sample_diverse_product_goals(server.all_products, server.goals, 1, 0)[0]
    start = testbed.construct_results_start_episode(server, selected, seed=0)
    return start, selected


def test_native_start_is_results_page_with_search_history_and_reachable_oracle(native_results_start):
    start, selected = native_results_start
    parts = extract_product_page_contexts([start.prompt])[0]
    session = start.env.server.user_sessions[start.env.session]

    assert start.env.server.get_page_name(start.env.browser.current_url) == "search_results"
    assert parts.current_step == 2 and parts.completed_steps == 1
    assert start.setup["search_action"] in parts.history_block
    assert selected.product["asin"] in start.setup["matching_asins"]
    assert session["asin"] is None and session["options"] == {}
    oracle = testbed.validate_reachable_full_reward(start, selected)
    assert oracle["full_reward"] and oracle["raw_reward"] == 1.0
    assert start.env.server.get_page_name(start.env.browser.current_url) == "search_results"


def test_sampler_excludes_goal_that_native_reward_cannot_fully_score(monkeypatch):
    server = testbed.create_server(testbed._DEFAULT_CATALOG, testbed._DEFAULT_ATTRIBUTES, 0, 1000)
    bad_goal = next(goal for goal in server.goals if goal["asin"] == "B07S7HDC88" and goal["goal_options"] == {"color": "r.brown2070", "size": "13"})
    sample = testbed.sample_diverse_product_goals
    captured = {}

    def capture(products, goals, count, seed):
        captured["goals"] = goals
        return sample(products, goals, count, seed)

    monkeypatch.setattr(testbed, "sample_diverse_product_goals", capture)
    selected = testbed.sample_full_reward_product_goals(server, 20, 0)

    assert bad_goal not in captured["goals"]
    assert all(item.goal["goal_options"] for item in selected)
    assert all(testbed.validate_reachable_full_reward(testbed.construct_results_start_episode(server, item, seed=index), item)["full_reward"] for index, item in enumerate(selected))


def test_rollout_records_first_full_reward_product_entry(native_results_start):
    start, selected = native_results_start
    forward_clicks = testbed.validate_reachable_full_reward(start, selected)["next_page_clicks"]
    assert forward_clicks < 14
    actions = iter([*["click[next >]"] * forward_clicks, start.setup["target_action"]])

    (trajectory,) = testbed.collect_rollouts(start, lambda prompts, seeds: [response(next(actions))], samples_per_page=1, max_steps=forward_clicks + 1)

    assert trajectory["entered_full_reward_product_page"]
    assert trajectory["first_full_reward_product_asin"] == selected.product["asin"]
    assert trajectory["first_full_reward_product_step"] == forward_clicks + 1
    assert trajectory["steps"][-1]["entered_full_reward_product_page"]
    assert trajectory["entered_later_page_full_reward_product"] == (forward_clicks > 0)
    assert not trajectory["purchased"]


def test_product_entry_excludes_products_that_cannot_earn_full_reward(monkeypatch):
    server = SimpleNamespace(get_page_name=lambda url: url, user_sessions={"session": {"asin": "BAD"}})
    env = SimpleNamespace(server=server, browser=SimpleNamespace(current_url="item_page"), session="session")
    monkeypatch.setattr(testbed, "_product_can_earn_full_reward", lambda state, asin: False)

    assert testbed._eligible_product_entry(env, "search_results", {}) is None
    assert testbed._eligible_product_entry(env, "item_sub_page", {}) is None


def test_collect_rollouts_defaults_to_fourteen_actions(monkeypatch):
    class Episode:
        def __init__(self):
            self.prompt = "prompt"
            self.env = object()

        def clone(self):
            return Episode()

        def advance(self, action):
            purchased = action == "click[buy now]"
            return {
                "raw_reward": float(purchased),
                "done": purchased,
                "purchased": purchased,
                "full_reward": purchased,
                "selected_options": {},
                "purchased_asin": "ITEM" if purchased else None,
                "reward_info": None,
                "observation": "observation",
            }

    seen_seeds = []

    def sample(prompts, seeds):
        seen_seeds.extend(seeds)
        return [response("click[buy now]" if seed == 13 else "nonsense") for seed in seeds]

    monkeypatch.setattr(testbed, "forbidden_exit_reason", lambda env, action: None)
    (trajectory,) = testbed.collect_rollouts(Episode(), sample, samples_per_page=1)

    assert seen_seeds == list(range(14))
    assert trajectory["termination_reason"] == "purchase"
    assert trajectory["full_reward"]
    assert [step["training_step"] for step in trajectory["steps"]] == list(range(2, 16))


def test_forbidden_paths_match_pseudo_estimator_scope(monkeypatch):
    engine = SimpleNamespace(
        BACK_TO_SEARCH="Back to Search",
        PREV_PAGE="< Prev",
        parse_action=lambda action: (action.split("[", 1)[0], action.split("[", 1)[1][:-1]) if "[" in action else (action, None),
    )
    env = SimpleNamespace(
        get_available_actions=lambda: {"clickables": ["back to search", "< prev"]},
        server=SimpleNamespace(get_page_name=lambda url: url),
        browser=SimpleNamespace(current_url="search_results"),
    )
    monkeypatch.setattr(testbed, "_webshop_modules", lambda: (engine, object()))

    assert testbed.forbidden_exit_reason(env, "search[different]") == "new_search"
    assert testbed.forbidden_exit_reason(env, "click[back to search]") == "back_to_search"
    assert testbed.forbidden_exit_reason(env, "click[< prev]") == "previous_results_page"
    env.browser.current_url = "item_page"
    assert testbed.forbidden_exit_reason(env, "click[< prev]") == "back_to_results"
    env.browser.current_url = "item_sub_page"
    assert testbed.forbidden_exit_reason(env, "click[< prev]") is None


def test_outcome_summary_compares_unconditional_full_reward_probability():
    trajectories = [
        {"purchased": True, "raw_reward": 1.0, "termination_reason": "purchase", "entered_full_reward_product_page": True, "entered_later_page_full_reward_product": True},
        {"purchased": True, "raw_reward": 0.5, "termination_reason": "purchase", "entered_full_reward_product_page": True},
        {"purchased": False, "raw_reward": 0.0, "termination_reason": "step_limit", "entered_full_reward_product_page": True},
        {"purchased": False, "raw_reward": 0.0, "termination_reason": "new_search", "entered_full_reward_product_page": False},
    ]

    result = testbed.summarize_outcomes(trajectories, 0.4, 0.8, 0.12)

    assert result["empirical_success_probability"] == 0.25
    assert result["pseudo_success_probability"] == 0.4
    assert result["pseudo_minus_empirical_success_probability"] == pytest.approx(0.15)
    assert result["purchase_count"] == 2
    assert result["full_reward_product_entry_count"] == 3
    assert result["empirical_full_reward_product_entry_probability"] == 0.75
    assert result["pseudo_minus_empirical_full_reward_product_entry_probability"] == pytest.approx(0.05)
    assert result["termination_counts"] == {"purchase": 2, "step_limit": 1, "new_search": 1}
    assert result["later_page_full_reward_product_entry_count"] == 1
    assert result["empirical_later_page_full_reward_product_entry_probability"] == 0.25
    assert result["pseudo_later_page_full_reward_product_entry_probability"] == 0.12


def test_argument_defaults_limits_and_report_serialization(tmp_path):
    parser = testbed.build_argument_parser()
    args = parser.parse_args([])
    testbed._validate_arguments(args)
    assert args.max_steps == 14
    assert args.path_probability_threshold == 1e-3
    assert args.data_parallel_size is None
    with pytest.raises(ValueError, match="at most 14"):
        testbed._validate_arguments(parser.parse_args(["--max-steps", "15"]))

    outcomes = testbed.summarize_outcomes([{"purchased": True, "raw_reward": 1.0, "termination_reason": "purchase", "entered_full_reward_product_page": True}], 0.7, 0.8)
    report = {
        "mode": "monte_carlo",
        "configuration": {"model": "example"},
        "pages": [{"asin": "ITEM", "setup": {"query": "query"}, "outcomes": outcomes}],
        "summary": {
            "page_mean_pseudo_success_probability": 0.7,
            "pooled_empirical_success_probability": 1.0,
            "page_mean_absolute_gap": 0.3,
            "page_root_mean_squared_gap": 0.3,
            "page_mean_pseudo_full_reward_product_entry_probability": 0.8,
            "pooled_empirical_full_reward_product_entry_probability": 1.0,
            "page_mean_absolute_full_reward_product_entry_gap": 0.2,
            "page_root_mean_squared_full_reward_product_entry_gap": 0.2,
        },
    }
    output = tmp_path / "report.json"
    testbed.write_report(report, output)
    assert json.loads(output.read_text()) == report
    markdown = output.with_suffix(".md").read_text()
    assert "Search-results continuation outcome testbed" in markdown
    assert "70.000%" in markdown and "100.000%" in markdown
    assert "Empirical product entry" in markdown


def test_gpu_groups_default_to_all_visible_tensor_parallel_groups():
    parser = testbed.build_argument_parser()

    args = parser.parse_args([])
    assert testbed._resolve_gpu_groups(args, ["0", "1", "2", "3"]) == (("0",), ("1",), ("2",), ("3",))

    args = parser.parse_args(["--tensor-parallel-size", "2"])
    assert testbed._resolve_gpu_groups(args, ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]) == (("GPU-a", "GPU-b"), ("GPU-c", "GPU-d"))

    args = parser.parse_args(["--data-parallel-size", "2"])
    assert testbed._resolve_gpu_groups(args, ["0", "1", "2", "3"]) == (("0",), ("1",))


def test_gpu_group_validation_rejects_impossible_parallelism():
    parser = testbed.build_argument_parser()

    with pytest.raises(ValueError, match="divisible"):
        testbed._resolve_gpu_groups(parser.parse_args(["--tensor-parallel-size", "2"]), ["0", "1", "2"])
    with pytest.raises(ValueError, match="only 2"):
        testbed._resolve_gpu_groups(parser.parse_args(["--data-parallel-size", "3"]), ["0", "1"])
    with pytest.raises(ValueError, match="requires visible CUDA"):
        testbed._resolve_gpu_groups(parser.parse_args(["--data-parallel-size", "2"]), [])


def test_start_sharding_is_balanced_round_robin_and_preserves_indices():
    selected = [SimpleNamespace(name=str(index)) for index in range(7)]

    shards = testbed._shard_selected_starts(selected, 3)

    assert [[index for index, _ in shard] for shard in shards] == [[0, 3, 6], [1, 4], [2, 5]]
    assert [item for shard in shards for _, item in shard] == [selected[0], selected[3], selected[6], selected[1], selected[4], selected[2], selected[5]]


def test_complete_testbed_orchestration_uses_same_start_and_default_horizon(native_results_start, monkeypatch, tmp_path):
    from transformers import AutoTokenizer

    start, selected = native_results_start
    tokenizer = SimpleNamespace()
    trajectory_rows = [
        {"purchased": True, "full_reward": True, "raw_reward": 1.0, "selected_options": {}, "termination_reason": "purchase", "steps": [], "entered_full_reward_product_page": True},
        {"purchased": False, "full_reward": False, "raw_reward": 0.0, "selected_options": {}, "termination_reason": "step_limit", "steps": [], "entered_full_reward_product_page": False},
    ]
    seen = {}

    class Engine:
        def __init__(self, **kwargs):
            seen["engine_kwargs"] = kwargs
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))

    def estimate(state, received_tokenizer, adapter, engine, threshold, **kwargs):
        seen["pseudo_prompt"] = state.prompt
        seen["threshold"] = threshold
        assert received_tokenizer is tokenizer
        return SimpleNamespace(success=0.4, product_entry=0.6, later_page_product_entry=0.02)

    def collect(received_start, policy, **kwargs):
        seen["rollout_prompt"] = received_start.prompt
        seen["max_steps"] = kwargs["max_steps"]
        return trajectory_rows

    monkeypatch.setattr(testbed, "create_server", lambda *args: start.env.server)
    monkeypatch.setattr(testbed, "sample_full_reward_product_goals", lambda *args: (selected,))
    monkeypatch.setattr(testbed, "construct_results_start_episode", lambda *args, **kwargs: start)
    monkeypatch.setattr(testbed, "validate_reachable_full_reward", lambda *args: {"full_reward": True, "raw_reward": 1.0})
    monkeypatch.setattr(testbed, "_configure_vllm_engine", lambda: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine))
    monkeypatch.setattr(testbed, "VllmPolicy", lambda *args: object())
    monkeypatch.setattr(testbed, "estimate_search_results_probabilities", estimate)
    monkeypatch.setattr(testbed, "collect_rollouts", collect)

    output = tmp_path / "complete.json"
    args = testbed.build_argument_parser().parse_args(["--num-pages", "1", "--samples-per-page", "2", "--output", str(output)])
    report = testbed.run_testbed(args)

    assert report["status"] == "complete" and report["pages_completed"] == 1
    assert seen["pseudo_prompt"] == seen["rollout_prompt"] == start.prompt
    assert seen["max_steps"] == 14 and seen["threshold"] == 1e-3
    assert seen["engine_kwargs"]["enable_prefix_caching"] is False
    assert report["pages"][0]["outcomes"]["empirical_success_probability"] == 0.5
    assert report["pages"][0]["pseudo_success_probability"] == 0.4
    assert report["pages"][0]["pseudo_full_reward_product_entry_probability"] == 0.6
    assert report["pages"][0]["outcomes"]["empirical_full_reward_product_entry_probability"] == 0.5
    assert report["summary"]["page_mean_absolute_gap"] == pytest.approx(0.1)
    assert report["summary"]["later_page_product_entry_above_one_percent"] == {"pages": 1, "pseudo_count": 1, "empirical_count": 0}
    assert "## Products beyond the starting results page" in output.with_suffix(".md").read_text()
    assert json.loads(output.read_text())["status"] == "complete"
