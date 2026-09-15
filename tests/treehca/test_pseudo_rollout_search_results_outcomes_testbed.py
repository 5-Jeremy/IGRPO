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
        {"purchased": True, "raw_reward": 1.0, "termination_reason": "purchase"},
        {"purchased": True, "raw_reward": 0.5, "termination_reason": "purchase"},
        {"purchased": False, "raw_reward": 0.0, "termination_reason": "step_limit"},
        {"purchased": False, "raw_reward": 0.0, "termination_reason": "new_search"},
    ]

    result = testbed.summarize_outcomes(trajectories, 0.4)

    assert result["empirical_success_probability"] == 0.25
    assert result["pseudo_success_probability"] == 0.4
    assert result["pseudo_minus_empirical_success_probability"] == pytest.approx(0.15)
    assert result["purchase_count"] == 2
    assert result["termination_counts"] == {"purchase": 2, "step_limit": 1, "new_search": 1}


def test_argument_defaults_limits_and_report_serialization(tmp_path):
    parser = testbed.build_argument_parser()
    args = parser.parse_args([])
    testbed._validate_arguments(args)
    assert args.max_steps == 14
    assert args.path_probability_threshold == 1e-6
    with pytest.raises(ValueError, match="at most 14"):
        testbed._validate_arguments(parser.parse_args(["--max-steps", "15"]))

    outcomes = testbed.summarize_outcomes([{"purchased": True, "raw_reward": 1.0, "termination_reason": "purchase"}], 0.7)
    report = {
        "mode": "monte_carlo",
        "configuration": {"model": "example"},
        "pages": [{"asin": "ITEM", "setup": {"query": "query"}, "outcomes": outcomes}],
        "summary": {
            "page_mean_pseudo_success_probability": 0.7,
            "pooled_empirical_success_probability": 1.0,
            "page_mean_absolute_gap": 0.3,
            "page_root_mean_squared_gap": 0.3,
        },
    }
    output = tmp_path / "report.json"
    testbed.write_report(report, output)
    assert json.loads(output.read_text()) == report
    markdown = output.with_suffix(".md").read_text()
    assert "Search-results continuation outcome testbed" in markdown
    assert "70.000%" in markdown and "100.000%" in markdown


def test_complete_testbed_orchestration_uses_same_start_and_default_horizon(native_results_start, monkeypatch, tmp_path):
    from transformers import AutoTokenizer

    start, selected = native_results_start
    tokenizer = SimpleNamespace()
    trajectory_rows = [
        {"purchased": True, "full_reward": True, "raw_reward": 1.0, "selected_options": {}, "termination_reason": "purchase", "steps": []},
        {"purchased": False, "full_reward": False, "raw_reward": 0.0, "selected_options": {}, "termination_reason": "step_limit", "steps": []},
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
        return 0.4

    def collect(received_start, policy, **kwargs):
        seen["rollout_prompt"] = received_start.prompt
        seen["max_steps"] = kwargs["max_steps"]
        return trajectory_rows

    monkeypatch.setattr(testbed, "create_server", lambda *args: start.env.server)
    monkeypatch.setattr(testbed, "sample_diverse_product_goals", lambda *args: (selected,))
    monkeypatch.setattr(testbed, "construct_results_start_episode", lambda *args, **kwargs: start)
    monkeypatch.setattr(testbed, "validate_reachable_full_reward", lambda *args: {"full_reward": True, "raw_reward": 1.0})
    monkeypatch.setattr(testbed, "_configure_vllm_engine", lambda: None)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine))
    monkeypatch.setattr(testbed, "VllmPolicy", lambda *args: object())
    monkeypatch.setattr(testbed, "estimate_search_results_success_probability", estimate)
    monkeypatch.setattr(testbed, "collect_rollouts", collect)

    output = tmp_path / "complete.json"
    args = testbed.build_argument_parser().parse_args(["--num-pages", "1", "--samples-per-page", "2", "--output", str(output)])
    report = testbed.run_testbed(args)

    assert report["status"] == "complete" and report["pages_completed"] == 1
    assert seen["pseudo_prompt"] == seen["rollout_prompt"] == start.prompt
    assert seen["max_steps"] == 14 and seen["threshold"] == 1e-6
    assert seen["engine_kwargs"]["enable_prefix_caching"] is False
    assert report["pages"][0]["outcomes"]["empirical_success_probability"] == 0.5
    assert report["pages"][0]["pseudo_success_probability"] == 0.4
    assert report["summary"]["page_mean_absolute_gap"] == pytest.approx(0.1)
    assert json.loads(output.read_text())["status"] == "complete"
