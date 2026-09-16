import copy
from types import SimpleNamespace

import pytest

from treehca import webshop_success_probability as estimator
from treehca.product_page_parser import ProductPageContextParts


class _FakeServer:
    def __init__(self, asins, correct_asins):
        self.product_item_dict = {asin: {"correct": asin in correct_asins} for asin in asins}
        self.product_prices = {asin: 1.0 for asin in asins}
        self.user_sessions = {"session": {"goal": {"goal_options": {}}}}

    @staticmethod
    def get_page_name(url):
        return "search_results" if url.startswith("results:") else "item_page" if url.startswith("product:") else ""


class _FakeState:
    def __init__(self, page_actions, correct_asins):
        asins = {
            action[len("click[") : -1].upper()
            for actions in page_actions
            for action in actions
            if action.startswith("click[") and action[len("click[") : -1].upper() not in {"NEXT >", "< PREV", "BACK TO SEARCH"}
        }
        self.page_actions = tuple(tuple(actions) for actions in page_actions)
        self.page_index = 0
        self.prompt = "results:0"
        self.env = SimpleNamespace(
            session="session",
            browser=SimpleNamespace(current_url="results:0"),
            server=_FakeServer(asins, correct_asins),
        )

    def clone(self):
        return copy.deepcopy(self)

    def advance(self, action):
        if action == "click[next >]":
            self.page_index += 1
            self.prompt = f"results:{self.page_index}"
            self.env.browser.current_url = self.prompt
            return
        asin = action[len("click[") : -1].upper()
        self.prompt = f"product:{asin}"
        self.env.browser.current_url = self.prompt
        self.env.server.user_sessions[self.env.session]["asin"] = asin


def _parts(observation, actions):
    return ProductPageContextParts(
        agent_introduction="agent",
        shopping_task="task",
        history_block=None,
        completed_steps=None,
        history_length=None,
        current_step=None,
        current_observation=observation,
        admissible_actions=tuple(actions),
        response_instructions="respond",
    )


def _install_fakes(monkeypatch, state, action_probabilities, group_probabilities):
    product_score_calls = []

    def extract(prompts):
        extracted = []
        for prompt in prompts:
            if prompt.startswith("results:"):
                page_index = int(prompt.split(":")[1])
                extracted.append(_parts(prompt, state.page_actions[page_index]))
            else:
                extracted.append(_parts(prompt, ("click[buy now]",)))
        return extracted

    def prepare(parts, tokenizer, goals):
        asin = parts[0].current_observation.split(":")[1]
        return tuple(SimpleNamespace(asin=asin, group_index=index) for index, _ in enumerate(group_probabilities[asin]))

    def score(engine, pseudo_rollouts, **kwargs):
        asin = pseudo_rollouts[0].asin
        product_score_calls.append(asin)
        return [
            SimpleNamespace(correct_probability=group_probabilities[asin][rollout.group_index], option_group=SimpleNamespace(name=f"group-{rollout.group_index}"))
            for rollout in pseudo_rollouts
        ]

    monkeypatch.setattr(estimator, "extract_product_page_contexts", extract)
    monkeypatch.setattr(estimator, "_product_can_earn_full_reward", lambda page_state, asin: page_state.env.server.product_item_dict[asin]["correct"])
    monkeypatch.setattr(estimator, "compute_results_page_answer_probability", lambda prompt, action, tokenizer, worker: action_probabilities[(prompt, action)])
    monkeypatch.setattr(estimator, "prepare_product_page_grouped_choice_rollouts", prepare)
    monkeypatch.setattr(estimator, "score_product_page_grouped_choice_rollouts", score)
    return product_score_calls


def _estimate(state, threshold):
    return estimator.estimate_search_results_success_probability(
        state,
        tokenizer=object(),
        actor_rollout_wg=object(),
        inference_engine=object(),
        path_probability_threshold=threshold,
    )


def _estimate_both(state, threshold):
    return estimator.estimate_search_results_probabilities(
        state,
        tokenizer=object(),
        actor_rollout_wg=object(),
        inference_engine=object(),
        path_probability_threshold=threshold,
    )


def test_sums_products_and_multiplies_forward_pagination(monkeypatch):
    state = _FakeState(
        [
            ("click[back to search]", "click[a]", "click[b]", "click[next >]"),
            ("click[< prev]", "click[c]"),
        ],
        {"A", "B", "C"},
    )
    probabilities = {
        ("results:0", "click[a]"): 0.2,
        ("results:0", "click[b]"): 0.3,
        ("results:0", "click[next >]"): 0.5,
        ("results:1", "click[c]"): 0.4,
    }
    calls = _install_fakes(monkeypatch, state, probabilities, {"A": (0.5,), "B": (0.5, 0.5), "C": (0.75,)})

    probabilities = _estimate_both(state, 0.0)

    assert probabilities.success == pytest.approx(0.2 * 0.5 + 0.3 * 0.5 * 0.5 + 0.5 * 0.4 * 0.75)
    assert probabilities.product_entry == pytest.approx(0.2 + 0.3 + 0.5 * 0.4)
    assert calls == ["A", "B", "C"]
    assert state.page_index == 0
    assert state.prompt == state.env.browser.current_url == "results:0"


def test_prunes_small_prefix_but_keeps_completed_terminal_below_threshold(monkeypatch):
    state = _FakeState(
        [("click[a]", "click[b]", "click[next >]"), ("click[c]",)],
        {"A", "B", "C"},
    )
    probabilities = {
        ("results:0", "click[a]"): 0.2,
        ("results:0", "click[b]"): 0.3,
        ("results:0", "click[next >]"): 0.5,
        ("results:1", "click[c]"): 0.4,
    }
    calls = _install_fakes(monkeypatch, state, probabilities, {"A": (1.0,), "B": (0.1,), "C": (1.0,)})

    result = _estimate(state, 0.25)

    # B is expanded at 0.3, then retained after its product factor lowers it to 0.03.
    assert result == pytest.approx(0.03)
    assert calls == ["B"]

    # Entry still includes branches pruned from the more expensive purchase score.
    assert _estimate_both(state, 0.25).product_entry == pytest.approx(0.2 + 0.3 + 0.5 * 0.4)


def test_prefix_equal_to_threshold_is_expanded(monkeypatch):
    state = _FakeState([("click[a]",)], {"A"})
    calls = _install_fakes(monkeypatch, state, {("results:0", "click[a]"): 0.25}, {"A": (0.2,)})

    assert _estimate(state, 0.25) == pytest.approx(0.05)
    assert calls == ["A"]


def test_fallback_follows_greatest_immediate_probability_without_breadth(monkeypatch):
    state = _FakeState(
        [("click[a]", "click[b]", "click[next >]"), ("click[c]",)],
        {"A", "B", "C"},
    )
    probabilities = {
        ("results:0", "click[a]"): 0.4,
        ("results:0", "click[b]"): 0.5,
        ("results:0", "click[next >]"): 0.6,
        ("results:1", "click[c]"): 0.5,
    }
    calls = _install_fakes(monkeypatch, state, probabilities, {"A": (1.0,), "B": (1.0,), "C": (0.2,)})

    result = _estimate(state, 0.9)

    assert result == pytest.approx(0.6 * 0.5 * 0.2)
    assert calls == ["C"]


def test_greedy_fallback_does_not_follow_next_page_without_a_future_success(monkeypatch):
    state = _FakeState([("click[a]", "click[next >]"), ("click[x]",)], {"A"})
    probabilities = {
        ("results:0", "click[a]"): 0.2,
        ("results:0", "click[next >]"): 0.99,
    }
    calls = _install_fakes(monkeypatch, state, probabilities, {"A": (0.5,), "X": (1.0,)})

    assert _estimate(state, 0.9) == pytest.approx(0.1)
    assert calls == ["A"]


def test_result_is_capped_and_optionless_products_have_conditional_mass_one(monkeypatch):
    state = _FakeState([("click[a]", "click[b]")], {"A", "B"})
    probabilities = {
        ("results:0", "click[a]"): 0.8,
        ("results:0", "click[b]"): 0.8,
    }
    calls = _install_fakes(monkeypatch, state, probabilities, {"A": (), "B": ()})

    assert _estimate(state, 0.0) == 1.0
    assert _estimate_both(state, 0.0).product_entry == 1.0
    # Empty grouped rollouts do not invoke the product scoring engine.
    assert calls == []


def test_no_correct_path_returns_zero_and_threshold_is_validated(monkeypatch):
    state = _FakeState([("click[x]",)], set())
    _install_fakes(monkeypatch, state, {}, {"X": (1.0,)})

    assert _estimate(state, 0.5) == 0.0
    assert _estimate_both(state, 0.5).product_entry == 0.0
    with pytest.raises(ValueError, match="path_probability_threshold"):
        _estimate(state, 1.01)


def test_native_results_metadata_overrides_unconditional_next_button():
    actions = ("click[item]", "click[next >]")
    assert not estimator._has_real_next_page(_parts("'Back to Search' [SEP] 'Page 1 (Total results: 5)'", actions))
    assert estimator._has_real_next_page(_parts("'Back to Search' [SEP] 'Page 1 (Total results: 11)'", actions))
    assert not estimator._has_real_next_page(_parts("'Back to Search' [SEP] 'Page 2 (Total results: 11)'", actions))
