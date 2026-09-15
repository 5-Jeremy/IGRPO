"""Estimate full-reward purchase probability from a WebShop results page."""

from __future__ import annotations

import copy
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from treehca.product_page_parser import ProductPageContextParts, extract_product_page_contexts
from treehca.pseudo_rollout_product_page import prepare_product_page_grouped_choice_rollouts, score_product_page_grouped_choice_rollouts
from treehca.pseudo_rollout_results_page import compute_results_page_answer_probability

_NEXT_PAGE_ACTION = "click[next >]"
_RESULTS_PAGE_SUMMARY = re.compile(r"'Page (?P<page>\d+) \(Total results: (?P<total>\d+)\)'")
_PRODUCTS_PER_RESULTS_PAGE = 10
logger = logging.getLogger(__name__)


class ClonableWebshopState(Protocol):
    """Minimum state interface used by the estimator."""

    env: Any
    prompt: str

    def clone(self) -> ClonableWebshopState: ...

    def advance(self, action: str) -> Any: ...


@dataclass
class WebshopProbabilityState:
    """A clonable native WebShop environment and its matching prompt manager.

    ``prompt`` must describe the current environment page. ``advance`` mirrors
    the production manager's history update so child prompts retain the exact
    action path used to reach them.
    """

    env: Any
    manager: Any
    prompt: str

    def clone(self) -> WebshopProbabilityState:
        # Catalog/search resources are immutable and too large to duplicate.
        server = copy.copy(self.env.server)
        server.user_sessions = copy.deepcopy(self.env.server.user_sessions)
        memo = {id(self.env.server): server}
        text_to_clickable = getattr(self.env, "text_to_clickable", None)
        if text_to_clickable is not None:
            # The environment rebuilds this cyclic BeautifulSoup cache on step.
            memo[id(text_to_clickable)] = None
        return copy.deepcopy(self, memo)

    def advance(self, action: str) -> Any:
        observation, reward, done, info = self.env.step(action)
        if done:
            raise ValueError("A probability-search transition terminated before the implied purchase")
        formatted_observation = self.manager.format_obs([observation])
        self.manager.memory.store({"text_obs": self.manager.pre_text_obs, "action": [action]})
        self.manager.pre_text_obs = formatted_observation
        page_info = {"available_actions": self.env.get_available_actions()}
        self.prompt = self.manager.build_text_obs(formatted_observation, [page_info])[0]
        return observation, reward, done, info


@dataclass(frozen=True)
class _ResultsPage:
    state: ClonableWebshopState
    parts: ProductPageContextParts
    correct_product_actions: tuple[str, ...]
    has_next_page: bool


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


def _session(state: ClonableWebshopState) -> dict[str, Any]:
    try:
        return state.env.server.user_sessions[state.env.session]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("state must expose the active native WebShop session") from error


def _product_can_earn_full_reward(state: ClonableWebshopState, asin: str) -> bool:
    """Use the native reward function to recognize successful product leaves."""
    server = state.env.server
    session = _session(state)
    try:
        item = server.product_item_dict[asin]
        price = server.product_prices[asin]
        goal = session["goal"]
    except KeyError as error:
        raise ValueError(f"WebShop state is missing product or goal data for {asin!r}") from error

    from web_agent_site.engine.goal import get_option_reward, get_reward

    option_groups = item.get("options", {})
    if any(not values for values in option_groups.values()):
        return False

    raw_targets = goal.get("goal_options", {})
    targets = tuple(raw_targets.items()) if isinstance(raw_targets, dict) else tuple(raw_targets)
    full_mask = (1 << len(targets)) - 1
    # Retain one witness per coverage mask instead of enumerating the option grid.
    reachable: dict[int, dict[str, str]] = {0: {}}
    for group_name, values in option_groups.items():
        value_masks = []
        for value in values:
            mask = sum(1 << index for index, target in enumerate(targets) if get_option_reward((value,), (target,))[1] == 1)
            value_masks.append((value, mask))
        updated = dict(reachable)  # Omitting this group is a valid product-page choice.
        for covered, selection in reachable.items():
            for value, value_mask in value_masks:
                updated.setdefault(covered | value_mask, {**selection, group_name: value})
        reachable = updated

    if full_mask not in reachable:
        return False
    reward = float(get_reward(item, goal, price=price, options=reachable[full_mask]))
    return math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-12)


def _results_page(state: ClonableWebshopState) -> _ResultsPage:
    try:
        page_name = state.env.server.get_page_name(state.env.browser.current_url)
    except AttributeError as error:
        raise ValueError("state must contain a native WebShop environment") from error
    if page_name != "search_results":
        raise ValueError(f"Expected a search-results page, found {page_name!r}")

    parts = extract_product_page_contexts([state.prompt])[0]
    product_by_lower_asin = {asin.lower(): asin for asin in state.env.server.product_item_dict}
    correct_actions = []
    for action in parts.admissible_actions:
        if not action.startswith("click[") or not action.endswith("]"):
            continue
        asin = product_by_lower_asin.get(action[len("click[") : -1].lower())
        if asin is not None and _product_can_earn_full_reward(state, asin):
            correct_actions.append(action)
    return _ResultsPage(
        state=state,
        parts=parts,
        correct_product_actions=tuple(correct_actions),
        has_next_page=_has_real_next_page(parts),
    )


def _has_real_next_page(parts: ProductPageContextParts) -> bool:
    """Ignore WebShop's unconditional Next button on the final results page."""
    if _NEXT_PAGE_ACTION not in parts.admissible_actions:
        return False
    matches = list(_RESULTS_PAGE_SUMMARY.finditer(parts.current_observation))
    if not matches:
        # Retain compatibility with synthetic callers that omit native page metadata.
        return True
    if len(matches) != 1:
        raise ValueError(f"Expected one results-page summary, found {len(matches)}")
    page_number = int(matches[0]["page"])
    total_results = int(matches[0]["total"])
    return page_number * _PRODUCTS_PER_RESULTS_PAGE < total_results


def _forward_results_pages(initial_state: ClonableWebshopState) -> tuple[_ResultsPage, ...]:
    """Materialize the forward pagination spine without scoring model actions."""
    current = initial_state.clone()
    pages = []
    visited_urls = set()
    while True:
        url = current.env.browser.current_url
        if url in visited_urls:
            raise ValueError(f"Forward pagination revisited URL {url!r}")
        visited_urls.add(url)
        page = _results_page(current)
        pages.append(page)
        if not page.has_next_page:
            return tuple(pages)
        child = current.clone()
        child.advance(_NEXT_PAGE_ACTION)
        current = child


def _results_action_probability(
    page: _ResultsPage,
    action: str,
    tokenizer: Any,
    actor_rollout_wg: Any,
) -> float:
    probability = compute_results_page_answer_probability(
        page.state.prompt,
        action,
        tokenizer,
        actor_rollout_wg,
    )
    if hasattr(probability, "item"):
        probability = probability.item()
    return _validate_probability(probability, f"results action probability for {action!r}")


def _product_success_probability(
    page: _ResultsPage,
    product_action: str,
    tokenizer: Any,
    inference_engine: Any,
    *,
    max_model_len: int | None,
    lora_request: Any,
) -> float:
    product_state = page.state.clone()
    product_state.advance(product_action)
    page_name = product_state.env.server.get_page_name(product_state.env.browser.current_url)
    if page_name != "item_page":
        raise ValueError(f"Product action {product_action!r} led to {page_name!r}, not an item page")

    product_parts = extract_product_page_contexts([product_state.prompt])
    goal_options = _session(product_state).get("goal", {}).get("goal_options", {})
    pseudo_rollouts = prepare_product_page_grouped_choice_rollouts(
        product_parts,
        tokenizer,
        [goal_options],
    )
    if not pseudo_rollouts:
        return 1.0

    scores = score_product_page_grouped_choice_rollouts(
        inference_engine,
        pseudo_rollouts,
        max_model_len=max_model_len,
        lora_request=lora_request,
    )
    if len(scores) != len(pseudo_rollouts) or any(score is None for score in scores):
        raise ValueError("Every product option group must have a pseudo-rollout score")
    group_probabilities = [
        _validate_probability(score.correct_probability, f"correct probability for option group {score.option_group.name!r}")
        for score in scores
    ]
    return math.prod(group_probabilities)


def estimate_search_results_success_probability(
    initial_state: ClonableWebshopState,
    tokenizer: Any,
    actor_rollout_wg: Any,
    inference_engine: Any,
    path_probability_threshold: float,
    *,
    max_model_len: int | None = None,
    lora_request: Any = None,
) -> float:
    """Estimate the probability of a correct purchase without backtracking.

    Results-page action probabilities are raw answer-sequence probabilities.
    Product success is the product of the grouped correct-choice masses and is
    conditional on subsequently purchasing. A partial branch is expanded when
    its cumulative probability is at least ``path_probability_threshold``.

    If pruning completes no product branch, one fallback path is followed by
    greedily choosing the success-capable action with the greatest immediate
    probability at each results page. The final sum is bounded to ``[0, 1]``.
    """
    threshold = _validate_probability(path_probability_threshold, "path_probability_threshold")
    if not isinstance(initial_state.prompt, str):
        raise ValueError("initial_state.prompt must be a string")

    pages = _forward_results_pages(initial_state)
    # A next-page edge is success-capable only when a later page has a correct product.
    suffix_has_product = [False] * (len(pages) + 1)
    for index in range(len(pages) - 1, -1, -1):
        suffix_has_product[index] = bool(pages[index].correct_product_actions) or suffix_has_product[index + 1]
    logger.info(
        "Pseudo success search discovered %d forward results page(s) and %d correct product branch(es)",
        len(pages),
        sum(len(page.correct_product_actions) for page in pages),
    )

    result_probability_cache: dict[tuple[int, str], float] = {}

    def result_probability(page_index: int, action: str) -> float:
        key = (page_index, action)
        if key not in result_probability_cache:
            logger.info("Scoring results page %d action %s", page_index + 1, action)
            result_probability_cache[key] = _results_action_probability(pages[page_index], action, tokenizer, actor_rollout_wg)
        return result_probability_cache[key]

    terminal_masses = []
    completed_paths = 0
    page_probability = 1.0
    for page_index, page in enumerate(pages):
        for product_action in page.correct_product_actions:
            product_prefix_probability = page_probability * result_probability(page_index, product_action)
            if product_prefix_probability < threshold:
                continue
            logger.info("Scoring product branch %s with cumulative prefix probability %.6g", product_action, product_prefix_probability)
            conditional_probability = _product_success_probability(
                page,
                product_action,
                tokenizer,
                inference_engine,
                max_model_len=max_model_len,
                lora_request=lora_request,
            )
            # Completion makes the path count even when this final factor puts it below threshold.
            terminal_masses.append(product_prefix_probability * conditional_probability)
            completed_paths += 1

        can_reach_later_product = page.has_next_page and suffix_has_product[page_index + 1]
        if not can_reach_later_product:
            break
        next_probability = page_probability * result_probability(page_index, _NEXT_PAGE_ACTION)
        if next_probability < threshold:
            break
        page_probability = next_probability

    if completed_paths == 0 and suffix_has_product[0]:
        # Below the threshold, follow only one locally most probable successful continuation.
        page_probability = 1.0
        for page_index, page in enumerate(pages):
            product_actions = set(page.correct_product_actions)
            can_reach_later_product = page.has_next_page and suffix_has_product[page_index + 1]
            eligible_actions = [
                action
                for action in page.parts.admissible_actions
                if action in product_actions or (action == _NEXT_PAGE_ACTION and can_reach_later_product)
            ]
            if not eligible_actions:
                break
            greedy_action = max(eligible_actions, key=lambda action: result_probability(page_index, action))
            page_probability *= result_probability(page_index, greedy_action)
            if greedy_action == _NEXT_PAGE_ACTION:
                continue
            logger.info("Scoring greedy fallback product branch %s", greedy_action)
            conditional_probability = _product_success_probability(
                page,
                greedy_action,
                tokenizer,
                inference_engine,
                max_model_len=max_model_len,
                lora_request=lora_request,
            )
            terminal_masses.append(page_probability * conditional_probability)
            break

    return min(1.0, max(0.0, math.fsum(terminal_masses)))
