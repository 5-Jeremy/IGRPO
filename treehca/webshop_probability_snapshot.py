"""Immutable WebShop turn snapshots and isolated native product rendering."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from treehca.product_page_parser import extract_product_page_contexts


@dataclass(frozen=True)
class WebshopTurnSnapshot:
    """Serializable turn data; catalog/model objects are deliberately excluded.

    ``previous_page_type`` must describe the immediately preceding observation.
    ``terminated`` must come from the step result because WebShop auto-resets.
    History contains only the configured prompt window, not the trajectory.
    """

    catalog_key: str
    goal_json: str
    shopping_task: str
    prompt: str
    page_type: str
    search_terms: tuple[str, ...] = ()
    results_page: int = 1
    visible_asins: tuple[str, ...] = ()
    asin: str | None = None
    selected_options: tuple[tuple[str, str], ...] = ()
    previous_page_type: str | None = None
    terminated: bool = False
    randomized_search: bool = False
    history: tuple[tuple[str, str], ...] = ()
    completed_steps: int = 0
    history_limit: int = 0

    @property
    def query_key(self) -> tuple[str, str]:
        """Compare the full task and reward goal before any probability reuse."""
        return self.shopping_task, self.goal_json

    @property
    def fresh_product_entry(self) -> bool:
        return self.page_type == "item_page" and self.previous_page_type == "search_results" and not self.selected_options and not self.terminated


class WebshopSnapshotSource:
    """Share one fixed native catalog/price table across lightweight snapshots.

    Create a new source when the catalog, prices, or rendering settings change.
    Native WebShop must already be importable, as when constructing its env.
    """

    def __init__(self, server: Any):
        self.server = server
        self.catalog_key = uuid4().hex

    def capture(
        self,
        env: Any,
        manager: Any,
        prompt: str,
        *,
        rollout_index: int = 0,
        previous_page_type: str | None = None,
        terminated: bool = False,
        randomized_search: bool | None = None,
    ) -> WebshopTurnSnapshot:
        """Capture before choosing the next action, without mutating the env."""
        if env.server.product_item_dict is not self.server.product_item_dict or env.server.product_prices is not self.server.product_prices:
            raise ValueError("Snapshot environment must share the source's catalog and prices")
        if env.observation_mode != "text" or env.num_prev_obs or env.num_prev_actions:
            raise ValueError("Snapshot extraction requires production WebShop text mode without embedded history")
        session = env.server.user_sessions[env.session]
        task = manager.tasks[rollout_index]
        page_type = env.server.get_page_name(env.browser.current_url)
        terms = tuple(session.get("keywords") or ())
        history_limit = max(0, int(manager.config.env.history_length))
        memory = manager.memory[rollout_index]
        history = tuple((record["text_obs"], record["action"]) for record in memory[-history_limit:]) if history_limit else ()
        visible = ()
        if not terminated and page_type in {"search_results", "item_page"}:
            parts = extract_product_page_contexts([prompt])[0]
            if parts.shopping_task != task or task != session["goal"]["instruction_text"]:
                raise ValueError("Prompt, manager, and environment must describe the same shopping query")
            if page_type == "search_results":
                products = {asin.lower(): asin for asin in self.server.product_item_dict}
                visible = tuple(products[action[6:-1].lower()] for action in parts.admissible_actions if action.startswith("click[") and action[6:-1].lower() in products)
        return WebshopTurnSnapshot(
            catalog_key=self.catalog_key,
            goal_json=json.dumps(session["goal"], sort_keys=True, separators=(",", ":")),
            shopping_task=task,
            prompt=prompt,
            page_type=page_type,
            search_terms=terms,
            results_page=int(session.get("page") or 1),
            visible_asins=visible,
            asin=session.get("asin"),
            selected_options=tuple(sorted(session.get("options", {}).items())),
            previous_page_type=previous_page_type,
            terminated=bool(terminated),
            randomized_search=bool(terms and terms[0] == "<r>") or bool(randomized_search),
            history=history,
            completed_steps=len(memory),
            history_limit=history_limit,
        )

    def product_entry(self, snapshot: WebshopTurnSnapshot, asin: str) -> WebshopTurnSnapshot:
        """Render a hypothetical current-page click without stepping a live env."""
        if snapshot.catalog_key != self.catalog_key or snapshot.page_type != "search_results" or asin not in snapshot.visible_asins:
            raise ValueError("Product entry must refer to a visible product in this source's results snapshot")
        from web_agent_site.engine import engine
        from web_agent_site.envs import web_agent_text_env as native

        from agent_system.environments.env_manager import WebshopEnvironmentManager

        parts = extract_product_page_contexts([snapshot.prompt])[0]
        action = next(action for action in parts.admissible_actions if action.lower() == f"click[{asin.lower()}]")
        # Only native rendering/conversion methods are used; no reset or RNG calls.
        with native.app.test_request_context("/"):
            html = engine.map_action_to_html(
                "click",
                session_id="pseudo-snapshot",
                product_info=self.server.product_item_dict[asin],
                keywords=list(snapshot.search_terms),
                page=snapshot.results_page,
                asin=asin,
                options={},
                instruction_text=snapshot.shopping_task,
                show_attrs=self.server.show_attrs,
            )
        env = object.__new__(native.WebAgentTextEnv)
        env.observation_mode = "text"
        env.browser = SimpleNamespace(page_source=html, current_url="/item_page/pseudo-snapshot")
        env.instruction_text = snapshot.shopping_task
        formatter = object.__new__(WebshopEnvironmentManager)
        formatter.tasks = [snapshot.shopping_task]
        observation = formatter.format_obs([env.observation])[0]
        actions = formatter.format_avail_actions(env.get_available_actions())
        history = (*snapshot.history, (parts.current_observation, action))[-snapshot.history_limit :] if snapshot.history_limit else ()
        child = replace(
            snapshot,
            page_type="item_page",
            previous_page_type="search_results",
            asin=asin,
            selected_options=(),
            visible_asins=(),
            history=history,
            completed_steps=snapshot.completed_steps + 1,
        )
        return replace(child, prompt=_product_entry_prompt(child, observation, actions))


def _product_entry_prompt(snapshot: WebshopTurnSnapshot, observation: str, actions: Sequence[str]) -> str:
    """Use production templates, including the long-history fallback."""
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS

    fields = dict(task_description=snapshot.shopping_task, current_observation=observation, available_actions="\n".join(f"'{action}'," for action in actions))
    if snapshot.history_limit:
        start = snapshot.completed_steps - len(snapshot.history) + 1
        history = "\n".join(f"[Observation {step}: '{obs}', Action {step}: '{action}']" for step, (obs, action) in enumerate(snapshot.history, start=start))
        prompt = WEBSHOP_TEMPLATE.format(**fields, step_count=snapshot.completed_steps, history_length=len(snapshot.history), action_history=history, current_step=snapshot.completed_steps + 1)
        if len(prompt) <= 13000:
            return prompt
    return WEBSHOP_TEMPLATE_NO_HIS.format(**fields)


def snapshot_webshop_rollouts(
    source: WebshopSnapshotSource,
    states: Sequence[Any],
    *,
    previous_page_types: Sequence[str | None],
    terminated: Sequence[bool] | None = None,
) -> tuple[WebshopTurnSnapshot, ...]:
    """Extract a batch from single-rollout objects exposing env/manager/prompt."""
    if len(states) != len(previous_page_types) or (terminated is not None and len(states) != len(terminated)):
        raise ValueError("Snapshot metadata must align with rollout states")
    return tuple(source.capture(state.env, state.manager, state.prompt, previous_page_type=previous_page_types[index], terminated=False if terminated is None else terminated[index]) for index, state in enumerate(states))
