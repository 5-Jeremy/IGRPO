"""Lightweight, client-owned WebShop episodes; no catalog or browser process."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from uuid import uuid4

from .catalog_data import SeedViewRef, native_imports
from .catalog_service import CatalogClient, CatalogMismatchError, InvalidPageStateError

native_imports()
from web_agent_site.engine.engine import ACTION_TO_TEMPLATE, BACK_TO_SEARCH, END_BUTTON, NEXT_PAGE, PREV_PAGE, parse_action  # noqa: E402
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa: E402


@dataclass
class EpisodeState:
    episode_id: str
    seed_view: SeedViewRef
    goal_index: int
    goal: dict
    page_type: str = ""
    current_url: str = ""
    keywords: tuple[str, ...] = ()
    page: int = 1
    asin: str | None = None
    visited_asins: frozenset[str] = frozenset()
    selected_options: tuple[tuple[str, str], ...] = ()
    action_counts: dict = field(default_factory=dict)
    done: bool = False
    reward: float | None = None
    instruction_text: str = ""
    rendered_html: str = ""
    previous_observations: tuple[str, ...] = ()
    previous_actions: tuple[str, ...] = ()
    random_search_count: int = 0
    random_token: str | None = None
    visible_asins: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExportedEpisode:
    schema_version: int
    episode_state: EpisodeState
    rollout_task_id: int | None = None


class RemoteWebAgentTextEnv(WebAgentTextEnv):
    """Reuse native text/DOM formatting, but implement transitions over typed RPC."""

    def __init__(self, catalog_service, seed=42, request_timeout_s=120, startup_timeout_s=1800, observation_mode="text", **kwargs):
        self.client = CatalogClient(catalog_service, startup_timeout_s)
        self.metadata = self.client.call("metadata")
        self.client.catalog_key = self.metadata.catalog_key
        self.initial_seed_view = self.client.call("get_seed_view", seed=int(seed), split=kwargs.get("goal_split", "train"))
        self.client_id = uuid4().hex
        self.client.call("connect", client_id=self.client_id)
        self.observation_mode = observation_mode
        self.num_prev_obs = int(kwargs.get("num_prev_obs", 0))
        self.num_prev_actions = int(kwargs.get("num_prev_actions", 0))
        self.browser = SimpleNamespace(current_url=None, page_source=None)
        self._actions = None
        self._closed = False
        self.reset()
        self.client.timeout = request_timeout_s

    @property
    def seed_view(self):
        return self.episode.seed_view

    @property
    def page_type(self):
        return self.episode.page_type

    @property
    def goal_count(self):
        return self.initial_seed_view.goal_count

    def _install_render(self, response):
        self.episode.rendered_html = response.html
        self.episode.current_url = response.url
        self.episode.page_type = response.page_type
        self.browser.page_source = response.html
        self.browser.current_url = response.url
        self._actions = None
        self.text_to_clickable = None

    def _call(self, operation, **payload):
        return self.client.call(operation, episode_id=self.session, **payload)

    def reset(self, session=None, instruction_text=None):
        episode_id = uuid4().hex
        ref = self.initial_seed_view
        index = int(session) if session is not None else self.client.call("sample_goal_index", seed_view=ref, random_token=episode_id)
        goal = self.client.call("get_goal", seed_view=ref, goal_index=index)
        if instruction_text is not None:
            goal["instruction_text"] = instruction_text
        rendered = self.client.call("render_start", episode_id=episode_id, goal=goal)
        self.episode = EpisodeState(episode_id, ref, index, goal, instruction_text=goal["instruction_text"])
        self.session = episode_id
        self.instruction_text = goal["instruction_text"]
        self._install_render(rendered)
        self.prev_obs = [self.observation]
        self.prev_actions = []
        return self.observation, None

    def get_available_actions(self):
        if self._actions is None:
            self._actions = super().get_available_actions()
        return copy.deepcopy(self._actions)

    def convert_html_to_text(self, html, simple=False):
        if simple:
            return super().convert_html_to_text(html, simple=True)
        # Native rich formatting only needs the visited-ASIN set, never a server.
        from bs4.element import Comment

        observation = ""
        for text in self._parse_html(html).find_all(string=True):
            if text == "\n" or text.parent.name in {"style", "script", "head", "title", "meta", "[document]"} or isinstance(text, Comment):
                continue
            if text.parent.name == "button":
                formatted = f"[button] {text} [button_]"
            elif text.parent.name == "label":
                if f'"{text}"' in self.state["url"]:
                    formatted = f"  [clicked button] {text} [clicked button_]"
                    observation = f"You have clicked {text}.\n" + observation
                else:
                    formatted = f"  [button] {text} [button_]"
            elif text.parent.get("class") == ["product-link"]:
                label = "clicked button" if str(text) in self.episode.visited_asins else "button"
                formatted = f"\n[{label}] {text} [{label}_]"
            else:
                formatted = str(text)
            observation += formatted + "\n"
        return observation

    def _search(self, keywords, page):
        state = self.episode
        token = state.random_token
        if keywords[0] == "<r>":
            token = f"{state.episode_id}:{state.random_search_count}"
        rendered, asins = self._call("search_and_render", seed_view=state.seed_view, goal=state.goal, keywords=keywords, page=page, random_token=token)
        state.keywords, state.page = tuple(keywords), page
        state.asin, state.selected_options, state.visible_asins = None, (), asins
        state.random_token = token
        if keywords[0] == "<r>":
            state.random_search_count += 1
        state.action_counts["search"] += 1
        self._install_render(rendered)

    def _item(self, asin, options, subpage=None):
        state = self.episode
        rendered = self._call("render_item", seed_view=state.seed_view, goal=state.goal, asin=asin, keywords=state.keywords, page=state.page, selected_options=options, subpage=subpage)
        state.asin, state.selected_options = asin, options
        self._install_render(rendered)

    def step(self, action):
        self.get_available_actions()
        state = self.episode
        state.action_counts = defaultdict(int, state.action_counts)
        name, argument = parse_action(action)
        argument = argument.lower() if argument is not None else None
        reward, done, verbose = 0.0, False, None
        if name == "search" and argument:
            self._search(tuple(argument.split(" ")), 1)
        elif name == "click" and argument in self.text_to_clickable and argument != "search":
            clickable = self.text_to_clickable[argument]
            if argument == END_BUTTON.lower():
                purchase, rendered = self._call("purchase_and_render", seed_view=state.seed_view, goal=state.goal, asin=state.asin, selected_options=state.selected_options)
                self._install_render(rendered)
                reward, done, verbose = purchase.reward, True, purchase.verbose_info
                state.reward, state.done = reward, done
                state.action_counts["purchase"] += 1
            elif argument == BACK_TO_SEARCH.lower():
                rendered = self._call("render_start", goal=state.goal)
                state.keywords, state.page, state.asin = (), 1, None
                state.visited_asins, state.selected_options, state.visible_asins = frozenset(), (), ()
                state.action_counts.clear()
                self._install_render(rendered)
            elif argument in (NEXT_PAGE.lower(), PREV_PAGE.lower()) and state.page_type == "search_results":
                self._search(state.keywords, state.page + (1 if argument == NEXT_PAGE.lower() else -1))
            elif argument == PREV_PAGE.lower() and state.page_type == "item_page":
                self._search(state.keywords, state.page)
            elif argument == PREV_PAGE.lower() and state.page_type == "item_sub_page":
                self._item(state.asin, state.selected_options)
            elif argument in {key.lower() for key in ACTION_TO_TEMPLATE}:
                subpage = next(key for key in ACTION_TO_TEMPLATE if key.lower() == argument)
                self._item(state.asin, state.selected_options, subpage)
                state.action_counts[subpage] += 1
            elif "product-link" in (clickable.get("class") or []):
                asin = argument.upper()
                self._item(asin, state.selected_options)
                state.visited_asins = state.visited_asins | {asin}
                state.action_counts["asin"] += 1
            elif clickable.get("name"):
                options = dict(state.selected_options)
                options[clickable["name"].lower()] = argument
                self._item(state.asin, tuple(options.items()))
                state.action_counts["options"] += 1
        observation = self.observation
        text = [observation]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text.append(self.prev_obs[-i])
        self.prev_obs.append(observation)
        # Keep only the configured embedded-history window. Manager owns rollout history.
        self.prev_obs = self.prev_obs[-max(1, self.num_prev_obs) :]
        self.prev_actions = self.prev_actions[-self.num_prev_actions :] if self.num_prev_actions else []
        info = dict(webshop_session_id=self.session, page_type=state.page_type, available_actions=self.get_available_actions())
        if verbose is not None:
            info["verbose_info"] = verbose
        if done:
            self.reset()
        return " [SEP] ".join(text[::-1]), reward, done, info

    def export_episode(self, rollout_task_id=None):
        state = copy.deepcopy(self.episode)
        state.previous_observations = tuple(self.prev_obs)
        state.previous_actions = tuple(self.prev_actions)
        state.action_counts = dict(state.action_counts)
        return ExportedEpisode(1, state, rollout_task_id)

    def import_episode(self, exported):
        if not isinstance(exported, ExportedEpisode) or exported.schema_version != 1:
            raise InvalidPageStateError("Unsupported episode schema")
        state = copy.deepcopy(exported.episode_state)
        if state.seed_view.catalog_key != self.metadata.catalog_key:
            raise CatalogMismatchError("Cannot import an episode from another catalog")
        # Also validates deterministic reconstruction after seed-view eviction.
        goal = self.client.call("get_goal", seed_view=state.seed_view, goal_index=state.goal_index)
        if goal != state.goal:
            raise InvalidPageStateError("Imported goal does not match seed view")
        self.episode = state
        self.session, self.instruction_text = state.episode_id, state.instruction_text
        self.browser.current_url, self.browser.page_source = state.current_url, state.rendered_html
        self.prev_obs, self.prev_actions = list(state.previous_observations), list(state.previous_actions)
        self._actions, self.text_to_clickable = None, None

    def get_scoring_products(self, asins):
        return self._call("get_scoring_products", seed_view=self.seed_view, asins=tuple(asins))

    def close(self):
        if not self._closed:
            self._closed = True
            self.client.call("disconnect", client_id=self.client_id)
