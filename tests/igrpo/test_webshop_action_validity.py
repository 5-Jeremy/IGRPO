from types import SimpleNamespace

import numpy as np
import pytest

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop.envs import (
    WebshopWorker,
    is_admissible_webshop_action,
)


@pytest.mark.parametrize(
    ("action", "expected", "reason"),
    [
        ("click[B08LKKSL8F]", True, None),
        ("click[B08LKKSLS8F]", False, "unavailable_click"),
        ("click[3x-large]", True, None),
        ("click[3x]", False, "unavailable_click"),
        ("click[b-c-navy blue]", False, "unavailable_click"),
        ("search[navy shirt]", False, "search_unavailable"),
    ],
)
def test_admissible_webshop_action_requires_a_current_clickable(action, expected, reason):
    available = {
        "has_search_bar": False,
        "clickables": ["b08lkksl8f", "3x-large", "c-navy blue"],
    }

    assert is_admissible_webshop_action(action, available) == (expected, reason)


def test_admissible_webshop_search_accepts_free_form_queries():
    available = {"has_search_bar": True, "clickables": ["search"]}

    assert is_admissible_webshop_action("search[navy shirt]", available) == (True, None)
    assert is_admissible_webshop_action("search[   ]", available) == (False, "search_unavailable")
    assert is_admissible_webshop_action("click[search]", available) == (False, "unavailable_click")


class _WorkerEnv:
    def __init__(self):
        self.unwrapped = SimpleNamespace(
            session="session",
            page_type="search_results",
            server=SimpleNamespace(get_page_name=lambda url: "search_results"),
            browser=SimpleNamespace(current_url="unused"),
        )

    def get_available_actions(self):
        return {"has_search_bar": False, "clickables": ["B08LKKSL8F"]}

    def step(self, action):
        return "unchanged", 0.0, False, {}


def test_worker_marks_nonexistent_asin_semantically_invalid():
    worker = object.__new__(WebshopWorker)
    worker.env = _WorkerEnv()
    worker._rollout_task_id = 42

    _, _, _, info = worker.step("click[b08lkksls8f]")

    assert info["is_action_admissible"] is False
    assert info["invalid_action_reason"] == "unavailable_click"


class _Memory:
    def store(self, value):
        self.value = value


class _VectorEnv:
    def step(self, actions):
        return ["unchanged"], [0.0], [False], [{"is_action_admissible": False}]


def test_manager_combines_format_and_semantic_validity():
    manager = object.__new__(WebshopEnvironmentManager)
    manager.projection_f = lambda responses: (["click[3x]"], [1])
    manager.envs = _VectorEnv()
    manager.memory = _Memory()
    manager.pre_text_obs = ["before"]
    manager.format_obs = lambda observations: observations
    manager.build_text_obs = lambda observations, infos: observations

    _, rewards, dones, infos = manager.step(["<think>x</think><action>click[3x]</action>"])

    assert infos[0]["is_action_format_valid"] == np.array(True)
    assert infos[0]["is_action_valid"] == np.array(False)
    assert rewards.tolist() == [0.0]
    assert dones.tolist() == [False]
