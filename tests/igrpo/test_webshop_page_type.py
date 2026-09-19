from types import SimpleNamespace

from agent_system.environments.env_package.webshop.envs import WebshopWorker


class FakeEnv:
    def __init__(self, page_type):
        server = SimpleNamespace(get_page_name=lambda url: page_type)
        self.unwrapped = SimpleNamespace(server=server, browser=SimpleNamespace(current_url="unused"))

    def step(self, action):
        return "observation", 0.0, False, None

    def reset(self, session):
        return "observation", None

    def get_available_actions(self):
        return {}


def test_worker_reports_current_page_type_after_step_and_reset():
    worker = object.__new__(WebshopWorker)
    worker.env = FakeEnv("item_sub_page")

    assert worker.step("click[description]")[3]["page_type"] == "item_sub_page"
    assert worker.reset(0)[1]["page_type"] == "item_sub_page"
