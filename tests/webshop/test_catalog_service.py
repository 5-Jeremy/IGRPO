"""Native small-catalog parity and bounded-state migration contracts."""

import copy
import json
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from agent_system.environments.env_package.webshop.catalog_data import WebshopCatalogData
from agent_system.environments.env_package.webshop.catalog_service import (
    CatalogMismatchError,
    CatalogRequest,
    InvalidGoalIndexError,
    InvalidPageStateError,
    ServiceOverloadedError,
    UnknownProductError,
    WebshopCatalogService,
    resolve_catalog_config,
)
from agent_system.environments.env_package.webshop.remote_text_env import RemoteWebAgentTextEnv, WebAgentTextEnv


@pytest.fixture(scope="module")
def service():
    return WebshopCatalogService(resolve_catalog_config({"num_products": 1000}, {"seed_view_cache_size": 2}))


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_legacy_prices_goals_observations_actions_rewards(service, seed):
    legacy = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False, seed=seed)
    remote = RemoteWebAgentTextEnv(service, seed=seed)
    view = service._view(remote.seed_view)
    assert view.prices == legacy.server.product_prices
    assert list(view.goals) == legacy.server.goals
    for index in (0, 50, 500):
        assert legacy.reset(index)[0] == remote.reset(index)[0]

        def step(action):
            before = remote.session
            left, right = legacy.step(action), remote.step(action)
            assert left[:3] == right[:3]
            if not left[2]:
                assert legacy.get_available_actions() == remote.get_available_actions()
                assert legacy.server.get_page_name(legacy.browser.current_url) == remote.page_type
            else:
                assert right[3]["webshop_session_id"] == before
                assert right[3]["page_type"] == "done"
                assert remote.session != before

        step("search[shoes]")
        step("click[invalid]")
        step("click[Next >]")
        step("click[< Prev]")
        asin = remote.episode.visible_asins[0]
        step(f"click[{asin}]")
        for values in service.data.product_item_dict[asin]["options"].values():
            if values:
                step(f"click[{values[0]}]")
        for subpage in ("Description", "Features"):
            step(f"click[{subpage}]")
            step("click[< Prev]")
        step("click[Buy Now]")
    remote.close()


def test_isolated_rng_eviction_and_reconstruction(service):
    before = random.getstate()
    first = service.get_seed_view(91)
    prices = copy.deepcopy(service._view(first).prices)
    goal = service.get_goal(first, 0)
    for seed in range(100, 104):
        service.get_seed_view(seed)
    assert 91 not in service._views
    assert service.get_seed_view(91) == first
    assert service.get_goal(first, 0) == goal
    assert service._view(first).prices == prices
    assert random.getstate() == before
    assert len(service._views) == 2
    assert service.health()["catalog_load_count"] == 1


def test_seeded_random_special_queries_and_pagination(service):
    ref = service.get_seed_view(42)
    first = service.search(ref, ("<r>",), 1, "episode:0")
    assert first == service.search(ref, ("<r>",), 1, "episode:0")
    assert first.visible_asins != service.search(ref, ("<r>",), 1, "episode:1").visible_asins
    product = service.data.all_products[0]
    for keywords, expected in (
        (("<a>", product["Attributes"][0]), [p for p in service.data.all_products if product["Attributes"][0] in p["Attributes"]]),
        (("<c>", product["category"]), [p for p in service.data.all_products if p["category"] == product["category"]]),
        (("<q>", product["query"]), [p for p in service.data.all_products if p["query"] == product["query"]]),
    ):
        result = service.search(ref, keywords, 1)
        assert result.visible_asins == tuple(p["asin"] for p in expected[:10])
        assert result.total == len(expected)


def test_concurrent_rendering_does_not_mutate_catalog_or_goals(service):
    ref = service.get_seed_view(0)
    goal = service.get_goal(ref, 0)
    asin = goal["asin"]
    before = pickle.dumps(service.data.product_item_dict[asin])
    goals_before = pickle.dumps(service._view(ref).goals)
    requests = [CatalogRequest("render_item", dict(seed_view=ref, goal={**goal, "instruction_text": f"private goal {i}"}, episode_id=f"episode-{i}", asin=asin, keywords=("shoes",), page=1)) for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(service.call, requests))
    for i, result in enumerate(results):
        assert f"private goal {i}<" in result.result.html
        assert result.request_id == requests[i].request_id
    assert pickle.dumps(service.data.product_item_dict[asin]) == before
    assert pickle.dumps(service._view(ref).goals) == goals_before


def test_typed_errors_bounds_and_catalog_keys(service):
    ref = service.get_seed_view(0)
    with pytest.raises(InvalidGoalIndexError):
        service.get_goal(ref, -1)
    with pytest.raises(UnknownProductError):
        service.get_scoring_products(ref, ("UNKNOWN",))
    with pytest.raises(InvalidPageStateError):
        service.get_scoring_products(ref, tuple(str(i) for i in range(12)))
    with pytest.raises(InvalidPageStateError):
        service.search(ref, ("<r>",))
    with pytest.raises(InvalidPageStateError):
        service.search(ref, (), 0)
    with pytest.raises(CatalogMismatchError):
        service.get_goal(replace(ref, catalog_key="wrong"), 0)
    with pytest.raises(ServiceOverloadedError):
        service.call_batch([CatalogRequest("health")] * (service.config["max_batch_size"] + 1))
    for _ in range(service.config["max_pending_requests"]):
        service._admission.acquire()
    try:
        with pytest.raises(ServiceOverloadedError):
            service.call(CatalogRequest("health"))
    finally:
        for _ in range(service.config["max_pending_requests"]):
            service._admission.release()


def test_fork_isolation_cross_seed_and_bounded_payload(service):
    source = RemoteWebAgentTextEnv(service, seed=0)
    target = RemoteWebAgentTextEnv(service, seed=42)
    source.reset(0)
    source.step("search[shoes]")
    source.step(f"click[{source.episode.visible_asins[0]}]")
    exported = source.export_episode(7)
    target.import_episode(exported)
    assert target.seed_view == source.seed_view
    assert target.observation == source.observation
    asin = source.episode.asin
    product = service.data.product_item_dict[asin]
    option = next(values[0] for values in product["options"].values() if values)
    target.step(f"click[{option}]")
    assert not source.episode.selected_options
    assert not exported.episode_state.selected_options
    assert target.episode.selected_options
    before = pickle.dumps(source.export_episode())
    service.data.product_item_dict["unrelated"] = {"large": "x" * 1000000}
    try:
        assert pickle.dumps(source.export_episode()) == before
        assert len(before) < 50000
    finally:
        del service.data.product_item_dict["unrelated"]
    assert not hasattr(source, "server")
    target.reset(0)
    assert target.seed_view.seed == 42
    target.import_episode(exported)
    source.step("search[<r>]")
    target.step("search[<r>]")
    assert source.episode.visible_asins == target.episode.visible_asins
    target.step("search[<r>]")
    assert source.episode.random_search_count == 1
    assert target.episode.random_search_count == 2
    source.close()
    target.close()


def test_scoring_subset_and_batch_order(service):
    ref = service.get_seed_view(0)
    asins = tuple(service.data.product_item_dict)[:3]
    requests = [CatalogRequest("get_scoring_products", dict(seed_view=ref, asins=(asin, asin))) for asin in asins]
    for request, response, asin in zip(requests, service.get_scoring_products_batch(requests), asins):
        assert response.request_id == request.request_id
        assert set(response.result["products"]) == {asin}
        response.result["products"][asin]["Title"] = "mutated response"
        assert service.data.product_item_dict[asin]["Title"] != "mutated response"


def test_browser_adapter_uses_existing_catalog(service):
    from bs4 import BeautifulSoup
    from web_agent_site.app import create_app

    app = create_app(service)
    browser = app.test_client()
    assert browser.get("/fixed_0").status_code == 200
    result = browser.post("/fixed_0", data={"search_query": "shoes"}, follow_redirects=True)
    assert result.status_code == 200
    soup = BeautifulSoup(result.data, "html.parser")
    link = soup.select_one(".product-link")["href"]
    item = browser.get(link)
    assert item.status_code == 200
    soup = BeautifulSoup(item.data, "html.parser")
    buy = next(form for form in soup.find_all("form") if "Buy Now" in form.get_text())
    assert browser.post(buy["action"]).status_code == 200
    assert service.health()["catalog_load_count"] == 1


def test_fingerprint_tracks_input_and_renderer(tmp_path, service, monkeypatch):
    import agent_system.environments.env_package.webshop.catalog_data as module

    config = dict(service.config)
    attributes = tmp_path / "attributes.json"
    attributes.write_bytes(open(config["attr_path"], "rb").read())
    config["attr_path"] = str(attributes)
    assert WebshopCatalogData.load(config).metadata.catalog_key == service.metadata().catalog_key
    attributes.write_text(attributes.read_text() + "\n")
    changed = WebshopCatalogData.load(config).metadata.catalog_key
    assert changed != service.metadata().catalog_key
    monkeypatch.setattr(module, "RENDERER_VERSION", "test-renderer")
    assert WebshopCatalogData.load(config).metadata.catalog_key != changed


def test_treehca_snapshot_subset_and_hypothetical_page_parity(service):
    from types import SimpleNamespace

    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from treehca.training_webshop_env import TreeHCAWebshopWorker
    from treehca.webshop_probability_snapshot import WebshopSnapshotSource

    legacy = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=False, seed=0)
    remote = RemoteWebAgentTextEnv(service, seed=0)
    legacy.reset(0)
    remote.reset(0)
    worker = object.__new__(TreeHCAWebshopWorker)
    worker.env = remote
    formatter = WebshopEnvironmentManager(None, None, SimpleNamespace(env=SimpleNamespace(history_length=0)))
    formatter.tasks = [remote.instruction_text]
    formatter.memory.reset(batch_size=1)
    source = WebshopSnapshotSource(legacy.server)
    actions = ["search[shoes]"]
    previous = ""
    for action in actions:
        legacy.step(action)
        remote.step(action)
    asin = remote.episode.visible_asins[0]
    for action in [None, f"click[{asin}]", "click[Description]", "click[< Prev]", "click[Back to Search]"]:
        if action:
            previous = remote.page_type
            legacy.step(action)
            remote.step(action)
        prompt = formatter.build_text_obs(formatter.format_obs([remote.observation]), [{"available_actions": remote.get_available_actions()}], init=True)[0]
        payload = worker.scoring_payload(prompt, remote.instruction_text, [], 0, 0, previous)
        expected = source.capture(legacy, formatter, prompt, previous_page_type=previous)
        assert replace(payload["snapshot"], catalog_key=expected.catalog_key) == expected
        expected_asins = set(expected.visible_asins) | ({expected.asin} if expected.asin else set())
        assert set(payload["products"]) == expected_asins
        assert payload["products"] == {a: legacy.server.product_item_dict[a] for a in expected_asins}
        assert payload["prices"] == {a: legacy.server.product_prices[a] for a in expected_asins}
        if remote.page_type == "search_results":
            subset_source = WebshopSnapshotSource(SimpleNamespace(product_item_dict=payload["products"], product_prices=payload["prices"], show_attrs=payload["show_attrs"]))
            subset_source.catalog_key = payload["snapshot"].catalog_key
            hypothetical = subset_source.product_entry(payload["snapshot"], asin)
            native_hypothetical = source.product_entry(expected, asin)
            assert replace(hypothetical, catalog_key=expected.catalog_key) == native_hypothetical
    remote.close()


def test_wrapper_full_reward_and_terminal_episode_identity(service):
    from treehca.training_webshop_env import TreeHCAWebshopWorker

    worker = object.__new__(TreeHCAWebshopWorker)
    worker.env = RemoteWebAgentTextEnv(service, seed=0)
    index = next(i for i, goal in enumerate(service._view(worker.env.seed_view).goals) if not goal["goal_options"])
    worker.reset(index)
    goal = worker.env.episode.goal
    worker.step(f"search[<q> {goal['query']}]")
    # Find the native target on its actual results page.
    while goal["asin"] not in worker.env.episode.visible_asins:
        worker.step("click[Next >]")
    worker.step(f"click[{goal['asin']}]")
    episode_id = worker.env.session
    _, reward, done, info = worker.step("click[Buy Now]")
    assert done and reward == 10.0
    assert info["won"] and info["task_score"] == 1.0
    assert info["page_type"] == "done"
    assert info["webshop_session_id"] == episode_id != worker.env.session
    assert info["webshop_task_id"] == index
    worker.close()


def test_human_goal_seed_parity():
    config = resolve_catalog_config({"num_products": 1000, "human_goals": True})
    data = WebshopCatalogData.load(config)
    state = random.getstate()
    view = data.seed_view(42)
    assert random.getstate() == state
    legacy = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=True, seed=42)
    assert view.prices == legacy.server.product_prices
    assert list(view.goals) == legacy.server.goals


def test_configuration_and_index_fail_before_worker_creation(service, tmp_path, monkeypatch):
    from web_agent_site.engine import engine

    from agent_system.environments.env_package.webshop.catalog_service import ConfigurationError

    with pytest.raises(ConfigurationError, match="Missing file_path"):
        resolve_catalog_config({"file_path": str(tmp_path / "missing.json")})
    with pytest.raises(ConfigurationError, match="limit_goals"):
        resolve_catalog_config({"limit_goals": 0})
    with pytest.raises(ConfigurationError, match="search_concurrency"):
        resolve_catalog_config({}, {"search_concurrency": 2})
    monkeypatch.setattr(WebshopCatalogData, "load", lambda config: service.data)
    from types import SimpleNamespace

    monkeypatch.setattr(engine, "init_search_engine", lambda *args, **kwargs: SimpleNamespace(num_docs=0))
    with pytest.raises(ConfigurationError, match="empty"):
        WebshopCatalogService(service.config)


def test_attachment_requires_exact_config_and_fingerprint(service, monkeypatch):
    from types import SimpleNamespace

    import ray

    from agent_system.environments.env_package.webshop.catalog_service import start_catalog_service

    actor = SimpleNamespace(call=SimpleNamespace(remote=service.call))
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_actor", lambda *args, **kwargs: actor)
    monkeypatch.setattr(ray, "get", lambda value, **kwargs: value)
    settings = {"attach": True, "actor_name": "existing-service", "seed_view_cache_size": 2}
    assert start_catalog_service({"num_products": 1000}, settings) is actor
    with pytest.raises(CatalogMismatchError):
        start_catalog_service({"num_products": 1000, "show_attrs": True}, settings)
    import agent_system.environments.env_package.webshop.catalog_service as module

    monkeypatch.setattr(module, "catalog_fingerprint", lambda config: ("changed-content", {}))
    with pytest.raises(CatalogMismatchError):
        start_catalog_service({"num_products": 1000}, settings)


@pytest.mark.parametrize("shuffle_goals", [False, True])
@pytest.mark.parametrize("measure_bytes", [False, True])
def test_optional_seed_shuffle_and_serialization(service, monkeypatch, shuffle_goals, measure_bytes):
    from web_agent_site.engine import engine
    from web_agent_site.engine.goal import get_goals

    from agent_system.environments.env_package.webshop.catalog_data import WebshopSeedView
    from agent_system.environments.env_package.webshop.catalog_service import SeedViewNotFoundError

    # Reuse the fixture's catalog/searcher to test service settings without another load.
    monkeypatch.setattr(WebshopCatalogData, "load", lambda config: service.data)
    monkeypatch.setattr(engine, "init_search_engine", lambda *args, **kwargs: service.search_engine)
    configured = WebshopCatalogService(resolve_catalog_config({"num_products": 1000}, {"shuffle_goals": shuffle_goals, "measure_seed_view_bytes": measure_bytes, "seed_view_cache_size": 1}))
    shuffled = service.data.seed_view(42)
    rng = random.Random(42)
    prices = engine.generate_product_prices(service.data.all_products, rng=rng)
    ordered_goals = get_goals(service.data.all_products, prices, False, rng=rng)
    original_shuffle, original_dumps = random.Random.shuffle, pickle.dumps
    shuffle_calls, serialized_views = [], []

    def shuffle(rng, values):
        shuffle_calls.append(len(values))
        return original_shuffle(rng, values)

    def dumps(value, *args, **kwargs):
        if isinstance(value, WebshopSeedView):
            serialized_views.append(value.ref.seed)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(random.Random, "shuffle", shuffle)
    monkeypatch.setattr(pickle, "dumps", dumps)
    ref = configured.get_seed_view(42)
    view = configured._view(ref)
    assert view.prices == prices == shuffled.prices
    assert view.goals == (shuffled.goals if shuffle_goals else tuple(ordered_goals))
    assert ref.price_profile_key == shuffled.ref.price_profile_key
    assert (ref.goal_set_key == shuffled.ref.goal_set_key) == shuffle_goals
    assert len(shuffle_calls) == int(shuffle_goals)
    assert serialized_views == ([42] if measure_bytes else [])
    assert (configured.health()["seed_view_bytes"] is not None) == measure_bytes
    # Cache hits do neither operation; reconstruction preserves the selected order/key.
    assert configured.get_seed_view(42) == ref
    configured.get_seed_view(43)
    assert configured.get_seed_view(42) == ref
    assert configured._view(ref).goals == view.goals
    if not shuffle_goals:
        assert not shuffle_calls
    if not measure_bytes:
        assert not serialized_views
    if not shuffle_goals:
        with pytest.raises(SeedViewNotFoundError, match="Seed view identity"):
            configured.get_goal(shuffled.ref, 0)


@pytest.mark.parametrize("human_goals", [False, True])
def test_fixed_human_validation_and_training_holdout(human_goals):
    from web_agent_site.engine.engine import generate_product_prices
    from web_agent_site.engine.goal import get_goals

    validation = dict(seed=233, shuffle_seed=233, count=5)
    service = WebshopCatalogService(resolve_catalog_config(dict(num_products=1000, human_goals=human_goals, validation=validation), dict(seed_view_cache_size=1, shuffle_goals=False, measure_seed_view_bytes=False)))
    ref = service.get_seed_view(42, split="validation")
    view = service._view(ref)
    rng = random.Random(233)
    prices = generate_product_prices(service.data.all_products, rng=rng)
    canonical = get_goals(service.data.all_products, prices, True, rng=rng)
    random.Random(233).shuffle(canonical)
    assert list(view.goals) == canonical[:5]
    assert view.prices == prices
    assert ref == service.get_seed_view(9999, split="validation")
    assert ref == service.data.seed_view(7, split="validation", shuffle_goals=False).ref

    def identity(goal):
        return (goal["asin"], goal["instruction_text"].split(", and price lower than")[0], json.dumps(goal["goal_options"], sort_keys=True))

    excluded = {identity(g) for g in view.goals}
    excluded_asins = {g["asin"] for g in view.goals}
    for seed in (42, 233, 1001):
        train = service._view(service.get_seed_view(seed))
        assert train.ref != ref
        assert not excluded.intersection(identity(g) for g in train.goals)
        if not human_goals:
            assert not excluded_asins.intersection(g["asin"] for g in train.goals)
    assert service._view(ref) is view  # Pinned outside the training LRU.
    assert service.health()["metrics"]["validation_view_builds"] == 1
    assert service.health()["validation_goal_count"] == 5

    legacy = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=human_goals, seed=9999, validation=validation, goal_split="validation")
    remote = RemoteWebAgentTextEnv(service, seed=8888, goal_split="validation")
    assert legacy.server.goals == list(view.goals)
    assert legacy.server.product_prices == view.prices
    legacy_train = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=human_goals, seed=42, validation=validation, goal_split="train")
    assert legacy_train.server.goals == list(service.data.seed_view(42).goals)
    assert legacy.reset(2)[0] == remote.reset(2)[0]
    exported = remote.export_episode()
    clone = RemoteWebAgentTextEnv(service, seed=42)
    clone.import_episode(exported)
    assert clone.export_episode() == exported
    remote.close()
    clone.close()


def test_fixed_validation_rejects_undersized_catalog(service):
    data = replace(service.data, validation=dict(seed=233, shuffle_seed=233, count=500))
    # The synthetic-only fixture has no human instructions loaded.
    with pytest.raises(ValueError, match="Use the full catalog"):
        data.seed_view(42, split="validation")


@pytest.mark.parametrize("backend", ["legacy", "centralized"])
def test_training_factory_uses_independent_validation_config(monkeypatch, backend):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from agent_system.environments import env_manager
    from agent_system.environments.env_package import webshop
    from agent_system.environments.env_package.webshop import catalog_service

    calls, rpc_calls = [], []

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    class Client:
        def __init__(self, *args):
            pass

        def call(self, operation, **kwargs):
            rpc_calls.append((operation, kwargs))

    monkeypatch.setattr(webshop, "build_webshop_envs", build)
    monkeypatch.setattr(env_manager, "WebshopEnvironmentManager", lambda env, *args: env)
    monkeypatch.setattr(catalog_service, "start_catalog_service", lambda *args: object())
    monkeypatch.setattr(catalog_service, "CatalogClient", Client)
    config = OmegaConf.create(
        dict(env=dict(env_name="webshop", seed=91, rollout=dict(n=2), resources_per_worker={}, webshop=dict(use_small=False, backend=backend, human_goals=False, catalog_service=dict(startup_smoke_test=False))), data=dict(train_batch_size=2, val_batch_size=500), algorithm=dict(adv_estimator="grpo"))
    )
    env_manager.make_envs(config)
    train, val = calls
    assert train["seed"] == 91 and val["seed"] == 233
    assert train["env_kwargs"]["goal_split"] == "train"
    assert val["env_kwargs"]["goal_split"] == "validation"
    assert val["env_kwargs"]["validation"] == dict(seed=233, shuffle_seed=233, count=500)
    assert train["env_kwargs"]["validation"] == val["env_kwargs"]["validation"]
    if backend == "centralized":
        assert train["env_kwargs"]["catalog_service"] is val["env_kwargs"]["catalog_service"]
        assert rpc_calls[0] == ("get_seed_view", dict(seed=233, split="validation"))
