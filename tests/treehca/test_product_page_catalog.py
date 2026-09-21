"""One catalog-wide case exercising the real WebShop page/prompt formatting path."""

import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from treehca.product_page_parser import ProductOptionGroup, extract_product_page_contexts, parse_product_page_batch, parse_product_page_fields

_ROOT = Path(__file__).resolve().parents[2]
_WEBSHOP = _ROOT / "agent_system/environments/env_package/webshop/webshop"
_CATALOG = _WEBSHOP / "data/items_shuffle_1000.json"
_CONTROLS = ("Back to Search", "< Prev", "Description", "Features", "Buy Now")


def test_all_1000_product_pages(monkeypatch):
    """Render every item and verify both parsers, with and without prompt history.

    The independent oracle and catalog anomalies are described
    in docs/treehca/product_page_catalog_test.md.
    """
    monkeypatch.syspath_prepend(str(_WEBSHOP))
    from web_agent_site.engine import engine
    from web_agent_site.envs import web_agent_text_env

    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS
    from agent_system.memory import SimpleMemory

    # Ensure the renderer and loader come from this checkout rather than another WebShop installation.
    assert Path(engine.__file__).resolve() == _WEBSHOP / "web_agent_site/engine/engine.py"
    assert Path(web_agent_text_env.__file__).resolve() == _WEBSHOP / "web_agent_site/envs/web_agent_text_env.py"
    source_items = json.loads(_CATALOG.read_text())
    assert len(source_items) == 1000
    random_state = random.getstate()
    try:
        products, _, _, _ = engine.load_products(str(_CATALOG), str(_WEBSHOP / "data/items_ins_v2_1000.json"), human_goals=True)
    finally:
        # Loading also samples reward prices, which are unused here; do not perturb other tests' RNG.
        random.setstate(random_state)
    assert [product["asin"] for product in products] == [item["asin"] for item in source_items], "The loader must not silently skip, reorder, or deduplicate catalog rows"

    # Rendering/formatting need no search index, running server, model, or environment reset.
    # Bypass constructors, but call the unchanged production methods (see the test documentation).
    text_env = object.__new__(web_agent_text_env.WebAgentTextEnv)
    text_env.observation_mode = "text"
    manager = object.__new__(WebshopEnvironmentManager)
    manager.config = SimpleNamespace(env=SimpleNamespace(history_length=0))
    manager.memory = SimpleMemory()
    manager.memory.reset(len(products))
    tasks = [f"Find the product with catalog identifier {product['asin']}." for product in products]
    raw_observations, infos, expected_products, expected_actions = [], [], [], []
    repeated_options = {}

    with web_agent_text_env.app.test_request_context("/"):
        for product, task in zip(products, tasks):
            html = engine.map_action_to_html(
                "click",
                session_id="parser-catalog-test",
                product_info=product,
                keywords=["catalog"],
                page=1,
                asin=product["asin"],
                options={},
                instruction_text=task,
                show_attrs=False,
            )
            text_env.browser = SimpleNamespace(page_source=html, current_url="/item_page/parser-catalog-test")
            text_env.instruction_text = task
            raw_observations.append(text_env.observation)
            infos.append({"available_actions": text_env.get_available_actions()})

            # Oracle: structured loader output, never fragments recovered from the rendered page.
            expected_products.append(
                {
                    "navigation_controls": ("Back to Search", "< Prev"),
                    "option_groups": tuple({"name": name, "values": tuple(dict.fromkeys(values)), "correct_options": ()} for name, values in product["options"].items()),
                    "title": product["Title"],
                    "price_text": product["Price"],
                    "rating_text": product["Rating"],
                    "detail_page_controls": ("Description", "Features"),
                    "purchase_control": "Buy Now",
                }
            )
            commands = [f"click[{control.lower()}]" for control in _CONTROLS]
            commands.extend(f"click[{value}]" for values in product["options"].values() for value in values)
            # WebShop stores clickables by command target, so repeated values share a command.
            expected_actions.append(tuple(dict.fromkeys(commands)))
            counts = Counter(fragment for name, values in product["options"].items() for fragment in (name, *values))
            duplicates = {fragment: count for fragment, count in counts.items() if count > 1}
            if duplicates:
                repeated_options[product["asin"]] = duplicates

    # These expectations come from the catalog data, not from observing parser failures.
    assert repeated_options == {
        "B08G14B779": {"26 inch x 10 inch": 2},
        "B099WH1RTM": {"12x18 inch": 2},
    }
    missing_titles = {product["asin"] for product in products if not product["Title"]}
    assert missing_titles == {"B09P71WY8C"}
    # Real rollouts retain tasks from the initial search page; extract_task is not used on product pages.
    manager.tasks = tasks
    observations = manager.format_obs(raw_observations)
    contexts_without_history = manager.build_text_obs(observations, infos)

    # Controlled history exercises extraction without requiring a real shopping trajectory.
    past_actions = [f"search[{product['asin']}]" for product in products]
    manager.memory.store({"text_obs": ["'Search'"] * len(products), "action": past_actions})
    manager.config.env.history_length = 1
    contexts_with_history = manager.build_text_obs(observations, infos)
    contexts = contexts_without_history + contexts_with_history
    parsed_contexts = extract_product_page_contexts(contexts)
    assert len(parsed_contexts) == 2000

    failures, successful_parses = [], 0
    for index, parts in enumerate(parsed_contexts):
        row = index % len(products)
        with_history = index >= len(products)
        asin = products[row]["asin"]
        label = f"{asin}, {'with' if with_history else 'without'} history"
        template = WEBSHOP_TEMPLATE if with_history else WEBSHOP_TEMPLATE_NO_HIS
        history_block = None
        if with_history:
            history_block = f"Prior to this step, you have already taken 1 step(s). Below are the most recent 1 observations and the corresponding actions you took: [Observation 1: ''Search'', Action 1: '{past_actions[row]}']"
        expected_context = {
            "agent_introduction": template.strip().splitlines()[0].rstrip(),
            "shopping_task": tasks[row],
            "history_block": history_block,
            "completed_steps": 1 if with_history else None,
            "history_length": 1 if with_history else None,
            "current_step": 2 if with_history else None,
            "current_observation": observations[row],
            "admissible_actions": expected_actions[row],
            "response_instructions": template[template.index("Now it's your turn") :].rstrip("\n"),
        }
        for field, expected in expected_context.items():
            actual = getattr(parts, field)
            if actual != expected:
                failures.append(f"{label}: context.{field}: expected {expected!r}, got {actual!r}")
        try:
            fields = parse_product_page_fields(parts.current_observation, parts.admissible_actions)
        except ValueError as error:
            failures.append(f"{label}: unexpected product parse rejection: {error}")
            continue
        actual_fields = asdict(fields)
        for field, expected in expected_products[row].items():
            if actual_fields[field] != expected:
                failures.append(f"{label}: product.{field}: expected {expected!r}, got {actual_fields[field]!r}")
        successful_parses += 1

    if failures:
        pytest.fail("\n".join(failures))
    assert successful_parses == 2000

    # The tolerant wrapper must report exactly the known failures without losing or shifting rows.
    results = parse_product_page_batch(contexts)
    assert len(results) == len(contexts)
    assert all(result.error_stage is None for result in results)
    for index, result in enumerate(results):
        row = index % len(products)
        asin = products[row]["asin"]
        assert result.index == index
        assert result.context_parts == parsed_contexts[index], (asin, index)
        assert result.error_stage is None and result.error_message is None, (asin, index)
        assert asdict(result.product_fields) == expected_products[row], (asin, index)


def test_catalog_canonicalizes_option_values_before_rendering(monkeypatch, tmp_path):
    """Option values must match after HTML text extraction and action discovery."""
    monkeypatch.syspath_prepend(str(_WEBSHOP))
    from web_agent_site.engine import engine
    from web_agent_site.envs import web_agent_text_env

    from agent_system.environments.env_manager import WebshopEnvironmentManager

    asin = "B000000001"
    task = "Find the example product."
    source_values = ("/Blue", "Antique White+acacia Wood/", "5 Pk-black/Navy/White/")
    expected_values = ("| blue", "antique white+acacia wood |", "5 pk-black | navy | white |")
    shared_value = "shared value"
    source_product = {
        "asin": asin,
        "category": "test",
        "query": "example",
        "product_category": "Test",
        "name": "Example product",
        "full_description": "",
        "small_description": [],
        "pricing": "$1.00",
        "customization_options": {
            "Removed Group": [{"value": shared_value, "image": None}],
            "Style": [{"value": shared_value, "image": None}, {"value": "Style Only", "image": None}],
            "Color": [{"value": value, "image": None} for value in (*source_values, shared_value)],
        },
        "images": [""],
    }
    catalog_path = tmp_path / "catalog.json"
    attributes_path = tmp_path / "attributes.json"
    catalog_path.write_text(json.dumps([source_product]))
    attributes_path.write_text(json.dumps({asin: {"attributes": [], "instruction": task, "instruction_attributes": []}}))

    products, _, _ = engine.load_catalog_products(str(catalog_path), str(attributes_path), human_goals=False)
    product = products[0]
    assert product["options"] == {
        "style": ["style only"],
        "color": [*expected_values, shared_value],
    }

    with web_agent_text_env.app.test_request_context("/"):
        html = engine.map_action_to_html(
            "click",
            session_id="parser-boundary-slashes",
            product_info=product,
            keywords=["example"],
            page=1,
            asin=asin,
            options={},
            instruction_text=task,
            show_attrs=False,
        )
    text_env = object.__new__(web_agent_text_env.WebAgentTextEnv)
    text_env.observation_mode = "text"
    text_env.browser = SimpleNamespace(page_source=html, current_url="/item_page/parser-boundary-slashes")
    text_env.instruction_text = task
    manager = object.__new__(WebshopEnvironmentManager)
    manager.tasks = [task]
    observation = manager.format_obs([text_env.observation])[0]
    actions = manager.format_avail_actions(text_env.get_available_actions())

    fields = parse_product_page_fields(observation, actions)
    assert fields.option_groups == (
        ProductOptionGroup("style", ("style only",)),
        ProductOptionGroup("color", (*expected_values, shared_value)),
    )
