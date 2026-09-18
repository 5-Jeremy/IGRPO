"""Contract tests for raw-context and main product-page extraction."""

import json
import runpy
from pathlib import Path

import pytest

from treehca.product_page_parser import ProductOptionGroup, extract_product_page_contexts, parse_product_page_batch, parse_product_page_fields

_ROOT = Path(__file__).resolve().parents[2]
# Read only the prompt constants, avoiding environment/GPU imports in unit tests.
_TEMPLATES = runpy.run_path(str(_ROOT / "agent_system/environments/prompts/webshop.py"))
_SAVED_CONTEXTS = json.loads((Path(__file__).parent / "fixtures/product_page_contexts.json").read_text())
_CONTROLS = ("Back to Search", "< Prev", "Description", "Features", "Buy Now")


def make_page(groups=(), *, title="A shopper's brush", price="$11.89", rating="N.A.", attributes=False):
    fragments = ["Back to Search", "< Prev"]
    option_actions = []
    for name, values in groups:
        fragments.extend((name, *values))
        option_actions.extend(f"click[{value}]" for value in values)
    fragments.extend((title, f"Price: {price}", f"Rating: {rating}", "Description", "Features"))
    if attributes:
        fragments.append("Attributes")
    fragments.append("Buy Now")
    controls = _CONTROLS + (("Attributes",) if attributes else ())
    actions = tuple(f"click[{control.lower()}]" for control in controls) + tuple(option_actions)
    return " [SEP] ".join(f"'{fragment}'" for fragment in fragments), actions


def make_context(observation, actions, *, history=None, task="Find a brush."):
    fields = {
        "task_description": task,
        "current_observation": observation,
        "available_actions": "\n".join(f"'{action}'," for action in actions),
    }
    if history is None:
        return _TEMPLATES["WEBSHOP_TEMPLATE_NO_HIS"].format(**fields)
    return _TEMPLATES["WEBSHOP_TEMPLATE"].format(**fields, step_count=11, history_length=2, current_step=12, action_history=history)


def test_extract_no_history_preserves_task_punctuation_and_observation():
    observation, actions = make_page()
    context = make_context(observation, actions, task="Find a shopper's brush.\nUnder $20.")
    parts = extract_product_page_contexts([context])[0]
    assert parts.shopping_task == "Find a shopper's brush.\nUnder $20."
    assert parts.current_observation == observation
    assert parts.admissible_actions == actions
    assert parts.history_block is None
    assert (parts.completed_steps, parts.history_length, parts.current_step) == (None, None, None)
    assert parts.agent_introduction == "You are an expert autonomous agent operating in the WebShop e‑commerce environment."
    assert parts.response_instructions == context[context.index("Now it's your turn") :].rstrip("\n")


def test_extract_history_is_separate_from_current_observation():
    old_observation, _ = make_page(title="Previous product")
    observation, actions = make_page(title="Current product")
    history = f"[Observation 10: '{old_observation}', Action 10: 'click[description]']\n[Observation 11: 'Description text', Action 11: 'click[< prev]']"
    parts = extract_product_page_contexts([make_context(observation, actions, history=history)])[0]
    assert parts.history_block == ("Prior to this step, you have already taken 11 step(s). Below are the most recent 2 observations and the corresponding actions you took: " + history)
    assert (parts.completed_steps, parts.history_length, parts.current_step) == (11, 2, 12)
    assert parts.current_observation == observation
    assert "Previous product" not in parts.current_observation


def test_extract_batch_order_and_empty_batch():
    observation, actions = make_page()
    contexts = [make_context(observation, actions, task=task) for task in ("First", "Second")]
    assert [part.shopping_task for part in extract_product_page_contexts(contexts)] == ["First", "Second"]
    assert extract_product_page_contexts([]) == []


def test_extract_batch_error_identifies_index():
    observation, actions = make_page()
    with pytest.raises(ValueError, match="batch index 1"):
        extract_product_page_contexts([make_context(observation, actions), "broken"])


@pytest.mark.parametrize("context", [None, 42, "system\nuser\nbody\nassistant"])
def test_extract_rejects_invalid_raw_context(context):
    with pytest.raises(ValueError, match="batch index 0"):
        extract_product_page_contexts([context])


def test_extract_rejects_a_string_instead_of_a_batch():
    with pytest.raises(ValueError, match="batch"):
        extract_product_page_contexts("a single context")


@pytest.mark.parametrize(
    "old,new",
    [
        ("Your current observation is: ", "Observation: "),
        ("Your admissible actions of the current situation are:", "Available actions:"),
        ("'click[buy now]',", "'click[buy now]'"),
        ("'click[buy now]',", "'purchase[buy now]',"),
        ("'Buy Now'.\nYour admissible", "'Buy Now'\nYour admissible"),
        ("Now it's your turn", "Choose an action"),
    ],
)
def test_extract_rejects_malformed_sections(old, new):
    observation, actions = make_page()
    context = make_context(observation, actions)
    assert old in context
    with pytest.raises(ValueError, match="batch index 0"):
        extract_product_page_contexts([context.replace(old, new)])


def test_extract_rejects_ambiguous_section_boundaries():
    observation, actions = make_page()
    context = make_context(observation, actions, task="First line\nYour current observation is: conflicting marker")
    with pytest.raises(ValueError, match="current-observation boundary, found 2"):
        extract_product_page_contexts([context])


def test_extract_rejects_inconsistent_step_counts():
    observation, actions = make_page()
    context = make_context(observation, actions, history="[Observation 11: 'page', Action 11: 'click[< prev]']")
    with pytest.raises(ValueError, match="Inconsistent"):
        extract_product_page_contexts([context.replace("at step 12", "at step 15")])


def test_extract_does_not_classify_page_from_history():
    observation, actions = make_page()
    context = make_context("'Back to Search' [SEP] 'Page 1 (Total results: 0)'", ("click[back to search]",), history=f"[Observation 11: '{observation}', Action 11: 'click[back to search]']")
    parts = extract_product_page_contexts([context])[0]
    assert "Buy Now" not in parts.current_observation
    with pytest.raises(ValueError, match="navigation"):
        parse_product_page_fields(parts.current_observation, parts.admissible_actions)


@pytest.mark.parametrize("example", _SAVED_CONTEXTS, ids=lambda example: example["name"])
def test_saved_product_page_examples(example):
    parts = extract_product_page_contexts([example["raw_context"]])[0]
    product = parse_product_page_fields(parts.current_observation, parts.admissible_actions)
    assert product.navigation_controls == ("Back to Search", "< Prev")
    assert product.detail_page_controls == ("Description", "Features")
    assert product.purchase_control == "Buy Now"
    assert product.rating_text == "N.A."
    assert parts.history_length == 2
    if example["name"] == "one_option_group":
        assert parts.current_step == 12
        assert product.option_groups == (ProductOptionGroup("size", ("0.6mm", "0.7mm", "0.8mm", "1.0mm", "1.2mm")),)
        assert product.title.endswith("Tight 0.8mm")
        assert product.price_text == "$11.89"
        # Prior selections do not alter the available values or the displayed title.
        assert "Action 10: 'click[0.7mm]'" in parts.history_block
        assert "Action 11: 'click[0.7mm]'" in parts.history_block
    elif example["name"] == "two_option_groups":
        assert parts.current_step == 5
        assert [group.name for group in product.option_groups] == ["size", "color"]
        assert [len(group.values) for group in product.option_groups] == [5, 14]
        assert "christmasgoo3302" in product.option_groups[1].values
        assert product.price_text == "$15.9"
    else:
        assert parts.current_step == 10
        assert product.option_groups == ()
        assert product.title.endswith("Hupo Guoba Snacks 760g/bag")
        assert product.price_text == "$43.99"


def test_parse_no_options_and_optional_attributes():
    observation, actions = make_page(attributes=True, rating="4.7", price="$11.89 to $20.00")
    product = parse_product_page_fields(observation, actions)
    assert product.option_groups == ()
    assert product.detail_page_controls == ("Description", "Features", "Attributes")
    assert product.price_text == "$11.89 to $20.00"
    assert product.rating_text == "4.7"


def test_parse_preserves_unicode_quotes_backslashes_and_option_case():
    groups = (("cut", ("Women's Tall", r"C:\sizes\XL")), ("色", ("藍", "White")))
    title = "'Quoted' shopper's \"brush\"."
    observation, actions = make_page(groups, title=title)
    parts = extract_product_page_contexts([make_context(observation, actions)])[0]
    product = parse_product_page_fields(parts.current_observation, parts.admissible_actions)
    assert product.title == title
    assert product.option_groups == tuple(ProductOptionGroup(name, values) for name, values in groups)
    assert parts.admissible_actions == actions


def test_parse_structural_looking_option_values_using_suffix():
    groups = (("label", ("Price: $1", "Rating: N.A.")),)
    observation, actions = make_page(groups, title="Price: Brush")
    product = parse_product_page_fields(observation, actions)
    assert product.option_groups == (ProductOptionGroup("label", ("Price: $1", "Rating: N.A.")),)
    assert product.title == "Price: Brush"
    assert product.price_text == "$11.89"


@pytest.mark.parametrize(
    "groups,reason",
    [
        ((("size", ()),), "no recognized clickable values"),
        ((("size", ("size", "large")),), "collides with an admissible action"),
        ((("size", ("small",)), ("fit", ("small",))), "Duplicate admissible actions"),
        ((("style", ("buy now",)),), "Duplicate admissible actions"),
        ((("style", ("Buy Now",)),), "collides with a page control"),
    ],
)
def test_parse_rejects_ambiguous_option_groups(groups, reason):
    observation, actions = make_page(groups)
    with pytest.raises(ValueError, match=reason):
        parse_product_page_fields(observation, actions)


def test_parse_collapses_repeated_values_within_one_group():
    observation, actions = make_page((("size", ("small", "small", "large")),))
    product = parse_product_page_fields(observation, tuple(dict.fromkeys(actions)))
    assert product.option_groups == (ProductOptionGroup("size", ("small", "large")),)


@pytest.mark.parametrize("bad_action", ["search[brush]", "click[unexplained value]"])
def test_parse_rejects_actions_not_explained_by_page(bad_action):
    observation, actions = make_page()
    with pytest.raises(ValueError, match="Actions do not match"):
        parse_product_page_fields(observation, (*actions, bad_action))


def test_parse_rejects_missing_control_action():
    observation, actions = make_page()
    with pytest.raises(ValueError, match="Missing page-control actions"):
        parse_product_page_fields(observation, [action for action in actions if action != "click[buy now]"])


def test_parse_rejects_missing_option_action():
    observation, actions = make_page((("size", ("small",)),))
    with pytest.raises(ValueError, match="no recognized clickable values"):
        parse_product_page_fields(observation, [action for action in actions if action != "click[small]"])


@pytest.mark.parametrize(
    "old,new",
    [
        ("'Back to Search'", "Back to Search"),
        (" [SEP] ", "[SEP]"),
        ("'Description' [SEP] ", ""),
        ("'Buy Now'", "'Purchase'"),
        ("'Price: $11.89'", "'Price: '"),
        ("'Rating: N.A.'", "'Rating: '"),
        ("'Price: $11.89' [SEP] 'Rating: N.A.'", "'Rating: N.A.' [SEP] 'Price: $11.89'"),
        ("'A shopper's brush'", "''"),
    ],
)
def test_parse_rejects_malformed_product_observation(old, new):
    observation, actions = make_page()
    with pytest.raises(ValueError):
        parse_product_page_fields(observation.replace(old, new), actions)


def test_batch_reports_both_failure_stages_and_preserves_successful_rows():
    observation, actions = make_page((("size", ("small", "large")),), title="Valid brush")
    valid_context = make_context(observation, actions)
    duplicate_observation, duplicate_actions = make_page((("size", ("small", "small")),))
    # Match WebShop's deduplicated action list while retaining repeated display values.
    duplicate_context = make_context(duplicate_observation, tuple(dict.fromkeys(duplicate_actions)))
    untitled_observation, untitled_actions = make_page(title="Remove this title")
    missing_title_context = make_context(untitled_observation.replace("'Remove this title' [SEP] ", ""), untitled_actions)
    history = "[Observation 11: 'page', Action 11: 'click[< prev]']"
    history_context = make_context(observation, actions, history=history)
    contexts = ["broken", valid_context, duplicate_context, missing_title_context, history_context, None, valid_context]

    results = parse_product_page_batch(contexts)

    assert len(results) == len(contexts)
    assert [result.index for result in results] == list(range(len(contexts)))
    assert [result.index for result in results if result.error_stage is not None] == [0, 5]
    for index in (0, 5):
        assert results[index].error_stage == "context"
        assert results[index].error_message
        assert results[index].context_parts is None
        assert results[index].product_fields is None
    assert results[2].product_fields.option_groups == (ProductOptionGroup("size", ("small",)),)
    assert results[3].product_fields.title == ""
    for index in (1, 2, 3, 4, 6):
        assert results[index].error_stage is None
        assert results[index].error_message is None
    for index in (1, 4, 6):
        assert results[index].error_stage is None
        assert results[index].error_message is None
        assert results[index].context_parts.current_observation == observation
        assert results[index].product_fields.title == "Valid brush"
        assert results[index].product_fields.option_groups == (ProductOptionGroup("size", ("small", "large")),)
    assert results[1].context_parts.history_block is None
    assert results[4].context_parts.history_block.endswith(history)
    assert results[6].context_parts.history_block is None


def test_batch_returns_every_failure_when_all_rows_are_invalid():
    results = parse_product_page_batch(["broken", "also broken"])
    assert [result.index for result in results] == [0, 1]
    assert all(result.error_stage == "context" and result.error_message for result in results)
    assert all(result.context_parts is None and result.product_fields is None for result in results)


def test_batch_accepts_empty_batch():
    assert parse_product_page_batch([]) == []


@pytest.mark.parametrize("invalid_batch", ["a single context", b"a single context"])
def test_batch_rejects_single_strings(invalid_batch):
    with pytest.raises(ValueError, match="batch of strings"):
        parse_product_page_batch(invalid_batch)


@pytest.mark.parametrize("stage_function", ["_extract_context", "parse_product_page_fields"])
def test_batch_does_not_swallow_unexpected_exceptions(monkeypatch, stage_function):
    observation, actions = make_page()

    def fail_unexpectedly(*args):
        raise RuntimeError("unexpected implementation failure")

    # Inject a programming failure to ensure the wrapper catches only recognized parsing errors.
    monkeypatch.setattr(f"treehca.product_page_parser.{stage_function}", fail_unexpectedly)
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        parse_product_page_batch([make_context(observation, actions)])
