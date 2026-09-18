"""Parse WebShop prompt bodies and main product pages without environment imports.

See docs/treehca/product_page_parser_assumptions.md for the supported format,
the agreed contracts, and the limits of option-group inference.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProductPageContextParts:
    """Outer prompt sections; absent history and step counts are represented by None."""

    agent_introduction: str
    shopping_task: str
    history_block: str | None
    completed_steps: int | None
    history_length: int | None
    current_step: int | None
    current_observation: str
    admissible_actions: tuple[str, ...]
    response_instructions: str


@dataclass(frozen=True)
class ProductOptionGroup:
    """Available values and any known full-credit choices, in display order."""

    name: str
    values: tuple[str, ...]
    correct_options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.correct_options)) != len(self.correct_options):
            raise ValueError("Correct options must not contain duplicates")
        if any(option not in self.values for option in self.correct_options):
            raise ValueError("Every correct option must be one of the group's available values")


@dataclass(frozen=True)
class ProductPageFields:
    """Main product-page fields with price and rating retained as text."""

    navigation_controls: tuple[str, ...]
    option_groups: tuple[ProductOptionGroup, ...]
    title: str
    price_text: str
    rating_text: str
    detail_page_controls: tuple[str, ...]
    purchase_control: str


@dataclass(frozen=True)
class ProductPageParseResult:
    """One batch row's parsed data or failure, retaining its original zero-based index."""

    index: int
    context_parts: ProductPageContextParts | None
    product_fields: ProductPageFields | None
    error_stage: Literal["context", "product"] | None
    error_message: str | None


_CURRENT_OBSERVATION = re.compile(r"\n(?:Your current observation is: |You are now at step (?P<step>\d+) and your current observation is: )")
_ACTION_BLOCK = re.compile(r"\nYour admissible actions of the current situation are:[ \t]*\n\[\n")
_RESPONSE_INSTRUCTIONS = re.compile(r"\n\]\.\n\n(?=Now it's your turn to take one action for the current step\.)")
_HISTORY_START = re.compile(r"\n(?=Prior to this step,)")
_HISTORY_HEADER = re.compile(
    r"Prior to this step, you have already taken (?P<completed>\d+) step\(s\)\. "
    r"Below are the most recent (?P<length>\d+) observations and the corresponding actions you took: .*",
    re.DOTALL,
)
_INTRODUCTION_AND_TASK = re.compile(
    r"(?P<introduction>You are an expert autonomous agent operating in the WebShop e[‑-]commerce environment\.)[ \t]*\n"
    r"Your task is to: (?P<task>.+)\.",
    re.DOTALL,
)
_COMMAND = re.compile(r"(?:click|search)\[.+\]")
_NAVIGATION = ("Back to Search", "< Prev")
_DETAIL_CONTROLS = ("Description", "Features")


def _split_once(text: str, pattern: re.Pattern, section: str) -> tuple[str, re.Match, str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected one {section} boundary, found {len(matches)}")
    match = matches[0]
    return text[: match.start()], match, text[match.end() :]


def _validate_actions(actions: Sequence[str]) -> tuple[str, ...]:
    if isinstance(actions, (str, bytes)):
        raise ValueError("admissible_actions must be a sequence of command strings")
    result = tuple(actions)
    if any(not isinstance(action, str) or _COMMAND.fullmatch(action) is None for action in result):
        raise ValueError("Admissible actions must use click[...] or search[...] syntax")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate admissible actions are ambiguous")
    return result


def _parse_action_lines(text: str) -> tuple[str, ...]:
    actions = []
    for line in text.splitlines():
        if not line.startswith("'") or not line.endswith("',"):
            raise ValueError("Each admissible action must be single-quoted and followed by a comma")
        # Display quotes are literal; see Text preservation in docs/treehca/product_page_parser_assumptions.md.
        actions.append(line[1:-2])
    return _validate_actions(actions)


def _extract_context(context: str) -> ProductPageContextParts:
    if not isinstance(context, str):
        raise ValueError("Raw context must be a string")
    # Only template boundaries are consumed; history remains opaque (see the assumptions document).
    before_actions, _, after_actions = _split_once(context.strip("\n"), _ACTION_BLOCK, "admissible-action")
    action_lines, _, response_instructions = _split_once(after_actions, _RESPONSE_INSTRUCTIONS, "response-instructions")
    before_observation, observation_match, observation_sentence = _split_once(before_actions, _CURRENT_OBSERVATION, "current-observation")
    if not observation_sentence.endswith(".") or len(observation_sentence) == 1:
        raise ValueError("Current observation must be nonempty and followed by the template's period")
    current_step = int(observation_match["step"]) if observation_match["step"] is not None else None

    history_block = None
    completed_steps = history_length = None
    if current_step is not None:
        introduction_and_task, _, history_block = _split_once(before_observation, _HISTORY_START, "history")
        history_match = _HISTORY_HEADER.fullmatch(history_block)
        if history_match is None:
            raise ValueError("Malformed history introduction")
        completed_steps = int(history_match["completed"])
        history_length = int(history_match["length"])
        if current_step != completed_steps + 1 or history_length > completed_steps:
            raise ValueError("Inconsistent completed-step, history-length, or current-step counts")
    else:
        introduction_and_task = before_observation
        if _HISTORY_START.search(before_observation):
            raise ValueError("History block requires the template's current-step number")

    header = _INTRODUCTION_AND_TASK.fullmatch(introduction_and_task)
    if header is None:
        raise ValueError("Expected a WebShop prompt body with an agent introduction and shopping task")
    return ProductPageContextParts(
        agent_introduction=header["introduction"],
        shopping_task=header["task"],
        history_block=history_block,
        completed_steps=completed_steps,
        history_length=history_length,
        current_step=current_step,
        current_observation=observation_sentence[:-1],
        admissible_actions=_parse_action_lines(action_lines),
        response_instructions=response_instructions,
    )


def extract_product_page_contexts(contexts: Sequence[str]) -> list[ProductPageContextParts]:
    """Extract outer sections from a batch of raw WebShop prompt bodies.

    The caller identifies product pages. This function neither classifies pages
    nor parses product fields. Results preserve input order; an empty batch
    returns []. A malformed row raises ValueError with its zero-based batch
    index, and no partial batch is returned.

    See docs/treehca/product_page_parser_assumptions.md#raw-context-extraction.
    """
    if isinstance(contexts, (str, bytes)):
        raise ValueError("contexts must be a batch of strings, not a single string")
    results = []
    for index, context in enumerate(contexts):
        try:
            results.append(_extract_context(context))
        except ValueError as error:
            raise ValueError(f"Invalid product-page context at batch index {index}: {error}") from error
    return results


def _observation_fragments(observation: str) -> tuple[str, ...]:
    if not isinstance(observation, str) or not observation:
        raise ValueError("current_observation must be a nonempty string")
    parts = observation.split(" [SEP] ")
    if any(len(part) < 3 or not part.startswith("'") or not part.endswith("'") for part in parts):
        raise ValueError("Observation must contain nonempty single-quoted fragments separated by ' [SEP] '")
    # Remove exactly one quote pair, preserving apostrophes/backslashes; see Text preservation in the assumptions document.
    return tuple(part[1:-1] for part in parts)


def _parse_option_groups(fragments: tuple[str, ...], action_set: set[str]) -> tuple[ProductOptionGroup, ...]:
    groups = []
    index = 0
    while index < len(fragments):
        name = fragments[index]
        if f"click[{name}]" in action_set or f"click[{name.lower()}]" in action_set:
            raise ValueError(f"Option-group name {name!r} collides with an admissible action")
        index += 1
        start = index
        # The environment preserves option value case in commands; see Option-group heuristic in the assumptions document.
        while index < len(fragments) and f"click[{fragments[index]}]" in action_set:
            index += 1
        if start == index:
            raise ValueError(f"Option group {name!r} has no recognized clickable values")
        # WebShop displays duplicate normalized values but exposes just one click command.
        groups.append(ProductOptionGroup(name=name, values=tuple(dict.fromkeys(fragments[start:index]))))
    names = [group.name for group in groups]
    values = [value for group in groups for value in group.values]
    if len(set(names)) != len(names) or len(set(values)) != len(values):
        raise ValueError("Repeated option names or values across groups make option grouping ambiguous")
    return tuple(groups)


def parse_product_page_fields(current_observation: str, admissible_actions: Sequence[str]) -> ProductPageFields:
    """Parse a main product page using its current observation and action list.

    Supports zero or more option groups and the optional Attributes control.
    Values retain display order and spelling. Price and rating remain text;
    selected options are not inferred. Unsupported or detectably ambiguous
    structure raises ValueError, including disagreement with the action list.

    See docs/treehca/product_page_parser_assumptions.md#product-field-extraction.
    """
    fragments = _observation_fragments(current_observation)
    action_set = set(_validate_actions(admissible_actions))
    if fragments[:2] != _NAVIGATION:
        raise ValueError("Expected main product-page navigation: Back to Search, < Prev")

    # Anchor fields from the fixed suffix instead of treating every Price: fragment as a boundary.
    # See Product-field extraction in docs/treehca/product_page_parser_assumptions.md.
    detail_controls = _DETAIL_CONTROLS
    if fragments[-2:] == ("Attributes", "Buy Now"):
        detail_controls += ("Attributes",)
    suffix = detail_controls + ("Buy Now",)
    if fragments[-len(suffix) :] != suffix:
        raise ValueError("Expected Description, Features, optional Attributes, and Buy Now at the end of the page")
    product = fragments[2 : -len(suffix)]
    if len(product) < 2 or not product[-2].startswith("Price: ") or not product[-1].startswith("Rating: "):
        raise ValueError("Expected a product title, Price: field, and Rating: field before the detail controls")
    price_field, rating_field = product[-2:]
    # An empty catalog title has no rendered fragment. A non-clickable final
    # fragment is a title; otherwise the page has no title.
    has_title = len(product) >= 3 and f"click[{product[-3]}]" not in action_set
    title = product[-3] if has_title else ""
    price_text, rating_text = price_field[len("Price: ") :], rating_field[len("Rating: ") :]
    if not price_text or not rating_text:
        raise ValueError("Price and rating fields must contain text")

    control_actions = {f"click[{control.lower()}]" for control in _NAVIGATION + suffix}
    if not control_actions <= action_set:
        raise ValueError(f"Missing page-control actions: {sorted(control_actions - action_set)}")
    option_fragments = product[:-3] if has_title else product[:-2]
    if any(f"click[{fragment.lower()}]" in control_actions for fragment in option_fragments):
        raise ValueError("Option name or value collides with a page control")
    option_groups = _parse_option_groups(option_fragments, action_set)
    option_actions = {f"click[{value}]" for group in option_groups for value in group.values}
    if action_set != control_actions | option_actions:
        raise ValueError(f"Actions do not match the displayed product fields: {sorted(action_set - control_actions - option_actions)}")

    return ProductPageFields(
        navigation_controls=_NAVIGATION,
        option_groups=option_groups,
        title=title,
        price_text=price_text,
        rating_text=rating_text,
        detail_page_controls=detail_controls,
        purchase_control="Buy Now",
    )


def parse_product_page_batch(contexts: Sequence[str]) -> list[ProductPageParseResult]:
    """Parse every row, recording recognized failures instead of aborting the batch.

    Returns one result per input in input order. Context failures leave both
    parsed fields None; product failures preserve the extracted context parts.
    Successful rows have error_stage and error_message set to None.

    Only parsing ValueErrors are captured. Invalid batch arguments and other
    exceptions still propagate. No score or training fallback is assigned.
    See docs/treehca/product_page_parser_assumptions.md#batch-error-reporting.
    """
    if isinstance(contexts, (str, bytes)):
        raise ValueError("contexts must be a batch of strings, not a single string")
    results = []
    for index, context in enumerate(contexts):
        # Use the same row extractor as the strict API; indices belong to this batch.
        try:
            parts = _extract_context(context)
        except ValueError as error:
            results.append(ProductPageParseResult(index=index, context_parts=None, product_fields=None, error_stage="context", error_message=str(error)))
            continue
        try:
            product = parse_product_page_fields(parts.current_observation, parts.admissible_actions)
        except ValueError as error:
            # Keep usable context data; fallback decisions are deferred (see Batch error reporting in the assumptions document).
            results.append(ProductPageParseResult(index=index, context_parts=parts, product_fields=None, error_stage="product", error_message=str(error)))
            continue
        results.append(ProductPageParseResult(index=index, context_parts=parts, product_fields=product, error_stage=None, error_message=None))
    return results
