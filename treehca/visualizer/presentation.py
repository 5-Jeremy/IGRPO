"""Prompt sections and user-facing fields for the rollout viewer."""

import re

from treehca.product_page_parser import extract_product_page_contexts

HIDDEN_FIELDS = {"input", "node_uid", "parent_node_uid", "parent_uid", "children", "child_uids", "children_uids", "node_path", "uid", "tree_uid", "advantages", "values", "webshop_session_id"}
HIDDEN_COLOR_FIELDS = {"sampled_expansion_count", "branch_logit", "webshop_session_id", "webshop_task_id"}
PROMPT_FIELDS = ("shopping_task", "history", "current_observation", "available_actions")


def prompt_sections(model_input):
    """Locate the prompt body inside a decoded wrapper, then use the shared parser."""
    if not isinstance(model_input, str):
        return {name: None for name in PROMPT_FIELDS}, "No saved WebShop input."
    start = re.search(r"(?m)^You are an expert autonomous agent operating in the WebShop e[‑-]commerce environment\.", model_input)
    if start is None:
        return {name: None for name in PROMPT_FIELDS}, "Unrecognized WebShop prompt introduction."
    try:
        # The four sections end before response instructions and the chat suffix.
        parts = extract_product_page_contexts([model_input[start.start() :]])[0]
    except ValueError as exc:
        return {name: None for name in PROMPT_FIELDS}, str(exc)
    return {
        "shopping_task": parts.shopping_task,
        "history": parts.history_block,
        "current_observation": parts.current_observation,
        "available_actions": list(parts.admissible_actions),
    }, None


def visible_fields(record):
    return {key: value for key, value in record.items() if key not in HIDDEN_FIELDS}


def detail_fields(node, record, pinned=()):
    """Keep pinned fields first even when an older record lacks their values."""
    fields = {"graph depth": node.depth, "child count": len(node.children), **visible_fields(record)}
    pins = [name for name in pinned if name not in HIDDEN_FIELDS]
    return {**{name: fields.get(name, "Not saved for this node") for name in pins}, **fields}
