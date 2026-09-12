"""Build action-choice prompt bodies; see docs/treehca/pseudo_rollout_prompts.md."""

from itertools import product
from string import ascii_uppercase
from typing import Any

from treehca.product_page_parser import ProductPageContextParts

_LABEL_CANDIDATES = tuple(ascii_uppercase) + tuple("".join(pair) for pair in product(ascii_uppercase, repeat=2))
_TURN_INSTRUCTION = "Now it's your turn to take one action for the current step."
_CHOICE_INSTRUCTIONS = "You must give the label corresponding to the action you want to take. You should only respond with a single label from the list"


def select_action_labels(tokenizer: Any, num_actions: int) -> tuple[str, ...]:
    """Select ordered uppercase labels with bare and space-prefixed token variants.

    Candidates are A through Z, then AA through ZZ. Both spellings must encode
    as one token and decode exactly, with distinct token IDs across all selected
    variants. Token IDs are discovered using the supplied tokenizer, not fixed
    vocabulary offsets. Raises ValueError if the candidate pool is insufficient.

    See docs/treehca/pseudo_rollout_prompts.md#why-these-letter-labels.
    """
    if isinstance(num_actions, bool) or not isinstance(num_actions, int) or not 0 <= num_actions <= len(_LABEL_CANDIDATES):
        raise ValueError(f"num_actions must be an integer between 0 and {len(_LABEL_CANDIDATES)}")
    if num_actions == 0:
        return ()
    labels = []
    used_token_ids = set()
    for label in _LABEL_CANDIDATES:
        variants = (label, f" {label}")
        encoded = [tokenizer.encode(variant, add_special_tokens=False) for variant in variants]
        if any(len(ids) != 1 for ids in encoded):
            continue
        token_ids = {ids[0] for ids in encoded}
        if len(token_ids) != 2 or token_ids & used_token_ids:
            continue
        # Round-trip checks avoid unknown/special tokens masquerading as a usable label.
        if any(tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != variant for ids, variant in zip(encoded, variants)):
            continue
        labels.append(label)
        used_token_ids.update(token_ids)
        if len(labels) == num_actions:
            return tuple(labels)
    raise ValueError(f"Tokenizer supports only {len(labels)} usable action labels in A..Z, AA..ZZ; requested {num_actions}")


def build_product_page_pseudo_rollout_prompt(parts: ProductPageContextParts, tokenizer: Any) -> tuple[str, dict[str, str]]:
    """Build one raw prompt body from parsed context components.

    Retains the task, history when present, current observation, and action order.
    Replaces the action entries with letter labels and the final response-format
    instructions with a request for one label. Returns (prompt, label_to_action),
    with canonical labels (no leading space) as dictionary keys in prompt order.
    The prompt is text before chat-template application; this function neither
    generates a response nor computes probabilities.

    ValueError reports empty actions, inconsistent history metadata, or an
    insufficient single-token label pool. The caller handles construction errors.
    See docs/treehca/pseudo_rollout_prompts.md for the contract and scoring rationale.
    """
    if not parts.admissible_actions:
        raise ValueError("Cannot construct an action-choice prompt without admissible actions")
    if (parts.history_block is None) != (parts.current_step is None):
        raise ValueError("History block and current-step number must either both be present or both be absent")
    labels = select_action_labels(tokenizer, len(parts.admissible_actions))
    # Preserve the local templates' introduction whitespace and sentence punctuation.
    introduction_suffix = " " if parts.history_block is None else ""
    prompt = f"\n{parts.agent_introduction}{introduction_suffix}\nYour task is to: {parts.shopping_task}.\n"
    if parts.history_block is None:
        prompt += f"Your current observation is: {parts.current_observation}.\n"
    else:
        # Retain history verbatim; history-removal policy is outside this builder's contract.
        prompt += f"{parts.history_block}\nYou are now at step {parts.current_step} and your current observation is: {parts.current_observation}.\n"
    # Use one mapping for the displayed choices and the caller; see the documented return contract.
    label_to_action = dict(zip(labels, parts.admissible_actions))
    action_choices = "\n".join(f"{label}: {action}" for label, action in label_to_action.items())
    prompt += f"Your admissible actions of the current situation are: \n[\n{action_choices}\n].\n\n"
    prompt += f"{_TURN_INSTRUCTION}\n{_CHOICE_INSTRUCTIONS}\n"
    return prompt, label_to_action
