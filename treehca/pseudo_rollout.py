"""Build and score action-choice prompts; see the corresponding documentation."""

import logging
import math
from dataclasses import dataclass, field
from itertools import product
from string import ascii_uppercase
from typing import Any, Sequence

from treehca.product_page_parser import ProductPageContextParts

_LABEL_CANDIDATES = tuple(ascii_uppercase) + tuple("".join(pair) for pair in product(ascii_uppercase, repeat=2))
_TURN_INSTRUCTION = "Now it's your turn to take one action for the current step."
_CHOICE_INSTRUCTIONS = "You must give the label corresponding to the action you want to take. You should only respond with a single label from the list"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionLabelTokens:
    """One displayed label and its two supported next-token spellings."""

    label: str
    bare_token_id: int
    spaced_token_id: int

    @property
    def token_ids(self) -> tuple[int, int]:
        return self.bare_token_id, self.spaced_token_id


@dataclass(frozen=True)
class ActionLabelCatalog:
    """Reusable tokenizer-specific label metadata, ordered by preference."""

    entries: tuple[ActionLabelTokens, ...]
    tokenizer: Any = field(repr=False, compare=False)
    _validated_chat_templates: set[str] = field(default_factory=set, repr=False, compare=False, hash=False)

    def prefix(self, num_actions: int) -> tuple[ActionLabelTokens, ...]:
        _validate_num_actions(num_actions)
        if num_actions > len(self.entries):
            raise ValueError(f"Action-label catalog contains {len(self.entries)} labels; requested {num_actions}")
        return self.entries[:num_actions]


@dataclass(frozen=True)
class ProductPagePseudoRollout:
    """A tokenized prompt plus the action and token ordering needed to score it."""

    prompt: str
    prompt_token_ids: tuple[int, ...]
    labels: tuple[str, ...]
    actions: tuple[str, ...]
    variant_token_ids: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("A product-page pseudo-rollout must contain at least one action")
        if not len(self.labels) == len(self.actions) == len(self.variant_token_ids):
            raise ValueError("Pseudo-rollout labels, actions, and token pairs must have equal lengths")
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in self.variant_token_ids):
            raise ValueError("Each action must have exactly two label-token variants")
        flattened_ids = [token_id for pair in self.variant_token_ids for token_id in pair]
        if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in flattened_ids):
            raise ValueError("Action-label token IDs must be integers")
        if len(set(flattened_ids)) != len(flattened_ids):
            raise ValueError("Action-label token IDs must be distinct")

    @property
    def label_to_action(self) -> dict[str, str]:
        return dict(zip(self.labels, self.actions))

    @property
    def allowed_token_ids(self) -> tuple[int, ...]:
        return tuple(token_id for pair in self.variant_token_ids for token_id in pair)


@dataclass(frozen=True)
class ActionChoiceScore:
    """Conditional probability assigned to one action and its label variants."""

    label: str
    action: str
    probability: float
    log_probability: float
    bare_log_probability: float
    spaced_log_probability: float


@dataclass(frozen=True)
class ProductPagePseudoRolloutScores:
    """Forced-choice action distribution returned for one pseudo-rollout."""

    choices: tuple[ActionChoiceScore, ...]

    @property
    def label_probabilities(self) -> dict[str, float]:
        return {choice.label: choice.probability for choice in self.choices}

    @property
    def action_probabilities(self) -> dict[str, float]:
        return {choice.action: choice.probability for choice in self.choices}

    @property
    def entropy(self) -> float:
        return -sum(choice.probability * choice.log_probability for choice in self.choices if choice.probability > 0.0)


def _validate_num_actions(num_actions: int) -> None:
    if isinstance(num_actions, bool) or not isinstance(num_actions, int) or not 0 <= num_actions <= len(_LABEL_CANDIDATES):
        raise ValueError(f"num_actions must be an integer between 0 and {len(_LABEL_CANDIDATES)}")


def build_action_label_catalog(tokenizer: Any, num_actions: int) -> ActionLabelCatalog:
    """Discover reusable single-token label variants for one tokenizer."""
    _validate_num_actions(num_actions)
    if num_actions == 0:
        return ActionLabelCatalog(entries=(), tokenizer=tokenizer)

    entries = []
    used_token_ids = set()
    for label in _LABEL_CANDIDATES:
        variants = (label, f" {label}")
        encoded = [tokenizer.encode(variant, add_special_tokens=False) for variant in variants]
        if any(len(ids) != 1 for ids in encoded):
            continue
        token_ids = (encoded[0][0], encoded[1][0])
        if token_ids[0] == token_ids[1] or used_token_ids.intersection(token_ids):
            continue
        # Round-trip checks avoid unknown/special tokens masquerading as a usable label.
        if any(tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != variant for ids, variant in zip(encoded, variants)):
            continue
        entries.append(ActionLabelTokens(label=label, bare_token_id=token_ids[0], spaced_token_id=token_ids[1]))
        used_token_ids.update(token_ids)
        if len(entries) == num_actions:
            return ActionLabelCatalog(entries=tuple(entries), tokenizer=tokenizer)
    raise ValueError(f"Tokenizer supports only {len(entries)} usable action labels in A..Z, AA..ZZ; requested {num_actions}")


def select_action_labels(tokenizer: Any, num_actions: int) -> tuple[str, ...]:
    """Select ordered uppercase labels with bare and space-prefixed token variants.

    Candidates are A through Z, then AA through ZZ. Both spellings must encode
    as one token and decode exactly, with distinct token IDs across all selected
    variants. Token IDs are discovered using the supplied tokenizer, not fixed
    vocabulary offsets. Raises ValueError if the candidate pool is insufficient.

    See docs/treehca/pseudo_rollout_prompts.md#why-these-letter-labels.
    """
    return tuple(entry.label for entry in build_action_label_catalog(tokenizer, num_actions).entries)


def build_product_page_pseudo_rollout_prompt(parts: ProductPageContextParts, tokenizer: Any, *, label_catalog: ActionLabelCatalog | None = None) -> tuple[str, dict[str, str]]:
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
    if label_catalog is not None and label_catalog.tokenizer is not tokenizer:
        raise ValueError("The action-label catalog was built for a different tokenizer instance")
    catalog = label_catalog or build_action_label_catalog(tokenizer, len(parts.admissible_actions))
    labels = tuple(entry.label for entry in catalog.prefix(len(parts.admissible_actions)))
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


def _tokenize_chat_prompt(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one tokenized conversation")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError("Chat template must return a flat list of integer token IDs")
    return tuple(token_ids)


def _validate_assistant_boundary(tokenizer: Any, prompt: str, prompt_token_ids: tuple[int, ...], entries: Sequence[ActionLabelTokens]) -> None:
    """Ensure isolated label tokens remain suffix tokens after the chat template."""
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    variants = [variant for entry in entries for variant in (entry.label, f" {entry.label}")]
    expected_ids = [token_id for entry in entries for token_id in entry.token_ids]
    encoded_with_variants = tokenizer([rendered_prompt + variant for variant in variants], add_special_tokens=False)["input_ids"]
    for variant, expected_id, full_ids in zip(variants, expected_ids, encoded_with_variants):
        if list(full_ids) != [*prompt_token_ids, expected_id]:
            raise ValueError(f"Label variant {variant!r} is not one token at the assistant-response boundary")


def prepare_product_page_pseudo_rollouts(
    parts: Sequence[ProductPageContextParts],
    tokenizer: Any,
    *,
    label_catalog: ActionLabelCatalog | None = None,
) -> tuple[ProductPagePseudoRollout, ...]:
    """Build and tokenize a batch, discovering label tokens only once."""
    if isinstance(parts, (str, bytes)):
        raise ValueError("parts must be a sequence of ProductPageContextParts")
    parts = tuple(parts)
    if not parts:
        return ()

    max_actions = max(len(item.admissible_actions) for item in parts)
    if label_catalog is not None and label_catalog.tokenizer is not tokenizer:
        raise ValueError("The action-label catalog was built for a different tokenizer instance")
    catalog = label_catalog or build_action_label_catalog(tokenizer, max_actions)
    if len(catalog.entries) < max_actions:
        raise ValueError(f"Action-label catalog contains {len(catalog.entries)} labels; requested {max_actions}")

    prepared = []
    for item in parts:
        prompt, label_to_action = build_product_page_pseudo_rollout_prompt(item, tokenizer, label_catalog=catalog)
        entries = catalog.prefix(len(label_to_action))
        prepared.append(
            ProductPagePseudoRollout(
                prompt=prompt,
                prompt_token_ids=_tokenize_chat_prompt(tokenizer, prompt),
                labels=tuple(label_to_action),
                actions=tuple(label_to_action.values()),
                variant_token_ids=tuple(entry.token_ids for entry in entries),
            )
        )

    # Every prompt has the same chat-template assistant suffix, so one check of
    # the largest label set establishes the boundary contract for this catalog.
    chat_template_key = str(getattr(tokenizer, "chat_template", None))
    if chat_template_key not in catalog._validated_chat_templates:
        boundary_probe = max(prepared, key=lambda item: len(item.labels))
        _validate_assistant_boundary(tokenizer, boundary_probe.prompt, boundary_probe.prompt_token_ids, catalog.prefix(len(boundary_probe.labels)))
        catalog._validated_chat_templates.add(chat_template_key)
    return tuple(prepared)


def required_max_logprobs(pseudo_rollouts: Sequence[ProductPagePseudoRollout]) -> int:
    """Return the vLLM ``max_logprobs`` required by a prepared batch."""
    return max((len(item.allowed_token_ids) for item in pseudo_rollouts), default=0)


def _resolve_max_model_len(inference_engine: Any, explicit_max_model_len: int | None) -> int:
    if explicit_max_model_len is not None:
        if isinstance(explicit_max_model_len, bool) or not isinstance(explicit_max_model_len, int) or explicit_max_model_len <= 0:
            raise ValueError("max_model_len must be a positive integer")
        return explicit_max_model_len
    model_config = getattr(getattr(inference_engine, "llm_engine", None), "model_config", None)
    max_model_len = getattr(model_config, "max_model_len", None)
    if isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len <= 0:
        raise ValueError("Could not discover max_model_len from the inference engine; pass it explicitly")
    return max_model_len


def _extract_log_probability(logprob: Any) -> float:
    value = getattr(logprob, "logprob", logprob)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Malformed vLLM logprob value {value!r}")
    return float(value)


def _logaddexp(left: float, right: float) -> float:
    maximum = max(left, right)
    if maximum == -math.inf:
        return -math.inf
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _scores_from_vllm_output(pseudo_rollout: ProductPagePseudoRollout, output: Any) -> ProductPagePseudoRolloutScores:
    completions = getattr(output, "outputs", None)
    if not isinstance(completions, list) or len(completions) != 1:
        raise ValueError("Expected exactly one vLLM completion per pseudo-rollout")
    logprob_steps = getattr(completions[0], "logprobs", None)
    if not isinstance(logprob_steps, list) or len(logprob_steps) != 1:
        raise ValueError("Expected logprobs for exactly one generated token")
    token_logprobs = logprob_steps[0]
    if not hasattr(token_logprobs, "__getitem__"):
        raise ValueError("Expected vLLM token logprobs to be a token-ID mapping")

    variant_logprobs = []
    for bare_id, spaced_id in pseudo_rollout.variant_token_ids:
        missing_ids = [token_id for token_id in (bare_id, spaced_id) if token_id not in token_logprobs]
        if missing_ids:
            raise ValueError(f"vLLM omitted requested action-label token IDs {missing_ids}")
        variant_logprobs.append((_extract_log_probability(token_logprobs[bare_id]), _extract_log_probability(token_logprobs[spaced_id])))

    action_logprobs = [_logaddexp(*pair) for pair in variant_logprobs]
    normalizer = action_logprobs[0]
    for action_logprob in action_logprobs[1:]:
        normalizer = _logaddexp(normalizer, action_logprob)
    if not math.isfinite(normalizer):
        raise ValueError("vLLM returned no finite probability mass for the allowed action-label tokens")
    choices = tuple(
        ActionChoiceScore(
            label=label,
            action=action,
            probability=math.exp(action_logprob - normalizer),
            log_probability=action_logprob - normalizer,
            bare_log_probability=bare_logprob,
            spaced_log_probability=spaced_logprob,
        )
        for label, action, action_logprob, (bare_logprob, spaced_logprob) in zip(
            pseudo_rollout.labels,
            pseudo_rollout.actions,
            action_logprobs,
            variant_logprobs,
        )
    )
    return ProductPagePseudoRolloutScores(choices=choices)


def score_product_page_pseudo_rollouts(
    inference_engine: Any,
    pseudo_rollouts: Sequence[ProductPagePseudoRollout],
    *,
    max_model_len: int | None = None,
    lora_request: Any = None,
    assistant_response_prefix_token_ids: Sequence[Sequence[int]] | None = None,
) -> list[ProductPagePseudoRolloutScores | None]:
    """Run one constrained vLLM step and return action distributions in input order.

    ``assistant_response_prefix_token_ids`` optionally supplies one already
    generated assistant-response prefix per row. The scorer appends each prefix
    after the pseudo prompt's assistant boundary and measures the next-token
    action distribution conditioned on it. This is useful for retaining sampled
    thinking before replacing the original ``<action>`` with a choice label.
    Prefixes must be token IDs from the same tokenizer as the prepared prompts.

    Rows without room for the required output token are skipped and represented
    by ``None``. The engine must be initialized with ``max_logprobs`` at least as
    large as :func:`required_max_logprobs` for the submitted batch.
    """
    pseudo_rollouts = tuple(pseudo_rollouts)
    if assistant_response_prefix_token_ids is None:
        response_prefixes = ((),) * len(pseudo_rollouts)
    else:
        if isinstance(assistant_response_prefix_token_ids, (str, bytes)):
            raise ValueError("assistant_response_prefix_token_ids must be a sequence of token-ID sequences")
        response_prefixes = tuple(tuple(prefix) for prefix in assistant_response_prefix_token_ids)
        if len(response_prefixes) != len(pseudo_rollouts):
            raise ValueError("assistant_response_prefix_token_ids must align with pseudo_rollouts")
        for row, prefix in enumerate(response_prefixes):
            if any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in prefix):
                raise ValueError(f"Assistant response prefix {row} must contain only integer token IDs")
    if not pseudo_rollouts:
        return []
    model_limit = _resolve_max_model_len(inference_engine, max_model_len)
    conditioned_prompt_token_ids = tuple((*item.prompt_token_ids, *prefix) for item, prefix in zip(pseudo_rollouts, response_prefixes))
    scorable_indices = [index for index, token_ids in enumerate(conditioned_prompt_token_ids) if len(token_ids) + 1 <= model_limit]
    scorable_index_set = set(scorable_indices)
    for index, item in enumerate(pseudo_rollouts):
        if index not in scorable_index_set:
            logger.warning("Pseudo-rollout %d has no room for its output token at max_model_len=%d; leaving its score unset", index, model_limit)
    if not scorable_indices:
        return [None] * len(pseudo_rollouts)

    engine_module = type(getattr(inference_engine, "llm_engine", None)).__module__
    if engine_module.startswith("vllm.v1."):
        raise ValueError("vLLM V1 returns top logprobs from the unconstrained vocabulary before applying allowed_token_ids, so it cannot reliably score every action label with this method; initialize vLLM with VLLM_USE_V1=0")

    selected = [pseudo_rollouts[index] for index in scorable_indices]
    model_config = getattr(getattr(inference_engine, "llm_engine", None), "model_config", None)
    configured_max_logprobs = getattr(model_config, "max_logprobs", None)
    needed_max_logprobs = required_max_logprobs(selected)
    if isinstance(configured_max_logprobs, int) and not isinstance(configured_max_logprobs, bool) and configured_max_logprobs < needed_max_logprobs:
        raise ValueError(f"vLLM max_logprobs is {configured_max_logprobs}, but this batch requires {needed_max_logprobs}")

    try:
        from vllm import SamplingParams
    except ImportError as error:
        raise RuntimeError("vLLM is required to score product-page pseudo-rollouts") from error

    prompts = [{"prompt_token_ids": list(conditioned_prompt_token_ids[index])} for index in scorable_indices]
    sampling_params = [
        SamplingParams(
            n=1,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            allowed_token_ids=list(item.allowed_token_ids),
            logprobs=len(item.allowed_token_ids),
            detokenize=False,
        )
        for item in selected
    ]
    generate_kwargs = {}
    if lora_request is not None:
        if isinstance(lora_request, (list, tuple)):
            if len(lora_request) != len(pseudo_rollouts):
                raise ValueError("A per-row lora_request sequence must align with pseudo_rollouts")
            generate_kwargs["lora_request"] = [lora_request[index] for index in scorable_indices]
        else:
            generate_kwargs["lora_request"] = lora_request
    outputs = inference_engine.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
        **generate_kwargs,
    )
    if len(outputs) != len(selected):
        raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(selected)} pseudo-rollouts")

    results: list[ProductPagePseudoRolloutScores | None] = [None] * len(pseudo_rollouts)
    for original_index, item, output in zip(scorable_indices, selected, outputs):
        results[original_index] = _scores_from_vllm_output(item, output)
    return results
