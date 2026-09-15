"""Prompt preservation and tokenizer-backed checks for action-choice labels."""

import json
import math
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_product_page import (
    build_action_label_catalog,
    build_product_page_grouped_choice_prompt,
    build_product_page_pseudo_rollout_prompt,
    prepare_product_page_grouped_choice_rollouts,
    prepare_product_page_pseudo_rollouts,
    required_max_logprobs,
    score_product_page_grouped_choice_rollouts,
    score_product_page_pseudo_rollouts,
    select_action_labels,
)

_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = runpy.run_path(str(_ROOT / "agent_system/environments/prompts/webshop.py"))
_EXAMPLES = json.loads((Path(__file__).parent / "fixtures/product_page_contexts.json").read_text())
_GOAL_OPTIONS = {
    "two_option_groups": {"size": "19.7x31.5in+19.7x63in", "color": "christmasgoo3302"},
    # Extra goal groups are valid when the current product page has no options.
    "no_options": {"color": "xnj-tshirt345-black", "size": "large"},
    "one_option_group": {"size": "0.7mm"},
}


@pytest.fixture(scope="module")
def tokenizer():
    # Use the configured model's cached tokenizer; no downloads or model weights are needed.
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)


@pytest.fixture
def parts():
    return extract_product_page_contexts([_EXAMPLES[0]["raw_context"]])[0]


def test_55_labels_have_distinct_bare_and_spaced_single_tokens(tokenizer, parts):
    labels = select_action_labels(tokenizer, 55)
    assert labels == tuple("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AZ BA BB BC BD".split())
    variants = [variant for label in labels for variant in (label, " " + label)]
    ids = [tokenizer.encode(variant, add_special_tokens=False) for variant in variants]
    assert all(len(tokens) == 1 for tokens in ids)
    assert len({tokens[0] for tokens in ids}) == 110

    components = replace(parts, admissible_actions=tuple(f"click[option {i}]" for i in range(55)))
    body, label_to_action = build_product_page_pseudo_rollout_prompt(components, tokenizer)
    assert tuple(label_to_action) == labels
    assert tuple(label_to_action.values()) == components.admissible_actions
    assert len(label_to_action) == 55
    assert "AY" not in label_to_action
    assert label_to_action["BD"] == "click[option 54]"
    assert "BD: click[option 54]" in body
    # Validate both spellings at the actual Qwen assistant generation boundary, not only in isolation.
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for variant, tokens in zip(variants, ids):
        assert tokenizer.encode(prompt + variant, add_special_tokens=False) == prompt_ids + tokens


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda example: example["name"])
@pytest.mark.parametrize("with_history", [False, True])
def test_only_action_entries_and_final_instructions_change(tokenizer, example, with_history):
    original = example["raw_context"]
    components = extract_product_page_contexts([original])[0]
    if not with_history:
        original = _TEMPLATES["WEBSHOP_TEMPLATE_NO_HIS"].format(
            task_description=components.shopping_task,
            current_observation=components.current_observation,
            available_actions="\n".join(f"'{action}'," for action in components.admissible_actions),
        )
        components = extract_product_page_contexts([original])[0]
    prompt, label_to_action = build_product_page_pseudo_rollout_prompt(components, tokenizer)

    labels = select_action_labels(tokenizer, len(components.admissible_actions))
    assert label_to_action == dict(zip(labels, components.admissible_actions))
    original_action_block = "[\n" + "\n".join(f"'{action}'," for action in components.admissible_actions) + "\n]."
    expected_action_block = "[\n" + "\n".join(f"{label}: {action}" for label, action in zip(labels, components.admissible_actions)) + "\n]."
    expected_footer = (
        "Now it's your turn to take one action for the current step.\n"
        'You must give the label corresponding to the action you want to take. You should think about what is the logically best next action to take, and finish your thought with "The best next action is:" followed by the label of your chosen action.'
    )
    # Independently edit the original body; all other bytes must agree apart from surrounding newlines.
    assert original.count(original_action_block) == 1
    expected = original.replace(original_action_block, expected_action_block).replace(components.response_instructions, expected_footer)
    assert prompt.strip("\n") == expected.strip("\n")


def test_labels_follow_action_order_without_sorting(tokenizer, parts):
    components = replace(parts, admissible_actions=("click[zebra]", "click[apple]", "click[buy now]"))
    prompt, label_to_action = build_product_page_pseudo_rollout_prompt(components, tokenizer)
    assert label_to_action == {"A": "click[zebra]", "B": "click[apple]", "C": "click[buy now]"}
    assert "[\nA: click[zebra]\nB: click[apple]\nC: click[buy now]\n]." in prompt


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda example: example["name"])
@pytest.mark.parametrize("with_history", [False, True])
def test_grouped_choice_prompt_builds_one_labeled_prompt_per_group(tokenizer, example, with_history):
    original = example["raw_context"]
    components = extract_product_page_contexts([original])[0]
    if not with_history:
        original = _TEMPLATES["WEBSHOP_TEMPLATE_NO_HIS"].format(
            task_description=components.shopping_task,
            current_observation=components.current_observation,
            available_actions="\n".join(f"'{action}'," for action in components.admissible_actions),
        )
        components = extract_product_page_contexts([original])[0]

    choice_prompts = build_product_page_grouped_choice_prompt(components, tokenizer, _GOAL_OPTIONS[example["name"]])

    expected_groups = {
        "two_option_groups": (
            ("size", ("19.7x31.5in+19.7x63in",), ("C",)),
            (
                "color",
                ("christmas-005goo7317", "christmas-010goo9911", "christmasgoo1729", "christmasgoo3302", "christmasgoo3848", "christmasgoo6658"),
                ("B", "C", "D", "E", "F", "G"),
            ),
        ),
        "no_options": (),
        "one_option_group": (("size", ("0.7mm",), ("B",)),),
    }
    assert tuple((choice.option_group.name, choice.option_group.correct_options, choice.correct_labels) for choice in choice_prompts) == expected_groups[example["name"]]

    original_action_block = "[\n" + "\n".join(f"'{action}'," for action in components.admissible_actions) + "\n]."
    for choice in choice_prompts:
        labeled_actions = "[\n" + "\n".join(f"{label}: {option if option is not None else 'none (do not select any option in this group)'}" for label, option in choice.label_to_option.items()) + "\n]."
        group_name = choice.option_group.name
        instructions = (
            f'Now you must select exactly one option for the "{group_name}" group.\n'
            'The "none" choice means never clicking any option in this group; it is not a click action.\n'
            "You must give the letter (e.g. A, B, C, AB) corresponding to the option you want to select (NOT the name of the option). "
            f'You should think about which option in the "{group_name}" group best satisfies the query, and finish your thought with '
            f'"The best choice for the {group_name} group corresponds to the label:" followed by the label of your chosen option.'
        )
        expected = original.replace(original_action_block, labeled_actions).replace(components.response_instructions, instructions)
        expected = expected.replace("Your admissible actions of the current situation are:", f"Your available options for {group_name} are:")
        assert choice.prompt.strip("\n") == expected.strip("\n")
        assert not any(f"{label}: click[" in choice.prompt for label in choice.labels)


def test_grouped_choice_prompt_rejects_empty_actions(tokenizer, parts):
    with pytest.raises(ValueError, match="without admissible actions"):
        build_product_page_grouped_choice_prompt(replace(parts, admissible_actions=()), tokenizer, {})


def test_grouped_choice_prompt_allows_unrequested_groups_but_rejects_unmatched_goals(tokenizer, parts):
    choices = build_product_page_grouped_choice_prompt(parts, tokenizer, {"size": "19.7x31.5in+19.7x63in"})
    color = next(choice for choice in choices if choice.option_group.name == "color")
    assert color.none_is_correct and color.labels[-1] in color.correct_labels
    with pytest.raises(ValueError, match="No displayed option"):
        build_product_page_grouped_choice_prompt(parts, tokenizer, {"size": "not offered", "color": "not offered"})


def test_grouped_choice_prompt_retains_every_fuzzy_matching_option(tokenizer, parts):
    observation = "'Back to Search' [SEP] '< Prev' [SEP] 'style' [SEP] 'bath towel' [SEP] 'flower bath towel' [SEP] 'Towel' [SEP] 'Price: $10.00' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Buy Now'"
    actions = ("click[back to search]", "click[< prev]", "click[bath towel]", "click[flower bath towel]", "click[description]", "click[features]", "click[buy now]")
    components = replace(parts, current_observation=observation, admissible_actions=actions)

    (choice,) = build_product_page_grouped_choice_prompt(components, tokenizer, {"style": "bath towel"})

    assert choice.option_group.correct_options == ("bath towel", "flower bath towel")
    assert choice.correct_labels == ("A", "B")


def test_label_selection_uses_the_supplied_tokenizer(tokenizer, monkeypatch):
    original_encode = tokenizer.encode

    def encode_with_split_space_a(text, **kwargs):
        if text == " A":
            return [1, 2]
        return original_encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", encode_with_split_space_a)
    assert select_action_labels(tokenizer, 2) == ("B", "C")


def test_insufficient_single_token_labels_raise(tokenizer, monkeypatch):
    monkeypatch.setattr(tokenizer, "encode", lambda *args, **kwargs: [1, 2])
    with pytest.raises(ValueError, match="only 0 usable action labels"):
        select_action_labels(tokenizer, 1)


def test_labels_must_round_trip_to_the_original_spelling(tokenizer, monkeypatch):
    monkeypatch.setattr(tokenizer, "decode", lambda *args, **kwargs: "<unk>")
    with pytest.raises(ValueError, match="only 0 usable action labels"):
        select_action_labels(tokenizer, 1)


def test_empty_actions_raise(tokenizer, parts):
    with pytest.raises(ValueError, match="without admissible actions"):
        build_product_page_pseudo_rollout_prompt(replace(parts, admissible_actions=()), tokenizer)


def test_inconsistent_history_metadata_raises(tokenizer, parts):
    with pytest.raises(ValueError, match="History block and current-step"):
        build_product_page_pseudo_rollout_prompt(replace(parts, current_step=None), tokenizer)


@pytest.mark.parametrize("count", [-1, 703, True])
def test_invalid_action_counts_raise(tokenizer, count):
    with pytest.raises(ValueError, match="num_actions must be an integer"):
        select_action_labels(tokenizer, count)


def test_zero_actions_selects_no_labels(tokenizer):
    assert select_action_labels(tokenizer, 0) == ()


def test_prepare_batch_reuses_catalog_and_tokenizes_chat_boundary(tokenizer, parts):
    catalog = build_action_label_catalog(tokenizer, 3)
    batch = prepare_product_page_pseudo_rollouts(
        [
            replace(parts, admissible_actions=("click[first]", "click[second]")),
            replace(parts, admissible_actions=("click[third]", "click[fourth]", "click[fifth]")),
        ],
        tokenizer,
        label_catalog=catalog,
    )

    assert len(batch) == 2
    assert batch[0].labels == ("A", "B")
    assert batch[0].actions == ("click[first]", "click[second]")
    assert batch[0].variant_token_ids == tuple(entry.token_ids for entry in catalog.entries[:2])
    assert len(batch[0].allowed_token_ids) == 4
    assert len(batch[1].allowed_token_ids) == 6
    assert required_max_logprobs(batch) == 6
    assert list(batch[0].prompt_token_ids) == tokenizer.apply_chat_template(
        [{"role": "user", "content": batch[0].prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )


def test_prepare_grouped_choice_batch_flattens_groups_and_preserves_page_identity(tokenizer):
    components = extract_product_page_contexts([example["raw_context"] for example in _EXAMPLES])
    goal_options = [_GOAL_OPTIONS[example["name"]] for example in _EXAMPLES]
    catalog = build_action_label_catalog(tokenizer, 15)

    prepared = prepare_product_page_grouped_choice_rollouts(components, tokenizer, goal_options, label_catalog=catalog)

    assert tuple((item.source_page_index, item.option_group.name) for item in prepared) == ((0, "size"), (0, "color"), (2, "size"))
    assert prepared[0].actions == tuple(f"click[{option}]" for option in prepared[0].option_group.values) + ("none",)
    assert prepared[0].correct_actions == ("click[19.7x31.5in+19.7x63in]",)
    assert prepared[0].correct_labels == ("C",)
    assert prepared[1].correct_actions == tuple(f"click[{option}]" for option in prepared[1].option_group.correct_options)
    assert prepared[1].correct_labels == ("B", "C", "D", "E", "F", "G")
    assert prepared[0].variant_token_ids == tuple(entry.token_ids for entry in catalog.entries[:6])
    assert required_max_logprobs(prepared) == 30
    assert list(prepared[0].prompt_token_ids) == tokenizer.apply_chat_template(
        [{"role": "user", "content": prepared[0].prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )


def test_prepare_grouped_choice_batch_validates_page_alignment(tokenizer, parts):
    with pytest.raises(ValueError, match="must align"):
        prepare_product_page_grouped_choice_rollouts([parts], tokenizer, [])
    assert prepare_product_page_grouped_choice_rollouts([], tokenizer, []) == ()


class _FakeInferenceEngine:
    def __init__(self, max_model_len=100_000, max_logprobs=110, omit_last_logprob=False):
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len, max_logprobs=max_logprobs))
        self.omit_last_logprob = omit_last_logprob
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        outputs = []
        for params in kwargs["sampling_params"]:
            assert params.n == 1
            assert params.max_tokens == 1
            assert params.temperature == 1.0
            assert params.top_p == 1.0
            assert params.top_k == -1
            assert params.min_p == 0.0
            assert params.presence_penalty == 0.0
            assert params.frequency_penalty == 0.0
            assert params.repetition_penalty == 1.0
            assert params.logprobs == len(params.allowed_token_ids)
            assert params.detokenize is False

            weights = list(range(1, len(params.allowed_token_ids) + 1))
            total = sum(weights)
            token_logprobs = {token_id: SimpleNamespace(logprob=math.log(weight / total)) for token_id, weight in zip(params.allowed_token_ids, weights)}
            if self.omit_last_logprob:
                token_logprobs.pop(params.allowed_token_ids[-1])
            completion = SimpleNamespace(token_ids=[params.allowed_token_ids[-1]], logprobs=[token_logprobs])
            outputs.append(SimpleNamespace(outputs=[completion]))
        return outputs


def test_grouped_choice_rollouts_use_existing_scorer_and_sum_all_correct_options(tokenizer):
    (components,) = extract_product_page_contexts([_EXAMPLES[0]["raw_context"]])
    prepared = prepare_product_page_grouped_choice_rollouts([components], tokenizer, [_GOAL_OPTIONS["two_option_groups"]])
    engine = _FakeInferenceEngine()

    scores = score_product_page_grouped_choice_rollouts(engine, prepared)

    assert len(engine.calls) == 1
    assert len(engine.calls[0]["prompts"]) == 2
    size_scores, color_scores = scores
    assert size_scores is not None
    assert size_scores.source_page_index == 0
    assert size_scores.option_group == prepared[0].option_group
    assert size_scores.correct_probability == pytest.approx(11 / 78)
    assert size_scores.correct_log_probability == pytest.approx(math.log(11 / 78))
    assert color_scores is not None
    assert color_scores.option_group == prepared[1].option_group
    # Labels B through G are all correct after color normalization.
    assert color_scores.correct_probability == pytest.approx((7 + 11 + 15 + 19 + 23 + 27) / 465)
    assert sum(color_scores.action_probabilities.values()) == pytest.approx(1.0)


def test_native_vllm_scoring_combines_bare_and_spaced_variants(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [replace(parts, admissible_actions=("click[first]", "click[second]", "click[third]"))],
        tokenizer,
    )
    engine = _FakeInferenceEngine()
    results = score_product_page_pseudo_rollouts(engine, prepared)

    assert len(engine.calls) == 1
    assert len(engine.calls[0]["prompts"]) == 1
    scores = results[0]
    assert scores is not None
    assert scores.label_probabilities == pytest.approx({"A": 3 / 21, "B": 7 / 21, "C": 11 / 21})
    assert scores.action_probabilities == pytest.approx({"click[first]": 3 / 21, "click[second]": 7 / 21, "click[third]": 11 / 21})
    assert sum(scores.label_probabilities.values()) == pytest.approx(1.0)
    assert scores.entropy > 0.0


def test_scoring_can_condition_each_row_on_assistant_response_prefix_tokens(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [replace(parts, admissible_actions=("click[first]", "click[second]"))],
        tokenizer,
    )
    engine = _FakeInferenceEngine()

    results = score_product_page_pseudo_rollouts(
        engine,
        prepared,
        assistant_response_prefix_token_ids=[(901, 902, 903)],
    )

    assert results[0] is not None
    assert engine.calls[0]["prompts"] == [{"prompt_token_ids": [*prepared[0].prompt_token_ids, 901, 902, 903]}]


def test_scoring_validates_assistant_response_prefix_alignment(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [replace(parts, admissible_actions=("click[first]", "click[second]"))],
        tokenizer,
    )

    with pytest.raises(ValueError, match="must align"):
        score_product_page_pseudo_rollouts(_FakeInferenceEngine(), prepared, assistant_response_prefix_token_ids=[])
    with pytest.raises(ValueError, match="integer token IDs"):
        score_product_page_pseudo_rollouts(_FakeInferenceEngine(), prepared, assistant_response_prefix_token_ids=[(True,)])


def test_overlength_pseudo_rollout_is_skipped_without_reaching_vllm(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [
            replace(parts, admissible_actions=("click[first]",)),
            replace(parts, admissible_actions=("click[second]",)),
        ],
        tokenizer,
    )
    max_model_len = len(prepared[0].prompt_token_ids)
    engine = _FakeInferenceEngine(max_model_len=max_model_len)
    shorter = replace(prepared[1], prompt_token_ids=prepared[1].prompt_token_ids[: max_model_len - 1])

    results = score_product_page_pseudo_rollouts(engine, [prepared[0], shorter])

    assert results[0] is None
    assert results[1] is not None
    assert len(engine.calls[0]["prompts"]) == 1


def test_scoring_rejects_missing_requested_logprob(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [replace(parts, admissible_actions=("click[first]", "click[second]"))],
        tokenizer,
    )
    with pytest.raises(ValueError, match="omitted requested action-label token IDs"):
        score_product_page_pseudo_rollouts(_FakeInferenceEngine(omit_last_logprob=True), prepared)


def test_scoring_checks_engine_max_logprobs_before_generation(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [replace(parts, admissible_actions=("click[first]", "click[second]"))],
        tokenizer,
    )
    engine = _FakeInferenceEngine(max_logprobs=3)

    with pytest.raises(ValueError, match="max_logprobs is 3, but this batch requires 4"):
        score_product_page_pseudo_rollouts(engine, prepared)
    assert engine.calls == []


def test_scoring_rejects_vllm_v1_unconstrained_logprobs(tokenizer, parts):
    prepared = prepare_product_page_pseudo_rollouts(
        [replace(parts, admissible_actions=("click[first]", "click[second]"))],
        tokenizer,
    )
    engine = _FakeInferenceEngine()
    v1_engine_type = type("LLMEngine", (), {"__module__": "vllm.v1.engine.llm_engine"})
    engine.llm_engine = v1_engine_type()
    engine.llm_engine.model_config = SimpleNamespace(max_model_len=100_000, max_logprobs=110)

    with pytest.raises(ValueError, match="VLLM_USE_V1=0"):
        score_product_page_pseudo_rollouts(engine, prepared)
    assert engine.calls == []


def test_none_is_distinct_from_a_literal_none_option_and_scores_optional_group(tokenizer, parts):
    observation = "'Back to Search' [SEP] '< Prev' [SEP] 'style' [SEP] 'none' [SEP] 'plain' [SEP] 'Towel' [SEP] 'Price: $10.00' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Buy Now'"
    actions = ("click[back to search]", "click[< prev]", "click[none]", "click[plain]", "click[description]", "click[features]", "click[buy now]")
    components = replace(parts, current_observation=observation, admissible_actions=actions)
    (choice,) = build_product_page_grouped_choice_prompt(components, tokenizer, {})
    assert choice.label_to_option == {"A": "none", "B": "plain", "C": None}
    assert choice.none_is_correct
    assert choice.correct_labels == ("A", "B", "C")
    (pseudo,) = prepare_product_page_grouped_choice_rollouts([components], tokenizer, [{}])
    assert pseudo.actions == ("click[none]", "click[plain]", "none")
    assert pseudo.correct_actions == pseudo.actions
    (score,) = score_product_page_grouped_choice_rollouts(_FakeInferenceEngine(), [pseudo])
    assert score.none_is_correct
    assert score.correct_probability == pytest.approx(1)
    assert score.action_probabilities["none"] > 0
    (required,) = build_product_page_grouped_choice_prompt(components, tokenizer, {"style": "plain"})
    assert not required.none_is_correct
    assert required.labels[-1] not in required.correct_labels


def test_none_correctness_uses_cross_group_coverage_without_selecting_two_values_from_one_group():
    from treehca.product_page_parser import ProductOptionGroup
    from treehca.pseudo_rollout_product_page import _none_can_satisfy_goal

    groups = (ProductOptionGroup("color", ("red", "blue")), ProductOptionGroup("style", ("red floral", "striped")))
    assert _none_can_satisfy_goal(groups, "color", {"color": "red"})
    assert _none_can_satisfy_goal(groups, "style", {"color": "red"})
    groups = (ProductOptionGroup("color", ("red", "blue")), ProductOptionGroup("size", ("large", "small")), ProductOptionGroup("combined", ("red", "large")))
    # Omitting color leaves a size group and a combined group: large + red works.
    assert _none_can_satisfy_goal(groups, "color", {"color": "red", "size": "large"})
    # Only one of red/large can be selected from combined; unioning all its
    # values would incorrectly claim the missing group is dispensable.
    limited = (ProductOptionGroup("required", ("red large",)), ProductOptionGroup("combined", ("red", "large")))
    assert not _none_can_satisfy_goal(limited, "required", {"color": "red", "size": "large"})
