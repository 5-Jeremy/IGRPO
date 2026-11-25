"""Verify token ownership, isolated advantage edits, and saved per-step results."""

import json

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.utils.rollout_logging import dump_generations
from agent_system.reasoning_steps import attach_reasoning_steps, ReasoningStepFormatter
from agent_system.reasoning_advantage import attach_reasoning_token_ids, adjust_reasoning_advantages
import agent_system.reasoning_advantage as module


class CharacterTokenizer:
    all_special_ids = [0]

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) + 1 for char in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, **kwargs):
        return "".join(chr(int(token) - 1) for token in ids if int(token))

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row) for row in rows]


def make_batch(outputs=None):
    outputs = outputs or ["<think>First é. Wait, reconsider.</think><action>Lima, Peru</action>",
                          "Other. However, retry.</think><action>Lima population</action>"]
    tokenizer = CharacterTokenizer()
    responses = torch.tensor([tokenizer(text)["input_ids"] + [0] * (max(map(len, outputs)) + 2 - len(text)) for text in outputs])
    prompts = torch.tensor([tokenizer("<think>")["input_ids"]] * len(outputs))
    advantages = torch.arange(responses.numel(), dtype=torch.float32).reshape(responses.shape) / 10 - 3
    mask = (responses != 0).long()
    batch = DataProto.from_dict(tensors={
        "responses": responses, "prompts": prompts, "advantages": advantages, "returns": advantages,
        "response_mask": mask, "attention_mask": torch.cat([torch.ones_like(prompts), mask], dim=1),
        "loss_mask": torch.cat([torch.ones_like(prompts), mask], dim=1),
    }, non_tensors={
        "id": np.array([f"turn-{i}" for i in range(len(outputs))], dtype=object),
        "previous_step_ids": np.array([[] for _ in outputs], dtype=object),
        "trajectory_id": np.array([f"traj-{i}" for i in range(len(outputs))], dtype=object),
        "trajectory_length": np.ones(len(outputs), dtype=int),
        "decision_model_probabilities": np.array([{"step_1": {"supports": .8}, "step_2": {"supports": .2}}] * len(outputs), dtype=object),
    })
    attach_reasoning_steps(batch, tokenizer)
    return batch, tokenizer


def test_spans_reference_original_output_with_prompt_tag_and_repeated_text():
    text = "  Same. Wait, Same. </think><action>Lima, Peru</action>"
    steps, action, spans = ReasoningStepFormatter().parse_with_spans(text, prompt="prefix <think>")
    assert [text[start:end] for start, end in spans] == steps == ["Same.", "Wait, Same."]
    assert action == "Lima, Peru"


@pytest.mark.parametrize("operation", ["action", "search", "answer", "code", "json"])
def test_sentence_newline_token_alignment_for_native_environment_formats(operation):
    thought = 'First "idea".\nOptions:\nSecond é 😀?'
    if operation == "json":
        output = json.dumps({"thoughts": thought, "action": "hit"})
    else:
        output = f"<think>{thought}</think><{operation}>chosen</{operation}>"
    batch, tokenizer = make_batch([output])
    attach_reasoning_steps(batch, tokenizer, None)
    attach_reasoning_token_ids(batch, tokenizer, None)
    assert batch.non_tensor_batch["reasoning_alignment_status"].tolist() == ["aligned"]
    assert batch.non_tensor_batch["reasoning_trace"][0]["steps"] == ['First "idea".', "Options:\nSecond é 😀?"]
    for number, expected in enumerate(batch.non_tensor_batch["reasoning_trace"][0]["steps"], 1):
        raw = tokenizer.decode(batch.batch["responses"][0, batch.batch["reasoning_step_ids"][0] == number])
        text = json.loads('"' + raw + '"') if operation == "json" else raw
        assert text == expected


def test_original_token_alignment_and_pass_through_preserve_all_tensors():
    batch, tokenizer = make_batch()
    snapshots = {key: value.clone() for key, value in batch.batch.items()}
    attach_reasoning_token_ids(batch, tokenizer)
    for row, trace in enumerate(batch.non_tensor_batch["reasoning_trace"]):
        for number, step in enumerate(trace["steps"], 1):
            tokens = batch.batch["responses"][row, batch.batch["reasoning_step_ids"][row] == number]
            assert tokenizer.decode(tokens) == step
    metrics = adjust_reasoning_advantages(batch)
    assert metrics["reasoning_advantage/scored_step_count"] == 4
    for key, value in snapshots.items():
        assert torch.equal(value, batch.batch[key])
    assert batch.batch["advantages"].data_ptr() != batch.batch["returns"].data_ptr()


def test_hook_receives_step_context_and_edits_only_unmasked_reasoning(monkeypatch):
    batch, tokenizer = make_batch()
    attach_reasoning_token_ids(batch, tokenizer)
    # Exclude a generated thinking token from the policy loss (e.g. a forced token).
    first = (batch.batch["reasoning_step_ids"][0] == 1).nonzero()[0].item()
    batch.batch["loss_mask"][0, batch.batch["prompts"].shape[1] + first] = 0
    before = batch.batch["advantages"].clone()
    calls = []

    def adjust(original_advantage, **kwargs):
        calls.append(kwargs)
        assert kwargs["step_probabilities"] == kwargs["all_step_probabilities"][f"step_{kwargs['step_number']}"]
        assert kwargs["step_text"] == kwargs["reasoning_steps"][kwargs["step_number"] - 1]
        # Even an in-place experimental rule must not mutate returns/baseline.
        return original_advantage.add_(kwargs["step_number"])

    monkeypatch.setattr(module, "apply_reasoning_advantage_adjustment", adjust)
    adjust_reasoning_advantages(batch)
    ids = batch.batch["reasoning_step_ids"]
    mask = batch.batch["loss_mask"][:, -ids.shape[1]:].bool()
    assert torch.equal(batch.batch["advantages"], before + ids * mask)
    assert torch.equal(batch.batch["returns"], before)
    assert [(call["turn_id"], call["step_number"]) for call in calls] == [("turn-0", 1), ("turn-0", 2), ("turn-1", 1), ("turn-1", 2)]


def test_failed_alignment_and_missing_trace_are_explicit_and_do_not_change_advantages():
    batch, tokenizer = make_batch(["<think>First.</think><action>Lima, Peru</action>", "<action>bad</action>"])
    class MismatchedTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            result["input_ids"][0] += 1
            return result
    before = batch.batch["advantages"].clone()
    attach_reasoning_token_ids(batch, MismatchedTokenizer())
    metrics = adjust_reasoning_advantages(batch)
    assert batch.non_tensor_batch["reasoning_alignment_status"].tolist() == ["token_id_mismatch", "no_trace"]
    assert metrics["reasoning_advantage/alignment_failure_count"] == 1
    assert batch.non_tensor_batch["reasoning_step_advantages"][0]["step_1"]["advantage"] is None
    assert batch.non_tensor_batch["reasoning_step_advantages"][1] is None
    assert torch.equal(batch.batch["advantages"], before)


def test_missing_probabilities_still_logs_advantages_and_zero_trainable_tokens(monkeypatch):
    batch, tokenizer = make_batch()
    attach_reasoning_token_ids(batch, tokenizer)
    del batch.non_tensor_batch["decision_model_probabilities"]
    batch.batch["loss_mask"][1] = 0
    def adjust(original_advantage, **kwargs):
        assert kwargs["step_probabilities"] is None
        assert kwargs["all_step_probabilities"] == {}
        return original_advantage
    monkeypatch.setattr(module, "apply_reasoning_advantage_adjustment", adjust)
    metrics = adjust_reasoning_advantages(batch)
    assert metrics["reasoning_advantage/scored_step_count"] == 2
    assert batch.non_tensor_batch["reasoning_step_advantages"][1]["step_1"]["token_count"] == 0


@pytest.mark.parametrize("bad_result", [lambda x: x[:0], lambda x: x * float('nan'), lambda x: 1.0])
def test_bad_hook_output_fails_before_replacing_advantages(monkeypatch, bad_result):
    batch, tokenizer = make_batch()
    attach_reasoning_token_ids(batch, tokenizer)
    original = batch.batch["advantages"]
    monkeypatch.setattr(module, "apply_reasoning_advantage_adjustment", lambda x, **kwargs: bad_result(x))
    with pytest.raises(ValueError, match="Invalid reasoning advantage result"):
        adjust_reasoning_advantages(batch)
    assert batch.batch["advantages"] is original


def test_reordering_padding_jsonl_and_visualizer_keep_final_step_advantages(tmp_path):
    from rollout_visualizer.data import read_document
    batch, tokenizer = make_batch()
    attach_reasoning_token_ids(batch, tokenizer)
    batch = batch.select_idxs([1, 0, 1])
    metrics = adjust_reasoning_advantages(batch)
    assert metrics["reasoning_advantage/step_count"] == 4  # Excludes duplicate padding.
    expected = dict(zip(batch.non_tensor_batch["id"], batch.non_tensor_batch["reasoning_step_advantages"]))
    path = tmp_path / "1.jsonl"
    dump_generations(path, ["prompt"] * 3, tokenizer.batch_decode(batch.batch["responses"]), [0] * 3, {}, batch.non_tensor_batch)
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(saved) == 2
    assert {row["id"]: row["reasoning_step_advantages"] for row in saved} == expected
    document = read_document(path)
    displayed = [row for group in document["groups"] for node in group["nodes"] for row in node["records"]]
    assert {row["id"]: row["reasoning_step_advantages"] for row in displayed} == expected


def test_byte_level_tokenizer_offsets_unicode_and_boundary_tokens():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    output = "<think>Café 🙂. Wait, café.</think><action>Lima, Peru</action>"
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([output], trainers.BpeTrainer(vocab_size=80, special_tokens=["[PAD]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", clean_up_tokenization_spaces=False)
    batch, _ = make_batch([output])
    responses = torch.tensor([tokenizer(output)["input_ids"] + [tokenizer.pad_token_id]])
    # Only alignment depends on responses; build a minimal batch at this token length.
    batch = DataProto.from_dict(tensors={"responses": responses}, non_tensors={
        "reasoning_trace": batch.non_tensor_batch["reasoning_trace"],
    })
    attach_reasoning_token_ids(batch, tokenizer)
    assert batch.non_tensor_batch["reasoning_alignment_status"].tolist() == ["aligned"]
    assert (batch.batch["reasoning_step_ids"] == 1).any()
    assert (batch.batch["reasoning_step_ids"] == 2).any()
    offsets = tokenizer(output, return_offsets_mapping=True)["offset_mapping"]
    spans = ReasoningStepFormatter().parse_with_spans(output)[2]
    for index, (start, end) in enumerate(offsets):
        number = batch.batch["reasoning_step_ids"][0, index].item()
        if number:
            assert spans[number - 1][0] <= start < end <= spans[number - 1][1]
    assert batch.batch["reasoning_step_ids"][0, -1] == 0
