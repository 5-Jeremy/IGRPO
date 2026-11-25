"""Check content entropy, alignment failures, and persistence through rollout logs."""

import json

import pytest
import torch

from agent_system.reasoning_entropy import answer_span, attach_reasoning_entropy
from agent_system.reasoning_steps import attach_reasoning_steps
from verl.utils.rollout_logging import dump_generations
from rollout_visualizer.data import read_document
from tests.trainer.ppo.test_reasoning_advantage import CharacterTokenizer, make_batch


@pytest.mark.parametrize("tag", ["action", "answer", "search", "code", "json", "fenced_json"])
def test_content_entropies_for_all_response_formats(tag):
    thought = 'First é.\nWait, "retry" 🙂.'
    output = (json.dumps({"thoughts": thought, "action": 'say "yes"'}) if 'json' in tag
              else f'<think>{thought}</think><{tag}> say "yes" </{tag}>')
    if tag == 'fenced_json':
        output = f'```json\n{output}\n```'
    batch, tokenizer = make_batch([output])
    attach_reasoning_steps(batch, tokenizer, None)
    entropys = torch.arange(batch.batch["responses"].numel(), dtype=torch.float32).reshape(1, -1) / 10
    # Formatting and padding carry large values and must not contaminate summaries.
    start, end = answer_span(output)
    entropys[0, start:end] = 2
    entropys[0, len(output):] = float('nan')
    batch.batch["loss_mask"].zero_()  # Entropy is about generated content, not loss weighting.
    batch.batch["response_mask"].zero_()
    snapshots = {k: v.clone() for k, v in batch.batch.items()}
    attach_reasoning_entropy(batch, tokenizer, entropys, None)
    assert batch.non_tensor_batch["entropy_alignment_status"].tolist() == ["aligned"]
    answer = batch.non_tensor_batch["answer_entropy"][0]
    assert answer == {"mean": 2, "sum": 2 * (end - start), "token_count": end - start}
    steps = batch.non_tensor_batch["reasoning_step_entropies"][0]
    assert set(steps) == {"step_1", "step_2"}
    for entry in steps.values():
        assert entry["token_count"] > 0
        assert entry["sum"] == pytest.approx(entry["mean"] * entry["token_count"])
    for key, value in snapshots.items():
        assert torch.equal(batch.batch[key], value)


def test_prompt_think_and_action_only_and_missing_content():
    outputs = ['First. Wait, retry.</think><action>look</action>', '<answer>zero</answer>', 'invalid']
    batch, tokenizer = make_batch(outputs)
    entropys = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
    attach_reasoning_entropy(batch, tokenizer, entropys)
    assert batch.non_tensor_batch["entropy_alignment_status"].tolist() == ['aligned', 'aligned', 'no_content']
    assert len(batch.non_tensor_batch["reasoning_step_entropies"][0]) == 2
    assert batch.non_tensor_batch["reasoning_step_entropies"][1] is None
    assert batch.non_tensor_batch["answer_entropy"][1] == {"mean": 0, "sum": 0, "token_count": 4}
    assert batch.non_tensor_batch["answer_entropy"][2] is None


@pytest.mark.parametrize("failure", ['mismatch', 'no_offsets'])
def test_unverifiable_offsets_do_not_fabricate_entropy(failure):
    batch, _ = make_batch()
    class BrokenTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            if failure == 'no_offsets':
                raise NotImplementedError
            result = super().__call__(text, **kwargs)
            result['input_ids'][0] += 1
            return result
    attach_reasoning_entropy(batch, BrokenTokenizer(), torch.ones_like(batch.batch['responses'], dtype=torch.float32))
    assert batch.non_tensor_batch['entropy_alignment_status'][0] == ('offsets_unavailable' if failure == 'no_offsets' else 'token_id_mismatch')
    assert batch.non_tensor_batch['answer_entropy'][0] is None
    assert batch.non_tensor_batch['reasoning_step_entropies'][0]['step_1']['mean'] is None


def test_masks_nonfinite_values_and_no_tokens_are_distinct_from_zero():
    output = '<think>One.</think><action>look</action>'
    batch, tokenizer = make_batch([output])
    entropys = torch.ones_like(batch.batch['responses'], dtype=torch.float32)
    start, end = answer_span(output)
    prompt_length = batch.batch['prompts'].shape[1]
    batch.batch['attention_mask'][0, prompt_length + start:prompt_length + end] = 0
    entropys[0, output.index('One')] = float('nan')
    attach_reasoning_entropy(batch, tokenizer, entropys)
    assert batch.non_tensor_batch['answer_entropy'][0] == {'mean': None, 'sum': None, 'token_count': 0}
    assert batch.non_tensor_batch['reasoning_step_entropies'][0]['step_1'] == {'mean': None, 'sum': None, 'token_count': 4}


def test_reordering_duplicate_padding_and_jsonl_reach_visualizer(tmp_path):
    batch, tokenizer = make_batch()
    path = tmp_path / '1.jsonl'
    def dump(current):
        dump_generations(path, ['prompt'] * len(current), tokenizer.batch_decode(current.batch['responses']),
                         [0] * len(current), {}, current.non_tensor_batch)
    dump(batch)
    assert all('answer_entropy' not in json.loads(line) for line in path.read_text().splitlines())
    entropys = torch.stack([torch.ones_like(batch.batch['responses'][0]), 2 * torch.ones_like(batch.batch['responses'][1])])
    attach_reasoning_entropy(batch, tokenizer, entropys)
    expected = dict(zip(batch.non_tensor_batch['id'], batch.non_tensor_batch['answer_entropy']))
    batch = batch.select_idxs([1, 0, 1])
    dump(batch)
    rows = [r for g in read_document(path)['groups'] for n in g['nodes'] for r in n['records']]
    assert len(rows) == 2
    assert {r['id']: r['answer_entropy'] for r in rows} == expected
    assert all(r['reasoning_step_entropies']['step_2']['mean'] == r['answer_entropy']['mean'] for r in rows)


def test_byte_tokenizer_includes_boundary_tokens_once():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    output = '<think>Café 🙂. Wait, café.</think><action>look</action>'
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator([output] * 20, trainers.BpeTrainer(vocab_size=320, special_tokens=['[PAD]'],
                                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]')
    from verl import DataProto
    tokens = tokenizer(output, add_special_tokens=False, return_offsets_mapping=True)
    responses = torch.tensor([tokens['input_ids'] + [tokenizer.pad_token_id]])
    batch = DataProto.from_dict(tensors={'responses': responses, 'attention_mask': (responses != tokenizer.pad_token_id).long()})
    attach_reasoning_steps(batch, tokenizer)
    attach_reasoning_entropy(batch, tokenizer, torch.ones_like(responses, dtype=torch.float32))
    summaries = [*batch.non_tensor_batch['reasoning_step_entropies'][0].values(), batch.non_tensor_batch['answer_entropy'][0]]
    assert all(entry['mean'] == 1 for entry in summaries)
    from agent_system.reasoning_steps import ReasoningStepFormatter
    spans = [*ReasoningStepFormatter().parse_with_spans(output)[2], answer_span(output)]
    count = sum(any(min(end, right) > max(start, left) for left, right in spans) for start, end in tokens['offset_mapping'])
    assert sum(entry['token_count'] for entry in summaries) == count


@pytest.mark.parametrize('calculate_entropy', [True, False])
def test_dynamic_microbatching_restores_entropy_row_order(monkeypatch, calculate_entropy):
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    import verl.workers.actor.dp_actor as actor_module

    monkeypatch.setattr('verl.utils.debug.performance._get_current_mem_info', lambda: (0, 0, 0, 0))
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = torch.nn.Linear(1, 1)
    actor.ulysses_sequence_parallel_size = 1
    responses = torch.tensor([[10, 11], [20, 21], [30, 31]])
    data = DataProto.from_dict(tensors={
        'responses': responses, 'input_ids': responses, 'attention_mask': torch.ones_like(responses),
        'position_ids': torch.zeros_like(responses),
    }, meta_info={'micro_batch_size': 2, 'temperature': 1, 'use_dynamic_bsz': True, 'max_token_len': 4})
    monkeypatch.setattr(actor_module, 'rearrange_micro_batches',
                        lambda batch, **kwargs: ([batch[[2, 0]], batch[[1]]], [[2, 0], [1]]))
    actor._forward_micro_batch = lambda micro_batch, **kwargs: (
        micro_batch['responses'].float() / 10 if calculate_entropy else None, -micro_batch['responses'].float())
    log_probs, entropys = actor.compute_log_prob(data, calculate_entropy=calculate_entropy)
    torch.testing.assert_close(log_probs, -responses.float())
    if calculate_entropy:
        torch.testing.assert_close(entropys, responses.float() / 10)
    else:
        assert entropys is None
