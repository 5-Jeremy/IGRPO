"""Training-format rollout records with native transitions and controlled logits."""

import copy
import json
import math
import random
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from treehca import webshop_turn_success_testbed as testbed
from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_product_page_grouped_choices_testbed import sample_diverse_product_goals


@pytest.fixture(scope="module")
def native():
    server = testbed.create_server(testbed._DEFAULT_CATALOG, testbed._DEFAULT_ATTRIBUTES, 0, 1000)
    selected = sample_diverse_product_goals(server.all_products, server.goals[500:], 1, 0)[0]
    index = next(index for index, goal in enumerate(server.goals) if goal is selected.goal)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    return server, selected, index, tokenizer


class ScriptedEngine:
    """Native oracle, malformed purchase, and timed-out rollouts in one batch."""

    def __init__(self, tokenizer, selected, args):
        self.tokenizer, self.selected, self.args = tokenizer, selected, args
        self.calls = []
        self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32768, max_logprobs=2048), cache_config=SimpleNamespace(enable_prefix_caching=False))

    def _action(self, text, rollout, step):
        if rollout == 1:
            return "not-a-command"
        if step == 0:
            action = f"search[<q> {self.selected.product['query']}]"
        elif "Your current observation is:" in text or "your current observation is:" in text:
            # Only the current observation/action block controls this script.
            user = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
            parts = extract_product_page_contexts([user])[0]
            if "click[buy now]" not in parts.admissible_actions:
                target = f"click[{self.selected.product['asin'].lower()}]"
                action = target if target in parts.admissible_actions else "click[next >]"
            else:
                # Rollout 2 buys without options, exercising native partial credit.
                if rollout == 2:
                    action = "click[buy now]"
                else:
                    history = parts.history_block or ""
                    remaining = [f"click[{value}]" for value in self.selected.goal["goal_options"].values() if f"Action {parts.completed_steps}: 'click[{value}]'" not in history]
                    # A long history in this fixture retains every prior selection.
                    remaining = [action for action in remaining if f"'{action}']" not in history]
                    action = remaining[0] if remaining else "click[buy now]"
        else:
            raise AssertionError(text)
        # A malformed but executable buy must earn 10 - 0.1 on this turn.
        if action == "click[buy now]" and rollout == 0:
            return "<action>click[buy now]</action>"
        return f"<think>Choose the next action.</think><action>{action}</action>"

    def generate(self, *, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params))
        outputs = []
        for prompt, params in zip(prompts, sampling_params):
            tokens = prompt["prompt_token_ids"]
            if params.allowed_token_ids:
                values = {token: -math.log(len(params.allowed_token_ids)) for token in params.allowed_token_ids}
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[values])]))
            elif params.prompt_logprobs == 0:
                outputs.append(SimpleNamespace(prompt_logprobs=[{token: -0.1} for token in tokens]))
            else:
                rollout, step = divmod(params.seed - self.args.seed - 10_000, self.args.max_steps)
                text = self._action(self.tokenizer.decode(tokens), rollout, step)
                response = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=response)]))
        return outputs


def arguments(native, tmp_path, *extra):
    return testbed.build_argument_parser().parse_args(
        [
            "--num-rollouts",
            "3",
            "--group-size",
            "3",
            "--goal-indices",
            str(native[2]),
            "--history-length",
            "15",
            "--output",
            str(tmp_path / "rollouts.jsonl"),
            *extra,
        ]
    )


def test_complete_testbed_matches_training_scores_and_preserves_rollout_identity(native, tmp_path, monkeypatch):
    server, selected, _, tokenizer = native
    args = arguments(native, tmp_path, "--pseudo-batch-size", "2")
    engine = ScriptedEngine(tokenizer, selected, args)
    monkeypatch.setattr(testbed, "create_server", lambda *_: server)
    monkeypatch.setattr(testbed, "_load_inference", lambda *_: (tokenizer, engine))
    result = testbed.run_testbed(args)
    assert result["status"] == "complete"
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    metadata = json.loads(args.output.with_suffix(".metadata.json").read_text())
    assert metadata == result
    assert metadata["pseudo_batch_size"] == 2
    assert all(len(prompts) <= 2 for prompts, params in engine.calls if params[0].prompt_logprobs is not None or params[0].allowed_token_ids)
    assert result["num_turns"] == len(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["rollout_index"]].append(row)
        assert "step" not in row
        assert row["data_source"] == "text"
        assert row["input"].startswith("system\nYou are Qwen")
        assert "<|im_start|>" not in row["input"] and "<|im_end|>" not in row["output"]
        assert row["pseudo_skip_reason"] != "pending"
    assert set(grouped) == {0, 1, 2}
    assert len({row["uid"] for row in rows}) == 1
    assert len({row["traj_uid"] for row in rows}) == 3
    for index, turns in grouped.items():
        assert [row["traj_step"] for row in turns] == list(range(len(turns)))
        assert all(row["episode_lengths"] == len(turns) for row in turns)
        assert turns[0]["pseudo_probability"] is None
        assert turns[0]["pseudo_skip_reason"] == "unsupported_page:"
        assert len(turns) == result["episodes"][index]["episode_lengths"]
    successful = grouped[0]
    assert successful[-1]["episode_rewards"] == successful[-1]["rewards"] == 10
    assert all(row["score"] == 10 for row in successful[:-1])
    assert successful[-1]["score"] == float(np.float32(10) - np.float32(0.1))
    assert not successful[-1]["is_action_valid"]
    assert successful[-1]["pseudo_probability"] == 1
    assert all(row["score"] == float(np.float32(-0.1)) for row in grouped[1])
    assert len(grouped[1]) == args.max_steps
    assert all(row["score"] == 0 for row in grouped[2])
    assert result["episodes"][0]["termination_reason"] == "purchase"
    assert result["episodes"][1]["termination_reason"] == "step_limit"
    assert result["episodes"][2]["termination_reason"] == "purchase"
    assert result["full_reward_rollouts"] == 1
    kinds = ["pseudo" if params[0].allowed_token_ids or params[0].prompt_logprobs == 0 else "rollout" for _, params in engine.calls]
    first_pseudo = kinds.index("pseudo")
    assert all(kind == "rollout" for kind in kinds[:first_pseudo])
    assert all(kind == "pseudo" for kind in kinds[first_pseudo:])
    assert len(engine.calls[0][0]) == args.num_rollouts
    assert len(engine.calls[first_pseudo - 1][0]) == 1
    assert all(len(prompts) <= 32 for prompts, _ in engine.calls[first_pseudo:])


def test_collection_uses_actual_training_manager_and_freezes_pre_action_states(native, tmp_path):
    server, selected, _, tokenizer = native
    args = arguments(native, tmp_path)
    records, episodes, source = testbed.collect_parallel_rollouts(server, testbed.TrainingRolloutPolicy(ScriptedEngine(tokenizer, selected, args), tokenizer, args), args)
    success = [row for row in records if row.rollout_index == 0]
    assert success[0].snapshot.page_type == ""
    assert success[1].snapshot.page_type == "search_results"
    product = next(row for row in success if row.snapshot.page_type == "item_page")
    assert product.snapshot.fresh_product_entry
    assert not product.snapshot.selected_options
    assert success[-1].snapshot.selected_options
    assert success[-1].done and not success[-1].snapshot.terminated
    assert all(row.snapshot.catalog_key == source.catalog_key for row in records)
    assert all(row.snapshot.goal_json == success[0].snapshot.goal_json for row in success)
    testbed.assign_training_scores(records, episodes, tokenizer, args)
    no_penalty = replace(success[-1], score=None)
    disabled = copy.copy(args)
    disabled.invalid_action_penalty = 0
    testbed.assign_training_scores([no_penalty], episodes, tokenizer, disabled)
    assert no_penalty.score == 10


def test_tokenization_matches_training_preprocessor_and_retains_real_output_ids(native, tmp_path):
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from verl import DataProto

    server, selected, _, tokenizer = native
    args = arguments(native, tmp_path, "--max-steps", "1")
    engine = ScriptedEngine(tokenizer, selected, args)
    records, _, _ = testbed.collect_parallel_rollouts(server, testbed.TrainingRolloutPolicy(engine, tokenizer, args), args)
    config = OmegaConf.create({"data": {"max_prompt_length": args.max_prompt_length, "truncation": "error", "return_raw_chat": True}})
    collector = TrajectoryCollector(config, tokenizer)
    gen_batch = DataProto.from_dict(
        tensors={"input_ids": torch.zeros((1, 1), dtype=torch.long)},
        non_tensors={"raw_prompt": np.array([[]], dtype=object), "data_source": np.array(["text"], dtype=object), "ground_truth": np.array([None], dtype=object)},
    )
    processed = collector.preprocess_single_sample(0, gen_batch, {"text": [records[0].snapshot.prompt], "image": None, "anchor": ["observation"]})
    expected = tuple(processed["input_ids"][processed["attention_mask"].bool()].tolist())
    assert records[0].generation.prompt_token_ids == expected
    assert records[0].generation.response_token_ids[-1] == tokenizer.eos_token_id


def test_group_sampling_matches_training_split_rng_and_partial_final_group(native, tmp_path):
    server, _, _, _ = native
    args = arguments(native, tmp_path)
    args.goal_indices = None
    args.num_rollouts, args.group_size = 5, 2
    expected = np.random.RandomState(0).choice(range(500, len(server.goals)), size=3, replace=False)
    assert testbed.select_goal_indices(server, args) == np.repeat(expected, 2).tolist()[:5]
    args.split = "test"
    assert all(index < 500 for index in testbed.select_goal_indices(server, args))
    args.goal_indices = [500, 501, 502]
    with pytest.raises(ValueError, match="selected split"):
        testbed.select_goal_indices(server, args)


def test_pseudo_failure_keeps_training_rollouts_with_pending_markers(native, tmp_path, monkeypatch):
    server, selected, _, tokenizer = native
    args = arguments(native, tmp_path, "--max-steps", "1")
    engine = ScriptedEngine(tokenizer, selected, args)
    monkeypatch.setattr(testbed, "create_server", lambda *_: server)
    monkeypatch.setattr(testbed, "_load_inference", lambda *_: (tokenizer, engine))

    def fail(*_, **__):
        raise ValueError("deliberate scoring failure")

    monkeypatch.setattr(testbed.WebshopTurnSuccessScorer, "score", fail)
    with pytest.raises(ValueError, match="deliberate"):
        testbed.run_testbed(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == 3 and all(row["score"] is not None for row in rows)
    assert all(row["pseudo_skip_reason"] == "pending" for row in rows)
    assert json.loads(args.output.with_suffix(".metadata.json").read_text())["status"] == "failed"


def test_rollout_batches_are_bounded_and_seeded_by_original_slot(native, tmp_path):
    server, selected, _, tokenizer = native
    args = arguments(native, tmp_path, "--rollout-batch-size", "2", "--max-steps", "2")
    engine = ScriptedEngine(tokenizer, selected, args)
    testbed.collect_parallel_rollouts(server, testbed.TrainingRolloutPolicy(engine, tokenizer, args), args)
    assert [len(prompts) for prompts, _ in engine.calls] == [2, 1, 2, 1]
    assert [[params.seed for params in row] for _, row in engine.calls] == [[10000, 10002], [10004], [10001, 10003], [10005]]


def test_local_sessions_isolate_random_searches_like_separate_workers(native):
    server, _, goal_index, _ = native
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    batch = testbed.LocalWebshopRolloutBatch(server, [goal_index, goal_index], group_size=2, seed=123)
    try:
        batch.reset()
        observations, _, _, _ = batch.step(["search[<r>]", "search[<r>]"])
        assert observations[0] == observations[1]
        second, _, _, _ = batch.step(["search[<r>]", "search[<r>]"])
        assert second[0] == second[1] and second != observations
    finally:
        batch.close()
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0] and actual_numpy[2:] == numpy_state[2:]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("name,value", [("num_rollouts", 0), ("group_size", 0), ("max_steps", 0), ("pseudo_batch_size", 0), ("pseudo_batch_size", 33), ("history_length", -1), ("invalid_action_penalty", -1), ("path_probability_threshold", 2), ("top_k", 0)])
def test_bad_arguments_fail_before_model_loading(native, tmp_path, name, value):
    args = arguments(native, tmp_path)
    setattr(args, name, value)
    with pytest.raises(ValueError):
        testbed._validate_arguments(args)
