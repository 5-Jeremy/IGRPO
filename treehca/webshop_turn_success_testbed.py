"""Run parallel WebShop trajectories, then annotate training-style JSONL turns.

See docs/treehca/webshop_turn_success_testbed.md for score and snapshot semantics.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from treehca.pseudo_rollout_batch import MixedPseudoRolloutScorer
from treehca.pseudo_rollout_product_page_outcomes_testbed import _DEFAULT_ATTRIBUTES, _DEFAULT_CATALOG, _webshop_modules, create_server
from treehca.webshop_probability_snapshot import WebshopSnapshotSource, WebshopTurnSnapshot
from treehca.webshop_turn_success import WebshopTurnSuccessScorer

logger = logging.getLogger(__name__)
_DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "pseudo_prob_test_results/webshop_turn_success_rollouts.jsonl"


@dataclass(frozen=True)
class GeneratedTurn:
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]


@dataclass
class RecordedTurn:
    rollout_index: int
    traj_step: int
    generation: GeneratedTurn
    snapshot: WebshopTurnSnapshot
    is_action_valid: bool
    reward: float
    task_score: float
    done: bool
    score: float | None = None
    pseudo_probability: float | None = None
    pseudo_skip_reason: str | None = "pending"


class TrainingRolloutPolicy:
    """Training chat tokenization and sampling, preserving actual generated IDs."""

    def __init__(self, engine: Any, tokenizer: Any, args: argparse.Namespace):
        self.engine, self.tokenizer, self.args = engine, tokenizer, args

    def generate(self, prompts: list[str], seeds: list[int]) -> list[GeneratedTurn]:
        from vllm import SamplingParams

        from verl.utils.torch_functional import tokenize_and_postprocess_data

        if len(prompts) != len(seeds):
            raise ValueError("Rollout seeds must align with prompts")
        turns = []
        limit = self.engine.llm_engine.model_config.max_model_len
        for start in range(0, len(prompts), self.args.rollout_batch_size):
            batch = prompts[start : start + self.args.rollout_batch_size]
            token_rows = []
            for prompt in batch:
                chat = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                ids, mask = tokenize_and_postprocess_data(
                    prompt=chat,
                    tokenizer=self.tokenizer,
                    max_length=self.args.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation="error",
                )
                tokens = tuple(ids[0][mask[0].bool()].tolist())
                if len(tokens) + self.args.max_new_tokens > limit:
                    raise ValueError("A rollout prompt plus response exceeds max_model_len")
                token_rows.append(tokens)
            params = [
                SamplingParams(
                    n=1,
                    logprobs=0,
                    max_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    min_p=0.0,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    repetition_penalty=1.0,
                    seed=seed,
                    detokenize=False,
                )
                for seed in seeds[start : start + len(batch)]
            ]
            outputs = self.engine.generate(prompts=[{"prompt_token_ids": list(tokens)} for tokens in token_rows], sampling_params=params, use_tqdm=False)
            if len(outputs) != len(batch) or any(len(output.outputs) != 1 for output in outputs):
                raise ValueError("vLLM returned misaligned rollout responses")
            for tokens, output in zip(token_rows, outputs):
                response = tuple(output.outputs[0].token_ids)
                if not response or len(response) > self.args.max_new_tokens:
                    raise ValueError("Every rollout response must contain 1..max_new_tokens tokens")
                turns.append(GeneratedTurn(tokens, response))
        return turns


class LocalWebshopRolloutBatch:
    """N independent native sessions with batched policy inference.

    Use the production worker's reset/step and reward conversion, without Ray
    or N separate copies of the catalog/search index. Finished slots are inert.
    """

    def __init__(self, server: Any, goal_indices: list[int], group_size: int, seed: int):
        from agent_system.environments.env_package.webshop.envs import WebshopWorker

        _, native = _webshop_modules()
        self.goal_indices = goal_indices
        self.workers = []
        self._python_rng_states = []
        self._numpy_rng_states = []
        self.finished = [False] * len(goal_indices)
        for index, goal_index in enumerate(goal_indices):
            private = copy.copy(server)
            private.user_sessions = {}
            # Bypass only the expensive worker constructor. The real native env
            # and production worker reset/step methods supply all transitions.
            worker = object.__new__(WebshopWorker)
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
                with torch.random.fork_rng(devices=devices):
                    worker.env = native.WebAgentTextEnv(observation_mode="text", server=private, seed=seed + index // group_size, session=goal_index)
                self._python_rng_states.append(random.getstate())
                self._numpy_rng_states.append(np.random.get_state())
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
            self.workers.append(worker)

    def _run_native(self, index, method, *args):
        # Ray workers normally isolate these globals. Preserve that separation
        # for random searches and purchase auto-reset in local sessions too.
        python_state, numpy_state = random.getstate(), np.random.get_state()
        random.setstate(self._python_rng_states[index])
        np.random.set_state(self._numpy_rng_states[index])
        try:
            return method(*args)
        finally:
            self._python_rng_states[index] = random.getstate()
            self._numpy_rng_states[index] = np.random.get_state()
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def reset(self):
        self.finished = [False] * len(self.workers)
        rows = [self._run_native(index, worker.reset, goal_index) for index, (worker, goal_index) in enumerate(zip(self.workers, self.goal_indices))]
        return [row[0] for row in rows], [row[1] for row in rows]

    def step(self, actions):
        if len(actions) != len(self.workers):
            raise ValueError("Actions must align with environment slots")
        rows = []
        for index, (worker, action) in enumerate(zip(self.workers, actions)):
            if self.finished[index]:
                rows.append((worker.env.observation, 0.0, True, {"available_actions": worker.get_available_actions(), "won": False, "task_score": 0.0}))
                continue
            row = self._run_native(index, worker.step, action)
            self.finished[index] = bool(row[2])
            rows.append(row)
        return tuple([row[column] for row in rows] for column in range(4))

    def close(self):
        for worker in self.workers:
            worker.close()


def select_goal_indices(server: Any, args: argparse.Namespace) -> list[int]:
    """Mirror training's index split and without-replacement group sampling."""
    group_count = math.ceil(args.num_rollouts / args.group_size)
    pool = list(range(500, len(server.goals))) if args.split == "train" else list(range(min(500, len(server.goals))))
    if args.goal_indices is None:
        if group_count > len(pool):
            raise ValueError(f"Requested {group_count} groups but {args.split} contains only {len(pool)} goals")
        indices = np.random.RandomState(args.seed).choice(pool, size=group_count, replace=False).tolist()
    else:
        indices = args.goal_indices
        if len(indices) != group_count or any(index not in pool for index in indices):
            raise ValueError("--goal-indices requires one index per group, all within the selected split")
    return np.repeat(indices, args.group_size).tolist()[: args.num_rollouts]


def collect_parallel_rollouts(server: Any, policy: TrainingRolloutPolicy, args: argparse.Namespace):
    """Start at search input, collect active turns, and freeze pre-action states."""
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop.projection import webshop_projection

    goal_indices = select_goal_indices(server, args)
    source = WebshopSnapshotSource(server)
    batch = LocalWebshopRolloutBatch(server, goal_indices, args.group_size, args.seed)
    manager = WebshopEnvironmentManager(batch, webshop_projection, SimpleNamespace(env=SimpleNamespace(history_length=args.history_length)))
    group_uids = [str(uuid4()) for _ in range(math.ceil(args.num_rollouts / args.group_size))]
    episodes = [
        {"rollout_index": index, "uid": group_uids[index // args.group_size], "traj_uid": str(uuid4()), "goal_index": goal_index, "goal": copy.deepcopy(server.goals[goal_index]), "episode_rewards": 0.0, "episode_lengths": 0, "task_score": 0.0, "termination_reason": "step_limit"}
        for index, goal_index in enumerate(goal_indices)
    ]
    previous_types = [None] * args.num_rollouts
    records = []
    try:
        observations, _ = manager.reset(None)
        for step in range(args.max_steps):
            active = [index for index, done in enumerate(batch.finished) if not done]
            if not active:
                break
            logger.info("Rollout turn %d/%d: %d active trajectories", step + 1, args.max_steps, len(active))
            snapshots = [source.capture(batch.workers[index].env, manager, observations["text"][index], rollout_index=index, previous_page_type=previous_types[index]) for index in active]
            # Per-trajectory seeds are stable when other trajectories terminate.
            seeds = [args.seed + 10_000 + index * args.max_steps + step for index in active]
            generated = policy.generate([snapshot.prompt for snapshot in snapshots], seeds)
            if len(generated) != len(active):
                raise ValueError("Generated turns must align with active rollout slots")
            responses = [""] * args.num_rollouts
            for index, turn in zip(active, generated):
                responses[index] = policy.tokenizer.decode(turn.response_token_ids, skip_special_tokens=True)
            observations, rewards, dones, infos = manager.step(responses)
            for index, snapshot, turn in zip(active, snapshots, generated):
                reward = float(rewards[index])
                valid = bool(infos[index]["is_action_valid"])
                records.append(RecordedTurn(index, step, turn, snapshot, valid, reward, float(infos[index]["task_score"]), bool(dones[index])))
                episode = episodes[index]
                episode["episode_rewards"] = float(np.float32(episode["episode_rewards"]) + np.float32(reward))
                episode["episode_lengths"] += 1
                episode["task_score"] = float(infos[index]["task_score"])
                if dones[index]:
                    episode["termination_reason"] = "purchase"
                previous_types[index] = snapshot.page_type
    finally:
        batch.close()
    return records, episodes, source


def assign_training_scores(records: list[RecordedTurn], episodes: list[dict], tokenizer: Any, args: argparse.Namespace) -> None:
    """Use the actual episode reward manager and trainer invalid-action penalty."""
    from agent_system.reward_manager import EpisodeRewardManager
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import apply_invalid_action_penalty
    from verl.utils.torch_functional import get_response_mask

    reward_manager = EpisodeRewardManager(tokenizer, num_examine=0, normalize_by_length=False)
    for start in range(0, len(records), 32):
        rows = records[start : start + 32]
        prompts = torch.full((len(rows), args.max_prompt_length), tokenizer.pad_token_id, dtype=torch.long)
        responses = torch.full((len(rows), args.max_new_tokens), tokenizer.pad_token_id, dtype=torch.long)
        prompt_mask = torch.zeros_like(prompts)
        for index, row in enumerate(rows):
            prompt_ids, response_ids = row.generation.prompt_token_ids, row.generation.response_token_ids
            prompts[index, -len(prompt_ids) :] = torch.tensor(prompt_ids)
            responses[index, : len(response_ids)] = torch.tensor(response_ids)
            prompt_mask[index, -len(prompt_ids) :] = 1
        batch = DataProto.from_dict(
            tensors={"prompts": prompts, "responses": responses, "attention_mask": torch.cat((prompt_mask, get_response_mask(responses, tokenizer.eos_token_id)), dim=1)},
            non_tensors={
                "data_source": np.array(["text"] * len(rows), dtype=object),
                "episode_rewards": np.array([episodes[row.rollout_index]["episode_rewards"] for row in rows], dtype=np.float32),
                "episode_lengths": np.array([episodes[row.rollout_index]["episode_lengths"] for row in rows], dtype=np.float32),
                "is_action_valid": np.array([row.is_action_valid for row in rows], dtype=bool),
            },
        )
        batch.batch["token_level_scores"] = reward_manager(batch)
        if args.invalid_action_penalty:
            batch, _ = apply_invalid_action_penalty(batch, args.invalid_action_penalty)
        for row, score in zip(rows, batch.batch["token_level_scores"].sum(-1).tolist()):
            row.score = score


def write_rollout_jsonl(records: list[RecordedTurn], episodes: list[dict], tokenizer: Any, output: Path) -> None:
    """Decode actual IDs as training does; group rows by trajectory and turn."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as stream:
        for row in sorted(records, key=lambda row: (row.rollout_index, row.traj_step)):
            episode = episodes[row.rollout_index]
            entry = {
                "input": tokenizer.decode(row.generation.prompt_token_ids, skip_special_tokens=True),
                "output": tokenizer.decode(row.generation.response_token_ids, skip_special_tokens=True),
                "score": row.score,
                "rollout_index": row.rollout_index,
                "pseudo_probability": row.pseudo_probability,
                "pseudo_skip_reason": row.pseudo_skip_reason,
                "uid": episode["uid"],
                "traj_uid": episode["traj_uid"],
                "traj_step": row.traj_step,
                "data_source": "text",
                "is_action_valid": row.is_action_valid,
                "rewards": row.reward,
                "episode_rewards": episode["episode_rewards"],
                "episode_lengths": episode["episode_lengths"],
                "tool_callings": 0.0,
            }
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, output)


def _write_metadata(metadata: dict, output: Path) -> None:
    path = output.with_suffix(".metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_inference(args: argparse.Namespace, server: Any):
    if "vllm" in sys.modules:
        raise RuntimeError("Run the testbed as a fresh Python module so vLLM V0 can be configured before import")
    os.environ["VLLM_USE_V1"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    max_choices = max((len(values) + 1 for item in server.product_item_dict.values() for values in item.get("options", {}).values()), default=1)
    kwargs = dict(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        seed=args.seed,
        max_logprobs=2 * max_choices,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return tokenizer, LLM(**kwargs)


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in ("num_rollouts", "group_size", "max_steps", "max_prompt_length", "max_new_tokens", "rollout_batch_size", "pseudo_batch_size", "num_products", "tensor_parallel_size"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if args.pseudo_batch_size > 32:
        raise ValueError("pseudo_batch_size must be at most 32")
    if args.history_length < 0:
        raise ValueError("history_length must be nonnegative")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    for name in ("temperature", "invalid_action_penalty"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or args.top_k < -1 or args.top_k == 0:
        raise ValueError("Require top_p in (0, 1] and top_k=-1 or a positive integer")
    if not math.isfinite(args.path_probability_threshold) or not 0 <= args.path_probability_threshold <= 1:
        raise ValueError("path_probability_threshold must be in [0, 1]")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.output.suffix != ".jsonl":
        raise ValueError("output must have a .jsonl extension")


def run_testbed(args: argparse.Namespace) -> dict:
    _validate_arguments(args)
    metadata = {
        "schema_version": 1,
        "status": "loading",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "score_definition": "episode reward (10 for full-reward purchase, otherwise 0), minus per-turn invalid-action penalty",
        "pseudo_observation": "before the logged action",
        "pseudo_batch_size": args.pseudo_batch_size,
        "environment_execution": "independent local sessions, batched policy inference",
    }
    _write_metadata(metadata, args.output)
    try:
        server = create_server(args.catalog, args.attributes, args.seed, args.num_products)
        select_goal_indices(server, args)  # Fail on invalid selection before loading weights.
        tokenizer, engine = _load_inference(args, server)
        metadata["effective_max_model_len"] = engine.llm_engine.model_config.max_model_len
        metadata["status"] = "collecting_rollouts"
        _write_metadata(metadata, args.output)
        records, episodes, source = collect_parallel_rollouts(server, TrainingRolloutPolicy(engine, tokenizer, args), args)
        assign_training_scores(records, episodes, tokenizer, args)
        # Keep the expensive empirical trajectories if pseudo scoring fails.
        write_rollout_jsonl(records, episodes, tokenizer, args.output)
        metadata.update(status="scoring_pseudo_probabilities", episodes=episodes, num_turns=len(records))
        _write_metadata(metadata, args.output)
        logger.info("Scoring %d saved pre-action snapshots from %d rollouts", len(records), len(episodes))
        scorer = WebshopTurnSuccessScorer(source, MixedPseudoRolloutScorer(engine, tokenizer, batch_size=args.pseudo_batch_size), path_probability_threshold=args.path_probability_threshold)
        scores = scorer.score([row.snapshot for row in records], policy_version=args.model)
        if len(scores) != len(records):
            raise ValueError("Pseudo probabilities must align with recorded turns")
        for row, score in zip(records, scores):
            row.pseudo_probability = score.probability
            row.pseudo_skip_reason = score.skipped_reason
        write_rollout_jsonl(records, episodes, tokenizer, args.output)
        metadata.update(
            status="complete",
            num_pseudo_scored_turns=sum(row.pseudo_probability is not None for row in records),
            full_reward_rollouts=sum(episode["episode_rewards"] == 10 for episode in episodes),
            mean_episode_reward=float(np.mean([episode["episode_rewards"] for episode in episodes])),
        )
        _write_metadata(metadata, args.output)
        return metadata
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_metadata(metadata, args.output)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokenizer")
    parser.add_argument("--num-rollouts", "-n", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8, help="Rollouts per shopping goal; the final group may be smaller")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--goal-indices", type=int, nargs="+", help="Optional native goal index per group within the selected split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--pseudo-batch-size", type=int, default=32, help="Maximum pseudo prompts per vLLM call; lower this to reduce prompt-logprob peak memory")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.1, help="Training format penalty; use 0 to disable")
    parser.add_argument("--path-probability-threshold", type=float, default=1e-3)
    parser.add_argument("--catalog", type=Path, default=_DEFAULT_CATALOG)
    parser.add_argument("--attributes", type=Path, default=_DEFAULT_ATTRIBUTES)
    parser.add_argument("--num-products", type=int, default=1000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argument_parser().parse_args()
    result = run_testbed(args)
    print(f"Wrote {result['num_turns']} turns from {args.num_rollouts} rollouts to {args.output}")


if __name__ == "__main__":
    main()
