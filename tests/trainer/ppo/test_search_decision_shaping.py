"""Exercise the Search trainer, including turn credit, failed requests and tree ancestry."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from agent_system.decision_model_requests import DecisionModelRequestBuilder, attach_decision_model_probabilities
from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.env_package.search.projection import search_projection
from agent_system.multi_turn_rollout import TrajectoryCollector
from agent_system.reward_manager import EpisodeRewardManager
from igpo import core_igpo
from rollout_visualizer.data import read_document
from tests.trainer.ppo.test_reasoning_advantage import CharacterTokenizer
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_unfinished_thinking_penalty
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.rollout_logging import dump_generations


@pytest.mark.parametrize("engine_reports_truncation", [False, True])
def test_unfinished_thinking_penalty_applies_only_at_the_last_generated_token(engine_reports_truncation):
    tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = tokenizer("~")["input_ids"][0]
    tokenizer.pad_token_id = 0
    limit = 8
    outputs = ["<think>x", "</THINK>", "short", "endsEOS~", "cutoff"]
    if engine_reports_truncation:
        outputs.append("stopped!")
    responses = torch.tensor([tokenizer(output)["input_ids"] + [0] * (limit - len(output)) for output in outputs])
    tensors = {
        "responses": responses,
        "attention_mask": (responses != 0).long(),
        "token_level_scores": torch.zeros_like(responses, dtype=torch.float32),
        "step_rewards": torch.ones(len(outputs)),
    }
    expected_penalties = [(0, 7)]
    if engine_reports_truncation:
        tensors["truncated_response_length"] = torch.tensor([8, 8, 0, 0, 6, 0])
        expected_penalties.append((4, 5))
    batch = DataProto.from_dict(tensors=tensors)

    count = apply_unfinished_thinking_penalty(batch, tokenizer, limit, -0.1)

    expected_scores = torch.zeros_like(tensors["token_level_scores"])
    for row, last_token in expected_penalties:
        expected_scores[row, last_token] = -0.1
    assert count == len(expected_penalties)
    torch.testing.assert_close(batch.batch["token_level_scores"], expected_scores)
    torch.testing.assert_close(batch.batch["step_rewards"], 1 + expected_scores.sum(-1))


@pytest.fixture
def decision_endpoint(request):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply({"models": [{"name": "kev-latest", "run": "test-judge"}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            answers = {}
            for step, question in payload["questions"].items():
                labels = list(question["criteria"])
                chosen = labels[1]  # Contradict the chosen operation.
                answers[step] = {"choice": chosen, "probabilities": {k: float(k == chosen) for k in labels}}
            self.reply({"answers": {} if request.param == "invalid" else answers})

        def reply(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/models", requests, request.param
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("decision_endpoint", ["ok", "selective", "invalid", "disabled", "unfinished_thinking"], indirect=True)
def test_fit_shapes_igpo_search_turns_before_actor_update(tmp_path, monkeypatch, decision_endpoint):
    url, requests, mode = decision_endpoint
    penalize = mode == "unfinished_thinking"
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.algorithm.adv_estimator = "igpo"
    config.algorithm.igpo.prob_diff_mode = False
    config.algorithm.reasoning_advantage.enabled = mode != "disabled"
    if penalize:
        config.algorithm.unfinished_thinking_penalty = -0.1
    config.env.update({"env_name": "search", "max_steps": 2, "history_length": 0})
    config.env.rollout.n = 2
    config.trainer.update({"resume_mode": "disable", "total_epochs": 1, "default_local_dir": str(tmp_path),
                           "balance_batch": False, "n_gpus_per_node": 1, "track_reasoning_entropy": True,
                           "rollout_data_dir": str(tmp_path / "rollouts")})
    config.trainer.decision_model.update({"enabled": mode != "disabled", "url": url,
                                          "selective_requests": mode == "selective"})
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.actor.use_kl_loss = False
    config.actor_rollout_ref.model.path = "Qwen2.5"

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config, trainer.total_training_steps = config, 1
    trainer.tokenizer = tokenizer = CharacterTokenizer()
    trainer.use_critic = trainer.use_reference_policy = trainer.use_rm = False
    trainer.val_reward_fn = None
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.reward_fn = EpisodeRewardManager(tokenizer, 0)
    trainer.train_dataloader = StatefulDataLoader([{
        "input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]), "position_ids": torch.tensor([0]),
        "raw_prompt_ids": [1], "data_source": "search", "ground_truth": {"target": ["Peru"]},
    }], batch_size=1, collate_fn=collate_fn)
    trainer.traj_collector = TrajectoryCollector(config, tokenizer)
    tracker = MagicMock()
    monkeypatch.setattr("verl.utils.tracking.Tracking", lambda **kwargs: tracker)

    def step(actions):
        done = actions[1].startswith("<answer>")
        if done:
            assert actions == ["<answer>Peru</answer>"] * 2
        else:
            assert actions == ["" if penalize else "<search>Lima</search>", "<search>Lima</search>"]
        infos = [{"won": done, "data_source": "search"} for _ in actions]
        return ["Lima is in Peru."] * 2, [float(done)] * 2, [done] * 2, infos

    backend = SimpleNamespace(reset=lambda **kwargs: (["Where is Lima?"] * 2, [{}, {}]), step=step)
    trainer.envs = SearchEnvironmentManager(backend, search_projection, config)

    def preprocess(**kwargs):
        return DataProto.from_dict(
            tensors={key: torch.ones((2, 1), dtype=torch.long)
                     for key in ("input_ids", "attention_mask", "position_ids")},
            non_tensors={"raw_prompt_ids": np.array([[1], [1]]), "data_source": np.array(["search"] * 2)},
        )

    trainer.traj_collector.preprocess_batch = preprocess
    operations = iter(["<search>Lima</search>", "<answer>Peru</answer>"])
    intervening_text = "\nHowever, proceed.\n"

    def generate(batch):
        action = next(operations)
        is_search = action.startswith("<search>")
        thought = "Find Lima." if is_search else "Found Lima. Wait, check. However, choose this."
        text = f"<think>{thought}</think>{intervening_text}{action}"
        tokens = tokenizer(text)["input_ids"]
        response = torch.tensor([tokens + [0] * (128 - len(tokens))] * len(batch))
        truncated_lengths = torch.zeros(len(batch), dtype=torch.long)
        if penalize and is_search:
            unfinished = "<think>I am still thinking.".ljust(127, "x") + "…"
            response[0] = torch.tensor(tokenizer(unfinished)["input_ids"])
            truncated_lengths[0] = len(unfinished)
        ids = torch.cat([batch.batch["input_ids"], response], dim=1)
        return DataProto.from_dict(tensors={"input_ids": ids, "prompts": ids[:, :1], "responses": response,
                                           "attention_mask": (ids != 0).long(), "truncated_response_length": truncated_lengths})

    worker = trainer.actor_rollout_wg = MagicMock(world_size=1)
    worker.generate_sequences.side_effect = generate
    worker.compute_log_prob.side_effect = lambda b: DataProto.from_dict(tensors={
        key: torch.ones_like(b.batch["responses"], dtype=torch.float32) for key in ("entropys", "old_log_probs")
    })
    worker.update_actor.return_value = SimpleNamespace(meta_info={"metrics": {}})

    def answer_log_probs(batch, **kwargs):
        # Identical episode outcomes, opposite information gains on the two searches.
        assert batch.non_tensor_batch["traj_step"].tolist() == [0, 1, 0, 1]
        batch.batch["avg_ans_log_probs"] = torch.tensor([-2., -1., -1., -2.])
        return batch

    monkeypatch.setattr(core_igpo, "compute_answer_block_avg_log_prob", answer_log_probs)
    if mode == "invalid":
        with pytest.raises(RuntimeError, match="Training halted before the update"):
            trainer.fit()
        worker.update_actor.assert_not_called()
        assert list(tmp_path.glob("emergency_*/global_step_0/emergency.json"))
        return

    trainer.fit()
    trained = worker.update_actor.call_args.args[0]
    first_reward = 1. - config.actor_rollout_ref.actor.invalid_action_penalty_coef - 0.1 if penalize else 1.
    assert trained.non_tensor_batch["igpo_step_reward"].tolist() == pytest.approx([first_reward, 1., -1., 1.])
    assert trained.batch["token_level_scores"][0, -1].item() == pytest.approx(first_reward if penalize else 0.0)
    metrics = tracker.log.call_args.kwargs["data"]
    assert metrics["episode/successful_length/mean"] == 2
    assert metrics["episode/all_failed_group_fraction"] == 0
    assert metrics["response_length/reasoning_step_count"] == (1.75 if penalize else 2.)
    assert metrics.get("reward/unfinished_thinking_penalized_rows", 0) == int(penalize)
    assert len(requests) == (0 if mode == "disabled" else 4 - int(penalize))
    if mode != "disabled":
        assert all(intervening_text.strip() not in json.dumps(payload) for payload in requests)
        assert metrics["decision_model/answer_count"] == 8 - int(penalize)
        assert trainer._decision_model_run == "test-judge"
        ids = trained.batch["reasoning_step_ids"]
        baseline = trained.batch["returns"]
        assert (ids == 1).any(dim=1).tolist() == [not penalize, True, True, True]
        scored_rows = slice(int(penalize), None)
        for response, step_ids in zip(trained.batch["responses"][scored_rows], ids[scored_rows], strict=True):
            start = tokenizer.decode(response).index(intervening_text)
            assert not step_ids[start:start + len(intervening_text)].any()
        expected = torch.where(ids > 0, -baseline, baseline)
        torch.testing.assert_close(trained.batch["advantages"], expected)
        assert trained.non_tensor_batch["igpo_turn_advantage"][0] > 0 > trained.non_tensor_batch["igpo_turn_advantage"][2]
    group, = read_document(tmp_path / "rollouts/1.jsonl")["groups"]
    first_action = "[No action tag]" if penalize else "<search>Lima</search>"
    assert sorted(n["label"] for n in group["nodes"][1:]) == sorted([first_action, "<search>Lima</search>", "<answer>Peru</answer>", "<answer>Peru</answer>"])
    for node in group["nodes"][1:]:
        record = node["records"][0]
        if node["label"] == "[No action tag]":
            assert record["score"] == pytest.approx(first_reward)
            assert record["answer_entropy"] is None
        else:
            assert record["answer_entropy"]["mean"] == 1


@pytest.mark.parametrize("mode", ["max", "full"])
def test_tree_prefixes_share_judgments_and_keep_each_training_record(mode, tmp_path):
    collector = TrajectoryCollector(OmegaConf.create({"algorithm": {"igrpo": {"reward_mode": mode}}}), None)
    trace = {"steps": ["First.", "Wait, search."], "action": "<search>Lima</search>"}
    nodes = []
    for uid, parent, depth in [("a", "root", 0), ("b", "a", 1), ("c", "a", 1), ("d", "b", 2)]:
        nodes.append(dict(node_uid=uid, parent_node_uid=parent, uid="group", traj_step=depth,
                          is_terminal=uid in ("c", "d"), deactivate=False, current_tool_callings=depth,
                          reasoning_trace=trace, rewards=float(uid == "d"), input_ids=torch.tensor([1])))
    batch = collector.gather_rollout_data_tree_structure(nodes, {"success_rate": [0, 1]}, global_steps=1)
    builder = DecisionModelRequestBuilder(selective_requests=True, max_turns_per_trajectory=2)
    assert set(builder.build_rollout_requests(batch)) == {"a", "b", "c"}
    attach_decision_model_probabilities(batch, {"a": {"step_2": {"support": 1.}}})
    size = len(batch)
    path = tmp_path / "1.jsonl"
    dump_generations(path, ["Question"] * size, [trace["action"]] * size, [0.] * size, {}, batch.non_tensor_batch)
    group, = read_document(path)["groups"]
    by_uid = {n["records"][0]["node_uid"]: n for n in group["nodes"][1:]}
    assert len(by_uid) == 4
    assert by_uid["c"]["parent"] == by_uid["b"]["parent"] == by_uid["a"]["id"]
    assert by_uid["d"]["parent"] == by_uid["b"]["id"]
    prefix = by_uid["a"]["records"]
    assert all(r["decision_model_probabilities"] == {"step_2": {"support": 1.}} for r in prefix)
    if mode == "full":
        assert {r["episode_success"] for r in prefix} == {True, False}
        assert len({r["trajectory_id"] for r in prefix}) == 2


def test_validation_rollouts_save_the_expanded_turns(tmp_path):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    trainer.config.trainer.validation_data_dir = str(tmp_path)
    trainer.config.trainer.decision_model.enabled = True
    trainer.config.actor_rollout_ref.rollout.val_kwargs.n = 1
    trainer.global_steps = 1
    tokenizer = trainer.tokenizer = CharacterTokenizer()
    tokenizer.eos_token_id = tokenizer.pad_token_id = 0
    trainer.val_envs = trainer.actor_rollout_wg = MagicMock()
    data = collate_fn([{"input_ids": torch.ones(1), "attention_mask": torch.ones(1), "position_ids": torch.zeros(1),
                        "ground_truth": "Peru", "raw_prompt_ids": [1], "data_source": "search"}])
    trainer.val_dataloader = [data]
    outputs = ["<think>Check.</think><search>Lima</search>", "<think>Found.</think><answer>Peru</answer>"]

    def collect(*, is_train, **kwargs):
        assert not is_train
        assert kwargs.get("on_turn_generated") is None
        responses = [tokenizer(o)["input_ids"] for o in outputs]
        tokens = torch.tensor([r + [0] * (60 - len(r)) for r in responses])
        return DataProto.from_dict(tensors={"responses": tokens, "prompts": torch.tensor([[82], [83]])}, non_tensors={
            "id": np.array(["s", "a"]), "previous_step_ids": np.array([[], ["s"]], dtype=object),
            "trajectory_id": np.array(["t", "t"]), "trajectory_length": np.array([1, 2]), "uid": np.array(["g", "g"]),
            "data_source": np.array(["search", "search"]), "traj_uid": np.array(["t", "t"]),
            "success_rate": np.ones(2), "tool_callings": np.ones(2),
        })

    trainer.traj_collector = SimpleNamespace(multi_turn_loop=collect)
    trainer.val_reward_fn = lambda *args, **kwargs: {"reward_tensor": torch.ones((2, 1))}
    training_log = tmp_path / "1.jsonl"
    training_log.write_text("training samples")
    metrics = trainer._validate()
    assert training_log.read_text() == "training samples"
    assert metrics["val/success_rate"] == 1
    records = [json.loads(line) for line in (tmp_path / "1.validation.jsonl").read_text().splitlines()]
    assert [r["input"] for r in records] == ["Q", "R"]
    assert [r["output"] for r in records] == outputs
    assert [r["reasoning_trace"]["action"] for r in records] == ["<search>Lima</search>", "<answer>Peru</answer>"]
