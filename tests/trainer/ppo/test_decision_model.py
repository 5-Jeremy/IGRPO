"""Decision-model API failures preserve a checkpoint that retries the failed batch."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from agent_system.decision_model import DecisionModelUnavailable, check_decision_model
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


URL = "http://127.0.0.1:8009/v1/models"


@pytest.mark.parametrize("failure,status,timed_out", [
    (TimeoutError("timed out"), 0, 1),
    (URLError(TimeoutError("timed out")), 0, 1),
    (URLError(ConnectionRefusedError("refused")), 0, 0),
    (HTTPError(URL, 503, "Unavailable", {}, None), 503, 0),
    (HTTPError(URL, 504, "Gateway timeout", {}, None), 504, 1),
])
def test_probe_failure_policy(monkeypatch, failure, status, timed_out):
    monkeypatch.setattr("agent_system.decision_model.urlopen", MagicMock(side_effect=failure))
    monkeypatch.setattr("agent_system.decision_model.perf_counter", MagicMock(side_effect=[10.0, 10.25]))
    with pytest.raises(DecisionModelUnavailable) as caught:
        check_decision_model(URL)
    metrics = caught.value.metrics
    assert metrics == {
        "decision_model/alive": 0, "decision_model/http_status": status,
        "decision_model/response_time_s": 0.25, "decision_model/timed_out": timed_out,
    }


def test_probe_success_and_body_read_timeout(monkeypatch):
    response = MagicMock()
    response.status = 200
    response.__enter__.return_value = response
    opener = MagicMock(return_value=response)
    monkeypatch.setattr("agent_system.decision_model.urlopen", opener)
    assert check_decision_model(URL, timeout=2.5)["decision_model/alive"] == 1
    assert opener.call_args.args[0].get_method() == "GET"
    assert opener.call_args.kwargs["timeout"] == 2.5
    response.read.side_effect = TimeoutError("body timed out")
    with pytest.raises(DecisionModelUnavailable) as caught:
        check_decision_model(URL)
    assert caught.value.metrics["decision_model/http_status"] == 200
    assert caught.value.metrics["decision_model/alive"] == 0
    assert caught.value.metrics["decision_model/timed_out"] == 1


def make_trainer(tmp_path, step=2, use_critic=False):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"trainer": {
        "default_local_dir": str(tmp_path), "default_hdfs_dir": None,
        "resume_mode": "resume_path", "resume_from_path": None,
        "del_local_ckpt_after_load": False, "save_freq": -1,
        "max_actor_ckpt_to_keep": 1, "max_critic_ckpt_to_keep": 1,
        "decision_model": {"enabled": True, "url": URL, "timeout": 1},
    }})
    trainer.global_steps = step
    trainer.use_critic = use_critic
    trainer.actor_rollout_wg = MagicMock()
    trainer.critic_wg = MagicMock()
    for worker in [trainer.actor_rollout_wg, trainer.critic_wg]:
        worker.save_checkpoint.side_effect = lambda local, *args, **kwargs: Path(local).mkdir(parents=True)
    trainer.train_dataloader = StatefulDataLoader(list(range(4)), batch_size=1, shuffle=False)
    return trainer


@pytest.mark.parametrize("override,expected", [
    (None, ("however,", "wait,", "but wait,", "alternatively,")),
    ([], []),
    (["next,"], ["next,"]),
])
def test_local_qwen_checkpoint_uses_tokenizer_family_and_honors_override(override, expected):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    trainer.config.actor_rollout_ref.model.path = "/checkpoints/step_100/actor"
    trainer.config.trainer.decision_model.thinking_split_phrases = override
    trainer.tokenizer = type("Qwen2TokenizerFast", (), {})()
    assert trainer._get_thinking_split_phrases() == expected


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("use_critic", [False, True])
def test_timeout_saves_and_resumes_failed_batch(tmp_path, monkeypatch, step, use_critic):
    trainer = make_trainer(tmp_path, step, use_critic)
    iterator = iter(trainer.train_dataloader)
    for _ in range(step - 1):
        next(iterator)
    snapshot = deepcopy(trainer.train_dataloader.state_dict())
    failed_batch = next(iterator)
    tracker = tmp_path / "latest_checkpointed_iteration.txt"
    tracker.write_text("previous checkpoint marker")
    monkeypatch.setattr("agent_system.decision_model.urlopen", MagicMock(side_effect=TimeoutError("timed out")))
    logger, metrics = MagicMock(), {}
    with pytest.raises(RuntimeError, match="Training halted before the update"):
        trainer._check_decision_model(metrics, logger, snapshot, epoch=2)
    folder, = tmp_path.glob("emergency_*/global_step_*")
    assert folder.name == f"global_step_{step - 1}"
    assert (folder / "actor").is_dir()
    assert (folder / "critic").exists() == use_critic
    assert tracker.read_text() == "previous checkpoint marker"
    assert json.loads((folder / "emergency.json").read_text())["failed_step"] == step
    for worker in [trainer.actor_rollout_wg] + ([trainer.critic_wg] if use_critic else []):
        assert worker.save_checkpoint.call_args.args[2] == step - 1
        assert worker.save_checkpoint.call_args.kwargs["max_ckpt_to_keep"] is None
    assert metrics["decision_model/emergency_checkpoint_saved"] == 1
    assert metrics["decision_model/timed_out"] == 1
    logger.log.assert_called_once_with(data=metrics, step=step)
    trainer.actor_rollout_wg.update_actor.assert_not_called()
    trainer.critic_wg.update_critic.assert_not_called()

    # Exercise the production loader with a fresh stateful dataloader.
    tracker.unlink()  # This deliberately is not a regular auto-resume checkpoint.
    resumed = make_trainer(tmp_path, use_critic=use_critic)
    resumed.config.trainer.resume_from_path = str(folder)
    resumed._load_checkpoint()
    assert resumed.global_steps == step - 1
    assert resumed._resume_epoch == 2
    assert torch.equal(next(iter(resumed.train_dataloader)), failed_batch)
    resumed.actor_rollout_wg.load_checkpoint.assert_called_once_with(
        str(folder / "actor"), del_local_after_load=False)


def test_logging_failure_does_not_prevent_emergency_save(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    monkeypatch.setattr("agent_system.decision_model.urlopen", MagicMock(side_effect=TimeoutError()))
    logger = MagicMock()
    logger.log.side_effect = RuntimeError("logging offline")
    with pytest.raises(RuntimeError, match="Training halted before the update"):
        trainer._check_decision_model({}, logger, trainer.train_dataloader.state_dict(), epoch=0)
    assert len(list(tmp_path.glob("emergency_*/global_step_*/emergency.json"))) == 1


def test_save_failure_halts_without_claiming_complete_checkpoint(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    trainer.actor_rollout_wg.save_checkpoint.side_effect = OSError("disk full")
    monkeypatch.setattr("agent_system.decision_model.urlopen", MagicMock(side_effect=TimeoutError()))
    with pytest.raises(RuntimeError, match="Emergency checkpoint saving also failed: disk full"):
        trainer._check_decision_model({}, MagicMock(), trainer.train_dataloader.state_dict(), epoch=0)
    assert not list(tmp_path.glob("emergency_*/global_step_*/emergency.json"))
