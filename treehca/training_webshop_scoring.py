"""DataProto adapter for WebShop turn success during TreeHCA training."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from treehca.pseudo_rollout_product_page import build_action_label_catalog
from treehca.training_pseudo_probes import TrainingPseudoProbeScorer
from treehca.webshop_probability_snapshot import WebshopSnapshotSource
from treehca.webshop_turn_success import WebshopTurnSuccessScorer


class _TrainingTokenizer:
    def __init__(self, tokenizer, kwargs):
        self.tokenizer, self.kwargs = tokenizer, dict(kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **(self.kwargs | kwargs))


def resolve_treehca_scorer(config):
    """Gate dispatch on TreeHCA; existing algorithms always keep their scorer."""
    if config.algorithm.adv_estimator != "treehca":
        return "answer_block"
    choice = config.algorithm.get("treehca", {}).get("pseudo_scorer", "auto")
    if choice == "auto":
        return "webshop" if "webshop" in config.env.env_name.lower() else "answer_block"
    if choice not in {"webshop", "answer_block"}:
        raise ValueError(f"Unknown TreeHCA pseudo_scorer: {choice!r}")
    return choice


class TrainingWebshopTurnSuccessScorer(WebshopTurnSuccessScorer):
    """Separate training specialization; the standalone scorer stays unchanged."""

    def add_catalog_payload(self, payload):
        self.source.server.product_item_dict.update(payload["products"])
        self.source.server.product_prices.update(payload["prices"])
        if self._label_catalog is None:
            self._label_catalog = build_action_label_catalog(self.probe_scorer.tokenizer, payload["max_choices"])


class WebshopInfoGainScorer:
    """Accept padded/reordered info_gain_batch rows and return aligned log scores.

    Missing probabilities remain None in non_tensor_batch and NaN in the
    tensor output. The explicit scored mask must be consulted by the caller.
    """

    def __init__(self, tokenizer, actor_rollout_wg, *, max_model_len, path_probability_threshold=1e-3, batch_size=32, apply_chat_template_kwargs=None):
        # Import native rendering only on the WebShop training path.
        native_root = Path(__file__).resolve().parents[1] / "agent_system/environments/env_package/webshop/webshop"
        if str(native_root) not in sys.path:
            sys.path.append(str(native_root))
        tokenizer = _TrainingTokenizer(tokenizer, apply_chat_template_kwargs or {})
        self.probes = TrainingPseudoProbeScorer(tokenizer, actor_rollout_wg, max_model_len=max_model_len, batch_size=batch_size)
        self.threshold = path_probability_threshold
        self.scorers = {}

    @staticmethod
    def require_active_scores(batch):
        missing = batch.non_tensor_batch["webshop_active"] & ~batch.batch["webshop_scored"].cpu().numpy()
        if missing.any():
            reasons = [(int(index), batch.non_tensor_batch["webshop_skipped_reason"][index]) for index in np.flatnonzero(missing)]
            raise ValueError(f"TreeHCA WebShop scoring is not implemented for these active rows: {reasons}. Their success probabilities remain None.")

    def compute(self, batch, *, policy_version):
        payloads = batch.non_tensor_batch["webshop_scoring_payload"]
        active = batch.non_tensor_batch["webshop_active"]
        terminated = batch.non_tensor_batch["webshop_terminated"]
        won = batch.non_tensor_batch["webshop_won"]
        probabilities = np.empty(len(batch), dtype=object)
        probabilities[:] = None
        reasons = np.empty(len(batch), dtype=object)
        reasons[:] = None
        groups = defaultdict(list)
        for index, payload in enumerate(payloads):
            if not active[index]:
                reasons[index] = "inactive"
            elif terminated[index]:
                probabilities[index] = float(bool(won[index]))
            elif payload is None:
                raise ValueError("Active nonterminal WebShop rows require a scoring snapshot")
            else:
                snapshot = payload["snapshot"]
                key = snapshot.catalog_key
                if key not in self.scorers:
                    source = WebshopSnapshotSource(SimpleNamespace(product_item_dict={}, product_prices={}, show_attrs=payload["show_attrs"]))
                    source.catalog_key = key
                    self.scorers[key] = TrainingWebshopTurnSuccessScorer(source, self.probes, path_probability_threshold=self.threshold)
                self.scorers[key].add_catalog_payload(payload)
                groups[key].append((index, snapshot))
        for key, rows in groups.items():
            scores = self.scorers[key].score([snapshot for _, snapshot in rows], policy_version=policy_version)
            for (index, _), score in zip(rows, scores):
                probabilities[index], reasons[index] = score.probability, score.skipped_reason
        device = batch.batch["prompts"].device
        logs = [math.nan if p is None else (-math.inf if p == 0 else math.log(p)) for p in probabilities]
        batch.batch["avg_ans_log_probs"] = torch.tensor(logs, dtype=torch.float64, device=device)
        batch.batch["webshop_scored"] = torch.tensor([p is not None for p in probabilities], dtype=torch.bool, device=device)
        batch.non_tensor_batch["webshop_success_probability"] = probabilities
        batch.non_tensor_batch["webshop_skipped_reason"] = reasons
        return batch
