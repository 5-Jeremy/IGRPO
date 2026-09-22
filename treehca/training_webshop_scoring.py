"""DataProto adapter for WebShop turn success during TreeHCA training."""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from treehca.product_page_parser import extract_product_page_contexts, parse_product_page_fields
from treehca.pseudo_rollout_product_page import GROUP_NONE_ACTION, _build_product_page_prompt
from treehca.pseudo_rollout_results_page import prepare_response_suffix_probe, prepare_results_page_answer_probe
from treehca.training_pseudo_probes import TrainingProductOptionProbe, TrainingPseudoProbeScorer
from treehca.webshop_probability_snapshot import WebshopSnapshotSource
from treehca.webshop_turn_success import WebshopTurnSuccessScorer


def _render_scoring_prompt(snapshot, history, parts=None):
    """Render a scorer-only prompt with a suffix of the captured history."""
    from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS

    parts = parts or extract_product_page_contexts([snapshot.prompt])[0]
    fields = dict(
        task_description=snapshot.shopping_task,
        current_observation=parts.current_observation,
        available_actions="\n".join(f"'{action}'," for action in parts.admissible_actions),
    )
    if history:
        start = snapshot.completed_steps - len(history) + 1
        action_history = "\n".join(f"[Observation {step}: '{observation}', Action {step}: '{action}']" for step, (observation, action) in enumerate(history, start=start))
        prompt = WEBSHOP_TEMPLATE.format(
            **fields,
            step_count=snapshot.completed_steps,
            history_length=len(history),
            action_history=action_history,
            current_step=snapshot.completed_steps + 1,
        )
    else:
        prompt = WEBSHOP_TEMPLATE_NO_HIS.format(**fields)
    return replace(snapshot, prompt=prompt, history=tuple(history))


def _history_trim_candidates(snapshot):
    """Yield the original snapshot, then suffixes formed by dropping oldest history."""
    yield snapshot
    parts = extract_product_page_contexts([snapshot.prompt])[0]
    if parts.history_block is None:
        return
    for removed in range(1, len(snapshot.history) + 1):
        yield _render_scoring_prompt(snapshot, snapshot.history[removed:], parts)


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

    def __init__(self, *args, prune_unsuccessful_choices=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(prune_unsuccessful_choices, bool):
            raise ValueError("prune_unsuccessful_choices must be a boolean")
        self.prune_unsuccessful_choices = prune_unsuccessful_choices

    def _request_fits(self, prompt, response):
        check = getattr(self.probe_scorer, "request_fits", None)
        if check is not None:
            return check(prompt, response)
        limit = getattr(self.probe_scorer, "max_model_len", None)
        return limit is None or len(prompt) + len(response) <= limit

    def _prepare_search_job(self, job):
        # Hypothetical product pages inherit the fitted history suffix.
        job.snapshot = self._fit_results_snapshot(job.snapshot, job.actions.values())

    def _fit_results_snapshot(self, snapshot, actions=None):
        tokenizer = self.probe_scorer.tokenizer
        if actions is None:
            visible = {asin.lower() for asin in snapshot.visible_asins}
            parts = extract_product_page_contexts([snapshot.prompt])[0]
            actions = [action for action in parts.admissible_actions if action.startswith("click[") and action[6:-1].lower() in visible]
        actions = tuple(actions)
        for candidate in _history_trim_candidates(snapshot):
            prompt_ids = None
            fits = True
            for action in actions:
                probe = prepare_results_page_answer_probe(candidate.prompt, action, tokenizer, prompt_token_ids=prompt_ids)
                prompt_ids = probe.prompt_token_ids
                if not self._request_fits(probe.prompt_token_ids, probe.response_token_ids):
                    fits = False
                    break
            if fits:
                return candidate
        raise ValueError(f"TreeHCA results-page pseudo probe exceeds max_model_len={self.probe_scorer.max_model_len} after removing all past observations")

    def add_catalog_payload(self, payload):
        self.source.server.product_item_dict.update(payload["products"])
        self.source.server.product_prices.update(payload["prices"])

    def _append_product_probes(self, jobs, probes, owners):
        for job in jobs:
            if job.probability is not None:
                continue
            for candidate in _history_trim_candidates(job.snapshot):
                prepared = self._prepare_product_probes(job, candidate)
                if all(self._request_fits(answer.prompt_token_ids, answer.response_token_ids) for probe, _ in prepared for answer in probe.answer_probes):
                    job.snapshot = candidate
                    for probe, group_name in prepared:
                        probes.append(probe)
                        owners.append((job, group_name))
                    break
            else:
                raise ValueError(f"TreeHCA product-page pseudo probe exceeds max_model_len={self.probe_scorer.max_model_len} after removing all past observations")

    def _prepare_product_probes(self, job, snapshot):
        tokenizer = self.probe_scorer.tokenizer
        parts = extract_product_page_contexts([snapshot.prompt])[0]
        fields = parse_product_page_fields(parts.current_observation, parts.admissible_actions)
        groups = {group.name: group for group in fields.option_groups}
        useful = job.plan.successful_actions() if self.prune_unsuccessful_choices else {group.name: group.actions for group in job.plan.groups}
        prepared = []
        for planned in job.plan.groups:
            group = groups.get(planned.name)
            if group is None or tuple(f"click[{value}]" for value in group.values) + (GROUP_NONE_ACTION,) != planned.actions:
                raise ValueError("Snapshot product options differ from the native catalog")
            # The synthetic none response represents leaving this group untouched.
            names = (*group.values, GROUP_NONE_ACTION)
            options = "\n".join((*group.values, "none (do not select any option in this group)"))
            instructions = (
                f'Now choose one option for the "{group.name}" group that best satisfies the user\'s needs. '
                'The "none" choice means never clicking an option in this group. '
                f'Think about what is the best choice inside <think>...</think> before giving your answer. '
            )
            prompt = _build_product_page_prompt(parts, options, instructions, selection_description=f"Your available options for {group.name} are:")
            answers = []
            prompt_ids = None
            scored_pairs = [(action, name) for action, name in zip(planned.actions, names) if action in useful[planned.name]]
            if not scored_pairs:
                raise ValueError(f"No full-reward choice is available for option group {planned.name!r}")
            response_prefix = f"<think> The best choice for the {group.name} group is "
            for _, name in scored_pairs:
                answer = prepare_response_suffix_probe(prompt, response_prefix, name, tokenizer, prompt_token_ids=prompt_ids)
                prompt_ids = answer.prompt_token_ids
                answers.append(answer)
            prepared.append((TrainingProductOptionProbe(tuple(action for action, _ in scored_pairs), tuple(name for _, name in scored_pairs), tuple(answers)), group.name))
        return prepared

    def _finish_products(self, jobs):
        for job in jobs:
            if job.probability is None:
                job.probability = job.plan.aggregate_success_mass(job.groups)
            if job.cache_asin is not None:
                self._product_cache.setdefault(job.snapshot.query_key, {})[job.cache_asin] = job.probability


class WebshopInfoGainScorer:
    """Accept padded/reordered info_gain_batch rows and return aligned log scores.

    Missing probabilities remain None in non_tensor_batch and NaN in the
    tensor output. The explicit scored mask must be consulted by the caller.
    """

    def __init__(self, tokenizer, actor_rollout_wg, *, max_model_len, path_probability_threshold=1e-3, batch_size=32, apply_chat_template_kwargs=None, prune_unsuccessful_choices=True):
        # Import native rendering only on the WebShop training path.
        native_root = Path(__file__).resolve().parents[1] / "agent_system/environments/env_package/webshop/webshop"
        if str(native_root) not in sys.path:
            sys.path.append(str(native_root))
        tokenizer = _TrainingTokenizer(tokenizer, apply_chat_template_kwargs or {})
        self.probes = TrainingPseudoProbeScorer(tokenizer, actor_rollout_wg, max_model_len=max_model_len, batch_size=batch_size)
        self.threshold = path_probability_threshold
        if not isinstance(prune_unsuccessful_choices, bool):
            raise ValueError("prune_unsuccessful_choices must be a boolean")
        self.prune_unsuccessful_choices = prune_unsuccessful_choices
        self.scorers = {}
        self.cache_reuses = 0
        self.scoring_time_seconds = 0.0
        self.probability_ranges = {"search_results": [math.inf, -math.inf], "item_page": [math.inf, -math.inf]}

    def metrics(self):
        """Rollout-wide metrics, including all scoring calls in this policy step."""
        metrics = {
            "scorer/cache_reuses": self.cache_reuses,
            "scorer/scoring_time_seconds": self.scoring_time_seconds,
            "scorer/forward_pass_time_seconds": self.probes.forward_pass_time_seconds,
        }
        for page, (minimum, maximum) in self.probability_ranges.items():
            if minimum <= maximum:
                metrics[f"scorer/{page}_probability_min"] = minimum
                metrics[f"scorer/{page}_probability_max"] = maximum
        return metrics

    @staticmethod
    def require_active_scores(batch):
        missing = batch.non_tensor_batch["webshop_active"] & ~batch.batch["webshop_scored"].cpu().numpy()
        if missing.any():
            reasons = [(int(index), batch.non_tensor_batch["webshop_skipped_reason"][index]) for index in np.flatnonzero(missing)]
            raise ValueError(f"TreeHCA WebShop scoring is not implemented for these active rows: {reasons}. Their success probabilities remain None.")

    def compute(self, batch, *, policy_version):
        scoring_start = time.perf_counter()
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
                    self.scorers[key] = TrainingWebshopTurnSuccessScorer(source, self.probes, path_probability_threshold=self.threshold, prune_unsuccessful_choices=self.prune_unsuccessful_choices)
                self.scorers[key].add_catalog_payload(payload)
                groups[key].append((index, snapshot))
        for key, rows in groups.items():
            scores = self.scorers[key].score([snapshot for _, snapshot in rows], policy_version=policy_version)
            self.cache_reuses += self.scorers[key].last_cache_reuses
            for (index, _), score in zip(rows, scores):
                probabilities[index], reasons[index] = score.probability, score.skipped_reason
        for payload, probability in zip(payloads, probabilities):
            if payload is None or probability is None:
                continue
            page = payload["snapshot"].page_type
            if page in self.probability_ranges:
                bounds = self.probability_ranges[page]
                bounds[0] = min(bounds[0], probability)
                bounds[1] = max(bounds[1], probability)
        device = batch.batch["prompts"].device
        logs = [math.nan if p is None else (-math.inf if p == 0 else math.log(p)) for p in probabilities]
        batch.batch["avg_ans_log_probs"] = torch.tensor(logs, dtype=torch.float64, device=device)
        batch.batch["webshop_scored"] = torch.tensor([p is not None for p in probabilities], dtype=torch.bool, device=device)
        batch.non_tensor_batch["webshop_success_probability"] = probabilities
        batch.non_tensor_batch["webshop_skipped_reason"] = reasons
        self.scoring_time_seconds += time.perf_counter() - scoring_start
        return batch
