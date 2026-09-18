"""Two-stage, query-scoped pseudo success scoring for batches of WebShop turns."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_batch import MixedPseudoRolloutScorer, PseudoProbe
from treehca.pseudo_rollout_product_page import ProductPagePseudoRolloutScores, build_action_label_catalog, prepare_product_page_grouped_choice_rollouts
from treehca.pseudo_rollout_results_page import prepare_results_page_answer_probe
from treehca.webshop_option_success import NativeOptionSuccessPlan, build_native_option_success_plan
from treehca.webshop_probability_snapshot import WebshopSnapshotSource, WebshopTurnSnapshot


@dataclass(frozen=True)
class TurnSuccessProbability:
    probability: float | None
    skipped_reason: str | None = None


@dataclass
class _ProductJob:
    snapshot: WebshopTurnSnapshot
    plan: NativeOptionSuccessPlan
    probability: float | None = None
    cache_asin: str | None = None
    groups: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class _SearchJob:
    snapshot: WebshopTurnSnapshot
    actions: dict[str, str]
    entry_probabilities: dict[str, float] = field(default_factory=dict)
    products: dict[str, _ProductJob] = field(default_factory=dict)


def _estimate_search_results_probabilities(job: _SearchJob, threshold: float) -> float:
    """Trimmed current-page estimator: sum retained entry × native success.

    No pagination, entry-only metrics, fallback, or option-level pruning.
    Pruned branches are excluded even if their product score is cached.
    """
    terms = []
    for asin, probability in job.entry_probabilities.items():
        if probability == 0 or probability < threshold:
            continue
        product = job.products[asin]
        if product.probability is None:
            raise ValueError("Retained product branches must be scored before aggregation")
        terms.append(probability * product.probability)
    return min(1.0, max(0.0, math.fsum(terms)))


class WebshopTurnSuccessScorer:
    """Score snapshots in input order, with persistent fresh-entry caches.

    ``policy_version`` is required on every call. Change it whenever weights,
    adapters, or model scoring settings change; this clears probability caches.
    One instance owns one immutable catalog source, tokenizer, and probe scorer.
    Calls are synchronous and must not overlap on the same instance.
    """

    def __init__(self, source: WebshopSnapshotSource, probe_scorer: MixedPseudoRolloutScorer, *, path_probability_threshold: float = 1e-3):
        threshold = float(path_probability_threshold)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("path_probability_threshold must be finite and in [0, 1]")
        self.source = source
        self.probe_scorer = probe_scorer
        self.threshold = threshold
        self._policy_version: Hashable = object()
        # Outer keys enforce shopping-query equality before every lookup.
        self._product_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._results_cache: dict[tuple[str, str], dict[tuple, float]] = {}
        self._plans: dict[tuple, NativeOptionSuccessPlan] = {}
        self._label_catalog = None
        self.last_cache_reuses = 0

    def clear_probability_cache(self) -> None:
        self._product_cache.clear()
        self._results_cache.clear()

    def _plan(self, snapshot: WebshopTurnSnapshot, asin: str, selected: tuple[tuple[str, str], ...]) -> NativeOptionSuccessPlan:
        key = (snapshot.query_key, asin, selected)
        if key not in self._plans:
            server = self.source.server
            self._plans[key] = build_native_option_success_plan(server.product_item_dict[asin], json.loads(snapshot.goal_json), server.product_prices[asin], dict(selected))
        return self._plans[key]

    @staticmethod
    def _results_key(snapshot: WebshopTurnSnapshot) -> tuple:
        # Include displayed content as a guard against changed result ordering.
        observation = extract_product_page_contexts([snapshot.prompt])[0].current_observation
        return snapshot.search_terms, snapshot.results_page, snapshot.visible_asins, observation

    def score(self, snapshots: Sequence[WebshopTurnSnapshot], *, policy_version: Hashable) -> list[TurnSuccessProbability]:
        hash(policy_version)
        self.last_cache_reuses = 0
        if policy_version != self._policy_version:
            self.clear_probability_cache()
            self._policy_version = policy_version
        results: list[TurnSuccessProbability | _ProductJob | _SearchJob] = []
        product_jobs: dict[tuple, _ProductJob] = {}
        search_jobs: dict[tuple, _SearchJob] = {}

        def product_job(snapshot: WebshopTurnSnapshot) -> _ProductJob:
            if snapshot.asin is None:
                raise ValueError("Product snapshots require an ASIN")
            key = (snapshot.query_key, snapshot.asin) if snapshot.fresh_product_entry else ("configured", snapshot)
            if key not in product_jobs:
                cached = self._product_cache.get(snapshot.query_key, {}).get(snapshot.asin) if snapshot.fresh_product_entry else None
                if cached is not None:
                    self.last_cache_reuses += 1
                    plan = NativeOptionSuccessPlan((), 0, 0, cached)
                else:
                    plan = self._plan(snapshot, snapshot.asin, snapshot.selected_options)
                product_jobs[key] = _ProductJob(snapshot, plan, plan.constant_probability, snapshot.asin if snapshot.fresh_product_entry else None)
            return product_jobs[key]

        for index, snapshot in enumerate(snapshots):
            if snapshot.catalog_key != self.source.catalog_key:
                raise ValueError(f"Snapshot {index} belongs to a different catalog source")
            if snapshot.terminated or snapshot.page_type not in {"search_results", "item_page"}:
                if snapshot.page_type == "": # Initial search page
                    results.append(TurnSuccessProbability(0.0, "deferred_initial_search_page"))
                elif snapshot.page_type == "item_sub_page":
                    results.append(TurnSuccessProbability(0.0, "deferred_item_sub_page"))
                else:
                    results.append(TurnSuccessProbability(None, "terminated" if snapshot.terminated else f"unsupported_page:{snapshot.page_type}"))
                continue
            parts = extract_product_page_contexts([snapshot.prompt])[0]
            if parts.shopping_task != snapshot.shopping_task or json.loads(snapshot.goal_json)["instruction_text"] != snapshot.shopping_task:
                raise ValueError("Snapshot prompt and goal must describe the same shopping query")
            if snapshot.page_type == "item_page":
                results.append(product_job(snapshot))
                continue
            page_key = self._results_key(snapshot)
            cached = None if snapshot.randomized_search else self._results_cache.get(snapshot.query_key, {}).get(page_key)
            if cached is not None:
                self.last_cache_reuses += 1
                results.append(TurnSuccessProbability(cached))
                continue
            # Random pages are not cached; identical input states can share work
            # within this call, but never across separate observations/calls.
            key = (snapshot.query_key, page_key, snapshot.prompt if snapshot.randomized_search else None)
            if key not in search_jobs:
                actions = {}
                for asin in snapshot.visible_asins:
                    if self._plan(snapshot, asin, ()).constant_probability == 0.0:
                        continue
                    action = next((a for a in parts.admissible_actions if a.lower() == f"click[{asin.lower()}]"), None)
                    if action is None:
                        raise ValueError(f"Visible product {asin!r} is missing from admissible actions")
                    actions[asin] = action
                search_jobs[key] = _SearchJob(snapshot, actions)
            results.append(search_jobs[key])

        probes: list[PseudoProbe] = []
        owners = []
        for job in search_jobs.values():
            prompt_ids = None
            for asin, action in job.actions.items():
                probe = prepare_results_page_answer_probe(job.snapshot.prompt, action, self.probe_scorer.tokenizer, prompt_token_ids=prompt_ids)
                prompt_ids = probe.prompt_token_ids
                probes.append(probe)
                owners.append((job, asin))
        self._append_product_probes(list(product_jobs.values()), probes, owners)
        self._run_probes(probes, owners)
        self._finish_products(product_jobs.values())

        new_products = []
        for job in search_jobs.values():
            for asin, probability in job.entry_probabilities.items():
                if probability == 0 or probability < self.threshold:
                    continue
                key = (job.snapshot.query_key, asin)
                if key not in product_jobs:
                    cached = self._product_cache.get(job.snapshot.query_key, {}).get(asin)
                    if cached is not None:
                        self.last_cache_reuses += 1
                    plan = self._plan(job.snapshot, asin, ())
                    probability = plan.constant_probability if cached is None else cached
                    if probability is not None:
                        # Resolved fresh-entry scores need no hypothetical HTML.
                        product_jobs[key] = _ProductJob(job.snapshot, plan, probability, asin)
                        new_products.append(product_jobs[key])
                    else:
                        child = self.source.product_entry(job.snapshot, asin)
                        new_products.append(product_job(child))
                job.products[asin] = product_jobs[key]
        probes, owners = [], []
        self._append_product_probes(new_products, probes, owners)
        self._run_probes(probes, owners)
        self._finish_products(new_products)

        answer = []
        for result in results:
            if isinstance(result, TurnSuccessProbability):
                answer.append(result)
            elif isinstance(result, _ProductJob):
                answer.append(TurnSuccessProbability(result.probability))
            else:
                probability = _estimate_search_results_probabilities(result, self.threshold)
                if not result.snapshot.randomized_search:
                    self._results_cache.setdefault(result.snapshot.query_key, {})[self._results_key(result.snapshot)] = probability
                answer.append(TurnSuccessProbability(probability))
        return answer

    def _append_product_probes(self, jobs: Sequence[_ProductJob], probes: list, owners: list) -> None:
        jobs = [job for job in jobs if job.probability is None]
        if not jobs:
            return
        tokenizer = self.probe_scorer.tokenizer
        # Reuse tokenizer-safe labels for all groups, including later stages.
        max_choices = max((len(values) + 1 for item in self.source.server.product_item_dict.values() for values in item.get("options", {}).values()), default=0)
        if self._label_catalog is None:
            self._label_catalog = build_action_label_catalog(tokenizer, max_choices)
        prepared = prepare_product_page_grouped_choice_rollouts(
            extract_product_page_contexts([job.snapshot.prompt for job in jobs]),
            tokenizer,
            [{} for _ in jobs],
            label_catalog=self._label_catalog,
            group_names_by_page=[[group.name for group in job.plan.groups] for job in jobs],
            annotate_correctness=False,
        )
        for probe in prepared:
            job = jobs[probe.source_page_index]
            group = next(group for group in job.plan.groups if group.name == probe.option_group.name)
            if probe.actions != group.actions:
                raise ValueError("Snapshot product options differ from the native catalog")
            probes.append(probe)
            owners.append((job, group.name))

    def _run_probes(self, probes: list, owners: list) -> None:
        scores = self.probe_scorer.score(probes)
        if len(scores) != len(probes):
            raise ValueError("Pseudo scores must align with their requests")
        for (job, name), score in zip(owners, scores):
            if isinstance(job, _SearchJob):
                probability = float(score)
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Results probabilities must be finite and in [0, 1]")
                job.entry_probabilities[name] = probability
            else:
                if not isinstance(score, ProductPagePseudoRolloutScores):
                    raise ValueError("Product probe must return a complete choice distribution")
                job.groups[name] = score.action_probabilities

    def _finish_products(self, jobs: Any) -> None:
        for job in jobs:
            if job.probability is None:
                job.probability = job.plan.aggregate(job.groups)
            if job.cache_asin is not None:
                self._product_cache.setdefault(job.snapshot.query_key, {})[job.cache_asin] = job.probability
