"""Resolve WebShop state scores before computing gains along tree edges."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

INITIAL_SEARCH = "deferred_initial_search_page"
ITEM_SUB_PAGE = "deferred_item_sub_page"


@dataclass
class _Node:
    group: str
    parent: str
    log_probability: float
    reason: str | None
    first_child: str | None = None
    row: dict | None = None


class WebshopDeferredScores:
    """Own one virtual root per prompt group and references to saved node rows.

    Scores are recomputed when a level arrives, including historical rows whose
    parent or deferred descendant changed. Branch decisions already made are
    not replayed. First children follow active rollout-slot order.
    """

    def __init__(self, groups, *, prob_diff_mode, prob_floor):
        if not math.isfinite(prob_floor) or not 0 < prob_floor <= 1:
            raise ValueError("WebShop deferred scoring requires 0 < prob_floor <= 1")
        self.prob_diff_mode = prob_diff_mode
        self.log_floor = math.log(prob_floor)
        self.nodes: dict[str, _Node] = {}
        self.root_children = defaultdict(list)
        # No root children exist before the first rollout iteration.
        self.roots = {group: self._values(-math.inf) for group in groups}
        for root in self.roots.values():
            root["info_gain"] = root["info_gain_sum"]

    def _values(self, log_probability):
        probability = math.exp(log_probability)
        # Collation keeps object arrays; reward metrics expect NumPy scalars
        # even when a saved score becomes a pruned node's reward.
        return {
            "avg_ans_log_probs": np.float64(log_probability),
            "webshop_success_probability": np.float64(probability),
            "info_gain_sum": np.float64(probability if self.prob_diff_mode else max(self.log_floor, log_probability)),
        }

    def update(self, batch, active, node_uid2info_gain_sum):
        """Resolve parents and saved rows, then populate the current batch gains."""
        data = batch.non_tensor_batch
        for index in np.flatnonzero(active):
            uid, parent, group = (data[key][index] for key in ("node_uid", "parent_node_uid", "uid"))
            if uid in self.nodes:
                raise ValueError(f"WebShop node {uid!r} was already registered")
            node = _Node(group, parent, float(data["avg_ans_log_probs"][index]), data["webshop_skipped_reason"][index])
            self.nodes[uid] = node
            if parent == "root":
                self.root_children[group].append(uid)
            else:
                parent_node = self.nodes[parent]
                if parent_node.group != group:
                    raise ValueError("WebShop tree edges must stay in the same prompt group")
                if parent_node.first_child is None:
                    parent_node.first_child = uid

        # Follow first-child chains once. None denotes dependence on the root
        # mean, rather than a model score or an unexpanded subpage placeholder.
        sources = {}

        def source(uid):
            if uid not in sources:
                node = self.nodes[uid]
                if node.reason == INITIAL_SEARCH:
                    sources[uid] = None
                elif node.reason == ITEM_SUB_PAGE and node.first_child is not None:
                    sources[uid] = source(node.first_child)
                else:
                    sources[uid] = node.log_probability
            return sources[uid]

        for group, children in self.root_children.items():
            independent = [source(uid) for uid in children if source(uid) is not None]
            # Solves m = mean(independent scores, m, ...). If all children
            # depend on m, the zero-probability placeholder remains unchanged.
            mean = sum(independent) / len(independent) if independent else -math.inf
            self.roots[group] = self._values(mean)
            self.roots[group]["info_gain"] = self.roots[group]["info_gain_sum"]

        values = {}
        for uid, node in self.nodes.items():
            log_probability = source(uid)
            if log_probability is None:
                log_probability = self.roots[node.group]["avg_ans_log_probs"]
            values[uid] = self._values(log_probability)
            node_uid2info_gain_sum[uid] = values[uid]["info_gain_sum"]

        # Insertion order is parent-first. Update historical gains before any
        # current child gains; all parent cumulative scores are already resolved.
        for uid, node in self.nodes.items():
            root = self.roots[node.group]
            parent_sum = root["info_gain_sum"] if node.parent == "root" else node_uid2info_gain_sum[node.parent]
            values[uid]["info_gain"] = values[uid]["info_gain_sum"] - parent_sum
            values[uid]["webshop_root_avg_ans_log_probs"] = root["avg_ans_log_probs"]
            if node.row is not None:
                node.row.update(values[uid])

        data["webshop_root_avg_ans_log_probs"] = np.full(len(batch), -math.inf)
        for index in np.flatnonzero(active):
            for key, value in values[data["node_uid"][index]].items():
                data[key][index] = value

    def register_rows(self, rows):
        """Keep the actual dictionaries that downstream training will consume."""
        for row in rows:
            self.nodes[row["node_uid"]].row = row
