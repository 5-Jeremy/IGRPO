"""Identify TreeHCA backup boundaries for allocation-pruned leaves.

Tree rollouts label a leaf ``pruned`` when it receives no expansion allocation
and ``success`` when it reaches a full-score environment terminal. A pruned
leaf may still train its own branch, but it must not cross an ancestor whose
subtree already contains a successful leaf. Turn-limit and other terminal
leaves are deliberately not classified as pruned.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrematureLeafFilter:
    """Tree nodes involved in the TreeHCA premature-leaf backup rule."""

    premature_leaves: frozenset
    full_score_leaves: frozenset
    protected_ancestors: frozenset

    @classmethod
    def from_rows(cls, node_uid, parent_node_uid, termination_reason=None):
        """Build the filter from possibly duplicated batch rows.

        ``adjust_batch`` can duplicate logical nodes. Classification therefore
        happens by node id, after deriving leaves from the stored tree edges.
        ``success`` is used instead of a numeric reward threshold because reward
        scales differ by environment (for example, WebShop full credit is 10).
        """
        if termination_reason is None:
            return cls(frozenset(), frozenset(), frozenset())
        if not (len(node_uid) == len(parent_node_uid) == len(termination_reason)):
            raise ValueError("TreeHCA premature-leaf inputs must have the same batch size")

        nodes = set(node_uid)
        parents = {}
        reasons = {}
        for node, parent, reason in zip(node_uid, parent_node_uid, termination_reason):
            parents.setdefault(node, parent)
            reasons.setdefault(node, reason)

        internal = {parent for parent in parents.values() if parent in nodes}
        leaves = nodes - internal
        premature = frozenset(node for node in leaves if reasons.get(node) == "pruned")
        full_score = frozenset(node for node in leaves if reasons.get(node) == "success")

        protected = set()
        for leaf in full_score:
            node = parents.get(leaf)
            seen = set()
            while node in nodes:
                if node in seen:
                    raise ValueError(f"Cycle in TreeHCA rollout tree at {node}")
                seen.add(node)
                protected.add(node)
                node = parents.get(node)

        return cls(premature, full_score, frozenset(protected))

    def stops(self, leaf, ancestor) -> bool:
        """Whether ``leaf`` must stop before contributing to ``ancestor``."""
        return leaf in self.premature_leaves and ancestor in self.protected_ancestors
