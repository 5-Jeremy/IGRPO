"""Native full-reward aggregation over independent option-choice probes."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from treehca.pseudo_rollout_product_page import GROUP_NONE_ACTION


@dataclass(frozen=True)
class OptionCoverageGroup:
    """Native target coverage of every choice, including the unchanged none label."""

    name: str
    actions: tuple[str, ...]
    masks: tuple[int, ...]


@dataclass(frozen=True)
class NativeOptionSuccessPlan:
    """Only uncertain groups that can change full-reward success remain."""

    groups: tuple[OptionCoverageGroup, ...]
    fixed_mask: int
    full_mask: int
    constant_probability: float | None = None

    def aggregate(self, distributions: Mapping[str, Mapping[str, float]]) -> float:
        if self.constant_probability is not None:
            return self.constant_probability
        if set(distributions) != {group.name for group in self.groups}:
            raise ValueError("Every required option group must have exactly one distribution")
        masses = {self.fixed_mask: 1.0}
        for group in self.groups:
            choices = distributions[group.name]
            if set(choices) != set(group.actions):
                raise ValueError(f"Distribution must contain every choice for {group.name!r}")
            probabilities = [float(choices[action]) for action in group.actions]
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities) or not math.isclose(math.fsum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Option probabilities must be finite, nonnegative, and sum to one")
            next_masses = defaultdict(list)
            for covered, mass in masses.items():
                for mask, probability in zip(group.masks, probabilities):
                    next_masses[covered | mask].append(mass * probability)
            masses = {mask: math.fsum(terms) for mask, terms in next_masses.items()}
        return min(1.0, max(0.0, masses.get(self.full_mask, 0.0)))


def build_native_option_success_plan(item: Mapping[str, Any], goal: Mapping[str, Any], price: float, selected_options: Mapping[str, str]) -> NativeOptionSuccessPlan:
    """Use native fuzzy matches and a native reward witness, without inference.

    Selected values covering a requested target are maintained when compatible
    with a full-reward completion. A partly matching but blocking selection
    must still be corrected. Mutable groups retain the original none choice.
    """
    from web_agent_site.engine.goal import get_option_reward, get_reward

    targets = tuple(goal["goal_options"].items()) if isinstance(goal["goal_options"], dict) else tuple(goal["goal_options"])
    full_mask = (1 << len(targets)) - 1
    options = item.get("options", {})
    names = {name.lower(): name for name in options}
    selected = {names.get(name.lower(), name): value.lower() for name, value in selected_options.items()}
    if any(name not in options or value not in [v.lower() for v in options[name]] for name, value in selected.items()):
        raise ValueError("Selected options must be valid choices for this product")

    def coverage(value: str) -> int:
        return sum(1 << index for index, target in enumerate(targets) if get_option_reward((value.lower(),), (target,))[1] == 1)

    all_groups = tuple(OptionCoverageGroup(name, tuple(f"click[{value}]" for value in values) + (GROUP_NONE_ACTION,), tuple(coverage(value) for value in values) + (0,)) for name, values in options.items())

    def witnesses_for(fixed: dict[str, str]) -> tuple[int, dict[int, dict[str, str]]]:
        fixed_mask = 0
        for value in fixed.values():
            fixed_mask |= coverage(value)
        # One witness per coverage state replaces the Cartesian product.
        witnesses = {fixed_mask: dict(fixed)}
        for group in all_groups:
            if group.name in fixed:
                continue
            updated = dict(witnesses)
            for covered, selection in witnesses.items():
                for action, mask in zip(group.actions[:-1], group.masks[:-1]):
                    updated.setdefault(covered | mask, {**selection, group.name.lower(): action[6:-1].lower()})
            witnesses = updated
        return fixed_mask, witnesses

    _, witnesses = witnesses_for({})
    if full_mask not in witnesses or float(get_reward(item, goal, price=price, options=witnesses[full_mask])) != 1.0:
        return NativeOptionSuccessPlan((), 0, full_mask, 0.0)
    fixed = {}
    # Catalog order makes coupled cross-group ties deterministic. Never freeze
    # a choice that would prevent full reward with the choices already kept.
    for group in all_groups:
        value = selected.get(group.name)
        if value is not None and coverage(value) and full_mask in witnesses_for({**fixed, group.name: value})[1]:
            fixed[group.name] = value
    fixed_mask, _ = witnesses_for(fixed)
    if fixed_mask == full_mask:
        return NativeOptionSuccessPlan((), fixed_mask, full_mask, 1.0)
    mutable = tuple(group for group in all_groups if group.name not in fixed)

    required = []
    for group in mutable:
        others = {fixed_mask}
        for other in mutable:
            if other.name != group.name:
                others = {covered | mask for covered in others for mask in other.masks}
        # Omission is one choice: skip a group only if *no* choice can change
        # success for any reachable assignment of all the other groups.
        if any(covered != full_mask and any((covered | mask) == full_mask for mask in group.masks) for covered in others):
            required.append(group)
    return NativeOptionSuccessPlan(tuple(required), fixed_mask, full_mask)
