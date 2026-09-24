"""
TreeHCA credit assignment.

The rollout is identical to IGRPO: the tree is grown by the information-gain
expansion scheme in ``igrpo.core_igrpo.TrajectoryNodeStateManagement``. Only the
credit assignment differs.

A leaf keeps its own (optionally group-baselined) outcome score. An internal node
takes a self-normalised importance-weighted mean of its children's advantages,

    w_c   = n_c * (1 / p_gt(c)) ** temp
    A_par = sum_c (w_c / sum_c' w_c') * A_c

where ``p_gt(c)`` is the IGPO ground-truth answer probability already stored on
every node as ``info_gain_sum``, and ``n_c`` is the number of leaves under child
``c``. Children that make the answer look unlikely get the most weight, so a
parent is credited for the branches that still had work left to do.

The ``n_c`` factor is what keeps the estimator comparable to IGRPO. IGRPO unrolls
every root-to-leaf path, so a node appears once per descendant leaf and its credit
is a flat mean over the leaves of its subtree. Weighting only by 1/p_gt would
instead be a flat mean over *siblings*, which silently discounts a child that
carries a large subtree. With the ``n_c`` factor, equal sibling p_gt reduces
exactly to the IGRPO backup, and equal subtree sizes reduce to pure 1/p_gt SNIS.

The resulting per-node scalar is broadcast over the response tokens and consumed
by the unchanged GRPO/PPO clipped loss.
"""

from collections import defaultdict

import numpy as np
import torch

from treehca.premature_leaf_filter import PrematureLeafFilter
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def cap_no_progress_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    node_uid: np.ndarray,
    parent_node_uid: np.ndarray,
    is_terminal: np.ndarray,
    info_gain: np.ndarray,
    successful_terminal: np.ndarray,
    turns_threshold: int = 3,
    info_gain_threshold: float = 0.05,
):
    """Cap advantages for sustained low-progress runs on successful paths.

    Every successful terminal defines one root-to-leaf rollout. A maximal
    contiguous run whose nodes all have information gain strictly below the
    configured threshold is capped when it contains enough turns. Logical
    nodes duplicated in the batch are modified together. Terminal nodes are
    excluded from both the run length and the modification.

    Returns:
        The modified advantages and diagnostics for the trainer logger.
    """
    if isinstance(turns_threshold, (bool, np.bool_)) or not isinstance(turns_threshold, (int, np.integer)) or turns_threshold < 1:
        raise ValueError(f"treehca.no_progress_turns_threshold must be an integer >= 1, got {turns_threshold!r}")
    if isinstance(info_gain_threshold, (bool, np.bool_)) or not np.isscalar(info_gain_threshold):
        raise ValueError(f"treehca.no_progress_info_gain_threshold must be a finite float >= 0, got {info_gain_threshold!r}")
    info_gain_threshold = float(info_gain_threshold)
    if not np.isfinite(info_gain_threshold) or info_gain_threshold < 0:
        raise ValueError(f"treehca.no_progress_info_gain_threshold must be a finite float >= 0, got {info_gain_threshold!r}")

    batch_size = advantages.shape[0]
    if any(len(values) != batch_size for values in (response_mask, node_uid, parent_node_uid, is_terminal, info_gain, successful_terminal)):
        raise ValueError("TreeHCA no-progress inputs must all have the same batch size")

    node_rows = defaultdict(list)
    for row, node in enumerate(node_uid):
        node_rows[node].append(row)
    row_of = {node: rows[0] for node, rows in node_rows.items()}

    nodes_to_cap = set()
    successful_rollouts = 0
    qualifying_runs = 0
    for terminal_node, terminal_rows in node_rows.items():
        if not any(bool(successful_terminal[row]) for row in terminal_rows):
            continue
        successful_rollouts += 1

        # Follow the stored tree edges, then scan the rollout chronologically.
        path = []
        node = terminal_node
        seen = set()
        while node in row_of:
            if node in seen:
                raise ValueError(f"Cycle in TreeHCA rollout tree at {node}")
            seen.add(node)
            path.append(node)
            node = parent_node_uid[row_of[node]]

        run = []
        for node in reversed(path):
            if bool(is_terminal[row_of[node]]):
                if len(run) >= turns_threshold:
                    nodes_to_cap.update(run)
                    qualifying_runs += 1
                run = []
                continue
            if float(info_gain[row_of[node]]) < info_gain_threshold:
                run.append(node)
                continue
            if len(run) >= turns_threshold:
                nodes_to_cap.update(run)
                qualifying_runs += 1
            run = []
        if len(run) >= turns_threshold:
            nodes_to_cap.update(run)
            qualifying_runs += 1

    modified = advantages.clone()
    capped_rows = [row for node in nodes_to_cap for row in node_rows[node]]
    if capped_rows:
        modified[capped_rows] = torch.minimum(modified[capped_rows], torch.zeros_like(modified[capped_rows]))
        # Retain the standard invariant that padding has zero advantage.
        modified[capped_rows] *= response_mask[capped_rows]

    metrics = {
        "treehca/no_progress/successful_rollout_count": float(successful_rollouts),
        "treehca/no_progress/qualifying_run_count": float(qualifying_runs),
        "treehca/no_progress/capped_node_count": float(len(nodes_to_cap)),
        "treehca/no_progress/capped_row_count": float(len(capped_rows)),
    }
    return modified, metrics


def _gt_prob(raw_value, prob_diff_mode: bool, prob_floor: float) -> float:
    """Ground-truth answer probability of a node.

    The rollout stores ``exp(avg answer log-prob)`` in ``info_gain_sum`` when
    ``algorithm.igrpo.prob_diff_mode`` is on, and the raw average log-prob
    otherwise.
    """
    value = float(raw_value)
    if not prob_diff_mode:
        value = float(np.exp(value))
    return float(np.clip(value, prob_floor, 1.0))


def compute_treehca_outcome_advantage(token_level_rewards: torch.Tensor,
                                      response_mask: torch.Tensor,
                                      uid: np.ndarray,
                                      node_uid: np.ndarray,
                                      parent_node_uid: np.ndarray,
                                      is_terminal: np.ndarray,
                                      info_gain_sum: np.ndarray,
                                      traj_step: np.ndarray,
                                      termination_reason: np.ndarray | None = None,
                                      filter_premature_leaves: bool = False,
                                      prob_diff_mode: bool = True,
                                      prob_floor: float = 1e-6,
                                      weight_temp: float = 1.0,
                                      max_weight_ratio: float = -1.0,  # cap w_c at this multiple of the mean sibling weight
                                      subtree_size_weight: bool = True,
                                      leaf_baseline: str = "group",
                                      norm_adv_by_std: bool = True,
                                      eps: float = 1e-6):
    """
    Args:
        token_level_rewards: `(torch.Tensor)` shape (bs, response_length)
        response_mask: `(torch.Tensor)` shape (bs, response_length)
        uid: prompt-group id of each row
        node_uid / parent_node_uid: tree edges. Rows duplicated by ``adjust_batch``
            share a ``node_uid`` and are collapsed to a single node here.
        is_terminal: leaf flag from the rollout, used for diagnostics only
        info_gain_sum: p_gt (or log p_gt) of the node, see :func:`_gt_prob`
        traj_step: tree depth of the node, used to walk the DAG bottom-up
        termination_reason: rollout stop classification used when
            ``filter_premature_leaves`` is enabled
        filter_premature_leaves: remove ``pruned`` leaves from backups at
            ancestors containing a ``success`` leaf. Disabled by default;
            ``turn_limit`` leaves continue to propagate when enabled
        subtree_size_weight: scale w_c by the number of leaves under child c, so
            the backup matches IGRPO's flat mean over subtree leaves
        leaf_baseline: "group" for a GRPO-style baseline over the leaves of a
            prompt group, "none" to use the raw outcome score

    Returns:
        advantages: `(torch.Tensor)` shape (bs, response_length)
        returns: `(torch.Tensor)` shape (bs, response_length)
        metrics: `(dict)` diagnostics for the logger
    """
    scores = token_level_rewards.sum(dim=-1)
    bs = scores.shape[0]

    with torch.no_grad():
        # collapse duplicated rows: one entry per tree node
        node_rows = defaultdict(list)
        for i in range(bs):
            node_rows[node_uid[i]].append(i)
        row_of = {node: rows[0] for node, rows in node_rows.items()}

        # a parent outside the batch ("root") makes the node a tree root
        children = defaultdict(list)
        for node, i in row_of.items():
            parent = parent_node_uid[i]
            if parent in node_rows:
                children[parent].append(node)

        leaves = [node for node in row_of if len(children[node]) == 0]
        leaf_filter = (
            PrematureLeafFilter.from_rows(node_uid, parent_node_uid, termination_reason)
            if filter_premature_leaves
            else PrematureLeafFilter.from_rows([], [])
        )

        node_adv = {}
        unpruned_node_adv = {}
        if leaf_baseline == "group":
            group_scores = defaultdict(list)
            unpruned_group_scores = defaultdict(list)
            for node in leaves:
                group_scores[uid[row_of[node]]].append(scores[row_of[node]])
                if node not in leaf_filter.premature_leaves:
                    unpruned_group_scores[uid[row_of[node]]].append(scores[row_of[node]])

            def group_stats(grouped_scores):
                means, stds = {}, {}
                for group, group_score in grouped_scores.items():
                    stacked = torch.stack(group_score)
                    if len(group_score) > 1:
                        means[group] = stacked.mean()
                        stds[group] = stacked.std()
                    else:
                        means[group] = torch.zeros_like(stacked[0])
                        stds[group] = torch.ones_like(stacked[0])
                return means, stds

            group_mean, group_std = group_stats(group_scores)
            unpruned_group_mean, unpruned_group_std = group_stats(unpruned_group_scores)
            for node in leaves:
                i = row_of[node]
                advantage = scores[i] - group_mean[uid[i]]
                if norm_adv_by_std:
                    advantage = advantage / (group_std[uid[i]] + eps)
                node_adv[node] = advantage
                if node not in leaf_filter.premature_leaves:
                    clean_advantage = scores[i] - unpruned_group_mean[uid[i]]
                    if norm_adv_by_std:
                        clean_advantage = clean_advantage / (unpruned_group_std[uid[i]] + eps)
                    unpruned_node_adv[node] = clean_advantage
        elif leaf_baseline == "none":
            for node in leaves:
                node_adv[node] = scores[row_of[node]]
                if node not in leaf_filter.premature_leaves:
                    unpruned_node_adv[node] = node_adv[node]
        else:
            raise ValueError(f"Invalid leaf_baseline: {leaf_baseline}, expected one of ['group', 'none']")

        # children sit one step deeper than their parent, so walking the nodes in
        # decreasing depth guarantees every child is resolved before its parent
        internal = [node for node in row_of if node not in node_adv]
        internal.sort(key=lambda node: -int(traj_step[row_of[node]]))

        subtree_leaves = {node: 1 for node in leaves}
        unpruned_subtree_leaves = {
            node: int(node not in leaf_filter.premature_leaves) for node in leaves
        }

        ess_ratios = []
        degenerate_weights = 0

        def combine(child_nodes, values, leaf_counts):
            weights = np.array(
                [1.0 / _gt_prob(info_gain_sum[row_of[child]], prob_diff_mode, prob_floor) for child in child_nodes],
                dtype=np.float64,
            )
            if weight_temp != 1.0:
                weights = weights ** weight_temp
            if subtree_size_weight:
                # a child standing in for many leaves speaks for all of them, as it
                # would under IGRPO's path unrolling
                weights = weights * np.array([leaf_counts[child] for child in child_nodes], dtype=np.float64)
            if max_weight_ratio > 0:
                # truncated importance sampling, caps a single child's influence
                weights = np.minimum(weights, max_weight_ratio * weights.mean())
            total = weights.sum()
            degenerate = False
            if not np.isfinite(total) or total <= 0.0:
                weights = np.ones_like(weights)
                total = weights.sum()
                degenerate = True
            weights = weights / total

            advantage = None
            for weight, child in zip(weights, child_nodes):
                term = values[child] * float(weight)
                advantage = term if advantage is None else advantage + term
            ess = 1.0 / (float(np.sum(weights ** 2)) * len(child_nodes))
            return advantage, ess, degenerate

        for node in internal:
            child_nodes = children[node]
            subtree_leaves[node] = sum(subtree_leaves[child] for child in child_nodes)
            unpruned_subtree_leaves[node] = sum(unpruned_subtree_leaves[child] for child in child_nodes)

            clean_children = [child for child in child_nodes if unpruned_subtree_leaves[child] > 0]
            if clean_children:
                clean_advantage, clean_ess, clean_degenerate = combine(
                    clean_children, unpruned_node_adv, unpruned_subtree_leaves
                )
                unpruned_node_adv[node] = clean_advantage

            if node in leaf_filter.protected_ancestors:
                # A successful descendant guarantees at least one clean child.
                node_adv[node] = unpruned_node_adv[node]
                ess, degenerate = clean_ess, clean_degenerate
            else:
                node_adv[node], ess, degenerate = combine(child_nodes, node_adv, subtree_leaves)
            ess_ratios.append(ess)
            degenerate_weights += int(degenerate)

        node_advantages = torch.zeros_like(scores)
        for i in range(bs):
            node_advantages[i] = node_adv[node_uid[i]]

        advantages = node_advantages.unsqueeze(-1) * response_mask

    leaf_rows = [row_of[node] for node in leaves]
    internal_rows = [row_of[node] for node in internal]
    non_terminal_leaves = sum(1 for node in leaves if not bool(is_terminal[row_of[node]]))
    metrics = {
        "treehca/node_count": float(len(row_of)),
        "treehca/leaf_count": float(len(leaves)),
        "treehca/internal_count": float(len(internal)),
        "treehca/branching_factor": float(np.mean([len(children[node]) for node in internal])) if internal else 0.0,
        "treehca/subtree_leaves/mean": float(np.mean([subtree_leaves[node] for node in internal])) if internal else 0.0,
        "treehca/subtree_leaves/max": float(max(subtree_leaves[node] for node in internal)) if internal else 0.0,
        # 1.0 means every child of a parent is weighted equally, 1/k means one child took everything
        "treehca/snis_ess_ratio": float(np.mean(ess_ratios)) if ess_ratios else 1.0,
        "treehca/degenerate_weight_count": float(degenerate_weights),
        "treehca/leaf_without_terminal_flag": float(non_terminal_leaves),
        "treehca/premature_leaf_filter/enabled": float(filter_premature_leaves),
        "treehca/premature_leaf_filter/pruned_leaf_count": float(len(leaf_filter.premature_leaves)),
        "treehca/premature_leaf_filter/full_score_leaf_count": float(len(leaf_filter.full_score_leaves)),
        "treehca/premature_leaf_filter/protected_ancestor_count": float(len(leaf_filter.protected_ancestors)),
        "treehca/leaf_adv/mean": node_advantages[leaf_rows].mean().item() if leaf_rows else 0.0,
        "treehca/internal_adv/mean": node_advantages[internal_rows].mean().item() if internal_rows else 0.0,
        "treehca/internal_adv/abs_mean": node_advantages[internal_rows].abs().mean().item() if internal_rows else 0.0,
    }

    return advantages, advantages, metrics


def compute_treehca_q_outcome_advantage(token_level_rewards: torch.Tensor,
                                        response_mask: torch.Tensor,
                                        uid: np.ndarray,
                                        node_uid: np.ndarray,
                                        parent_node_uid: np.ndarray,
                                        is_terminal: np.ndarray,
                                        info_gain_sum: np.ndarray,
                                        traj_step: np.ndarray,
                                        termination_reason: np.ndarray | None = None,
                                        filter_premature_leaves: bool = False,
                                        prob_diff_mode: bool = True,
                                        prob_floor: float = 1e-6,
                                        max_inv_ratio: float = 2.0,
                                        grpo_weight: float = 0.7,
                                        q_weight: float = 0.3,
                                        aux_mode: str = "hindsight",
                                        norm_adv_by_std: bool = True):
    """GRPO advantage blended with a hindsight-weighted Q term.

        A = grpo_weight * A_grpo + q_weight * A_aux

    The primary signal is the ordinary GRPO advantage, computed by the shared
    :func:`compute_grpo_outcome_advantage` so it is identical to what IGRPO uses.
    It keeps the majority of the weight because the auxiliary term is the noisier
    of the two: it depends on p_gt ratios that move with the policy.

    The auxiliary term backs the outcome reward up the tree, a leaf keeping its own
    reward and a parent averaging its children, which gives every node a Q. That Q is
    then scaled by how much the node improved the answer's odds,

        A_aux = Q * (1 - 1 / h),   h = p_gt(node) / p_gt(parent)

    A node that made the gold answer more likely has h > 1, so the factor is positive
    and it keeps its Q. A node that made it less likely has h < 1 and the factor turns
    negative, so the same Q is charged against it. ``1 / h`` is clipped at
    ``max_inv_ratio``: at the default of 2 the factor spans [-1, 1] and the term
    therefore ranges from -Q to +Q, which stops one collapsed child probability from
    dominating the batch.

    With ``aux_mode="td"`` the hindsight factor is dropped and the auxiliary term is the
    temporal-difference residual instead,

        A_aux = Q(node) - Q(parent)

    Q(parent) is the mean of its children, so this sums to zero over every sibling set
    by construction rather than approximately, at the cost of no longer using p_gt.
    At a protected parent, both sides of the residual use the pruned-leaf-free Q:
    an all-pruned child subtree receives zero on the crossing edge, while retained
    child residuals still sum to zero. Ordinary Q residuals remain in use below that
    boundary so the incomplete rollout can still provide local supervision.

    Roots get no auxiliary term under either form. P(gold | question) is never measured,
    so there is no prior for the first turn to be scored against, and a root has no
    parent Q either.

    Args:
        max_inv_ratio: cap on 1 / h, see above
        aux_mode: "hindsight" for Q * (1 - 1/h), "td" for Q(node) - Q(parent)
        grpo_weight: coefficient on the GRPO term
        q_weight: coefficient on the auxiliary term. The two are independent, so they
            need not sum to 1: 0.7/0.3 is a convex blend favouring GRPO, 1.0/1.0 adds
            the aux term at full strength on top of an unscaled GRPO advantage
        termination_reason: rollout stop classification used when
            ``filter_premature_leaves`` is enabled
        filter_premature_leaves: enable the premature-leaf filter shared with
            the SNIS backup; disabled by default

    Returns:
        advantages / returns: `(torch.Tensor)` shape (bs, response_length)
        metrics: `(dict)` diagnostics for the logger
    """
    if max_inv_ratio <= 0:
        raise ValueError(f"treehca.max_inv_ratio must be > 0, got {max_inv_ratio}")
    if q_weight < 0.0:
        raise ValueError(f"treehca.q_weight must be >= 0, got {q_weight}")
    if grpo_weight < 0.0:
        raise ValueError(f"treehca.grpo_weight must be >= 0, got {grpo_weight}")
    if aux_mode not in ("hindsight", "td"):
        raise ValueError(f"Invalid treehca.aux_mode: {aux_mode}, expected one of ['hindsight', 'td']")

    grpo_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=uid,
        traj_index=uid,  # unused, compute_mean_std_cross_steps ignores it
        norm_adv_by_std_in_grpo=norm_adv_by_std,
        compute_mean_std_cross_steps=True,
        # adjust_batch may append random copies solely for worker divisibility.
        # They are training rows, but must not reweight the prompt-group baseline.
        deduplicate_by=node_uid,
    )

    scores = token_level_rewards.sum(dim=-1)
    bs = scores.shape[0]

    with torch.no_grad():
        # collapse duplicated rows: one entry per tree node
        node_rows = defaultdict(list)
        for i in range(bs):
            node_rows[node_uid[i]].append(i)
        row_of = {node: rows[0] for node, rows in node_rows.items()}

        children = defaultdict(list)
        for node, i in row_of.items():
            parent = parent_node_uid[i]
            if parent in node_rows:
                children[parent].append(node)

        leaf_filter = (
            PrematureLeafFilter.from_rows(node_uid, parent_node_uid, termination_reason)
            if filter_premature_leaves
            else PrematureLeafFilter.from_rows([], [])
        )

        # Keep a second, pruned-leaf-free value for every subtree. A protected
        # ancestor uses that value; an unprotected branch retains its original Q
        # so the pruned leaf can still provide local supervision.
        q = {}
        unpruned_q = {}
        for node in sorted(row_of, key=lambda n: -int(traj_step[row_of[n]])):
            child_nodes = children[node]
            if child_nodes:
                clean_values = [unpruned_q[child] for child in child_nodes if child in unpruned_q]
                if clean_values:
                    unpruned_q[node] = sum(clean_values) / len(clean_values)
                if node in leaf_filter.protected_ancestors:
                    # A successful descendant guarantees a clean value.
                    q[node] = unpruned_q[node]
                else:
                    q[node] = sum(q[child] for child in child_nodes) / len(child_nodes)
            else:
                q[node] = float(scores[row_of[node]])
                if node not in leaf_filter.premature_leaves:
                    unpruned_q[node] = q[node]

        node_aux = {}
        factors, clipped, roots = [], 0, 0
        for node, i in row_of.items():
            parent = parent_node_uid[i]
            if parent not in row_of:
                # P(gold | question) is never measured and a root has no parent Q, so
                # there is nothing to credit the first turn against
                node_aux[node] = 0.0
                roots += 1
                continue
            if aux_mode == "td":
                if parent in leaf_filter.protected_ancestors:
                    # The protected parent's Q is a mean over clean children only.
                    # Use that same population for its TD edges: an all-pruned
                    # child stops here, while eligible sibling residuals remain
                    # exactly zero-sum around their clean parent mean.
                    node_aux[node] = (
                        unpruned_q[node] - unpruned_q[parent]
                        if node in unpruned_q
                        else 0.0
                    )
                else:
                    node_aux[node] = q[node] - q[parent]
                continue
            p = _gt_prob(info_gain_sum[i], prob_diff_mode, prob_floor)
            p_parent = _gt_prob(info_gain_sum[row_of[parent]], prob_diff_mode, prob_floor)
            inv_ratio = p_parent / p  # _gt_prob floors p, so this cannot divide by zero
            if inv_ratio > max_inv_ratio:
                inv_ratio = max_inv_ratio
                clipped += 1
            factor = 1.0 - inv_ratio
            factors.append(factor)
            node_aux[node] = q[node] * factor

        aux = torch.zeros_like(scores)
        for i in range(bs):
            aux[i] = node_aux[node_uid[i]]

        grpo_term = grpo_weight * grpo_advantages
        aux_term = q_weight * aux.unsqueeze(-1) * response_mask
        advantages = grpo_term + aux_term

    leaves = [node for node in row_of if len(children[node]) == 0]
    token_count = response_mask.sum().clamp(min=1.0)
    # signed and token-weighted, i.e. the push the term actually applies in the loss.
    # an abs mean hides the sign and a plain mean is diluted by padding.
    aux_mean = (aux_term.sum() / token_count).item()
    aux_var = ((aux_term - aux_mean) ** 2 * response_mask).sum() / token_count
    aux_std = float(aux_var.sqrt().item())
    metrics = {
        "treehca/node_count": float(len(row_of)),
        "treehca/leaf_count": float(len(leaves)),
        "treehca/root_frac": float(roots) / max(len(row_of), 1),
        "treehca/q/mean": float(np.mean(list(q.values()))) if q else 0.0,
        "treehca/hindsight_factor/mean": float(np.mean(factors)) if factors else 0.0,
        "treehca/hindsight_factor/neg_frac": float(np.mean([f < 0 for f in factors])) if factors else 0.0,
        "treehca/hindsight_factor/clipped_frac": float(clipped) / max(len(row_of), 1),
        "treehca/premature_leaf_filter/enabled": float(filter_premature_leaves),
        "treehca/premature_leaf_filter/pruned_leaf_count": float(len(leaf_filter.premature_leaves)),
        "treehca/premature_leaf_filter/full_score_leaf_count": float(len(leaf_filter.full_score_leaves)),
        "treehca/premature_leaf_filter/protected_ancestor_count": float(len(leaf_filter.protected_ancestors)),
        # post-blend, so the two are directly comparable as contributions to the loss
        "treehca/grpo_adv/abs_mean": grpo_term.abs().mean().item(),
        "treehca/aux_adv/abs_mean": aux_term.abs().mean().item(),
        "treehca/aux_adv/mean": aux_mean,
        # directional push relative to the term's own spread, keep it under ~0.05
        "treehca/aux_adv/bias_ratio": abs(aux_mean) / aux_std if aux_std > 0 else 0.0,
    }

    return advantages, advantages, metrics
