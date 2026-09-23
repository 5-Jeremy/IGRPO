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

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


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

        node_adv = {}
        if leaf_baseline == "group":
            group_scores = defaultdict(list)
            for node in leaves:
                group_scores[uid[row_of[node]]].append(scores[row_of[node]])
            group_mean, group_std = {}, {}
            for group, group_score in group_scores.items():
                stacked = torch.stack(group_score)
                if len(group_score) > 1:
                    group_mean[group] = stacked.mean()
                    group_std[group] = stacked.std()
                else:
                    group_mean[group] = torch.zeros_like(stacked[0])
                    group_std[group] = torch.ones_like(stacked[0])
            for node in leaves:
                i = row_of[node]
                advantage = scores[i] - group_mean[uid[i]]
                if norm_adv_by_std:
                    advantage = advantage / (group_std[uid[i]] + eps)
                node_adv[node] = advantage
        elif leaf_baseline == "none":
            for node in leaves:
                node_adv[node] = scores[row_of[node]]
        else:
            raise ValueError(f"Invalid leaf_baseline: {leaf_baseline}, expected one of ['group', 'none']")

        # children sit one step deeper than their parent, so walking the nodes in
        # decreasing depth guarantees every child is resolved before its parent
        internal = [node for node in row_of if node not in node_adv]
        internal.sort(key=lambda node: -int(traj_step[row_of[node]]))

        subtree_leaves = {node: 1 for node in leaves}

        ess_ratios = []
        degenerate_weights = 0
        for node in internal:
            child_nodes = children[node]
            subtree_leaves[node] = sum(subtree_leaves[child] for child in child_nodes)
            weights = np.array(
                [1.0 / _gt_prob(info_gain_sum[row_of[child]], prob_diff_mode, prob_floor) for child in child_nodes],
                dtype=np.float64,
            )
            if weight_temp != 1.0:
                weights = weights ** weight_temp
            if subtree_size_weight:
                # a child standing in for many leaves speaks for all of them, as it
                # would under IGRPO's path unrolling
                weights = weights * np.array([subtree_leaves[child] for child in child_nodes], dtype=np.float64)
            if max_weight_ratio > 0:
                # truncated importance sampling, caps a single child's influence
                weights = np.minimum(weights, max_weight_ratio * weights.mean())
            total = weights.sum()
            if not np.isfinite(total) or total <= 0.0:
                weights = np.ones_like(weights)
                total = weights.sum()
                degenerate_weights += 1
            weights = weights / total

            advantage = None
            for weight, child in zip(weights, child_nodes):
                term = node_adv[child] * float(weight)
                advantage = term if advantage is None else advantage + term
            node_adv[node] = advantage
            ess_ratios.append(1.0 / (float(np.sum(weights ** 2)) * len(child_nodes)))

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

        # Q backup: a leaf keeps its outcome reward, a parent averages its children.
        # Deepest first, so every child is resolved before its parent is read.
        q = {}
        for node in sorted(row_of, key=lambda n: -int(traj_step[row_of[n]])):
            child_nodes = children[node]
            if child_nodes:
                q[node] = sum(q[child] for child in child_nodes) / len(child_nodes)
            else:
                q[node] = float(scores[row_of[node]])

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
        # post-blend, so the two are directly comparable as contributions to the loss
        "treehca/grpo_adv/abs_mean": grpo_term.abs().mean().item(),
        "treehca/aux_adv/abs_mean": aux_term.abs().mean().item(),
        "treehca/aux_adv/mean": aux_mean,
        # directional push relative to the term's own spread, keep it under ~0.05
        "treehca/aux_adv/bias_ratio": abs(aux_mean) / aux_std if aux_std > 0 else 0.0,
    }

    return advantages, advantages, metrics
