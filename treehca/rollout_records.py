"""Fields added to saved TreeHCA training rollouts."""

import math

import numpy as np


def capture_branch_logging(*, next_obs, infos, dones, is_last_step, expand_prob, expand_num, branch_score, gamma):
    """Snapshot diagnostics before frontier slots are reused; do not change decisions."""
    reasons = []
    for i, info in enumerate(infos):
        if dones[i]:
            reason = "success" if info.get("won") else "environment_failure" if "won" in info or info.get("error") else "environment_terminal"
        elif is_last_step:
            reason = "turn_limit"
        elif expand_num[i] == 0:
            reason = "pruned"
        else:
            reason = None
        reasons.append(reason)
    observations = next_obs.get("anchor")
    if observations is None:
        observations = next_obs.get("text")
    return {
        "termination_reason": np.asarray(reasons, dtype=object),
        "environment_done": np.asarray(dones, dtype=bool).copy(),
        "expansion_probability": np.asarray(expand_prob).copy(),
        # The final step still samples allocations, but never executes them.
        "sampled_expansion_count": np.asarray(expand_num).copy(),
        "expansion_count": np.zeros_like(expand_num) if is_last_step else np.asarray(expand_num).copy(),
        "branch_score": np.asarray(branch_score).copy(),
        "branch_logit": np.asarray(branch_score).copy() * gamma,
        "post_action_observation": np.asarray(list(observations) if observations is not None else [None] * len(infos), dtype=object),
        "webshop_task_id": np.asarray([info.get("webshop_task_id") for info in infos], dtype=object),
        "webshop_session_id": np.asarray([info.get("webshop_session_id") for info in infos], dtype=object),
    }


def _json_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def build_treehca_rollout_fields(non_tensor_batch, tensor_batch=None):
    """Return JSON-ready columns in the same order as the rollout batch."""
    node_uids = non_tensor_batch["node_uid"]
    parent_uids = non_tensor_batch["parent_node_uid"]
    parents = dict(zip(node_uids, parent_uids))

    paths = []
    for node_uid in node_uids:
        path = [node_uid]
        while path[-1] != "root":
            parent = parents[path[-1]]
            if parent in path:
                raise ValueError(f"Cycle in TreeHCA rollout tree at {parent}")
            path.append(parent)
        paths.append(path)

    fields = {
        "node_path": paths,
        "avg_ans_log_probs": ["-Infinity" if float(value) == -math.inf else float(value) for value in non_tensor_batch["avg_ans_log_probs"]],
        "info_gain": [float(value) for value in non_tensor_batch["info_gain"]],
        "info_gain_sum": [float(value) for value in non_tensor_batch["info_gain_sum"]],
        "page_type": list(non_tensor_batch["page_type"]),
    }
    columns = (
        "uid",
        "node_uid",
        "parent_node_uid",
        "traj_step",
        "is_terminal",
        "deactivate",
        "termination_reason",
        "environment_done",
        "expansion_probability",
        "expansion_count",
        "sampled_expansion_count",
        "branch_score",
        "branch_logit",
        "post_action_observation",
        "webshop_task_id",
        "webshop_session_id",
        "is_action_valid",
        "rewards",
    )
    for name in columns:
        if name in non_tensor_batch:
            fields[name] = [_json_value(value) for value in non_tensor_batch[name]]
    # Values need a critic; TreeHCA normally has no value tensor. Preserve that absence.
    for source, summary in (("values", "value"), ("advantages", "advantage")):
        fields[source] = [None] * len(node_uids)
        fields[summary] = [None] * len(node_uids)
        if tensor_batch is not None and source in tensor_batch:
            mask = tensor_batch["response_mask"].detach().cpu().bool()
            tensor = tensor_batch[source].detach().cpu()
            for i in range(len(node_uids)):
                tokens = tensor[i][mask[i]]
                fields[source][i] = _json_value(tokens.tolist())
                fields[summary][i] = _json_value(tokens.float().mean().item()) if tokens.numel() else None
    return fields
