"""Shared diagnostics and fields added to saved training rollouts."""

import math

import numpy as np


def capture_step_logging(*, next_obs, infos, dones, is_last_step):
    """Snapshot diagnostics before frontier slots are reused; do not change decisions."""
    reasons = []
    for i, info in enumerate(infos):
        if dones[i]:
            reason = "success" if info.get("won") else "environment_failure" if "won" in info or info.get("error") else "environment_terminal"
        elif is_last_step:
            reason = "turn_limit"
        else:
            reason = None
        reasons.append(reason)
    observations = next_obs.get("anchor")
    if observations is None:
        observations = next_obs.get("text")
    return {
        "termination_reason": np.asarray(reasons, dtype=object),
        "environment_done": np.asarray(dones, dtype=bool).copy(),
        "post_action_observation": np.asarray(list(observations) if observations is not None else [None] * len(infos), dtype=object),
        "webshop_task_id": np.asarray([info.get("webshop_task_id") for info in infos], dtype=object),
        "webshop_session_id": np.asarray([info.get("webshop_session_id") for info in infos], dtype=object),
        # Preserve WebShop's native score so the invalid-response adjustment can
        # distinguish full success from partial purchases on environment terminals.
        "webshop_task_score": np.asarray([info.get("task_score") for info in infos], dtype=object),
    }


def capture_branch_logging(*, next_obs, infos, dones, is_last_step, expand_prob, expand_num, branch_score, gamma):
    """Snapshot shared diagnostics and branch decisions before frontier reuse."""
    fields = capture_step_logging(next_obs=next_obs, infos=infos, dones=dones, is_last_step=is_last_step)
    for i, reason in enumerate(fields["termination_reason"]):
        if reason is None and expand_num[i] == 0:
            fields["termination_reason"][i] = "pruned"
    fields.update({
        "expansion_probability": np.asarray(expand_prob).copy(),
        # The final step still samples allocations, but never executes them.
        "sampled_expansion_count": np.asarray(expand_num).copy(),
        "expansion_count": np.zeros_like(expand_num) if is_last_step else np.asarray(expand_num).copy(),
        "branch_score": np.asarray(branch_score).copy(),
        "branch_logit": np.asarray(branch_score).copy() * gamma,
    })
    return fields


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


def build_rollout_fields(non_tensor_batch, tensor_batch=None):
    """Return JSON-ready columns in the same order as the rollout batch."""
    non_tensor_batch = dict(non_tensor_batch)
    if "node_uid" not in non_tensor_batch:
        # GiGPO has independent chains within each group, with no virtual root.
        trajectories = non_tensor_batch["traj_uid"]
        steps = non_tensor_batch["traj_step"]
        non_tensor_batch["node_uid"] = [f"{traj}:{int(step)}" for traj, step in zip(trajectories, steps)]
        non_tensor_batch["parent_node_uid"] = [f"{traj}:{int(step) - 1}" if step else None for traj, step in zip(trajectories, steps)]
        paths = [[f"{traj}:{i}" for i in range(int(step), -1, -1)] for traj, step in zip(trajectories, steps)]
    else:
        parents = dict(zip(non_tensor_batch["node_uid"], non_tensor_batch["parent_node_uid"]))
        paths = []
        for node_uid in non_tensor_batch["node_uid"]:
            path = [node_uid]
            while path[-1] != "root":
                parent = parents[path[-1]]
                if parent is None:
                    break
                if parent in path:
                    raise ValueError(f"Cycle in rollout tree at {parent}")
                path.append(parent)
            paths.append(path)

    node_uids = non_tensor_batch["node_uid"]
    fields = {
        "node_path": paths,
    }
    for name in ("avg_ans_log_probs", "info_gain", "info_gain_sum", "page_type"):
        fields[name] = [_json_value(value) for value in non_tensor_batch.get(name, [None] * len(node_uids))]
    columns = (
        "uid",
        "traj_uid",
        "episode_rewards",
        "episode_lengths",
        "tool_callings",
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
        "webshop_task_score",
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


# Preserve the existing public entry point.
build_treehca_rollout_fields = build_rollout_fields
