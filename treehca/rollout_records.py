"""Fields added to saved TreeHCA training rollouts."""

import math


def build_treehca_rollout_fields(non_tensor_batch):
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

    return {
        "node_path": paths,
        "avg_ans_log_probs": ["-Infinity" if float(value) == -math.inf else float(value) for value in non_tensor_batch["avg_ans_log_probs"]],
        "info_gain": [float(value) for value in non_tensor_batch["info_gain"]],
        "info_gain_sum": [float(value) for value in non_tensor_batch["info_gain_sum"]],
        "page_type": list(non_tensor_batch["page_type"]),
    }
