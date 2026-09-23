"""Reconstruct the two Q-hindsight advantage terms from trusted debug batches.

Measures actual weighted contributions, verifies their sum against saved tensors,
and separates token-weighted magnitudes from deduplicated action diagnostics.
Does not run models or alter training.
"""
import argparse
from collections import defaultdict
import csv
import gc
import json
from pathlib import Path
import re

import numpy as np
import torch

from scripts.audit_saved_search_trajectories import CPUUnpickler, normalize
from agent_system.environments.env_package.search.projection import search_projection
from treehca.core_treehca import _gt_prob, compute_treehca_q_outcome_advantage
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def analyze(batch, model, step):
    nt, bt = batch.non_tensor_batch, batch.batch
    mask = bt["response_mask"]
    grpo, _ = compute_grpo_outcome_advantage(
        bt["token_level_rewards"], mask, nt["uid"], nt["uid"],
        norm_adv_by_std_in_grpo=True, compute_mean_std_cross_steps=True,
    )
    scores = bt["token_level_rewards"].sum(-1)
    rows = {}
    for i, node in enumerate(nt["node_uid"]):
        rows.setdefault(str(node), i)
    children = defaultdict(list)
    for node, i in rows.items():
        parent = str(nt["parent_node_uid"][i])
        if parent in rows:
            children[parent].append(node)
    q, factor, parent_prob, prob = {}, {}, {}, {}
    for node in sorted(rows, key=lambda n: -int(nt["traj_step"][rows[n]])):
        q[node] = (sum(q[c] for c in children[node]) / len(children[node])
                   if children[node] else float(scores[rows[node]]))
    for node, i in rows.items():
        parent = str(nt["parent_node_uid"][i])
        prob[node] = _gt_prob(nt["info_gain_sum"][i], True, 1e-6)
        parent_prob[node] = _gt_prob(nt["info_gain_sum"][rows[parent]], True, 1e-6) if parent in rows else 1e-6
        factor[node] = 1.0 - min(parent_prob[node] / prob[node], 2.0)
    aux = torch.tensor([q[str(n)] * factor[str(n)] for n in nt["node_uid"]], dtype=scores.dtype)
    g = 0.7 * grpo
    h = 0.3 * aux.unsqueeze(-1) * mask
    total = g + h
    max_error = float((total - bt["advantages"]).abs().max())
    assert max_error < 1e-6, (model, step, max_error)
    if step == 200:
        production, _, _ = compute_treehca_q_outcome_advantage(
            bt["token_level_rewards"], mask, nt["uid"], nt["node_uid"],
            nt["parent_node_uid"], nt["is_terminal"], nt["info_gain_sum"], nt["traj_step"],
        )
        assert torch.allclose(total, production, atol=1e-7, rtol=1e-6)
    active_index = mask.to(torch.int64).argmax(-1).unsqueeze(-1)
    g_scalar = g.gather(1, active_index).squeeze(-1).numpy()
    h_scalar = h.gather(1, active_index).squeeze(-1).numpy()
    total_scalar = total.gather(1, active_index).squeeze(-1).numpy()
    num_tokens = mask.sum(-1).numpy()
    actions = search_projection([str(a) for a in nt["text_actions"]])[0]
    query_by_node = {}
    for node, i in rows.items():
        matches = re.findall(r"<search>(.*?)</search>", actions[i], re.S)
        query_by_node[node] = matches[0].strip() if matches else None
    records = []
    for node, i in rows.items():
        parent = str(nt["parent_node_uid"][i])
        query = query_by_node[node]
        ancestor = parent
        ancestor_queries = []
        while ancestor in rows:
            old_query = query_by_node[ancestor]
            if old_query:
                ancestor_queries.append(old_query)
            ancestor = str(nt["parent_node_uid"][rows[ancestor]])
        raw = "\n".join(str(m["content"]) for m in nt["raw_prompt"][i])
        question = re.search(r"Your question:\s*(.*?)\n", raw).group(1).strip()
        record = {
            "model": model, "training_step": step, "node_id": node,
            "question": question, "source": str(nt["data_source"][i]),
            "turn": int(nt["traj_step"][i]) + 1,
            "query": query, "action": actions[i],
            "root": parent not in rows, "pruned": bool(nt["deactivate"][i]),
            "terminal": bool(nt["is_terminal"][i]), "is_search": query is not None,
            "executed_search": query is not None and int(nt["traj_step"][i]) < 3,
            "repeated_query": bool(query and normalize(query) in {normalize(x) for x in ancestor_queries}),
            "literal_repeated_query": bool(query and query in ancestor_queries),
            "valid_action": bool(nt["is_action_valid"][i]),
            "reward_score": float(scores[i]), "Q": q[node],
            "p_parent": parent_prob[node], "p_after": prob[node],
            "h_ratio": prob[node] / parent_prob[node], "hindsight_factor": factor[node],
            "grpo_unweighted": float(g_scalar[i]) / 0.7,
            "hindsight_unweighted": float(aux[i]),
            "grpo_weighted": float(g_scalar[i]), "hindsight_weighted": float(h_scalar[i]),
            "total": float(total_scalar[i]), "response_tokens": int(num_tokens[i]),
        }
        records.append(record)

    def group_stats(selected):
        if not selected:
            return {"nodes": 0}
        gg = np.array([r["grpo_weighted"] for r in selected])
        hh = np.array([r["hindsight_weighted"] for r in selected])
        tt = np.array([r["total"] for r in selected])
        conf = (gg > 1e-8) & (hh < -1e-8)
        stats = {"nodes": len(selected),
                 "grpo_abs_mean": float(np.abs(gg).mean()),
                 "hindsight_abs_mean": float(np.abs(hh).mean()),
                 "grpo_mean": float(gg.mean()), "hindsight_mean": float(hh.mean()),
                 "hindsight_positive": int((hh > 1e-8).sum()),
                 "hindsight_negative": int((hh < -1e-8).sum()),
                 "hindsight_zero": int((np.abs(hh) <= 1e-8).sum()),
                 "total_positive": int((tt > 1e-8).sum()), "total_negative": int((tt < -1e-8).sum()),
                 "positive_grpo_negative_hindsight": int(conf.sum()),
                 "positive_grpo_flipped_negative": int((conf & (tt < -1e-8)).sum()),
                 "negative_hindsight_total_positive": int(((hh < -1e-8) & (tt > 1e-8)).sum()),
                 "q_zero": sum(abs(r["Q"]) <= 1e-8 for r in selected),
                 "q_negative": sum(r["Q"] < -1e-8 for r in selected)}
        for label, arr in [("grpo", gg), ("hindsight", hh)]:
            stats[label + "_abs_quantiles"] = dict(zip(["p50", "p90", "p99", "max"], np.quantile(np.abs(arr), [.5, .9, .99, 1]).tolist()))
        return stats

    groups = {
        "all": records,
        "root": [r for r in records if r["root"]],
        "nonroot": [r for r in records if not r["root"]],
        "executed_search": [r for r in records if r["executed_search"]],
        "nonroot_executed_search": [r for r in records if not r["root"] and r["executed_search"]],
        "repeated_executed_search": [r for r in records if r["executed_search"] and r["repeated_query"]],
        "literal_repeated_executed_search": [r for r in records if r["executed_search"] and r["literal_repeated_query"]],
        "decreased_p": [r for r in records if not r["root"] and r["h_ratio"] < 1],
        "decreased_p_executed_search": [r for r in records if not r["root"] and r["h_ratio"] < 1 and r["executed_search"]],
        "halved_p_executed_search": [r for r in records if not r["root"] and r["h_ratio"] <= .5 and r["executed_search"]],
        "unchanged_p": [r for r in records if not r["root"] and abs(r["h_ratio"] - 1) < .01],
        "pruned": [r for r in records if r["pruned"]],
        "completed_failure": [r for r in records if r["terminal"] and not r["pruned"] and r["reward_score"] <= 0],
        "final_unexecuted_search": [r for r in records if r["is_search"] and not r["executed_search"]],
    }
    stats = {name: group_stats(rr) for name, rr in groups.items()}
    mass_g, mass_h = float(g.abs().sum()), float(h.abs().sum())
    active_tokens = int(mask.sum())
    summary = {
        "model": model, "training_step": step, "rows": len(batch),
        "max_saved_advantage_error": max_error, "active_response_tokens": active_tokens,
        "weighted_grpo_abs_token_mean": mass_g / active_tokens,
        "weighted_hindsight_abs_token_mean": mass_h / active_tokens,
        "hindsight_to_grpo_abs_ratio": mass_h / mass_g,
        "weighted_grpo_abs_padded_mean": float(g.abs().mean()),
        "weighted_hindsight_abs_padded_mean": float(h.abs().mean()),
        "root_share_hindsight_absolute_mass": sum(abs(float(h_scalar[i])) * float(num_tokens[i]) for i,n in enumerate(nt["node_uid"]) if str(nt["parent_node_uid"][i]) not in rows) / mass_h,
        "groups": stats,
    }
    for key, value in [("treehca/grpo_adv/abs_mean", summary["weighted_grpo_abs_padded_mean"]), ("treehca/aux_adv/abs_mean", summary["weighted_hindsight_abs_padded_mean"])]:
        assert abs(value - batch.meta_info["treehca_metrics"][key]) < 1e-7
    return records, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 100, 150, 200])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in ["treehca-qh-infoval-7B", "treehca-qh-logratio-7B"]:
        for step in args.steps:
            paths = list((args.root / model / "debug_batches").rglob(f"batch_debug_after_adv_{step}.pkl"))
            assert len(paths) == 1, paths
            print(f"Loading {model} {step}", flush=True)
            with paths[0].open("rb") as f:
                batch = CPUUnpickler(f).load()
            records, summary = analyze(batch, model, step)
            with (args.out / f"{model}-step{step}-advantages.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            summaries.append(summary)
            (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
            print(json.dumps({k: v for k,v in summary.items() if k != "groups"}), flush=True)
            print(json.dumps({k:summary['groups'][k] for k in ['decreased_p_executed_search','repeated_executed_search']}), flush=True)
            del batch, records
            gc.collect()


if __name__ == "__main__":
    main()
