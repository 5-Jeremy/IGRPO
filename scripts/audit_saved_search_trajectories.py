"""Extract inspectable, deduplicated search trees from trusted local debug pickles.

These are sampled TRAINING rollouts, not held-out evaluation trajectories.
Pruned leaves are reported separately from completed environment episodes.
"""
import argparse
from collections import Counter
import gc
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import string

import numpy as np
import torch
from agent_system.environments.env_package.search.projection import search_projection


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def normalize(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def extract(batch, model, step):
    nt = batch.non_tensor_batch
    nodes = {}
    for i, node_id in enumerate(nt["node_uid"]):
        node_id = str(node_id)
        if node_id in nodes:
            continue
        raw = nt["raw_prompt"][i]
        prompt = "\n".join(str(m["content"]) for m in raw)
        match = re.search(r"Your question:\s*(.*?)\n", prompt)
        question = match.group(1).strip() if match else prompt.splitlines()[0]
        response = str(nt["text_actions"][i])
        targets = nt["ground_truth"][i]["target"]
        targets = [targets] if isinstance(targets, str) else list(targets)
        projected_action = search_projection([response])[0][0]
        answers = re.findall(r"<answer>(.*?)</answer>", projected_action, re.S)
        pred = answers[-1].strip() if answers else None
        prompt_width = batch.batch["prompts"].shape[-1]
        mask = batch.batch["attention_mask"][i]
        response_mask = batch.batch["response_mask"][i]
        adv = batch.batch["advantages"][i]
        source = str(nt["data_source"][i])
        nodes[node_id] = {
            "node_id": node_id, "parent_id": str(nt["parent_node_uid"][i]),
            "question": question, "source": source, "targets": targets,
            "question_id": hashlib.sha256((source + "\n" + question).encode()).hexdigest()[:20],
            "turn": int(nt["traj_step"][i]), "response": response,
            "queries": [q.strip() for q in re.findall(r"<search>(.*?)</search>", projected_action, re.S)],
            "prediction": pred,
            "answer_exact_match": pred is not None and normalize(pred) in {normalize(t) for t in targets},
            "terminal": bool(nt["is_terminal"][i]), "pruned": bool(nt["deactivate"][i]),
            "tool_calls": int(nt["current_tool_callings"][i]),
            "reward": float(nt["rewards"][i]), "gold_probability": float(nt["info_gain_sum"][i]),
            "info_gain": float(nt["info_gain"][i]),
            "advantage": float((adv * response_mask).sum() / response_mask.sum().clamp_min(1)),
            "prompt_tokens": int(mask[:prompt_width].sum()),
            "response_tokens": int(mask[prompt_width:].sum()),
            "prompt": prompt,
        }
    trajectories = []
    for node in nodes.values():
        if not node["terminal"]:
            continue
        chain = [node]
        while chain[-1]["parent_id"] in nodes:
            chain.append(nodes[chain[-1]["parent_id"]])
        chain.reverse()
        queries = [q for n in chain for q in n["queries"]]
        normalized_queries = [normalize(q) for q in queries]
        evidence_blocks = []
        for block in re.findall(r"<information>(.*?)</information>", node["prompt"], re.S):
            try:
                block = json.loads(block)["result"]
            except (ValueError, KeyError, TypeError):
                pass
            evidence_blocks.append(block)
        evidence = "\n".join(evidence_blocks)
        evidence_norm = " " + normalize(evidence) + " "
        alias_present = any(" " + normalize(t) + " " in evidence_norm for t in node["targets"] if normalize(t))
        trajectories.append({
            "model": model, "training_step": step, "question_id": node["question_id"],
            "question": node["question"], "source": node["source"], "targets": node["targets"],
            "leaf_id": node["node_id"], "pruned": node["pruned"],
            "complete_ancestry": chain[0]["parent_id"] == "root", "turns": node["turn"] + 1,
            "queries": queries, "repeated_query": len(set(normalized_queries)) < len(normalized_queries),
            "tool_calls": node["tool_calls"],
            "executed_repeated_query": len(set(normalized_queries[:node["tool_calls"]])) < len(normalized_queries[:node["tool_calls"]]),
            "prediction": node["prediction"], "exact_match": node["reward"] >= 1.0,
            "prediction_matches_target": node["answer_exact_match"],
            "gold_alias_in_final_evidence": alias_present,
            "no_answer_at_limit": node["turn"] == 3 and node["prediction"] is None,
            "last_prompt": node["prompt"],
            "steps": [{k: n[k] for k in ["node_id", "turn", "response", "queries", "prediction", "gold_probability", "info_gain", "advantage", "prompt_tokens", "response_tokens"]} for n in chain],
        })
    completed = [t for t in trajectories if not t["pruned"]]
    failed = [t for t in completed if not t["exact_match"]]
    summary = {
        "model": model, "training_step": step, "rows": len(batch), "unique_nodes": len(nodes),
        "questions": len({n["question_id"] for n in nodes.values()}),
        "question_sources": dict(Counter({n["question_id"]: n["source"] for n in nodes.values()}.values())),
        "leaves": len(trajectories), "pruned_leaves": sum(t["pruned"] for t in trajectories),
        "completed_leaves": len(completed), "complete_ancestry": sum(t["complete_ancestry"] for t in trajectories),
        "completed_exact_match": sum(t["exact_match"] for t in completed),
        "completed_repeated_query": sum(t["repeated_query"] for t in completed),
        "completed_executed_repeated_query": sum(t["executed_repeated_query"] for t in completed),
        "completed_no_answer_at_limit": sum(t["no_answer_at_limit"] for t in completed),
        "completed_mean_queries": float(np.mean([len(t["queries"]) for t in completed])),
        "completed_mean_tool_calls": float(np.mean([t["tool_calls"] for t in completed])),
        "reward_prediction_disagreements": sum(t["exact_match"] != t["prediction_matches_target"] for t in completed),
        "completed_turn_histogram": dict(Counter(t["turns"] for t in completed)),
        "failed_gold_alias_in_evidence": sum(t["gold_alias_in_final_evidence"] for t in failed),
        "failed_leaves": len(failed),
        "nodes_at_prompt_limit": sum(n["prompt_tokens"] == 4096 for n in nodes.values()),
        "nodes_at_response_limit": sum(n["response_tokens"] == 512 for n in nodes.values()),
        "metrics": batch.meta_info.get("treehca_metrics", {}),
    }
    return list(nodes.values()), trajectories, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=[200])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in args.models:
        for step in args.steps:
            paths = list((args.root / model / "debug_batches").rglob(f"batch_debug_after_adv_{step}.pkl"))
            if len(paths) != 1:
                print(f"SKIP {model} step {step}: {len(paths)} files", flush=True)
                continue
            print(f"Loading {paths[0]}", flush=True)
            with paths[0].open("rb") as f:
                batch = CPUUnpickler(f).load()
            nodes, trajectories, summary = extract(batch, model, step)
            summary["input_path"] = str(paths[0])
            for kind, records in [("nodes", nodes), ("trajectories", trajectories)]:
                with (args.out / f"{model}-step{step}-{kind}.jsonl").open("w") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            summaries.append(summary)
            (args.out / "training_summary.json").write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summary), flush=True)
            del batch, nodes, trajectories
            gc.collect()


if __name__ == "__main__":
    main()
