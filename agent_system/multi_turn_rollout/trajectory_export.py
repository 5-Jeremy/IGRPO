"""Optional, question-level export of search evaluation trajectories."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from agent_system.environments.env_package.search.projection import search_projection


def export_search_trajectories(batch, tokenizer, path, metadata):
    """Write one record per episode, retaining each actual model input and action."""
    nt = batch.non_tensor_batch
    groups = defaultdict(list)
    for i, uid in enumerate(nt["traj_uid"]):
        groups[str(uid)].append(i)
    prompt_width = batch.batch["prompts"].shape[-1]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for uid, indices in groups.items():
            indices.sort(key=lambda i: int(nt["traj_step"][i]))
            steps = []
            for i in indices:
                mask = batch.batch["attention_mask"][i]
                prompt_mask = mask[:prompt_width].bool()
                response_mask = mask[prompt_width:].bool()
                prompt = tokenizer.decode(batch.batch["prompts"][i][prompt_mask].tolist(), skip_special_tokens=True)
                response = str(nt["text_actions"][i])
                projected_action = search_projection([response])[0][0]
                steps.append({
                    "turn": int(nt["traj_step"][i]), "prompt": prompt,
                    "response": response,
                    "projected_action": projected_action,
                    "prompt_tokens": int(prompt_mask.sum()),
                    "response_tokens": int(response_mask.sum()),
                    "valid_action": bool(nt["is_action_valid"][i]),
                    "reward": float(nt["rewards"][i]),
                })
            first, last = indices[0], indices[-1]
            question = str(nt["anchor_obs"][first])
            source = str(nt["data_source"][first])
            targets = nt["ground_truth"][first]["target"]
            targets = [targets] if isinstance(targets, str) else list(targets)
            answers = re.findall(r"<answer>(.*?)</answer>", steps[-1]["projected_action"], re.S)
            record = {
                **metadata,
                "question_id": hashlib.sha256((source + "\n" + question).encode()).hexdigest()[:20],
                "trajectory_id": uid, "question": question, "source": source,
                "targets": targets, "prediction": answers[-1].strip() if answers else None,
                "episode_reward": float(nt["episode_rewards"][last]),
                "episode_length": int(nt["episode_lengths"][last]),
                "tool_calls": int(nt["tool_callings"][last]), "steps": steps,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(groups)
