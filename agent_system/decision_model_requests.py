"""Build decision-model questions about reasoning within each action trace."""

import json
import uuid
from pathlib import Path

import numpy as np

from agent_system.reasoning_steps import DEFAULT_THINKING_SPLIT_PHRASES, attach_reasoning_steps, format_reasoning_trace

DEFAULT_TEMPLATE = Path(__file__).parent / "decision_model_inputs/questions.json"


def decision_turn_ids(batch):
    """A shared IGRPO node has one judgment across all its training paths."""
    metadata = batch.non_tensor_batch
    return metadata["node_uid"] if "node_uid" in metadata else metadata["id"]


def attach_decision_model_probabilities(batch, probabilities):
    """Keep every reasoning step's probabilities attached to its stable turn ID."""
    values = []
    for turn_id in decision_turn_ids(batch):
        answers = probabilities.get(turn_id)
        values.append({qid: dict(probs) for qid, probs in answers.items()} if answers is not None else None)
    batch.non_tensor_batch["decision_model_probabilities"] = np.array(values, dtype=object)


class DecisionModelRequestBuilder:
    """Load editable templates once; build one request per generated action trace."""

    def __init__(self, template_path=None, model="kev-latest", selective_requests=False,
                 max_turns_per_trajectory=10):
        if (isinstance(max_turns_per_trajectory, bool) or not isinstance(max_turns_per_trajectory, int)
                or max_turns_per_trajectory < 0):
            raise ValueError("max_turns_per_trajectory must be a nonnegative integer")
        path = Path(template_path) if template_path else DEFAULT_TEMPLATE
        self.template = json.loads(path.read_text(encoding="utf-8"))
        self.background = (path.parent / self.template["background_info_file"]).read_text(encoding="utf-8").strip()
        self.model = model
        self.selective_requests = selective_requests
        self.max_turns_per_trajectory = max_turns_per_trajectory
        self._submitted_turn_counts = {}
        criteria = self.template["criteria"]
        if not isinstance(criteria, list) or not criteria or any(not isinstance(c, str) or not c for c in criteria):
            raise ValueError("Question criteria must be a nonempty list of strings")
        if len(set(criteria)) != len(criteria):
            raise ValueError("Question criteria must be unique")

    def build_request(self, steps, action):
        """Ask about every reasoning step in the turn, starting with step 1."""
        if not isinstance(steps, list) or not steps or any(not isinstance(step, str) or not step.strip() for step in steps):
            raise ValueError("A request requires nonempty reasoning steps")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("A request requires a chosen action")
        questions = {
            f"step_{number}": {
                "type": "choice",
                "instructions": self.template["instructions_template"].format(step_number=number, step_text=step),
                "criteria": dict.fromkeys(self.template["criteria"]),
            }
            for number, step in enumerate(steps, 1)
        }
        return {
            "model": self.model,
            "state": self.template["state_template"].format(
                background_info=self.background, reasoning_steps=format_reasoning_trace(steps, action),
            ),
            "questions": questions,
        }

    def _requests_for_rows(self, batch, active_masks, rows, counts):
        metadata = batch.non_tensor_batch
        ids = decision_turn_ids(batch)
        tree = "node_uid" in metadata
        seen = {}
        for row in rows:
            if not active_masks[row]:
                continue
            turn_id = ids[row]
            trace = metadata["reasoning_trace"][row]
            if turn_id in seen:
                if seen[turn_id] != trace:
                    raise ValueError(f"Conflicting reasoning traces for turn {turn_id}")
                continue
            seen[turn_id] = trace
            if self.selective_requests:
                if tree:
                    # Each branch inherits the number of requests on its prefix.
                    parent = metadata["parent_node_uid"][row]
                    counts[turn_id] = 0 if parent == "root" else counts[parent]
                    quota_id = turn_id
                else:
                    quota_id = metadata["traj_uid"][row]
                if counts.get(quota_id, 0) >= self.max_turns_per_trajectory:
                    continue
            if trace is None:
                continue
            request = self.build_request(trace["steps"], trace["action"])
            if self.selective_requests:
                counts[quota_id] = counts.get(quota_id, 0) + 1
            yield turn_id, request

    def build_rollout_requests(self, batch):
        """Rebuild requests before training padding, visiting parents first."""
        metadata = batch.non_tensor_batch
        ids = decision_turn_ids(batch)
        rows = np.argsort(metadata["traj_step"], kind="stable") if "traj_step" in metadata else range(len(ids))
        return dict(self._requests_for_rows(batch, metadata.get("active_masks", [True] * len(ids)), rows, {}))

    def submit_turns(self, batch, outputs, active_masks, tokenizer, pending,
                     thinking_split_phrases=DEFAULT_THINKING_SPLIT_PHRASES):
        """Enqueue active turns after the environment has reported action validity.

        The environment owns action validation; ReasoningStepFormatter extracts
        reasoning and the chosen action from tagged or JSON model output.
        Selective quotas persist across calls for each traj_uid. Create a new
        builder for each training batch, as the trainer does.
        """
        if "node_uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["id"] = np.array([str(uuid.uuid4()) for _ in outputs], dtype=object)
        attach_reasoning_steps(batch, tokenizer, thinking_split_phrases, outputs=outputs)
        for turn_id, request in self._requests_for_rows(
            batch, active_masks, range(len(outputs)), self._submitted_turn_counts
        ):
            pending.submit(turn_id, request)
