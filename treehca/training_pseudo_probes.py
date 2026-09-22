"""Teacher-forced WebShop probes using the training actor's current weights."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from treehca.pseudo_rollout_product_page import ActionChoiceScore, ProductOptionGroupPseudoRollout, ProductPagePseudoRolloutScores
from treehca.pseudo_rollout_results_page import ResultsPageAnswerProbe
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.utils.model import compute_position_id_with_mask


@dataclass(frozen=True)
class TrainingProductOptionProbe:
    """One group of exact option responses, including the synthetic none choice."""

    actions: tuple[str, ...]
    option_names: tuple[str, ...]
    answer_probes: tuple[ResultsPageAnswerProbe, ...]


class TrainingPseudoProbeScorer:
    """Use teacher-forced answer-token logprobs for training probes.

    Each label variant becomes a one-token teacher-forcing row. This requires
    more forward passes than a vocabulary-logit API but no new worker RPC.
    """

    def __init__(self, tokenizer, actor_rollout_wg, *, max_model_len, batch_size=32):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Probe batch_size must be a positive integer")
        self.tokenizer = tokenizer
        self.actor = actor_rollout_wg
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.forward_pass_time_seconds = 0.0

    def score(self, probes):
        requests, lookup, owners = [], {}, []

        def request(prompt, response, indices):
            key = (tuple(prompt), tuple(response), tuple(indices))
            if not prompt or not response:
                raise ValueError("Empty TreeHCA pseudo probe")
            if not self.request_fits(prompt, response):
                raise ValueError(f"TreeHCA pseudo probe is overflowing: requires {len(prompt) + len(response)} tokens, exceeding max_model_len={self.max_model_len}")
            if key not in lookup:
                lookup[key] = len(requests)
                requests.append(key)
            return lookup[key]

        for probe in probes:
            if isinstance(probe, ResultsPageAnswerProbe):
                owners.append([request(probe.prompt_token_ids, probe.response_token_ids, probe.answer_token_indices)])
            elif isinstance(probe, ProductOptionGroupPseudoRollout):
                prefix = tuple(self.tokenizer.encode(f"<think>The best choice for the {probe.option_group.name} group corresponds to the label:", add_special_tokens=False))
                owners.append([request(probe.prompt_token_ids + prefix, (token,), (0,)) for token in probe.allowed_token_ids])
            elif isinstance(probe, TrainingProductOptionProbe):
                owners.append([request(answer.prompt_token_ids, answer.response_token_ids, answer.answer_token_indices) for answer in probe.answer_probes])
            else:
                raise TypeError(f"Unsupported pseudo probe: {type(probe).__name__}")

        values = []
        for start in range(0, len(requests), self.batch_size):
            rows = requests[start : start + self.batch_size]
            # Avoid an artificial context overflow from mixing unequal lengths.
            width = max(len(prompt) + len(response) for prompt, response, _ in rows)
            response_width = max(len(response) for _, response, _ in rows)
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
            if pad is None:
                raise ValueError("Pseudo scoring requires a pad or EOS token")
            ids = torch.full((len(rows), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for index, (prompt, response, _) in enumerate(rows):
                tokens = prompt + response
                ids[index, -len(tokens) :] = torch.tensor(tokens)
                mask[index, -len(tokens) :] = 1
            data = DataProto.from_dict(
                tensors={"input_ids": ids, "responses": ids[:, -response_width:], "attention_mask": mask, "position_ids": compute_position_id_with_mask(mask)},
                meta_info={"treehca_probe_temperature": 1.0},
            )
            data, _ = pad_dataproto_to_divisor(data, self.actor.world_size)
            forward_start = time.perf_counter()
            try:
                output = self.actor.compute_log_prob(data)
            finally:
                self.forward_pass_time_seconds += time.perf_counter() - forward_start
            if output.meta_info.get("temperature") != 1.0:
                raise ValueError("Training worker must honor treehca_probe_temperature=1.0")
            logs = output.batch["old_log_probs"]
            if logs.shape != (len(data), response_width):
                raise ValueError("Actor returned misaligned pseudo-probe log probabilities")
            for index, (_, response, positions) in enumerate(rows):
                selected = logs[index, [response_width - len(response) + pos for pos in positions]].double()
                if torch.isnan(selected).any() or (selected > 0).any():
                    raise ValueError("Probe log probabilities must be nonpositive and not NaN")
                values.append(float(selected.sum()))

        scores = []
        for probe, indices in zip(probes, owners):
            if isinstance(probe, ResultsPageAnswerProbe):
                scores.append(math.exp(values[indices[0]]))
                continue
            if isinstance(probe, TrainingProductOptionProbe):
                logs = torch.tensor([values[index] for index in indices], dtype=torch.float64)
                choices = tuple(
                    ActionChoiceScore(name, action, math.exp(float(logp)), float(logp), float(logp), -math.inf)
                    for name, action, logp in zip(probe.option_names, probe.actions, logs)
                )
                scores.append(ProductPagePseudoRolloutScores(choices))
                continue
            logs = torch.tensor([values[index] for index in indices], dtype=torch.float64).reshape(-1, 2)
            normalizer = torch.logsumexp(logs.flatten(), dim=0)
            if not torch.isfinite(normalizer):
                raise ValueError("No finite mass for product labels")
            action_logs = torch.logsumexp(logs, dim=1) - normalizer
            choices = tuple(ActionChoiceScore(label, action, math.exp(float(logp)), float(logp), float(pair[0] - normalizer), float(pair[1] - normalizer)) for label, action, logp, pair in zip(probe.labels, probe.actions, action_logs, logs))
            scores.append(ProductPagePseudoRolloutScores(choices))
        return scores

    def request_fits(self, prompt, response) -> bool:
        """Return whether one nonempty teacher-forcing row fits the actor context."""
        return bool(prompt) and bool(response) and len(prompt) + len(response) <= self.max_model_len
