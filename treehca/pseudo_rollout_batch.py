"""Batch results-action and product-group probes in the same vLLM calls."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from treehca.pseudo_rollout_product_page import ProductOptionGroupPseudoRollout, ProductPagePseudoRolloutScores, _resolve_max_model_len, _scores_from_vllm_output
from treehca.pseudo_rollout_results_page import ResultsPageAnswerProbe

PseudoProbe = ResultsPageAnswerProbe | ProductOptionGroupPseudoRollout
ProbeScore = float | ProductPagePseudoRolloutScores


class MixedPseudoRolloutScorer:
    """Score up to 32 unique pseudo prompts per engine call, without padding.

    Product rows always use the product outcome testbed's default fixed cue.
    Results rows retain raw joint answer-token probabilities. All rows must
    fit the model context; missing/overflowed rows are errors, never zeros.
    """

    def __init__(self, inference_engine: Any, tokenizer: Any, *, max_model_len: int | None = None, batch_size: int = 32, lora_request: Any = None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be between 1 and 32 pseudo prompts")
        if isinstance(lora_request, (list, tuple)):
            raise ValueError("Use one policy/LoRA adapter per scorer")
        self.engine = inference_engine
        self.tokenizer = tokenizer
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        self.lora_request = lora_request
        self._prefix_tokens: dict[str, tuple[int, ...]] = {}

    def _product_prefix(self, group_name: str) -> tuple[int, ...]:
        if group_name not in self._prefix_tokens:
            text = f"<think>The best choice for the {group_name} group corresponds to the label:"
            self._prefix_tokens[group_name] = tuple(self.tokenizer.encode(text, add_special_tokens=False))
        return self._prefix_tokens[group_name]

    def score(self, probes: Sequence[PseudoProbe]) -> list[ProbeScore]:
        if not probes:
            return []
        from vllm import SamplingParams

        llm_engine = getattr(self.engine, "llm_engine", None)
        if type(llm_engine).__module__.startswith("vllm.v1."):
            raise ValueError("Constrained label scoring requires vLLM V0 (VLLM_USE_V1=0)")
        if any(isinstance(probe, ResultsPageAnswerProbe) for probe in probes) and getattr(getattr(llm_engine, "cache_config", None), "enable_prefix_caching", False):
            raise ValueError("Disable vLLM V0 prefix caching when scoring results prompt logprobs")
        limit = _resolve_max_model_len(self.engine, self.max_model_len)
        configured_max_logprobs = getattr(getattr(llm_engine, "model_config", None), "max_logprobs", None)
        unique, indices, prompts, params = [], [], [], []
        seen = {}
        for probe in probes:
            if isinstance(probe, ResultsPageAnswerProbe):
                tokens = probe.input_token_ids
                key = ("results", tokens, len(probe.prompt_token_ids), probe.answer_token_indices)
                sampling = dict(temperature=0.0, prompt_logprobs=0)
            elif isinstance(probe, ProductOptionGroupPseudoRollout):
                tokens = probe.prompt_token_ids + self._product_prefix(probe.option_group.name)
                key = ("product", tokens, probe.allowed_token_ids, probe.actions)
                needed = len(probe.allowed_token_ids)
                if isinstance(configured_max_logprobs, int) and configured_max_logprobs < needed:
                    raise ValueError(f"vLLM max_logprobs must be at least {needed}")
                sampling = dict(temperature=1.0, allowed_token_ids=list(probe.allowed_token_ids), logprobs=needed)
            else:
                raise TypeError(f"Unsupported pseudo probe {type(probe).__name__}")
            if len(tokens) + 1 > limit:
                raise ValueError(f"Pseudo prompt requires {len(tokens) + 1} tokens, exceeding max_model_len={limit}")
            if key not in seen:
                seen[key] = len(unique)
                unique.append(probe)
                prompts.append({"prompt_token_ids": list(tokens)})
                params.append(SamplingParams(n=1, max_tokens=1, top_p=1.0, top_k=-1, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, detokenize=False, **sampling))
            indices.append(seen[key])

        scores = []
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            kwargs = {} if self.lora_request is None else {"lora_request": self.lora_request}
            outputs = self.engine.generate(prompts=prompts[start : start + len(batch)], sampling_params=params[start : start + len(batch)], use_tqdm=False, **kwargs)
            if len(outputs) != len(batch):
                raise ValueError("vLLM returned a misaligned pseudo-rollout batch")
            for probe, output in zip(batch, outputs):
                scores.append(_decode_results_probability(probe, output) if isinstance(probe, ResultsPageAnswerProbe) else _scores_from_vllm_output(probe, output))
        return [scores[index] for index in indices]


def _decode_results_probability(probe: ResultsPageAnswerProbe, output: Any) -> float:
    logprobs = getattr(output, "prompt_logprobs", None)
    if not isinstance(logprobs, list) or len(logprobs) != len(probe.input_token_ids):
        raise ValueError("vLLM must return a logprob entry for every results probe token")
    values = []
    for index in probe.answer_token_indices:
        position = len(probe.prompt_token_ids) + index
        token = probe.response_token_ids[index]
        entry = logprobs[position]
        if entry is None or token not in entry:
            raise ValueError(f"Missing results action logprob at token position {position}")
        value = float(getattr(entry[token], "logprob", entry[token]))
        if math.isnan(value) or value > 0:
            raise ValueError("Results action log probabilities must be nonpositive")
        values.append(value)
    return math.exp(math.fsum(values))
