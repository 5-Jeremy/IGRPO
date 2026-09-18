# Action-choice pseudo-rollout prompts and scoring

[`pseudo_rollout_product_page.py`](../../treehca/pseudo_rollout_product_page.py) builds a raw prompt body
from the components returned by the [product-page parser](product_page_parser_assumptions.md).
It also prepares tokenized batches and uses one constrained vLLM generation step
to compute a forced-choice distribution over the admissible actions.

## Interface and prompt changes

```python
build_product_page_pseudo_rollout_prompt(
    parts: ProductPageContextParts,
    tokenizer,
) -> tuple[str, dict[str, str]]

build_product_page_grouped_choice_prompt(
    parts: ProductPageContextParts,
    tokenizer,
    goal_options: Mapping[str, str],
    *,
    label_catalog=None,
) -> tuple[ProductOptionGroupChoicePrompt, ...]

prepare_product_page_grouped_choice_rollouts(
    parts,
    tokenizer,
    goal_options_by_page,
    *,
    label_catalog=None,
) -> tuple[ProductOptionGroupPseudoRollout, ...]

select_action_labels(tokenizer, num_actions: int) -> tuple[str, ...]

build_action_label_catalog(tokenizer, num_actions: int) -> ActionLabelCatalog

prepare_product_page_pseudo_rollouts(
    parts,
    tokenizer,
    *,
    label_catalog=None,
) -> tuple[ProductPagePseudoRollout, ...]

required_max_logprobs(pseudo_rollouts) -> int

score_product_page_pseudo_rollouts(
    inference_engine,
    pseudo_rollouts,
    *,
    max_model_len=None,
    lora_request=None,
) -> list[ProductPagePseudoRolloutScores | None]

score_product_page_grouped_choice_rollouts(
    inference_engine,
    pseudo_rollouts,
    *,
    max_model_len=None,
    lora_request=None,
    assistant_response_prefix_token_ids=None,
) -> list[ProductOptionGroupPseudoRolloutScores | None]
```

The tokenizer is supplied by the caller. Label selection requires Hugging
Face-style `encode` and `decode` methods; batch preparation additionally uses
`apply_chat_template` and batched tokenization. The module does not load a
tokenizer, download files, or hardcode vocabulary IDs. The implementation lives
separately from the parser, leaving its strict and batch error-reporting
interfaces unchanged.

The builder returns `(prompt, label_to_action)`. The dictionary maps every
displayed label to its exact action string, in prompt order, for example
`{"A": "click[back to search]", "B": "click[< prev]"}`. Keys are canonical
labels without leading whitespace. Bare and space-prefixed token spellings
represent the same choice; they do not create separate dictionary entries.
Skipped labels such as `AY` are absent from both the prompt and the mapping.

The builder preserves the agent introduction, shopping task, current
observation, history when present, and step information. It preserves the
original action order; it does not sort actions by their text, shuffle them,
filter them, or infer the correct action. The output follows the local WebShop
templates, including their surrounding newlines and punctuation. Whitespace
already discarded during parsing cannot be recovered from arbitrary inputs.

There are two substantive prompt changes:

1. Each quoted, comma-terminated action entry becomes a labeled entry:

   ```text
   Your admissible actions of the current situation are:
   [
   A: click[back to search]
   B: click[< prev]
   C: click[description]
   D: click[features]
   E: click[buy now]
   ...
   ].
   ```

2. The instructions after the retained turn-introduction sentence are replaced
   with the following text:

   ```text
   Now it's your turn to take one action for the current step.
   You must give the label corresponding to the action you want to take. You should think about what is the logically best next action to take, and finish your thought with "The best next action is:" followed by the label of your chosen action.
   ```

The wording uses **label** instead of **capital letter** because a label may
contain two uppercase letters. It does not forbid a leading space, since the
scoring method includes the space-prefixed token variant.

The returned prompt string is the raw user-message body. It does not include a chat
wrapper, assistant prefix, sampled response, or appended answer label. Applying
the scoring model's actual chat template belongs to
`prepare_product_page_pseudo_rollouts`.
History removal is also outside this builder's current contract.

### Group-specific option prompts

This section describes the standalone vLLM grouped-choice probes. TreeHCA
training uses exact option-name responses and answer-token log probabilities;
see [WebShop training scoring](training_webshop_scoring.md#inference).

`build_product_page_grouped_choice_prompt` builds one prompt for each parsed
product option group. Each prompt retains the task and observation but replaces
the admissible-action block with letter-labeled bare option values for only that
group. Its selection description is `Your available options for {group_name} are:`
(for example, `Your available options for size are:`). Choices such as `A: red`
do not use the environment's
`click[...]` wrapper. Its instructions name the group explicitly and request
one label. Every group appends a final labeled choice:

```text
C: none (do not select any option in this group)
```

The label depends on the number of catalog options. `none` means never clicking
an option in this group, not executing `click[none]`. The prompt explains this
explicitly. A real catalog value named `none` remains an ordinary clickable
option and is distinct from the final synthetic choice. Non-grouped pseudo
prompts are unchanged. A product page without option groups returns an empty tuple.

The caller supplies the active synthetic goal's `goal_options` mapping. For
each group, `ProductOptionGroup.correct_options` contains every displayed value
that matches the canonical goal value under WebShop's color normalization and
fuzzy token-set threshold. `ProductOptionGroupChoicePrompt.correct_labels`
provides the corresponding response labels in prompt order, including the
synthetic label when `none_is_correct` is true. A group not named in the goal is
allowed. Unmatched goals still raise `ValueError` if neither its displayed
choices nor omission can satisfy the option requirements.

Omission correctness asks whether **some** selection of at most one value from
each other group covers every native option-reward target. The code uses
WebShop's color normalization and fuzzy token-set threshold, with the same
`goal_options.items()` target representation as native synthetic rewards. A
bitmask dynamic program enforces one value per group; merely taking the union
of all available values would incorrectly permit selecting two values in one
group. No requested options means omission is correct for every group.
Cross-group fuzzy matches can also make a named group dispensable.

This is an option-reward feasibility test, assuming the product's other reward
components are satisfied. In the outcome testbed that assumption is verified
by the native oracle purchase. For general callers, prompt text alone does not
prove the product's attribute/type/price eligibility. If omission is feasible,
adding any option in the omitted group cannot reduce native option coverage,
so those displayed values are also marked correct. `none_is_correct` is stored
separately from `ProductOptionGroup.correct_options`; the parser's catalog
values are never modified.

These per-group correctness labels are existential, not a guarantee that every
combination of independently correct choices earns full reward. Two groups may
each be dispensable when the other is selected, while omitting both loses
credit. The outcome testbed therefore evaluates each joint combination with the
native reward function rather than relying on a product of correctness labels.

`prepare_product_page_grouped_choice_rollouts` accepts one goal mapping per
source page and flattens the page/group hierarchy in page order, then group
display order. Each `ProductOptionGroupPseudoRollout` records its
`source_page_index` and `option_group`, while otherwise satisfying the ordinary
`ProductPagePseudoRollout` contract. It can therefore be submitted directly to
`score_product_page_pseudo_rollouts`. Assistant-response prefixes, when used,
must align with this flattened group-rollout order.

The prepared rollout's real-option `actions` remain canonical WebShop
`click[option]` strings; the last action is the reserved pseudo sentinel `none`.
`ProductOptionGroupChoicePrompt.label_to_option` maps that last label to Python
`None`, which cannot collide with the literal catalog string `"none"`.
`include_none` and `none_is_correct` are retained in prepared metadata; scoring
propagates `none_is_correct` and adds the sentinel's probability to correct mass
when appropriate. Label capacity and required logprob counts include the extra
label's bare and leading-space tokens. The sentinel is never an environment
command. Real-option metadata keeps scores aligned with projected empirical
environment actions.

`score_product_page_grouped_choice_rollouts` is a thin group-aware wrapper over
the unchanged generic scorer. Its results retain the source-page and group
metadata and expose `correct_probability`, the sum of the probabilities of all
fuzzy-equivalent correct options, and `correct_log_probability`.

## Why these letter labels

We want an approximate distribution over every admissible action from the
model's next-token logits at one response position. The limiting condition is
**one token per output variant**, not one character per label. Inspecting more
vocabulary entries does not require additional autoregressive decoding steps.

Twenty-six single capital letters are insufficient for this catalog: 23
products have more than 26 distinct admissible actions, and the largest has 55.
For the currently configured `Qwen/Qwen2.5-1.5B-Instruct` tokenizer, many
two-letter strings are also single tokens. Integer labels do not provide the
same solution: labels such as `10` are split into two digit tokens.

The model may output a label either directly or with a leading space. We
therefore require **both** spellings to be single tokens:

```text
"A"    and " A"
"AA"   and " AA"
```

`select_action_labels` checks candidates in this order:

```text
A, B, ..., Z, AA, AB, ..., AZ, BA, ..., ZZ
```

For each candidate it checks that:

- Both spellings encode to exactly one token with special-token insertion off.
- Each token decodes exactly to its original spelling, without text cleanup.
- The two IDs are distinct and do not overlap IDs already selected for other
  labels. This avoids double-counting the same token under different actions.

It skips a candidate that fails any check and stops after finding the requested
number of labels. With the configured Qwen tokenizer, the first 55 labels are:

```text
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX
AZ BA BB BC BD
```

**AY is deliberately absent.** Although `AY` is one token, ` AY` splits into
` A` and `Y`. Skipping it and extending through `BD` gives every one of the 55
actions both a bare and a space-prefixed single-token representation: 110
distinct token IDs in total. The code discovers this through the tokenizer
checks; `AY` is not a hardcoded exception. A different tokenizer can select
different labels.

Only one- and two-letter candidates are considered, a finite pool of 702
strings. The number that actually qualifies depends on the tokenizer. Requests
for more usable labels than available raise `ValueError`; no actions are
silently dropped. Empty action lists and inconsistent history/current-step
presence also raise `ValueError` in the builder. These construction errors are
not caught by `parse_product_page_batch`, which only covers parsing. The
training fallback policy remains undecided.

## Implemented probability calculation

For action `i`, let `z[i, 0]` be the logit of its bare label token and `z[i, 1]`
the logit of its space-prefixed token. With selected raw logits, the equivalent
calculation is:

```python
variant_logits = next_token_logits[label_token_ids]  # [num_actions, 2]
action_probs = variant_logits.logsumexp(dim=-1).softmax(dim=-1)
```

This adds the probability mass of the two token spellings for each action,
then normalizes across all offered actions. It is a forced-choice estimate
under the rewritten prompt, conditioned on the next token being one of these
variants. It is not the probability of generating an entire original
`click[...]` response or of terminating immediately after the label.

The score does not include a standalone space token followed by a label, or
other multi-token whitespace sequences. Scoring those paths requires another
decoding step. Label and position preferences can also affect the estimate.

The implementation obtains this distribution through vLLM V0's public API. For
each row it supplies all `2 * num_actions` token IDs as `allowed_token_ids`,
requests the same number of output `logprobs`, and performs exactly one
generation step. V0 computes the returned logprobs after its allowed-token
logits processor, so the mapping contains every requested variant and is
conditioned on that set. The two values for each action are combined with
`logaddexp`, followed by a final normalization over actions to remove
floating-point drift:

```python
action_logprob = logaddexp(bare_logprob, spaced_logprob)
action_probs = softmax(action_logprobs)
```

The generated token itself is ignored. The dedicated sampling parameters use
temperature 1, unrestricted top-p/top-k/min-p, and neutral penalties. This
preserves the model's relative logits and prevents the training rollout's
sampling configuration from changing the probe.

The engine must be initialized with `max_logprobs` at least as large as
`required_max_logprobs(prepared_batch)`. The current catalog maximum of 55
actions requires 110. In the IGRPO rollout configuration this can be supplied
through:

```yaml
engine_kwargs:
  vllm:
    max_logprobs: 110
```

Native logprobs deliberately replace the original custom logits processor and
IPC queue. This avoids code tied to a particular vLLM internal API. Native
vLLM performs a top-k operation to return the requested entries, so a custom
indexed-logit gather may be faster if profiling later shows that these probes
are a material bottleneck.

This method is not compatible with vLLM 0.8.5's V1 engine. V1 snapshots raw
full-vocabulary logprobs before applying `allowed_token_ids`; its returned map
therefore contains the sampled allowed token plus the globally most likely
tokens, not necessarily every allowed token. Requesting more top logprobs does
not provide a correctness guarantee. The scorer detects a V1 engine and raises
a targeted error. Initialize vLLM before import with:

```bash
export VLLM_USE_V1=0
```

The standalone testbed sets this automatically. Supporting V1 efficiently
would require reinstating an indexed-logit capture mechanism or changing vLLM;
forcing each variant in a separate request would be public-API compatible but
substantially more expensive.

### Optional assistant-response conditioning

`score_product_page_pseudo_rollouts` normally scores the first assistant token
immediately after the pseudo prompt, preserving its original behavior. Its
optional `assistant_response_prefix_token_ids` argument instead accepts one
token-ID sequence per pseudo rollout. Each sequence is appended after that
row's assistant-generation boundary before the constrained label token is
scored. This allows callers to preserve an already-sampled `<think>...</think>`
prefix while replacing the response's original action with a multiple-choice
label.

The caller must provide token IDs generated with the same tokenizer as the
prepared prompt. Passing token IDs, rather than decoded text, preserves the
actual sampled tokenization and avoids re-tokenizing every reasoning trace.
Length filtering includes the prefix. Empty prefixes are valid, and omitting
the argument entirely follows the original path.

Before scoring with a new model or chat template, check the actual tokenized
assistant-response boundary. The builder only returns a raw body, so it cannot
validate a chat wrapper that has not yet been supplied. Batch preparation now
performs this validation for the largest label set the first time a catalog is
used with a chat template. Reusing that catalog avoids repeating the check in
later batches. The tests also validate both spellings for all 55 labels at the
configured Qwen assistant boundary.

## Correspondence with the original TreeHCA implementation

The original reference is
`verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools_tree_offline_treehca_turn.py`
and its `multiple_choice` helper package in the separate TreeHCA checkout. The
new implementation keeps its overall pipeline but replaces most representation
and engine-specific details.

| Original component | New component | Relationship |
| --- | --- | --- |
| `prepare_mc_prompt` | `build_product_page_pseudo_rollout_prompt` and `build_action_label_catalog` | Both construct labeled choices. The original constructs answer/decoy choices once per QA tree; the new code labels the already-known admissible WebShop actions. There is consequently no correct-answer, decoy, or “None of the above” metadata. |
| `prepare_mc_scoring_batch` | `prepare_product_page_pseudo_rollouts` | Both produce model-ready token IDs and the token metadata needed for scoring. The original decodes an existing rollout, splits on `"\nuser"`, replaces its system prompt, and appends a pseudo-response. The new prompt is already complete, so it applies the tokenizer's chat template directly and validates the actual assistant boundary. |
| Repeated construction of `" A"` through `" E"` IDs | `ActionLabelCatalog` | The catalog generalizes the token metadata to up to 55 actions and retains both bare and space-prefixed variants. It is reusable across batches so the tokenizer checks need not be repeated for every product page. |
| `truncate_overlength_rollouts` | Length filtering inside `score_product_page_pseudo_rollouts` | Both keep result positions aligned and leave unscorable rows unset. The new code never sends a truncated prompt to vLLM; it checks that the intact prompt plus one output token fits and returns `None` otherwise. |
| `_capture_mc_choice_logits` | The generation portion of `score_product_page_pseudo_rollouts` | Both submit one-token, per-row constrained requests. The original passes capture IDs through `SamplingParams.extra_args` and receives raw selected logits through a custom processor and IPC queue. The new code requests native vLLM V0 output logprobs and reads them directly from each `RequestOutput`. |
| `CaptureLogitsProcessor` and `logit_capture_records.py` | No direct equivalent | Capture lifecycle, worker identity, stale-record draining, and record association disappear because vLLM returns each row's logprob mapping in request order. |
| `compute_mc_metrics` | `_scores_from_vllm_output` and the score dataclasses | Both normalize choice scores and expose entropy/probabilities. The original assumes one spaced token per choice and hardcodes `A` through `E`; the new code combines two variants per action and preserves arbitrary selected labels and original action strings. |
| `get_ground_truth_probs_mc` and `_get_ground_truth_probs_mc_batch` | `score_product_page_pseudo_rollouts` | This is the closest high-level correspondence: validate and filter a batch, perform model inference, calculate distributions, and restore original row alignment. vLLM's scheduler handles engine-level batching, so there is no additional Python chunk loop here. |
| `_compute_gt_probs_for_all_nodes` | Caller integration, not implemented in this module | The original traverses tree nodes and decides when to score them. This module is intentionally independent of the rollout-tree lifecycle; its caller chooses product-page rows and stores the returned action distributions. |

One semantic distinction is important: native returned logprobs are normalized
after the allowed-token constraint, whereas the original capture returned raw
selected logits. They produce the same forced-choice probabilities and entropy.
If absolute raw logits are needed for another metric, native logprobs are not a
drop-in substitute and an indexed-logit capture path would again be required.

## Efficiency choices

- Build one `ActionLabelCatalog` for the maximum supported action count when the
  rollout worker is initialized, then pass it to every preparation call.
- Prepare all eligible rows together. Label discovery occurs once rather than
  once per page, and a reused catalog caches assistant-boundary validation for
  the tokenizer's chat template across batches.
- The chat template is applied with `tokenize=True`, avoiding a rendered-text
  round trip during ordinary prompt preparation.
- Token IDs are stored as ordered `(bare, spaced)` pairs. Scoring combines them
  positionally rather than repeatedly searching a token/string list.
- A single `generate` call submits all scorable rows with row-specific sampling
  parameters. vLLM applies its own `max_num_seqs` and token-budget scheduling.
- Prefix caching is already enabled by IGRPO's vLLM rollout and can reuse exact
  prompt-prefix blocks where product-page prompts share them.

## Usage

```python
from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout_product_page import (
    build_action_label_catalog,
    prepare_product_page_pseudo_rollouts,
    required_max_logprobs,
    score_product_page_pseudo_rollouts,
)

parts = extract_product_page_contexts(raw_contexts)
label_catalog = build_action_label_catalog(
    tokenizer,
    max(len(part.admissible_actions) for part in parts),
)
prepared = prepare_product_page_pseudo_rollouts(
    parts,
    tokenizer,
    label_catalog=label_catalog,
)

# Configure the engine with at least this value before constructing it.
print(required_max_logprobs(prepared))

# The result list remains aligned with `prepared`; an overlength row is None.
scores = score_product_page_pseudo_rollouts(inference_engine, prepared)
first_action_distribution = scores[0].action_probabilities
```

The builder consumes context components, not `ProductPageFields`; it preserves
the current observation directly rather than reconstructing it from product
fields. Deciding whether product-field failures should prevent prompt
construction remains a caller policy. No such training policy is added here.

For an end-to-end empirical check of whether these probabilities agree with
actions sampled from ordinary WebShop prompts, see the
[pseudo-rollout probability testbed](pseudo_rollout_product_page_choices_testbed.md).
The corresponding experiment for per-group option probabilities is documented
in the [grouped-choice probability testbed](pseudo_rollout_product_page_grouped_choices_testbed.md).

## Validation

[`test_pseudo_rollout_product_page.py`](../../tests/treehca/test_pseudo_rollout_product_page.py) verifies:

- All 55 labels and their 110 distinct single-token variants, using the cached
  configured tokenizer, including the actual assistant-response boundary.
- Exact preservation of the original saved prompt bodies apart from the two
  requested replacements and surrounding newlines, for both history variants.
- Original action order, tokenizer-dependent candidate rejection, round-trip
  checks, and errors for insufficient labels or invalid components.
- Returned label-to-action mappings, including all 55 choices and the skipped
  `AY` label, agreeing with the prompt entries and original action order.
- Batch preparation, reuse of a label catalog, direct chat-template tokenization,
  and computation of the required vLLM `max_logprobs` value.
- Native-vLLM request construction, two-variant probability aggregation,
  overlength row alignment, and rejection of incomplete logprob output, using a
  CPU-only fake inference engine.

Tests run in the `webshop` conda environment and load the tokenizer with
`local_files_only=True`; they do not download artifacts or load model weights.
