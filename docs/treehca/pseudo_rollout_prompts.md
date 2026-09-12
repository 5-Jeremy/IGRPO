# Action-choice pseudo-rollout prompts

[`pseudo_rollout.py`](../../treehca/pseudo_rollout.py) builds a raw prompt body
from the components returned by the [product-page parser](product_page_parser_assumptions.md).
It implements the agreed action-label probe; it does not call a model or compute
probabilities yet.

## Interface and prompt changes

```python
build_product_page_pseudo_rollout_prompt(
    parts: ProductPageContextParts,
    tokenizer,
) -> tuple[str, dict[str, str]]

select_action_labels(tokenizer, num_actions: int) -> tuple[str, ...]
```

The tokenizer is supplied by the caller and must provide Hugging Face-style
`encode` and `decode` methods. The builder does not load a tokenizer, download
files, or hardcode vocabulary IDs. The implementation lives separately from the
parser, leaving its strict and batch error-reporting interfaces unchanged.

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
   You must give the label corresponding to the action you want to take. You should only respond with a single label from the list
   ```

The wording uses **label** instead of **capital letter** because a label may
contain two uppercase letters. It does not forbid a leading space, since the
planned scoring method includes the space-prefixed token variant.

The returned prompt string is the raw user-message body. It does not include a chat
wrapper, assistant prefix, sampled response, or appended answer label. Applying
the scoring model's actual chat template belongs to the later scoring step.
History removal is also outside this builder's current contract.

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

## Intended probability calculation

For action `i`, let `z[i, 0]` be the logit of its bare label token and `z[i, 1]`
the logit of its space-prefixed token. The intended calculation is:

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

Before scoring with a new model or chat template, check the actual tokenized
assistant-response boundary. The builder only returns a raw body, so it cannot
validate a chat wrapper that has not yet been supplied. The tests validate both
spellings for all 55 labels at the configured Qwen assistant boundary.

## Usage

```python
from treehca.product_page_parser import extract_product_page_contexts
from treehca.pseudo_rollout import build_product_page_pseudo_rollout_prompt

parts = extract_product_page_contexts(raw_contexts)
prompt_results = [build_product_page_pseudo_rollout_prompt(part, tokenizer) for part in parts]
prompts = [prompt for prompt, _ in prompt_results]

# The mapping is returned with its corresponding prompt; no reselection is needed.
prompt, label_to_action = prompt_results[0]
labels = tuple(label_to_action)
label_token_ids = [
    [
        tokenizer.encode(label, add_special_tokens=False)[0],
        tokenizer.encode(" " + label, add_special_tokens=False)[0],
    ]
    for label in labels
]
```

The builder consumes context components, not `ProductPageFields`; it preserves
the current observation directly rather than reconstructing it from product
fields. Deciding whether product-field failures should prevent prompt
construction remains a caller policy. No such training policy is added here.

## Validation

[`test_pseudo_rollout.py`](../../tests/treehca/test_pseudo_rollout.py) verifies:

- All 55 labels and their 110 distinct single-token variants, using the cached
  configured tokenizer, including the actual assistant-response boundary.
- Exact preservation of the original saved prompt bodies apart from the two
  requested replacements and surrounding newlines, for both history variants.
- Original action order, tokenizer-dependent candidate rejection, round-trip
  checks, and errors for insufficient labels or invalid components.
- Returned label-to-action mappings, including all 55 choices and the skipped
  `AY` label, agreeing with the prompt entries and original action order.

Tests run in the `webshop` conda environment and load the tokenizer with
`local_files_only=True`; they do not download artifacts or load model weights.
