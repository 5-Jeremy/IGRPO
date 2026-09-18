# Product-page parser assumptions and contracts

The user approved these interfaces and their usage contracts before
implementation. The implementation is
[`product_page_parser.py`](../../treehca/product_page_parser.py); vocabulary and
observed examples are in [prompt_terminology.md](prompt_terminology.md).

These functions only extract data. They do not construct pseudo-rollouts, call a
model or environment, or change rollout behavior.

The separate [pseudo-rollout prompt builder](pseudo_rollout_prompts.md) consumes
the parsed context components to create action-choice prompts. That document
explains the tokenizer-checked letter labels and the intended treatment of
space-prefixed answers.

## Raw-context extraction

```python
extract_product_page_contexts(contexts: Sequence[str]) -> list[ProductPageContextParts]
```

- Each input is a **raw context**, meaning the WebShop user-message body. Chat
  wrappers and current generated responses are outside this contract.
- The caller identifies product pages. Extraction does not classify pages or
  invoke the product-field parser. The page-classification system remains a
  separate, unresolved decision.
- Supported bodies follow the current `WEBSHOP_TEMPLATE` or
  `WEBSHOP_TEMPLATE_NO_HIS` in
  [`webshop.py`](../../agent_system/environments/prompts/webshop.py). The agent
  introduction also accepts the ASCII hyphen used in the terminology examples.
- Each result corresponds to the input at the same index. An empty input batch
  produces an empty list. The functions do not mutate their inputs.
- Unexpected or detectably ambiguous formatting raises `ValueError` identifying
  the zero-based batch index. No partial batch is returned. A single string is
  rejected where a batch is required.
- Template markers must identify unique section boundaries. Repeated markers
  in data that conflict with those boundaries are rejected rather than guessed.
- `history_block` contains its complete introductory sentence and history
  entries. Entries remain opaque text; their contents are not parsed or counted.
- Without history, `history_block`, `completed_steps`, `history_length`, and
  `current_step` are all `None`. In particular, an absent step count is not
  inferred to be zero or one.
- With history, integer counts come from the template. The current step must be
  the completed-step count plus one, and history length must not exceed completed
  steps, matching the prompt builder. This does not verify the content of the
  history entries.
- History is exposed independently so later pseudo-rollout construction can
  omit it. No history-removal policy is implemented here.

The result fields are `agent_introduction`, `shopping_task`, `history_block`,
`completed_steps`, `history_length`, `current_step`, `current_observation`,
`admissible_actions`, and `response_instructions`.

## Product-field extraction

```python
parse_product_page_fields(
    current_observation: str,
    admissible_actions: Sequence[str],
) -> ProductPageFields
```

- Input is the isolated current observation, without the template's surrounding
  sentence, together with the actions belonging to that same page.
- Supported pages use the current local main
  [`item_page.html`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/templates/item_page.html)
  layout. Description/Features subpages, search pages, and purchase-result pages
  are outside this function's contract.
- The expected fragment sequence is `Back to Search`, `< Prev`, zero or more
  option groups, optional title, `Price: ...`, `Rating: ...`, `Description`, `Features`,
  optional `Attributes`, and `Buy Now`.
- The fixed trailing sequence anchors title, price, and rating. A price-like
  option value therefore does not automatically become a price-field boundary.
- Price and rating remain strings after removal of their labels. No numeric
  conversion is performed; for example, `N.A.` remains `N.A.`.
- Options retain their names, distinct clickable values, and display order.
  Repeated values within one group collapse to one choice, matching WebShop's
  deduplicated click commands. No options yields an empty tuple. Selected values
  are never inferred.
- Duplicate commands, missing control commands, unexplained actions, empty
  required fields, and unsupported or detectably ambiguous structures raise
  `ValueError`.

`ProductPageFields` contains `navigation_controls`, `option_groups`, `title`,
`price_text`, `rating_text`, `detail_page_controls`, and `purchase_control`.
Each `ProductOptionGroup` contains `name`, `values`, and `correct_options`.
Context-only parsing leaves `correct_options` empty because goal metadata is
not present in the rendered prompt; grouped pseudo-rollout construction fills
it later. Both result types, as well as `ProductPageContextParts`, are frozen
dataclasses; sequence fields are tuples.

The functions can be composed explicitly:

```python
from treehca.product_page_parser import (
    extract_product_page_contexts,
    parse_product_page_fields,
)

parts = extract_product_page_contexts(raw_contexts)
products = [
    parse_product_page_fields(part.current_observation, part.admissible_actions)
    for part in parts
]
```

## Batch error reporting

```python
parse_product_page_batch(contexts: Sequence[str]) -> list[ProductPageParseResult]
```

This additional, user-approved wrapper runs both parsing stages independently
for each input row. The existing strict functions retain their raising behavior.
All of the input-format and page-type assumptions above still apply.

The wrapper returns one frozen `ProductPageParseResult` per input, including
failed rows, in the original input order. `index` is the zero-based position in
the batch passed to this call, not a product ID or a position in an earlier
unfiltered batch. An empty batch returns an empty list.

| Outcome | `context_parts` | `product_fields` | `error_stage` | `error_message` |
| --- | --- | --- | --- | --- |
| Both stages succeed | Extracted context | Parsed product | `None` | `None` |
| Context extraction fails | `None` | `None` | `"context"` | Parser error text |
| Product parsing fails | Extracted context | `None` | `"product"` | Parser error text |

Each stage captures its parsing `ValueError` and continues to the next input.
The wrapper preserves successfully extracted context parts when product
parsing fails. It does not return partial product fields or infer missing data.
Callers should use `error_stage` for success/failure handling; the diagnostic
message is readable text rather than a stable machine-readable error code.

Example of obtaining failed indices and retaining the aligned results:

```python
from treehca.product_page_parser import parse_product_page_batch

results = parse_product_page_batch(raw_contexts)
failed_indices = [
    result.index for result in results if result.error_stage is not None
]
failures = [results[index] for index in failed_indices]
successful_results = [result for result in results if result.error_stage is None]
```

A malformed individual row, including a non-string row, is a context failure.
A single string or bytes object passed instead of a batch still raises
`ValueError` immediately. Other exceptions propagate: programming errors,
resource failures, and invalid non-iterable batch arguments are not silently
converted into parsing failures. This wrapper handles recognized bad inputs;
it does not guarantee that arbitrary training failures cannot occur.

### Training integration remains undecided

No trainer call sites, score assignment, or fallback policy are added here.
The wrapper does not assign `gt_prob_mc = 0`, remove rollout rows, or decide
whether a failed row should be omitted from pseudo-rollout scoring. Training
must explicitly consume the failure information and apply an agreed policy.
Using the strict functions directly will still raise on these same inputs.

The catalog test can detect the known data anomalies before training, while
this wrapper can report recognized failures in runtime batches. No automatic
catalog preflight has been added to training.

## Option-group heuristic

The approved heuristic uses the page's admissible actions to separate group
names from clickable values. Within the option section, a non-clickable group
name must be followed by one or more clickable values. Group names are not
restricted to a hardcoded list such as `size` or `color`.

The environment's
[`get_available_actions`](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py)
lowercases page-control commands and preserves option values in their commands.
Consequently, page controls are checked through their lowercase command names,
while an option value requires the exact command `click[{value}]`.

The parser rejects repeated group names, values shared across recognized groups,
group names colliding with commands, and options colliding with page controls.
Repeated values within one group produce one choice. Every supplied action must
correspond to a parsed option value or expected page control. An empty catalog
title produces an empty `title` field when no title fragment is rendered.

This remains a heuristic over an unescaped text representation. It cannot prove
that the caller supplied the correct action list. For example, removing a
value's action could make that value look like another group name. A group name
that also appears as a prior group's clickable value cannot be distinguished
from a duplicate value using prompt text alone. Rejection of detectable
ambiguities is not a guarantee that every corrupted input is detectable.
Environment metadata would be needed to eliminate this limitation.

## Text preservation

- Observation fragments use the literal separator ` [SEP] ` and one enclosing
  single-quote pair. The parser removes exactly that pair, leaving internal
  apostrophes, quotation marks, case, Unicode, and backslashes untouched.
- Admissible actions in raw contexts use one quoted command per line, followed
  by a comma. These display strings are not Python or JSON literals and are
  never evaluated or escape-decoded. Embedded apostrophes remain literal.
- Supported commands use `click[...]` or `search[...]` syntax. A main product
  page's field parser will reject search commands because they do not match its
  controls or options. Commands containing newlines are unsupported by the
  line-oriented action representation.
- Shopping-task and current-observation extraction removes the one period
  added by the prompt template, preserving any punctuation belonging to the
  original content.
- Boundary newlines and template-only whitespace around the introduction are
  excluded from extracted sections. History contents and response-instruction
  text are otherwise retained; response instructions include the opening
  `Now it's your turn...` sentence. Their wording is not semantically parsed.
- The source format does not escape separators or structural markers inside
  page text. The parser supports normal template output with unambiguous
  separators, not arbitrary lossless serialization of all possible strings.

## Validation

Tests and saved-example fixtures live under
[`tests/treehca`](../../tests/treehca). Run in the repository's `webshop` conda
environment:

```bash
conda run --no-capture-output -n webshop python -m pytest tests/treehca
```

Tests cover both prompt variants, independent history extraction, batch order
and error indexing, real saved product pages, zero/one/multiple option groups,
optional Attributes, text preservation, and malformed or ambiguous inputs.
Wrapper tests cover mixed successes and failures at both stages, preserved
context data, failed indices, empty and entirely invalid batches, and propagation
of unexpected exceptions.

The additional [catalog test](product_page_catalog_test.md) renders all 1,000
products through WebShop's actual page and prompt formatting methods in one
test case. Its independent field checks and handling of three catalog
anomalies (two duplicate-value products and one untitled product) are explained
in that document.
