# WebShop prompt terminology

This document records the agreed terminology for adapting TreeHCA probability
scoring to IGRPO. Prompt instructions quoted below describe the input data; they
are not instructions to the reader or to a coding assistant.

## Context boundaries

**Raw context** means the **prompt body**: the contents of the WebShop user
message, before application of the tokenizer's chat template. It excludes the
chat wrapper and the current generated response.

| Term | Meaning |
| --- | --- |
| Full model input | The raw context after application of the tokenizer's chat template, including the assistant generation prefix. |
| Chat wrapper | Role delimiters, any system message, and the assistant generation prefix supplied by the chat template. |
| Raw context / prompt body | The user-message contents containing the WebShop instructions, shopping task, history when present, current observation, and admissible actions. |
| Agent introduction | The opening sentence identifying the agent and the WebShop environment. |
| Shopping task | The requested product characteristics and constraints following `Your task is to:`. |
| History block | Recent observation/action pairs together with the introductory completed-step and history-length counts. This block is optional. |
| History entry | An `Observation i` and the `Action i` taken from that observation. |
| Current observation | The current page's quoted, ` [SEP] `-separated text, excluding the surrounding prompt sentence. |
| Admissible-action block | The explicit list of currently available commands and its introductory sentence. |
| Response instructions | The final instructions requesting reasoning inside `<think>` tags and an action inside `<action>` tags. |
| Current response | The assistant's generated response for this step, separate from the raw context. |

The current-step sentence introduces the current observation. Its step number is
distinct from the number of completed steps and the number of retained history
entries.

## Generic raw-context templates

Braces denote variable contents. The placeholders for response instructions and
repeated entries abbreviate the real text; they are not literal prompt syntax.
Whitespace in these examples is illustrative. The authoritative templates are in
[`webshop.py`](../../agent_system/environments/prompts/webshop.py).

### With history

```text
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {shopping_task}.
Prior to this step, you have already taken {completed_steps} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: [Observation {i}: '{past_observation_i}', Action {i}: '{past_action_i}']
{additional history entries, one per line}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
'{available_action_1}',
'{available_action_2}',
{additional available actions}
].

Now it's your turn to take one action for the current step.
{reasoning and action-format instructions}
```

Each history entry contains the observation before its paired action. The
observation already contains quoted page fragments, so enclosing it in the
history entry's quotes produces text such as:

```text
[Observation 10: ''Back to Search' [SEP] '< Prev' [SEP] ... [SEP] 'Buy Now'', Action 10: 'click[0.7mm]']
```

### Without history

```text
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {shopping_task}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
'{available_action_1}',
'{available_action_2}',
{additional available actions}
].

Now it's your turn to take one action for the current step.
{reasoning and action-format instructions}
```

The no-history template omits the step counts as well as history entries.
[`build_text_obs`](../../agent_system/environments/env_manager.py) uses it for
initial observations, when history is disabled, and as a fallback when the
history-bearing prompt exceeds 13,000 characters. Absence of history therefore
does not establish that this is the first step.

Some pseudo-rollouts will need to exclude history. The extraction design must
make the optional history block separately accessible; the policy for retaining
or excluding it belongs to pseudo-rollout construction.

## Product-page observation anatomy

For the current local main product-page template, the observation has this order:

```text
{navigation controls}
[SEP] {option groups, if present}
[SEP] {product title}
[SEP] {price field}
[SEP] {rating field}
[SEP] {detail-page controls}
[SEP] {purchase control}
```

The line breaks above are explanatory. The observation is normally one string
whose fragments are individually enclosed in single quotes and joined by
` [SEP] `.

| Part | Examples / structure |
| --- | --- |
| Navigation controls | `'Back to Search' [SEP] '< Prev'` |
| Option group | A group name followed by its available values; zero or more groups. |
| Option-group name | `size`, `color` |
| Option value | `0.7mm`, `christmasgoo3302` |
| Product title | The displayed product name. |
| Price field | `'Price: $11.89'` |
| Rating field | `'Rating: N.A.'` |
| Detail-page controls | `'Description' [SEP] 'Features'`; `'Attributes'` is conditional in the local template. |
| Purchase control | `'Buy Now'` |

Group names and values share the same separator as other page fragments; there
is no separate structural delimiter for option groups in this text format.

The admissible-action block is a separate representation of available commands.
For example, the observation displays `Buy Now`, whereas its command is
`click[buy now]`. Option values produce commands such as `click[0.7mm]`.

## Observed examples

Examples below are from
[`1.jsonl`](../../rollouts/gigpo_qwen2.5_1.5b_no_kl/20260911T192303.459664528-3509967/1.jsonl).
Line numbers refer to JSONL records, not environment steps.

| JSONL line | Product-page example |
| --- | --- |
| 175 | Interdental brushes; one `size` group with values `0.6mm`, `0.7mm`, `0.8mm`, `1.0mm`, and `1.2mm`; price `$11.89`. |
| 10 | Kitchen mats; `size` and `color` option groups; price `$15.9`. |
| 156 | Snacks; no option groups; price `$43.99`. |

The brush example's current observation, with only its long title abbreviated:

```text
'Back to Search' [SEP] '< Prev' [SEP] 'size' [SEP] '0.6mm' [SEP] '0.7mm' [SEP] '0.8mm' [SEP] '1.0mm' [SEP] '1.2mm' [SEP] 'meyarn Interdental Brush ... Tight 0.8mm' [SEP] 'Price: $11.89' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Buy Now'
```

The JSONL `input` includes a decoded chat wrapper. Training rollout logging uses
`skip_special_tokens=True`, so its visible `system`, `user`, and `assistant`
lines do not expose the complete token-level wrapper. Only the user-message body
is the raw context under this document's definition. The JSONL `output` contains
the separate current response.

## Environment properties and open decisions

- Selected options are not explicitly marked in the plain WebShop text
  observation. This is a property of the environment's representation. In the
  brush example, history contains two `click[0.7mm]` actions while the current
  observation still lists all sizes identically and its title still says
  `0.8mm`.
- These main product-page observations do not expose a product ID. An ID may
  appear in a historical search result and its corresponding click action.
- History can contain product pages even when the current page is of another
  type. A context-wide search for `Buy Now` is insufficient for classification.
- Page-type identification remains undecided. We will need either information
  from the underlying environment or a reliable heuristic over the current
  context. This document does not establish a classifier or decide whether
  product detail subpages belong in the same category as main product pages.
- The user approved separate functions for outer-context extraction and
  product-field parsing. Their contracts and parsing limitations are recorded
  in [product_page_parser_assumptions.md](product_page_parser_assumptions.md).
  New usage assumptions still require the user's confirmation.

## Source references

- [Prompt templates](../../agent_system/environments/prompts/webshop.py)
- [Observation and prompt formatting](../../agent_system/environments/env_manager.py)
- [History formatting](../../agent_system/memory/memory.py)
- [Chat-template application](../../agent_system/multi_turn_rollout/rollout_loop.py)
- [Product-page HTML template](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/templates/item_page.html)
- [HTML-to-text conversion and available actions](../../agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py)
- [Training rollout logging](../../verl/trainer/ppo/ray_trainer.py)
