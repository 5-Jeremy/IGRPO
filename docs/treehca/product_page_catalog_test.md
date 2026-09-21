# Testing every product in the 1,000-item catalog

The catalog-wide test case
[`test_all_1000_product_pages`](../../tests/treehca/test_product_page_catalog.py)
uses every row in
[`items_shuffle_1000.json`](../../agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json).
It is one pytest case with internal loops, not a parametrized family of cases.
The same file also contains a focused synthetic-catalog regression for option
normalization before rendering.

Run it from the IGRPO root:

```bash
conda run --no-capture-output -n webshop python -m pytest tests/treehca/test_product_page_catalog.py -q
```

## Constructing the inputs

1. Read the source JSON and assert that it has 1,000 rows. Call WebShop's real
   `load_products` and assert that its ordered ASIN list exactly matches the
   source list. No source item may silently disappear through loader filtering
   or deduplication.
2. For every loaded product, call the real `map_action_to_html` function, which
   renders the local `item_page.html` template. Use an empty selected-options
   dictionary and `show_attrs=False`, matching the default main product-page
   configuration. The dataset's original `is_selected` fields are not supplied
   as environment selections.
3. Call the real text environment's plain-text observation conversion and
   `get_available_actions` on that HTML. Set the environment manager's retained
   task to the known rendering input, and call its real `format_obs` method to
   obtain the prompt-ready observation. In real rollouts the task is extracted
   once from the initial search page; `extract_task` is not a product-page
   method.
4. Use the real `build_text_obs` method to build a batch without history and a
   batch with history. The latter uses `SimpleMemory` containing one controlled
   observation/action pair for each product: observation `'Search'` and action
   `search[ASIN]`. This is a formatting fixture, not a simulated navigation
   trajectory. A distinct controlled shopping task identifies each product.
5. Pass all 2,000 raw contexts, in a known order, to
   `extract_product_page_contexts`. Pass each extracted observation/action list
   to `parse_product_page_fields`.
6. Pass the same contexts to `parse_product_page_batch`. Check that every
   original index and extracted context is preserved and all products match
   the same structured expectations. This remains part of the
   same single test case and does not render any additional pages.

The test uses the installed `webshop` environment and imports the real local
WebShop modules. It creates a Flask request context only to render URLs in the
HTML template; it does not start a web server or fetch those URLs. Constructors
are bypassed for the text environment and environment manager because only
their formatting methods are needed. No search index, browser process, rollout,
or model inference is initialized. Importing the existing WebShop modules still
loads their dependencies, including Java and spaCy. The loader's unrelated
reward-price sampling has its global Python RNG state restored after loading.

## Independent correctness checks

The expected product fields are built directly from the structured product
dictionary returned by WebShop's loader, before parsing any observation. They
are not recovered from the text by another copy of the parser.

| Parsed field | Expected value |
| --- | --- |
| `title` | `product['Title']` |
| `price_text` | `product['Price']` |
| `rating_text` | `product['Rating']` |
| `option_groups` | Ordered names and distinct clickable values from `product['options']` |
| `navigation_controls` | `('Back to Search', '< Prev')` |
| `detail_page_controls` | `('Description', 'Features')` |
| `purchase_control` | `'Buy Now'` |

The loader is the appropriate source because these are the fields sent to the
renderer. For example, it lowercases option names and values, replaces `/` in
values with ` | `, formats price strings, substitutes `$100.0` for missing
pricing, and currently supplies `N.A.` ratings. The check targets parser
correctness relative to what WebShop displays; it does not independently test
the loader's normalization logic or reward calculations.

Every outer-context field is also checked:

- The task must equal the known task supplied before rendering. Distinct ASINs
  expose incorrect batch ordering.
- The current observation must equal the observation supplied to the prompt
  builder, byte for byte.
- The action sequence must equal the expected controls followed by the loaded
  option values. Identical commands are deduplicated in insertion order,
  matching WebShop's dictionary of clickables.
- Without history, history and all step counts must be `None`.
- With history, the complete history block must equal the independently
  constructed fixture text, and counts must be `(completed=1, history=1,
  current=2)`. Thus unexpected omission of history also fails this test.
- The introduction and response instructions must equal the corresponding
  static portions of the actual prompt template, with the documented boundary
  whitespace removed.

On a mismatch, diagnostics identify the ASIN, prompt variant, and field, with
expected and actual values. Field mismatches and unexpected product-parser
rejections are accumulated so the final failure reports all affected rows.

## Three catalog anomalies handled by the parser

Inspection of the structured catalog identifies two products with repeated
options:

| ASIN | Repeated normalized size | Source of repetition |
| --- | --- | --- |
| `B08G14B779` | `26 inch x 10 inch` | Raw values `26 inch x 10 inch` and `26 Inch x 10 Inch` become equal after lowercasing. |
| `B099WH1RTM` | `12x18 inch` | Raw values `12x18 Inch` and `12x18 inch` become equal after lowercasing. |

The third product, `B09P71WY8C`, has `name: ""` in the source JSON and an empty
`Title` after loading. Its rendered page has no title fragment. The parser
returns an empty title. The test independently computes missing titles from
the structured data and asserts that this is the only affected ASIN.

The test computes repeated fragments from the structured data and asserts that
the complete repetition map is exactly the table above. It checks that the two
duplicate-value products expose one parsed choice per distinct click command
and the untitled product exposes an empty title. All three pages are rendered
and their outer contexts are fully checked.

The final totals must be:

- 1,000 products rendered.
- 2,000 raw contexts extracted and checked.
- 2,000 product parses checked against the complete expected fields.

The batch wrapper is checked against those same totals: 2,000 aligned,
successful results. The focused parser tests separately exercise batches
containing context-stage failures.

The focused normalization regression separately checks leading and trailing
slashes and values repeated across option groups. Cross-group duplicates retain
only their last group occurrence, matching WebShop's value-keyed DOM lookup;
groups made empty by that rule are omitted. It then renders the normalized
product and verifies that observation fragments, actions, and parsed groups
agree.

This validates the parser's treatment of the three catalog anomalies. Distinct
display entries with the same click command count as one actionable choice.

## Coverage limits

This case covers every catalog product in the default main-page configuration.
It does not enumerate all selected-option combinations or enable Attributes.
Selection inference remains outside the parser contract, and the existing
focused tests cover the optional Attributes control. It does not classify pages
or test search results, detail subpages, or purchase-result pages.
