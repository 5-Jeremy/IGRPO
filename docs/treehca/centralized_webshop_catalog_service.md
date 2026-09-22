# Centralized WebShop Catalog Service

Status: proposed design

## Summary

Replace the per-environment `SimServer` catalog with one centralized catalog
service. The service owns the parsed product catalog, Lucene searcher, goal
definitions, and seed-specific price/goal views. Environment workers retain only
small, mutable episode state and request catalog operations through a narrow RPC
interface.

The training implementation should be Ray-native because the trainer and
environment workers already run in Ray. A Flask adapter may expose the same
service for interactive browser use, but the training path must not use Flask,
Selenium, or HTML form navigation as its transport.

The intended ownership model is:

```text
One WebshopCatalogService actor
  - one parsed product catalog
  - one Lucene searcher
  - immutable catalog metadata
  - bounded caches for seed-specific prices and goals

Many lightweight WebShop environment workers
  - current goal reference
  - search terms and result page
  - current ASIN and selected options
  - current rendered observation and URL
  - episode RNG/counters
  - TreeHCA branch-local history
```

This removes the current multiplication of the full catalog by
`train_batch_size * env.rollout.n` while preserving independent episodes and
TreeHCA branch copying.

## Motivation

`WebAgentTextEnv` currently constructs a `SimServer` whenever no server object is
provided. Every `SimServer` calls `load_products`, constructs a product dictionary,
opens a Lucene searcher, constructs prices and goals, and keeps all of that state
for its lifetime. `WebshopMultiProcessEnv` creates one Ray actor per rollout slot,
so the full catalog is replicated in every actor.

For the current TreeHCA WebShop example:

- `data.train_batch_size = 16`
- `env.rollout.n = 8`
- 128 training environment actors are created
- `env.resources_per_worker.num_cpus = 0.1`, allowing all of them to initialize
  concurrently
- the full `items_shuffle.json` is approximately 5.48 GB before Python object
  expansion

The result is node-level memory exhaustion during actor construction. Merely
limiting concurrent construction does not solve the steady-state duplication.

## Goals

The design must:

1. Keep exactly one parsed copy of the product catalog per configured service
   instance.
2. Keep exactly one search-engine instance per service instance unless a measured
   concurrency requirement justifies a small, explicitly configured searcher pool.
3. Preserve the Gym-facing behavior expected by `WebshopWorker`, including text
   observations, available actions, rewards, terminal flags, and page types.
4. Preserve deterministic goal selection and price generation for a given seed.
5. Preserve TreeHCA episode export/import, branch cloning, and scoring snapshots.
6. Prevent mutable session data from modifying shared catalog records.
7. Support both the 1,000-product catalog and the full catalog through explicit
   configuration.
8. Fail during startup, before creating rollout workers, if catalog loading or
   search-index validation fails.
9. Expose enough metrics to confirm that only one catalog was loaded and to locate
   service bottlenecks.
10. Permit a browser-facing WebShop application to reuse the same catalog core
    without making browser automation part of training.

## Non-goals

The first implementation will not:

- Preserve the internal shape of `SimServer` as a remotely mutable Python object.
- Expose the entire product dictionary, goals list, or price dictionary to every
  worker.
- Use Selenium or Chrome in training.
- Use HTML routes as the primary training API.
- Scale catalog replicas automatically across nodes. Multi-node replication must
  be an explicit configuration decision.
- Disable Ray's memory monitor or raise the OOM threshold.
- Make the Flask development server a production training dependency.

## Existing components and reuse decisions

### `web_agent_site/app.py`

`app.py` is a browser-facing Flask implementation with process-global catalog and
session state. Its central-catalog idea is sound, but its module should not become
the training service directly.

| Element | Decision | Rationale |
| --- | --- | --- |
| One process-global catalog | Incorporate conceptually | This is the required memory ownership model. Implement it inside a lifecycle-managed service object rather than module globals. |
| `load_products` | Incorporate after refactoring | Product parsing and normalization should remain the source of truth, but loading must be separated from seed-specific prices and goals. |
| `init_search_engine` | Incorporate | The service should own the Lucene searcher and validate the selected index during startup. |
| `get_top_n_product_from_keywords` | Incorporate behind a service method | Preserve search semantics, including special `<r>`, `<a>`, `<c>`, and `<q>` queries. Random search must use request/session RNG rather than module-global randomness. |
| `get_product_per_page` | Incorporate | Preserve existing pagination semantics. |
| `map_action_to_html` and templates | Incorporate as a renderer | Reuse to maintain observation compatibility. Rendering should receive copied/read-only product data and must not mutate catalog records. |
| `get_reward` | Incorporate behind a service method | It is the native reward implementation and should remain authoritative. |
| Fixed goal selection by numeric index | Incorporate as an explicit API | Training reset already selects a goal index. Do not encode this contract in a special session-name convention such as `fixed_17`. |
| `user_sessions` module-global dictionary | Do not incorporate | Mutable episode state belongs in lightweight clients or in an explicit session store, not alongside the catalog as an unbounded global. |
| Lazy loading on the first request | Do not incorporate | It races under concurrent requests and delays failure until rollout startup. Load and validate eagerly. |
| Module-global `random` use | Do not incorporate | It couples sessions and makes concurrent behavior nondeterministic. Use isolated `random.Random` instances or deterministic seed derivation. |
| Mutation of `product_info['goal_instruction']` | Do not incorporate | It writes session-specific data into a shared catalog object and can leak across sessions. Pass the instruction to the renderer separately. |
| URL-encoded Python values plus `literal_eval` | Do not incorporate into the RPC API | Use typed payloads. The browser adapter may retain compatible URLs, but it must validate and normalize inputs. |
| Per-session Python loggers stored in the logging registry | Do not incorporate into training | Use structured service metrics/events. Optional human-session trajectory logging belongs in the browser adapter. |
| `app.run(...)` development server | Do not incorporate into training | It provides neither the lifecycle nor concurrency guarantees required by the trainer. |
| HTML responses as the only API | Do not incorporate | Training also requires typed page state, reward metadata, products for scoring, and branch-safe state transfer. |

There is also a correctness issue in the current `app.py`: its `load_products`
call does not supply the required `attrpath`. The shared service must take both
paths from resolved configuration and validate them at startup.

### `WebAgentSiteEnv`

`WebAgentSiteEnv` is a Gym wrapper around a real Chrome browser pointed at the
Flask application.

| Element | Decision | Rationale |
| --- | --- | --- |
| Unique session identifier per environment | Incorporate | Session identity is required for diagnostics and optional server-side state. Use opaque generated IDs, not bare rollout indices. |
| Gym-style `reset`, `step`, observation, reward, and done behavior | Incorporate | This is the compatibility boundary expected by higher-level code. |
| Parsing available actions from rendered HTML | Incorporate initially | It minimizes behavior drift. A later typed-action renderer may replace it after parity tests exist. |
| Converting HTML to text observations | Incorporate from `WebAgentTextEnv` | The trainer consumes text observations. Keep the exact separator and visibility behavior during migration. |
| HTTP/browser separation between client and catalog owner | Incorporate conceptually | Clients must no longer own the catalog. For training, use typed Ray RPC rather than browser navigation. |
| Selenium WebDriver and one Chrome process per environment | Do not incorporate into training | It would replace catalog OOM with browser memory and process overhead. |
| DOM clicking and form submission | Do not incorporate into training | Actions are already structured strings and can be executed directly. |
| Hard-coded `http://127.0.0.1:3000` | Do not incorporate | Service discovery must come from configuration and support Ray actor lookup. |
| Scraping reward from the done-page HTML | Do not incorporate | Reward must be returned as typed data from the authoritative native reward call. |
| Browser page source as authoritative state | Do not incorporate | Typed episode state is authoritative; HTML is a derived observation. |
| Silent exception handling around the search bar | Do not incorporate | Invalid state and rendering errors must be explicit and observable. |

`WebAgentSiteEnv` should remain available for manual interaction and browser-level
regression testing. It should not be instantiated by rollout workers.

## Architecture

### Components

#### 1. `WebshopCatalogData`

An immutable in-process data object constructed once by the service. It owns:

- `all_products`
- `product_item_dict`
- `attribute_to_asins`
- catalog ordering
- parsed product pricing ranges
- catalog-wide metadata such as maximum option-group width
- a deterministic catalog fingerprint

Product mappings exposed internally should be treated as read-only. Rendering
code must never add session-specific fields to product dictionaries.

The fingerprint must include, at minimum:

- content digests of the product and attribute files
- `human_goals`
- `num_products`
- renderer/schema version

It must not serialize and hash the complete catalog separately in every worker.

#### 2. `WebshopSeedView`

A seed-specific, immutable view containing:

- a price profile identifier
- product prices for that seed
- goals derived from those prices
- the deterministic shuffled goal order
- goal sampling weights and cumulative weights

The service creates at most one view for each distinct environment seed and keeps
it in a bounded cache. Rollout replicas in the same group share a seed and should
therefore share a view.

The initial compatibility implementation should reproduce the legacy operation
order with an isolated RNG:

1. Generate prices in catalog order.
2. Construct goals and their price limits.
3. Shuffle goals.

No step may read or modify Python's module-global RNG. Golden tests must compare
selected goals, prices, and rewards against the legacy `SimServer` for fixed seeds.

If exact reproduction proves impossible because legacy code relies on shared
global RNG state outside the environment, define and version a new deterministic
scheme rather than silently changing results.

#### 3. `WebshopCatalogService`

The service owns `WebshopCatalogData`, the Lucene searcher, seed views, and bounded
read caches. The preferred implementation is a named Ray actor created before
environment workers.

Required properties:

- Eager, all-or-nothing initialization.
- A stable name scoped to the training job.
- Zero GPUs.
- An explicit CPU allocation large enough to prevent accidental oversubscription.
- No actor restart after initialization failure unless the owner deliberately
  recreates it.
- Health and metadata calls that do not transfer the catalog.
- Bounded concurrency around the Lucene searcher until its thread-safety has been
  established by tests.

The first version should favor correctness with serialized or lock-protected
searches. Rendering and reward evaluation may run concurrently only after proving
that their dependencies are read-only and thread-safe.

#### 4. `RemoteWebAgentTextEnv`

A lightweight environment used inside each existing `WebshopWorker`. It owns an
`EpisodeState` but no catalog or searcher. It preserves the current Gym-facing API:

```python
reset(session: int) -> tuple[str, dict | None]
step(action: str) -> tuple[str, float, bool, dict | None]
get_available_actions() -> dict
```

The environment may continue to render HTML and use the existing BeautifulSoup
action extraction so observation text and clickables remain compatible. It must
not pretend that the remote service is a mutable `SimServer` object. Code that
currently reaches through `env.server` must move to explicit methods.

#### 5. Optional Flask presentation adapter

The Flask application becomes an adapter over the same catalog core or service.
It may preserve the existing human-facing routes and templates, including MTurk
logging. It must not own a second catalog when attached to a running training
service.

Two supported deployment modes are acceptable:

- Standalone browser mode: Flask constructs one local service core.
- Attached mode: Flask forwards typed calls to the named Ray catalog service.

Running multiple WSGI worker processes in standalone mode creates one catalog per
process and must be documented as incompatible with the single-copy guarantee.

## State model

### Immutable shared state

```python
CatalogMetadata(
    catalog_key: str,
    product_count: int,
    human_goals: bool,
    num_products: int | None,
    max_choices: int,
    renderer_version: str,
)
```

The catalog key identifies product content and rendering semantics. A separate
seed-view key identifies prices and goals:

```python
SeedViewRef(
    catalog_key: str,
    seed: int,
    price_profile_key: str,
    goal_set_key: str,
    goal_count: int,
)
```

### Mutable episode state

Each environment worker owns a serializable state object:

```python
EpisodeState(
    episode_id: str,
    seed_view: SeedViewRef,
    goal_index: int,
    goal: dict,
    page_type: str,
    current_url: str,
    keywords: tuple[str, ...] | None,
    page: int | None,
    asin: str | None,
    visited_asins: frozenset[str],
    selected_options: tuple[tuple[str, str], ...],
    action_counts: dict[str, int],
    done: bool,
    reward: float | None,
    instruction_text: str,
    rendered_html: str,
    previous_observations: tuple[str, ...],
    previous_actions: tuple[str, ...],
    random_search_count: int,
)
```

Use immutable tuples/frozen sets in exported state. The live environment may use
mutable equivalents internally, but `export_episode` must return a deep-copy-safe,
versioned payload.

Episode state should not contain:

- the full catalog
- the Lucene searcher
- all goals
- a full product-price dictionary
- Flask, browser, or Ray implementation objects

### Branching invariant

TreeHCA branch cloning copies only `EpisodeState` and manager history. Both source
and destination continue to reference the same immutable `SeedViewRef`. Importing
a branch from a different seed changes that reference; it never copies a complete
price table into the destination worker.

## Service interface

The interface below is logical. It may be implemented as Ray actor methods and
wrapped by JSON endpoints for browser mode.

### Lifecycle and metadata

```python
health() -> HealthResponse
metadata() -> CatalogMetadata
get_seed_view(seed: int) -> SeedViewRef
get_goal(seed_view: SeedViewRef, goal_index: int) -> GoalResponse
```

`health` reports initialization state, catalog key, process RSS, cache counts,
search readiness, and service version. It must not report healthy until loading and
index validation are complete.

`get_goal` validates the index and returns one copied goal record. Workers obtain
goal counts from `SeedViewRef`, replacing direct access to `server.goals`.

### Search

```python
search(
    seed_view: SeedViewRef,
    keywords: tuple[str, ...],
    page: int,
    random_token: str | None,
) -> SearchResponse
```

`SearchResponse` contains:

- ordered visible ASINs
- total hit count
- copied render fields for the visible products, or rendered HTML
- normalized keywords and page
- catalog and seed-view keys

For `<r>` searches, the caller supplies an idempotent `random_token` derived from
episode ID and random-search count. The service derives a local RNG seed from that
token. Retrying the same request must return the same products.

### Product rendering

```python
render_start(goal: dict, episode_id: str) -> RenderResponse
render_search(goal: dict, episode_id: str, search: SearchResponse) -> RenderResponse
render_item(
    seed_view: SeedViewRef,
    goal: dict,
    episode_id: str,
    asin: str,
    keywords: tuple[str, ...],
    page: int,
    selected_options: tuple[tuple[str, str], ...],
    show_attrs: bool,
) -> RenderResponse
render_subpage(...) -> RenderResponse
```

`RenderResponse` contains HTML, canonical page type, and canonical URL. Passing the
goal instruction separately avoids modifying the shared product record.

Returning rendered HTML initially provides the lowest-risk compatibility path.
After parity is established, the service may return typed page models and let the
client render locally, but that is a separate optimization requiring golden
observation tests.

### Reward

```python
evaluate_purchase(
    seed_view: SeedViewRef,
    goal: dict,
    asin: str,
    selected_options: tuple[tuple[str, str], ...],
) -> PurchaseResponse
```

The response contains native reward, verbose reward information, effective price,
and copied product fields needed for the done page. The service must call the
existing native `get_reward` implementation.

### Scoring support

```python
get_scoring_products(
    seed_view: SeedViewRef,
    asins: tuple[str, ...],
) -> ScoringProductsResponse
```

This returns only requested products and prices, plus `show_attrs`, `max_choices`,
catalog key, and price-profile key. It replaces direct reads of
`server.product_item_dict` and `server.product_prices` in TreeHCA workers.

The response must reject unknown ASINs and duplicate keys must be normalized.
Payload size and latency must be measured because this call occurs during rollout
scoring. A bounded service-side cache may memoize serialized product subsets.

### Optional batching

Every high-frequency method should have a batch form before performance tuning:

```python
search_batch(requests: list[SearchRequest]) -> list[SearchResponse]
render_batch(requests: list[RenderRequest]) -> list[RenderResponse]
evaluate_purchase_batch(requests: list[PurchaseRequest]) -> list[PurchaseResponse]
get_scoring_products_batch(requests: list[ScoringProductsRequest]) -> list[ScoringProductsResponse]
```

The vector environment should prefer batch calls so one rollout step does not
generate hundreds of tiny RPCs. Responses must retain request order and include a
request ID for diagnostics.

## Environment behavior

### Reset

`reset(goal_index)` performs the following:

1. Resolve the worker's `SeedViewRef`.
2. Validate `0 <= goal_index < goal_count`.
3. Fetch the selected goal.
4. Create a new opaque episode ID.
5. Initialize local `EpisodeState` to the start page.
6. Request or locally perform start-page rendering.
7. Clear previous observation/action buffers.
8. Return the same text observation shape as the legacy environment.

The training/validation split remains owned by `WebshopMultiProcessEnv`; it chooses
goal indices exactly as it does now.

### Step

The environment parses the existing action string with `parse_action`, validates
it against current available actions, updates local episode state, and invokes only
the necessary service operation:

- `search[...]`: search and render results.
- Product click: render the item page.
- Option click: update local options and re-render the item page.
- Description/features/reviews/attributes: render the subpage.
- Navigation: update page state and render the destination.
- `Buy Now`: evaluate purchase and render the terminal page.

Invalid actions preserve current state and return zero reward, matching existing
behavior.

The wrapper continues converting a native full-reward terminal purchase to reward
`10.0` and all other steps to `0`, while preserving the native score in
`info['task_score']`.

### Terminal auto-reset

The legacy text environment resets internally after a terminal purchase. The new
environment may preserve this behavior during migration, but it must return the
terminal observation and metadata before replacing local state. The implementation
must retain the pre-reset episode ID in `info['webshop_session_id']`.

A later cleanup may remove implicit auto-reset, but only with coordinated changes
to rollout management and explicit regression tests.

### Available actions

Initially, derive available actions from rendered HTML with the existing
BeautifulSoup logic. This preserves casing, labels, and option ordering. Cache the
result in `EpisodeState` after each render so repeated calls do not reparse HTML.

## TreeHCA compatibility

### Replace direct server access

The following current accesses must become explicit environment/service methods:

| Current access | Replacement |
| --- | --- |
| `env.server.goals` | `seed_view.goal_count` and `get_goal()` |
| `server.get_page_name(url)` | canonical `EpisodeState.page_type` |
| `server.product_item_dict[asin]` | `get_scoring_products()` or product subset already returned by search/render |
| `server.product_prices[asin]` | seed-view price returned with the product subset |
| `server.show_attrs` | immutable service/config metadata |
| `server.user_sessions[session]` | local `EpisodeState` |
| assignment to `server.product_prices` | assignment of `SeedViewRef` during branch import |

No compatibility proxy may lazily download the complete catalog to satisfy these
attribute reads.

### Snapshot capture

`WebshopSnapshotSource` should be split into:

- Pure snapshot construction from `EpisodeState`, prompt, task, and history.
- Product-subset retrieval through the catalog service.
- Pure hypothetical product-page rendering using the returned subset.

The snapshot catalog identity should combine:

```text
catalog_key + price_profile_key + show_attrs + renderer_version
```

This replaces the current per-worker `pickle.dumps` hash of complete catalog and
price dictionaries.

`scoring_payload` must continue transferring only products visible on the current
results page plus the current item ASIN. It must never request or serialize the
complete catalog.

### Episode export and import

`export_episode()` returns:

```python
ExportedEpisode(
    schema_version: int,
    episode_state: EpisodeState,
    rollout_task_id: int | None,
)
```

Manager memory, tasks, and previous text observations remain copied by
`TreeHCAWebshopEnvironmentManager.fork_from` as they are today.

`import_episode()` validates:

- schema version
- catalog key
- availability of the referenced seed view
- goal index and goal-set key

It then replaces local episode state and clears derived caches such as parsed
clickables and the current probability source. It does not transfer prices or
products.

### Branch isolation

After a fork:

- Changing options in one branch must not affect another branch.
- Random search in one branch must not advance another branch's RNG state.
- Both branches may share catalog and price-profile references.
- Service methods must not mutate catalog records or goal dictionaries.

## Configuration

Add a resolved configuration block similar to:

```yaml
env:
  webshop:
    backend: centralized       # centralized | legacy
    use_small: true
    human_goals: false
    catalog_service:
      transport: ray           # ray; http reserved for browser/debug clients
      actor_name: webshop_catalog
      actor_namespace: null     # default to current Ray job namespace
      num_cpus: 4
      search_concurrency: 1
      request_timeout_s: 120
      startup_timeout_s: 1800
      seed_view_cache_size: 64
      product_subset_cache_size: 4096
      batch_requests: true
```

Requirements:

- `file_path` and `attr_path` are resolved once by `make_envs` and passed to the
  service constructor.
- `backend=legacy` remains available temporarily for parity tests and the small
  catalog.
- The actor name must include a run-unique suffix unless deliberately attaching to
  an existing service.
- Attaching requires an exact match on catalog fingerprint and configuration.
- Worker `resources_per_worker.num_cpus` controls lightweight episode workers only;
  it no longer controls catalog loading concurrency.
- Do not expose Ray memory-monitor threshold changes as a WebShop solution.

## Startup and shutdown

Startup order:

1. Resolve and validate catalog, attribute, human-goal, and search-index paths.
2. Create the catalog service actor.
3. Load and normalize the catalog once.
4. Open and probe the search index.
5. Create required seed views for training and validation seeds, or validate lazy
   seed-view creation with a bounded cache.
6. Verify service health and metadata.
7. Create lightweight environment workers.
8. Verify one reset and one deterministic search before model initialization if a
   fail-fast smoke test is enabled.

Shutdown order:

1. Stop issuing rollout requests.
2. Close lightweight environment workers.
3. Flush metrics and optional trajectory logs.
4. Terminate the run-owned service actor.

Cleanup must be idempotent. A destructor must not synchronously call remote actors
during interpreter shutdown; use explicit `close()` ownership from the trainer.

## Concurrency and performance

### Single-copy guarantee

The guarantee applies per service process. Do not configure multiple actor replicas
or WSGI worker processes unless accepting one catalog copy per replica.

Tests and logs must record:

- service PID
- catalog load count
- catalog fingerprint
- service RSS after load
- number of connected environment workers
- number and memory size of cached seed views

### Search concurrency

`LuceneSearcher` thread safety must not be assumed. Begin with a lock or one search
execution lane. Measure throughput using representative rollout batches. If search
is the bottleneck, consider in this order:

1. Batch requests.
2. Cache deterministic search results by normalized query and page.
3. Establish safe concurrent reads through stress testing.
4. Add a small searcher pool, documenting that it duplicates only searcher state,
   not the Python product catalog.
5. Add explicit service replicas only as a last resort.

### Backpressure

The service must bound queued requests and expose queue latency. When overloaded,
it should return a typed overload error rather than allowing unbounded memory
growth. The vector environment may retry idempotent reads with bounded exponential
backoff.

### Payload discipline

- Never return `all_products`, `product_item_dict`, all goals, or all prices.
- Return ASINs and only the visible/scoring product records needed by the caller.
- Avoid Ray object-store placement of the complete Python catalog because worker
  deserialization would recreate private copies.
- Track response byte sizes, especially scoring payloads and rendered HTML.

## Reliability and error handling

Every request carries:

- request ID
- catalog key
- seed-view key when applicable
- episode ID when applicable
- operation name

Read and render operations must be idempotent. Purchase evaluation must also be
pure; marking an episode done happens in the client only after a successful
response.

Errors are categorized as:

- `ConfigurationError`: missing/mismatched files or unsupported settings.
- `CatalogMismatchError`: client and service catalog keys differ.
- `SeedViewNotFoundError`: evicted or unknown price/goal view.
- `InvalidGoalIndexError`.
- `UnknownProductError`.
- `InvalidPageStateError`.
- `ServiceOverloadedError`.
- `ServiceUnavailableError`.

Environment workers must not convert service errors into invalid-action rewards.
They should fail the rollout with the request and episode identifiers preserved.

If a seed view is evicted, the service may reconstruct it deterministically. If
the service actor itself dies, the trainer should fail rather than silently create
a new catalog with potentially different state. Transparent actor retries can be
added only after deterministic reconstruction tests exist.

## Observability

Expose counters and distributions for:

- catalog loads and load duration
- service RSS and seed-view cache memory
- calls, failures, retries, queue time, and execution time by method
- search cache hit rate
- seed-view cache hit rate and rebuilds
- products and bytes returned per scoring request
- render duration by page type
- reward evaluation duration
- active episode count as reported by clients or the optional session adapter

Log one startup record containing resolved paths, file digests, product count,
goal counts for initialized seed views, catalog key, and service PID. Do not log
entire goals, product objects, or prompts by default.

## Security and input validation

Even when used on a trusted cluster:

- Bind any HTTP adapter to loopback by default.
- Do not use `eval`; avoid `literal_eval` in the typed API.
- Validate ASINs, pages, option group names, and option values.
- Bound query length, option count, batch size, and rendered response size.
- Escape browser-rendered content through the existing Jinja behavior.
- Use opaque session IDs and avoid using them directly as filesystem paths.
- Keep catalog and index paths server-side; clients cannot select arbitrary paths.

## File-level implementation plan

Suggested organization:

```text
agent_system/environments/env_package/webshop/
  catalog_data.py          # immutable load/normalization and fingerprints
  catalog_service.py       # Ray actor and typed request/response contracts
  remote_text_env.py       # lightweight episode environment
  envs.py                  # worker/vector integration and backend selection

agent_system/environments/env_package/webshop/webshop/web_agent_site/
  app.py                   # browser adapter; no independent training state
  engine/engine.py         # native parsing/search/render helpers
  envs/web_agent_site_env.py
                            # manual/browser compatibility only

treehca/
  training_webshop_env.py  # explicit remote methods and lightweight branch copy
  webshop_probability_snapshot.py
                            # snapshot construction without full server access
```

Refactor `load_products` so catalog normalization is separable from price and goal
generation. Keep a compatibility wrapper with the existing signature while legacy
tests and tools still use `SimServer`.

## Migration plan

### Phase 1: Deterministic core extraction

1. Add immutable catalog loading and fingerprinting.
2. Extract seed-specific price and goal construction using isolated RNG objects.
3. Add parity tests against legacy `SimServer` for small-catalog seeds.
4. Fix `app.py` to supply `attrpath` and consume the extracted core in standalone
   browser mode.

No trainer behavior changes in this phase.

### Phase 2: Service and lightweight environment

1. Add typed service contracts and a named Ray actor.
2. Add `RemoteWebAgentTextEnv` with reset, step, rendering, and action parity.
3. Add `env.webshop.backend` and retain `legacy` as the default until parity tests
   pass.
4. Run baseline WebShop rollouts using the small catalog.

### Phase 3: TreeHCA integration

1. Replace direct `env.server` access with explicit state/service methods.
2. Change snapshot catalog keys to metadata-based identities.
3. Change episode export/import to transfer only versioned lightweight state.
4. Confirm fork isolation and scoring payload parity.
5. Run TreeHCA end-to-end on the small catalog.

### Phase 4: Full-catalog rollout

1. Measure service RSS after one full-catalog load.
2. Start the original 128 lightweight environment workers.
3. Confirm that worker RSS no longer scales with catalog size.
4. Tune batching, search caching, and service CPU allocation.
5. Make `centralized` the default for full-catalog training.

### Phase 5: Cleanup

After sustained parity:

- Remove training dependencies on `SimServer` internals.
- Retain `SimServer` only for unit tests, standalone tools, or deprecate it with a
  stated removal window.
- Keep `WebAgentSiteEnv` for browser regression tests and demonstrations.

## Test specification

### Unit tests

- Catalog files are loaded exactly once per service process.
- Catalog records are not mutated by rendering different goals concurrently.
- Catalog fingerprint changes when relevant input content or renderer version
  changes.
- The same seed produces identical prices, goals, and goal order.
- Different seeds produce the expected distinct seed views.
- Seed-view eviction and reconstruction are deterministic.
- Search and pagination match legacy results for fixed ordinary queries.
- Special search operators match legacy behavior.
- Repeated `<r>` requests with the same random token are identical.
- Reward and verbose reward details match legacy behavior.
- Invalid requests return typed errors.

### Environment parity tests

For fixed seeds, goal indices, and action sequences, compare legacy and centralized
backends on:

- instruction text
- normalized text observation
- available actions and ordering
- canonical page type
- native reward
- wrapper reward
- done flag
- task score and win flag

HTML may differ only where explicitly normalized and approved; trainer-facing text
must remain equal during the initial migration.

### TreeHCA tests

- Snapshot fields match the current implementation for every page type.
- Scoring payloads contain exactly the required visible/current products.
- Catalog and price-profile keys prevent invalid cache sharing.
- Exported episode size is independent of full catalog size.
- Forked branches are isolated.
- Import across seed views switches the reference without transferring price maps.
- Terminal auto-reset retains the purchased episode ID and is not scored as the
  new reset state.
- Existing product-page and search-results pseudo-rollout tests pass under both
  backends where applicable.

### Concurrency tests

- Simultaneous first use cannot trigger multiple catalog loads.
- Concurrent rendering cannot leak goal instructions or selected options.
- Concurrent searches return deterministic results.
- Service queue limits and overload errors behave as configured.
- Killing a client does not leak server-side session state, because training state
  is client-owned.

### Memory acceptance test

With the full catalog and the production actor count:

1. Record service RSS after initialization.
2. Create all training workers.
3. Reset every worker.
4. Execute representative search, product, option, subpage, and purchase actions.
5. Exercise TreeHCA scoring and branch copying.

Pass conditions:

- Catalog load count remains one.
- No worker contains the full catalog or Lucene searcher.
- Aggregate worker RSS grows with episode state, not catalog size.
- Node memory remains below a configured safety target, recommended at most 80%
  under steady-state load.
- No Ray worker is killed for memory pressure.

## Acceptance criteria

The design is complete when all of the following hold:

1. Full-catalog TreeHCA startup succeeds with the current 16-by-8 rollout layout
   on the target node.
2. Exactly one service process reports loading the catalog.
3. Legacy and centralized small-catalog trajectories pass deterministic parity
   tests for an agreed seed/action corpus.
4. TreeHCA branch export payload size is bounded and does not increase with catalog
   size.
5. TreeHCA scoring never fetches the complete catalog.
6. Browser mode remains functional through the Flask adapter.
7. No training worker launches Chrome or accesses port 3000.
8. Configuration mismatches fail before rollout actors are created.
9. Service latency and queue metrics are present in run diagnostics.
10. The legacy backend can be removed without any remaining direct training access
    to `env.server` catalog internals.

## Rejected alternatives

### Launch `app.py` and switch workers to `WebAgentSiteEnv`

This shares the catalog but launches a Chrome process for each environment, adds
DOM and HTTP overhead, lacks TreeHCA state APIs, and preserves unsafe global session
state. It is appropriate for interactive testing, not high-parallelism training.

### Pass one `SimServer` object to all Ray actors

Normal Python objects are serialized into each process, recreating the memory
duplication. A Ray object reference to the catalog also risks deserializing a full
private copy in each worker.

### Increase `resources_per_worker.num_cpus`

This reduces simultaneous loading but does not eliminate one resident catalog per
actor.

### Raise or disable Ray's memory threshold

The node already approaches physical capacity. Disabling protection changes the
failure mode from a controlled Ray eviction to kernel OOM or severe swapping.

### Store all mutable sessions in the catalog service

This resembles `app.py`, but it makes TreeHCA branch cloning and cleanup more
complex, turns every action into shared-state mutation, and makes retries harder to
reason about. Client-owned episode state plus a stateless catalog API is simpler
and more robust.

### Run multiple service replicas immediately

Replicas weaken the single-copy memory guarantee and introduce cache and routing
complexity before throughput has been measured. Begin with one service and add
explicit replicas only if batching and caching are insufficient.

## Fixed human validation split

Training now uses `env.webshop.human_goals` only for training tasks. Validation
always constructs human goals in catalog order, shuffles their positions with
seed 233, and selects the first 500. No exported goal JSON is needed. The defaults
in `ppo_trainer.yaml` are:

```yaml
env:
  webshop:
    validation:
      mode: fixed_human
      seed: 233
      shuffle_seed: 233
      count: 500
```

Set `env.webshop.validation.mode=legacy` to restore the original validation
behavior. It uses the configured training seed plus 1000, constructs the same goal
kind as training (`env.webshop.human_goals`), and validates on positions 0--499 of
each worker's seeded goal ordering. This mode works with the 1,000-product catalog
and does not load or construct human validation goals. It is useful for small
catalog smoke runs, but it is not the fixed official test split.

`validation.seed` fixes prices and goal price limits. `validation.shuffle_seed`
fixes membership and ordering independently of the training seed. Reproducing the
same goals requires the same catalog/human-annotation files and construction code;
seed 233 alone does not pin data revisions. The service fingerprints these files
and the split configuration.

Use `env.webshop.use_small=False` for the official 500-goal test split. The small
1,000-product catalog contains only 13 constructed human goals and cannot reproduce
that split. For a small smoke test explicitly set `env.webshop.validation.count=5`
and `data.val_batch_size=5` (or fewer). An insufficient catalog fails rather than
silently truncating validation. A validation batch of 500 covers the entire official
split; smaller batches sample without replacement from that fixed pool using the
validation seed, independently of training randomness.

The centralized backend constructs and retains one validation view for the life of
the service, shared by all validation workers and outside the training LRU cache.
It does not pickle the complete validation view to measure its size. Health reports
`validation_goal_count` and `metrics.validation_view_builds` separately from training
seed-view counts/bytes. Its one-time canonical shuffle is always applied even when
training-view shuffling is disabled. The standalone acceptance script's existing
skip flags and default legacy parity mode remain available.

Human-goal training excludes the held-out construction positions before applying
its own shuffle. Synthetic training excludes goals whose target ASIN is among the
held-out human goals, since synthetic and human instructions have no shared goal
identity. All products remain searchable. Training uses the full remaining task
pool, without dropping another 500 positions. The legacy backend applies the same
split and seeds but still builds a private catalog/view in each worker; use the
centralized backend to share validation startup work.
