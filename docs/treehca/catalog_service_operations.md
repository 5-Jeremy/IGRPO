# Running the centralized WebShop catalog

The implementation of [the migration design](centralized_webshop_catalog_service.md)
uses one named Ray catalog actor for both training and validation. Each rollout
worker owns its episode state. The catalog and Lucene index remain in the service
process; TreeHCA scoring transfers at most ten visible products plus the current
item. Branch exports contain a versioned episode and seed-view reference.

## Configuration

Full-catalog runs select `centralized` automatically when `env.webshop.backend`
is null. Small-catalog runs retain the legacy default for migration comparisons.
To exercise the centralized small catalog, add:

```text
env.webshop.backend=centralized env.webshop.use_small=true
```

`make_envs` resolves the product, attribute, human-instruction and index paths
before creating workers. Small catalogs use the 1,000-product index. The service
loads and probes the index eagerly, and the factory warms training seed views
(up to the configured cache capacity) before starting workers. The default service
reserves four CPUs and zero GPUs. Worker CPU reservations apply only to episode
workers. Cache sizes, request/startup timeouts, queue and batch bounds are under
`env.webshop.catalog_service` in `ppo_trainer.yaml`.

Actor names receive a unique run suffix. Explicit attachment requires
`catalog_service.attach=true` and the complete existing actor name/namespace;
the requested content fingerprint and resolved configuration must match exactly.
Attached actors are not terminated by the attaching trainer. Run-owned actors
are terminated after both environment groups close, including initialization or
training failure. There are no destructor RPCs or automatic actor restarts.

Seed scheme `legacy-isolated-v1` reproduces price generation, goal construction,
and the legacy reseeding immediately before goal shuffling. It leaves global RNG
state untouched. Random `<r>` searches instead use a versioned episode's opaque
ID and local counter: retrying the same request returns identical results, and
forks do not advance each other's counters. Arbitrary legacy global-RNG random
search sequences are not a reproducibility contract.

Search, rendering and reward operations are serialized around native dependencies.
Scoring subset requests are batched across the vector environment when
`catalog_service.batch_requests=true`; each action also combines search/render or
purchase/render into one request. Admission and per-handle pending calls are
bounded. Rejected requests receive bounded backoff; actor failure and ordinary
service errors fail the rollout. Evicted seed views reconstruct deterministically.

Service health reports PID, catalog load count, RSS, cached seed counts and
serialized seed-view bytes, connected workers, method calls/failures, queue and
execution-time histograms, retries, response bytes and cache hits. Startup, each
new seed view, and close emit structured JSON records. Worker diagnostics report RSS, serialized episode size, and whether the
worker owns a catalog or has imported Lucene. Connected-worker counts reflect
explicit connect/close reports; an abruptly killed client may leave a stale count,
but never leaves episode state in the service.

## Validation

Run native parity, browser, isolation, scoring and request-validation tests with:

```bash
conda run --no-capture-output -n webshop env PYTHONPATH=. \
  python -m pytest tests/webshop -q
```

The model-free acceptance command exercises real Ray actors, TreeHCA snapshots,
batched scoring payloads, branch copying, options, subpages and terminal purchases:

```bash
conda run --no-capture-output -n webshop env TMPDIR=/tmp PYTHONPATH=. \
  python -m agent_system.environments.env_package.webshop.catalog_smoke \
  --output /tmp/webshop-small-acceptance.json

conda run --no-capture-output -n webshop env TMPDIR=/tmp PYTHONPATH=. \
  python -m agent_system.environments.env_package.webshop.catalog_smoke \
  --full --groups 16 --replicas 8 --output /tmp/webshop-full-acceptance.json
```

To isolate startup work, either flag can be supplied independently, or together:

```bash
conda run --no-capture-output -n webshop env TMPDIR=/tmp PYTHONPATH=. \
  python -m agent_system.environments.env_package.webshop.catalog_smoke \
  --full --groups 16 --replicas 8 --skip-serialization --skip-shuffling \
  --output /tmp/webshop-full-acceptance-skipped.json
```

`--skip-serialization` omits the full seed-view `pickle.dumps` used for size
measurement. The report records `seed_view_bytes: null` rather than claiming zero
memory use. Ray transport and the small episode/scoring payload measurements still
serialize their payloads.

`--skip-shuffling` retains catalog goal order. Prices and goal contents are
unchanged, but numeric goal indices and index-based train/validation splits refer
to different goals. The goal-set identity records this ordering so shuffled and
unshuffled episodes cannot be mixed. This flag applies to the entire acceptance
service, not just validation workers in an actual training job. Both skips default
to false and are recorded in the report; ordinary training defaults are unchanged.

The full command requires the production dataset/index and enough memory for one
catalog, sixteen cached seed views and 128 Python workers. It asserts one catalog
load, no catalog/Lucene in workers, small episode exports, completed purchases,
and node memory below `--memory-target` (default 0.8). Reports separate catalog,
seed-cache and worker RSS. This does not initialize an LLM or perform optimization.

## Browser mode

Standalone mode constructs one local service core eagerly:

```bash
PYTHONPATH=.:agent_system/environments/env_package/webshop/webshop \
  python -m web_agent_site.app
```

Use `--full`, `--attrs` or `--log` as needed. Attach to a running training service
with `--actor-name <complete-name> --actor-namespace <namespace>`. Attached mode
uses typed Ray calls and does not load another catalog. The development server
binds to loopback. A WSGI deployment can use `web_agent_site.app:create_app()`;
standalone mode must use one process to retain the single-copy guarantee.

The legacy `SimServer` and browser environment remain available for regression
tests and standalone tools. Legacy-only TreeHCA transport retains its old server
accesses until that backend is retired; the centralized path never invokes them.

## Validation recorded during migration

The 1,000-product parity corpus covers seeds 0, 42 and 123 and goal indices 0,
50 and 500, with ordinary search, pagination, invalid actions, option selection,
subpages and purchases. Additional checks cover human goals, seeded random/special
search, immutable rendering, eviction/reconstruction, terminal reward conversion,
TreeHCA snapshot and hypothetical-page parity, browser routes and attachment
validation. All 16 migration tests passed together in the final migration-suite run.

The broader `tests/treehca` run with migration tests and the worker page-type test
reported 310 passes and two failures. Both failures reproduce in a clean export
of the pre-migration commit:

- `test_mixed_results_decoder_matches_scalar_action_span`: expected probability
  0.123, obtained 0.24732617117726066.
- `test_complete_testbed_matches_training_scores_and_preserves_rollout_identity`:
  expected an initial probability of `None`, obtained 0.0.

A four-worker small-catalog Ray acceptance run passed. Service RSS was 0.88 GiB
after catalog loading and 1.01 GiB after exercising two seed views and the complete
transition sequence. Worker RSS was 575–582 MiB; exported reset episodes were
3.3–3.6 KiB. Separate factory validation confirmed training/validation share one
actor and explicit close terminates the run-owned actor idempotently.
