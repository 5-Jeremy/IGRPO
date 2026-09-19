"""A single catalog owner with bounded caches and a typed, stateless Ray API.

Only ``call`` and ``call_batch`` are used over Ray. The lock covers native Lucene,
Jinja and reward dependencies; actor threads provide bounded admission, not
concurrent access to these dependencies.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import random
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from .catalog_data import WebshopCatalogData, catalog_fingerprint, native_imports


class ConfigurationError(ValueError):
    pass


class CatalogMismatchError(ValueError):
    pass


class SeedViewNotFoundError(ValueError):
    pass


class InvalidGoalIndexError(ValueError):
    pass


class UnknownProductError(ValueError):
    pass


class InvalidPageStateError(ValueError):
    pass


class ServiceOverloadedError(RuntimeError):
    pass


class ServiceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogRequest:
    operation: str
    payload: dict = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    catalog_key: str | None = None
    episode_id: str | None = None
    submitted_at: float = field(default_factory=time.time)
    retry_count: int = 0


@dataclass(frozen=True)
class CatalogResponse:
    request_id: str
    result: object


@dataclass(frozen=True)
class SearchResponse:
    visible_asins: tuple[str, ...]
    total: int
    keywords: tuple[str, ...]
    page: int
    catalog_key: str
    price_profile_key: str
    products: tuple[dict, ...]


@dataclass(frozen=True)
class RenderResponse:
    html: str
    page_type: str
    url: str


@dataclass(frozen=True)
class PurchaseResponse:
    reward: float
    verbose_info: dict
    price: float
    product: dict


def resolve_catalog_config(env_kwargs, settings=None):
    native_imports()
    from web_agent_site.engine.engine import search_index_path
    from web_agent_site.utils import DEFAULT_ATTR_PATH, DEFAULT_FILE_PATH, HUMAN_ATTR_PATH

    settings = dict(settings or {})
    if settings.get("transport", "ray") != "ray" or settings.get("search_concurrency", 1) != 1:
        raise ConfigurationError("Only Ray transport with search_concurrency=1 is supported")
    for key, default in (("filter_goals", None), ("limit_goals", -1), ("get_image", 0)):
        if env_kwargs.get(key, default) != default:
            raise ConfigurationError(f"Centralized backend does not support {key}")
    config = dict(
        file_path=env_kwargs.get("file_path", DEFAULT_FILE_PATH),
        attr_path=env_kwargs.get("attr_path", DEFAULT_ATTR_PATH),
        human_attr_path=settings.get("human_attr_path") or HUMAN_ATTR_PATH,
        validation=env_kwargs.get("validation"),
        human_goals=bool(env_kwargs.get("human_goals", False)),
        num_products=env_kwargs.get("num_products"),
        show_attrs=bool(env_kwargs.get("show_attrs", False)),
        index_path=settings.get("index_path") or search_index_path(env_kwargs.get("num_products")),
        shuffle_goals=settings.get("shuffle_goals", True),
        measure_seed_view_bytes=settings.get("measure_seed_view_bytes", True),
        seed_view_cache_size=int(settings.get("seed_view_cache_size", 64)),
        product_subset_cache_size=int(settings.get("product_subset_cache_size", 4096)),
        search_cache_size=int(settings.get("search_cache_size", 1024)),
        max_pending_requests=int(settings.get("max_pending_requests", 64)),
        max_batch_size=int(settings.get("max_batch_size", 256)),
        max_response_bytes=int(settings.get("max_response_bytes", 16 * 1024 * 1024)),
    )
    if config["validation"] is not None:
        validation = dict(config["validation"])
        if set(validation) != {"seed", "shuffle_seed", "count"} or any(type(v) is not int for v in validation.values()) or validation["count"] < 1:
            raise ConfigurationError("validation requires integer seed, shuffle_seed and positive count")
        config["validation"] = validation
    for key in ("shuffle_goals", "measure_seed_view_bytes"):
        if not isinstance(config[key], bool):
            raise ConfigurationError(f"{key} must be a boolean")
    for key in ("file_path", "attr_path", "human_attr_path", "index_path"):
        config[key] = str(Path(config[key]).resolve())
        if key == "human_attr_path" and not (config["human_goals"] or config["validation"]):
            continue
        path = Path(config[key])
        if not (path.is_dir() if key == "index_path" else path.is_file()):
            raise ConfigurationError(f"Missing {key}: {path}")
    for key in ("seed_view_cache_size", "product_subset_cache_size", "search_cache_size", "max_pending_requests", "max_batch_size", "max_response_bytes"):
        if config[key] < 1:
            raise ConfigurationError(f"{key} must be positive")
    return config


class WebshopCatalogService:
    def __init__(self, config):
        started = time.monotonic()
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self.config = dict(config)
        self._lock = threading.RLock()
        self._admission = threading.BoundedSemaphore(config["max_pending_requests"])
        self._views, self._searches, self._subsets = OrderedDict(), OrderedDict(), OrderedDict()
        self._view_sizes = {}
        self._clients = set()
        self._metrics = Counter()
        self._ready = False
        self.data = WebshopCatalogData.load(config)
        from flask import Flask
        from web_agent_site.engine.engine import init_search_engine

        self._app = Flask(__name__)
        # URL generation only. No HTTP server is started by the training service.
        routes = {
            "index": "/<session_id>",
            "search_results": "/search_results/<session_id>/<keywords>/<page>",
            "item_page": "/item_page/<session_id>/<asin>/<keywords>/<page>/<options>",
            "item_sub_page": "/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>",
            "done": "/done/<session_id>/<asin>/<options>",
        }
        for endpoint, route in routes.items():
            self._app.add_url_rule(route, endpoint=endpoint, view_func=lambda: None)

        self.search_engine = init_search_engine(config["num_products"], index_path=config["index_path"])
        if self.search_engine.num_docs <= 0:
            raise ConfigurationError("Search index is empty")
        # Probe both query execution and document decoding before any workers exist.
        self.search_engine.search("webshop", k=1)
        probe = self.search_engine.doc(0)
        if probe is None or "id" not in json.loads(probe.raw()):
            raise ConfigurationError("Search index does not contain WebShop documents")
        self._metrics["catalog_loads"] = 1
        self._metrics["load_seconds"] = time.monotonic() - started
        self._ready = True
        self._logger.info(json.dumps(dict(event="webshop_catalog_startup", digests=self.data.digests, metadata=asdict(self.metadata()), health=self.health())))

    def metadata(self):
        return self.data.metadata

    def health(self):
        import psutil

        return dict(
            ready=self._ready,
            search_ready=self._ready,
            service_version=1,
            pid=os.getpid(),
            catalog_key=self.metadata().catalog_key,
            rss_bytes=psutil.Process().memory_info().rss,
            catalog_load_count=self._metrics["catalog_loads"],
            connected_workers=len(self._clients),
            validation_goal_count=self._validation_view.ref.goal_count if hasattr(self, "_validation_view") else 0,
            seed_view_count=len(self._views),
            seed_view_bytes=sum(self._view_sizes.values()) if self.config["measure_seed_view_bytes"] else None,
            seed_goal_counts={seed: view.ref.goal_count for seed, view in self._views.items()},
            metrics=dict(self._metrics),
            config=self.config.copy(),
        )

    def flush_metrics(self):
        health = self.health()
        self._logger.info(json.dumps(dict(event="webshop_catalog_metrics", health=health)))
        return health

    def connect(self, client_id):
        self._clients.add(client_id)

    def disconnect(self, client_id):
        self._clients.discard(client_id)

    def _cache(self, cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def get_seed_view(self, seed, split="train"):
        if type(seed) is not int:
            raise ConfigurationError("Seed must be an integer")
        if split not in {"train", "validation"}:
            raise ConfigurationError(f"Unknown goal split: {split}")
        if split == "validation":
            if not self.config["validation"]:
                raise ConfigurationError("Fixed validation is not configured")
            if not hasattr(self, "_validation_view"):
                started = time.monotonic()
                self._validation_view = self.data.seed_view(self.config["validation"]["seed"], split="validation")
                self._metrics["validation_view_builds"] += 1
                self._logger.info(json.dumps(dict(event="webshop_validation_view_ready", seed=self._validation_view.ref.seed, goal_count=self._validation_view.ref.goal_count, goal_set_key=self._validation_view.ref.goal_set_key, build_seconds=time.monotonic() - started)))
            return self._validation_view.ref
        if seed in self._views:
            self._metrics["seed_cache_hits"] += 1
            self._views.move_to_end(seed)
        else:
            self._metrics["seed_cache_builds"] += 1
            started = time.monotonic()
            view = self.data.seed_view(seed, shuffle_goals=self.config["shuffle_goals"])
            self._cache(self._views, seed, view, self.config["seed_view_cache_size"])
            self._view_sizes = {key: size for key, size in self._view_sizes.items() if key in self._views}
            if self.config["measure_seed_view_bytes"]:
                self._view_sizes[seed] = len(pickle.dumps(view))
            self._logger.info(json.dumps(dict(event="webshop_seed_view_ready", seed=seed, goal_count=view.ref.goal_count, cached_views=len(self._views), build_seconds=time.monotonic() - started, serialized_bytes=self._view_sizes.get(seed))))
        return self._views[seed].ref

    def _view(self, seed_view):
        if seed_view.catalog_key != self.metadata().catalog_key:
            raise CatalogMismatchError("Seed view belongs to another catalog")
        if self.get_seed_view(seed_view.seed, split=seed_view.split) != seed_view:
            raise SeedViewNotFoundError("Seed view identity does not match deterministic reconstruction")
        return self._validation_view if seed_view.split == "validation" else self._views[seed_view.seed]

    def get_goal(self, seed_view, goal_index):
        view = self._view(seed_view)
        if type(goal_index) is not int or not 0 <= goal_index < len(view.goals):
            raise InvalidGoalIndexError(str(goal_index))
        return copy.deepcopy(view.goals[goal_index])

    def sample_goal_index(self, seed_view, random_token):
        # Preserve legacy random_idx's cumulative-weight convention.
        import bisect

        weights = self._view(seed_view).cumulative_weights
        position = random.Random(random_token).uniform(0, weights[-1])
        return min(bisect.bisect(weights, position), len(weights) - 2)

    def _product(self, asin):
        try:
            return self.data.product_item_dict[asin]
        except (KeyError, TypeError):
            raise UnknownProductError(str(asin)) from None

    def _options(self, asin, selected_options):
        product = self._product(asin)
        if len(selected_options) > 100:
            raise InvalidPageStateError("Too many selected options")
        options = dict(selected_options)
        if len(options) != len(selected_options) or any(k not in product["options"] or v not in product["options"][k] for k, v in options.items()):
            raise InvalidPageStateError("Invalid product options")
        return options

    @staticmethod
    def _query(keywords, page):
        if type(page) is not int or page < 1 or not isinstance(keywords, (list, tuple)) or not keywords or any(not isinstance(k, str) for k in keywords) or len(" ".join(keywords)) > 2048:
            raise InvalidPageStateError("Invalid keywords or page")
        if keywords[0] in ("<a>", "<c>", "<q>") and len(keywords) < 2:
            raise InvalidPageStateError("Special search requires an argument")
        return tuple(keywords)

    def search(self, seed_view, keywords, page=1, random_token=None):
        from web_agent_site.engine.engine import get_product_per_page, get_top_n_product_from_keywords

        self._view(seed_view)
        keywords = self._query(keywords, page)
        if keywords[0] == "<r>" and not random_token:
            raise InvalidPageStateError("Random search requires an idempotency token")
        key = (keywords, random_token if keywords[0] == "<r>" else None)
        if key in self._searches:
            self._metrics["search_cache_hits"] += 1
            self._searches.move_to_end(key)
            asins = self._searches[key]
        else:
            products = get_top_n_product_from_keywords(list(keywords), self.search_engine, self.data.all_products, self.data.product_item_dict, self.data.attribute_to_asins, rng=random.Random(random_token))
            asins = tuple(p["asin"] for p in products)
            # Special queries may match the full catalog. Do not cache full result lists.
            if len(asins) <= 1000:
                self._cache(self._searches, key, asins, self.config["search_cache_size"])
        visible = tuple(get_product_per_page(asins, page))
        return SearchResponse(visible, len(asins), keywords, page, self.metadata().catalog_key, seed_view.price_profile_key, tuple(copy.deepcopy(self._product(a)) for a in visible))

    def _render(self, action, page_type, path, **kwargs):
        from web_agent_site.engine.engine import map_action_to_html

        with self._app.app_context(), self._app.test_request_context():
            html = map_action_to_html(action, **kwargs)
        if len(html.encode()) > self.config["max_response_bytes"]:
            raise InvalidPageStateError("Rendered response exceeds configured bound")
        return RenderResponse(html, page_type, "http://127.0.0.1:3000/" + path)

    def render_start(self, goal, episode_id):
        return self._render("start", "", episode_id, session_id=episode_id, instruction_text=goal["instruction_text"])

    def render_search(self, goal, episode_id, search):
        if search.catalog_key != self.metadata().catalog_key:
            raise CatalogMismatchError("Search belongs to another catalog")
        return self._render("search", "search_results", f"search_results/{episode_id}/{'+'.join(search.keywords)}/{search.page}", session_id=episode_id, instruction_text=goal["instruction_text"], products=search.products, keywords=list(search.keywords), page=search.page, total=search.total)

    def search_and_render(self, seed_view, goal, episode_id, keywords, page=1, random_token=None):
        result = self.search(seed_view, keywords, page, random_token)
        return self.render_search(goal, episode_id, result), result.visible_asins

    def render_item(self, seed_view, goal, episode_id, asin, keywords, page, selected_options=(), subpage=None):
        from web_agent_site.engine.engine import ACTION_TO_TEMPLATE

        self._view(seed_view)
        self._query(keywords, page)
        options = self._options(asin, selected_options)
        if subpage is not None and subpage not in ACTION_TO_TEMPLATE:
            raise InvalidPageStateError("Unknown subpage")
        page_type = "item_sub_page" if subpage else "item_page"
        suffix = f"{subpage}/{options}" if subpage else json.dumps(options)
        return self._render(
            f"click[{subpage}]" if subpage else "click",
            page_type,
            f"{page_type}/{episode_id}/{asin}/{'+'.join(keywords)}/{page}/{suffix}",
            session_id=episode_id,
            instruction_text=goal["instruction_text"],
            product_info=self._product(asin),
            keywords=list(keywords),
            page=page,
            asin=asin,
            options=options,
            show_attrs=self.config["show_attrs"],
        )

    def render_subpage(self, **kwargs):
        return self.render_item(**kwargs)

    def evaluate_purchase(self, seed_view, goal, asin, selected_options=()):
        from web_agent_site.engine.goal import get_reward

        view = self._view(seed_view)
        options = self._options(asin, selected_options)
        product = self._product(asin)
        reward, info = get_reward(product, goal, price=view.prices[asin], options=options, verbose=True)
        return PurchaseResponse(reward, info, view.prices[asin], copy.deepcopy(product))

    def purchase_and_render(self, seed_view, goal, episode_id, asin, selected_options=()):
        purchase = self.evaluate_purchase(seed_view, goal, asin, selected_options)
        options = dict(selected_options)
        rendered = self._render("click[Buy Now]", "done", f"done/{episode_id}/{asin}/{options}", session_id=episode_id, instruction_text=goal["instruction_text"], reward=purchase.reward, asin=asin, options=options)
        return purchase, rendered

    def get_scoring_products(self, seed_view, asins):
        view = self._view(seed_view)
        if len(asins) > 11:
            raise InvalidPageStateError("Scoring requests are limited to one results page and current item")
        asins = tuple(sorted(set(asins)))
        key = (seed_view.price_profile_key, asins)
        if key not in self._subsets:
            payload = dict(products={a: copy.deepcopy(self._product(a)) for a in asins}, prices={a: view.prices[a] for a in asins}, show_attrs=self.config["show_attrs"], max_choices=self.metadata().max_choices, catalog_key=self.metadata().catalog_key, price_profile_key=seed_view.price_profile_key)
            self._cache(self._subsets, key, payload, self.config["product_subset_cache_size"])
        self._subsets.move_to_end(key)
        result = copy.deepcopy(self._subsets[key])
        self._metrics["scoring_products"] += len(asins)
        self._metrics["scoring_bytes"] += len(pickle.dumps(result))
        return result

    OPERATIONS = frozenset(
        ("health", "metadata", "flush_metrics", "connect", "disconnect", "get_seed_view", "get_goal", "sample_goal_index", "search", "render_start", "render_search", "search_and_render", "render_item", "render_subpage", "evaluate_purchase", "purchase_and_render", "get_scoring_products")
    )

    def _observe_duration(self, name, seconds):
        self._metrics[name] += seconds
        self._metrics[f"{name}/max"] = max(self._metrics[f"{name}/max"], seconds)
        for upper in (0.001, 0.01, 0.1, 1, 10, 100, 1000, float("inf")):
            if seconds <= upper:
                self._metrics[f"{name}/bucket_{upper}"] += 1
                break

    def call(self, request):
        if not self._admission.acquire(blocking=False):
            raise ServiceOverloadedError(f"request={request.request_id} episode={request.episode_id}")
        try:
            with self._lock:
                started = time.monotonic()
                operation = request.operation if request.operation in self.OPERATIONS else "invalid_operation"
                self._observe_duration(f"{operation}/queue_seconds", max(0, time.time() - request.submitted_at))
                self._metrics[f"{operation}/retries"] += request.retry_count
                self._metrics[f"{operation}/calls"] += 1
                try:
                    if request.catalog_key is not None and request.catalog_key != self.metadata().catalog_key:
                        raise CatalogMismatchError("Request catalog mismatch")
                    if operation not in self.OPERATIONS:
                        raise InvalidPageStateError("Unknown catalog operation")
                    result = getattr(self, operation)(**request.payload)
                    size = len(pickle.dumps(result))
                    if size > self.config["max_response_bytes"]:
                        raise InvalidPageStateError("Response exceeds configured bound")
                    self._metrics[f"{operation}/bytes"] += size
                    return CatalogResponse(request.request_id, result)
                except Exception as exc:
                    self._metrics[f"{operation}/failures"] += 1
                    exc.args = (*exc.args, f"request={request.request_id} episode={request.episode_id} operation={operation}")
                    raise
                finally:
                    self._observe_duration(f"{operation}/execution_seconds", time.monotonic() - started)
        finally:
            self._admission.release()

    def call_batch(self, requests):
        if len(requests) > self.config["max_batch_size"]:
            raise ServiceOverloadedError("Batch exceeds configured bound")
        responses = []
        total_bytes = 0
        for request in requests:
            response = self.call(request)
            total_bytes += len(pickle.dumps(response))
            if total_bytes > self.config["max_response_bytes"]:
                raise ServiceOverloadedError("Batch response exceeds configured byte bound")
            responses.append(response)
        return responses

    def search_batch(self, requests):
        return self.call_batch(requests)

    render_batch = search_batch
    evaluate_purchase_batch = search_batch
    get_scoring_products_batch = search_batch


class CatalogClient:
    """RPC facade with no mutable catalog attributes or implicit bulk downloads."""

    def __init__(self, service, timeout=120):
        self.service = service
        self.timeout = timeout
        self.catalog_key = None

    def call(self, operation, episode_id=None, **payload):
        request = CatalogRequest(operation, payload, catalog_key=self.catalog_key, episode_id=episode_id)
        # Rendering operations also take the episode ID as an explicit argument.
        if operation.startswith("render_") or operation in ("search_and_render", "purchase_and_render"):
            request.payload["episode_id"] = episode_id
        if isinstance(self.service, WebshopCatalogService):
            return self.service.call(request).result
        import ray

        deadline = time.monotonic() + self.timeout
        for attempt in range(10):
            try:
                return ray.get(self.service.call.remote(request), timeout=max(0.001, deadline - time.monotonic())).result
            except ray.exceptions.RayTaskError as exc:
                failure = exc.as_instanceof_cause()
                if not isinstance(failure, ServiceOverloadedError):
                    raise failure from exc
            except ray.exceptions.PendingCallsLimitExceeded as exc:
                failure = ServiceOverloadedError(f"request={request.request_id} episode={episode_id}: {exc}")
            except (ray.exceptions.RayError, TimeoutError) as exc:
                raise ServiceUnavailableError(f"request={request.request_id} episode={episode_id} operation={operation}: {exc}") from exc
            delay = min(0.05 * 2**attempt, 2.0)
            if attempt == 9 or time.monotonic() + delay >= deadline:
                raise failure
            # Only rejected requests are retried. Their original request ID/token is retained.
            time.sleep(delay)
            request = replace(request, retry_count=attempt + 1)


def start_catalog_service(env_kwargs, settings=None):
    """Validate paths, eagerly initialize and probe one run-owned actor."""
    import ray

    settings = dict(settings or {})
    config = resolve_catalog_config(env_kwargs, settings)
    if float(settings.get("num_cpus", 4)) <= 0:
        raise ConfigurationError("Catalog service needs a positive CPU allocation")
    if not ray.is_initialized():
        ray.init()
    if settings.get("attach", False):
        actor = ray.get_actor(settings.get("actor_name", "webshop_catalog"), namespace=settings.get("actor_namespace"))
        health = ray.get(actor.call.remote(CatalogRequest("health")), timeout=settings.get("startup_timeout_s", 1800)).result
        key, _ = catalog_fingerprint(config)
        if not health["ready"] or health["config"] != config or health["catalog_key"] != key:
            raise CatalogMismatchError("Attached service configuration or file fingerprint differs")
        return actor
    options = dict(name=f"{settings.get('actor_name', 'webshop_catalog')}_{uuid4().hex}", num_cpus=settings.get("num_cpus", 4), num_gpus=0, max_restarts=0, max_task_retries=0, max_pending_calls=config["max_pending_requests"], max_concurrency=config["max_pending_requests"] + 1)
    if settings.get("actor_namespace"):
        options["namespace"] = settings["actor_namespace"]
    actor = ray.remote(WebshopCatalogService).options(**options).remote(config)
    try:
        health = ray.get(actor.call.remote(CatalogRequest("health")), timeout=settings.get("startup_timeout_s", 1800)).result
        if not health["ready"]:
            raise ServiceUnavailableError("Catalog did not become ready")
        logging.getLogger(__name__).info("WebShop service ready: %s", health)
        return actor
    except BaseException:
        ray.kill(actor, no_restart=True)
        raise
