"""Browser presentation adapter over the same catalog core used by training.

Use create_app() for standalone WSGI, or pass a Ray actor handle to attach without
loading another catalog. Standalone mode supports one WSGI process only.
"""
import argparse
import json
import logging
import random
from ast import literal_eval
from pathlib import Path
from uuid import uuid4

from flask import Flask, abort, redirect, request, url_for


def create_app(service=None, *, env_kwargs=None, settings=None, seed=233, log_dir=None):
    from agent_system.environments.env_package.webshop.catalog_service import CatalogClient, WebshopCatalogService, resolve_catalog_config
    from web_agent_site.utils import setup_logger, generate_mturk_code
    from web_agent_site.engine.engine import map_action_to_html

    if service is None:
        service = WebshopCatalogService(resolve_catalog_config(env_kwargs or {}, settings))
    client = CatalogClient(service)
    metadata = client.call("metadata")
    client.catalog_key = metadata.catalog_key
    ref = client.call("get_seed_view", seed=seed)
    app = Flask(__name__)
    sessions = {}
    rng = random.Random(seed)
    app.extensions["webshop_catalog"] = client

    def session_for(session_id):
        if not session_id or len(session_id) > 128 or not all(c.isalnum() or c in "_-" for c in session_id):
            abort(400, "Invalid session ID")
        if session_id not in sessions:
            if session_id.startswith("fixed_"):
                try:
                    index = int(session_id[6:])
                except ValueError:
                    abort(400, "Invalid goal index")
            else:
                index = client.call("sample_goal_index", seed_view=ref, random_token=str(rng.getrandbits(128)))
            if not 0 <= index < ref.goal_count:
                abort(400, "Goal index out of range")
            opaque_id = uuid4().hex
            sessions[session_id] = dict(goal=client.call("get_goal", seed_view=ref, goal_index=index), opaque_id=opaque_id)
            if log_dir:
                setup_logger(opaque_id, Path(log_dir))
        return sessions[session_id]

    def options_from_url(value):
        if len(value) > 4096:
            abort(400, "Options too long")
        try:
            # Native templates use Python dict representations in browser URLs.
            options = literal_eval(value)
            if not isinstance(options, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in options.items()):
                raise ValueError()
            return tuple(options.items())
        except (ValueError, SyntaxError):
            abort(400, "Invalid options")

    def query_from_url(keywords, page):
        if len(keywords) > 4096:
            abort(400, "Query too long")
        try:
            terms = literal_eval(keywords) if keywords.startswith("[") else keywords.replace("+", " ").split()
            return tuple(terms), int(page)
        except (ValueError, SyntaxError, TypeError):
            abort(400, "Invalid query")

    def event(session, page, **details):
        if log_dir:
            logging.getLogger(session["opaque_id"]).info(json.dumps(dict(page=page, url=request.url, **details)))

    @app.route("/")
    def home():
        return redirect(url_for("index", session_id=uuid4().hex))

    @app.route("/<session_id>", methods=["GET", "POST"])
    def index(session_id):
        session = session_for(session_id)
        if request.method == "POST" and "search_query" in request.form:
            return redirect(url_for("search_results", session_id=session_id, keywords=request.form["search_query"].lower().split(" "), page=1))
        event(session, "index")
        return client.call("render_start", episode_id=session_id, goal=session["goal"]).html

    @app.route("/search_results/<session_id>/<keywords>/<page>", methods=["GET", "POST"])
    def search_results(session_id, keywords, page):
        session = session_for(session_id)
        terms, page = query_from_url(keywords, page)
        rendered, asins = client.call("search_and_render", episode_id=session_id, seed_view=ref, goal=session["goal"], keywords=terms, page=page,
                                     random_token=f"{session['opaque_id']}:{keywords}:{page}")
        event(session, "search_results", search_result_asins=asins)
        return rendered.html

    @app.route("/item_page/<session_id>/<asin>/<keywords>/<page>/<options>", methods=["GET", "POST"])
    def item_page(session_id, asin, keywords, page, options):
        return render_product(session_id, asin, keywords, page, options)

    @app.route("/item_sub_page/<session_id>/<asin>/<keywords>/<page>/<sub_page>/<options>", methods=["GET", "POST"])
    def item_sub_page(session_id, asin, keywords, page, sub_page, options):
        return render_product(session_id, asin, keywords, page, options, sub_page)

    def render_product(session_id, asin, keywords, page, options, subpage=None):
        session = session_for(session_id)
        terms, page = query_from_url(keywords, page)
        response = client.call("render_item", episode_id=session_id, seed_view=ref, goal=session["goal"], asin=asin,
                               keywords=terms, page=page, selected_options=options_from_url(options), subpage=subpage)
        event(session, response.page_type, asin=asin)
        return response.html

    @app.route("/done/<session_id>/<asin>/<options>", methods=["GET", "POST"])
    def done(session_id, asin, options):
        session = session_for(session_id)
        purchase, rendered = client.call("purchase_and_render", episode_id=session_id, seed_view=ref, goal=session["goal"], asin=asin, selected_options=options_from_url(options))
        event(session, "done", reward=purchase.reward, reward_info=purchase.verbose_info)
        product = purchase.product
        options = options_from_url(options)
        return map_action_to_html("click[Buy Now]", session_id=session_id, reward=purchase.reward, asin=asin,
            options=dict(options), reward_info=purchase.verbose_info, query=product["query"], category=product["category"],
            product_category=product["product_category"], goal_attrs=session["goal"]["attributes"], purchased_attrs=product["Attributes"],
            goal=session["goal"], mturk_code=generate_mturk_code(session_id))

    return app


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    parser = argparse.ArgumentParser(description="WebShop browser catalog adapter")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--attrs", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--actor-name", help="Attach to an existing Ray catalog actor")
    parser.add_argument("--actor-namespace")
    args = parser.parse_args()
    service = None
    if args.actor_name:
        import ray
        ray.init(address="auto", namespace=args.actor_namespace)
        service = ray.get_actor(args.actor_name, namespace=args.actor_namespace)
    data = Path(__file__).resolve().parent.parent / "data"
    suffix = "" if args.full else "_1000"
    log_dir = Path("user_session_logs/mturk") if args.log else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    app = create_app(service, env_kwargs=dict(file_path=str(data / f"items_shuffle{suffix}.json"), attr_path=str(data / f"items_ins_v2{suffix}.json"),
                     num_products=None if args.full else 1000, human_goals=True, show_attrs=args.attrs), log_dir=log_dir)
    app.run(host="127.0.0.1", port=3000)
