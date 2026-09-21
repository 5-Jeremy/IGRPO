"""Opt-in Ray/memory acceptance run without model weights.

Small: python -m agent_system.environments.env_package.webshop.catalog_smoke
Full 16x8: append --full --groups 16 --replicas 8
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import psutil
import ray

from agent_system.environments.env_package.webshop.catalog_service import CatalogClient, start_catalog_service
from agent_system.environments.env_package.webshop.envs import build_webshop_envs
from agent_system.environments.env_package.webshop.projection import webshop_projection
from treehca.training_webshop_env import TreeHCAWebshopEnvironmentManager, TreeHCAWebshopWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--skip-serialization", action="store_true", help="Skip full seed-view serialization used only for size measurement; Ray and episode/scoring checks still serialize small payloads")
    parser.add_argument("--skip-shuffling", action="store_true", help="Keep goals in catalog order instead of legacy shuffled order; changes goal-index semantics")
    parser.add_argument("--memory-target", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.groups < 1 or args.replicas < 1 or not 0 < args.memory_target < 1:
        parser.error("groups/replicas must be positive and memory-target must be between 0 and 1")
    ray.init(num_cpus=max(8, math.ceil(args.groups * args.replicas * 0.1) + 4), include_dashboard=False, _temp_dir="/tmp/ray-webshop-acceptance")
    data = Path(__file__).parent / "webshop/data"
    suffix = "" if args.full else "_1000"
    kwargs = dict(file_path=str(data / f"items_shuffle{suffix}.json"), attr_path=str(data / f"items_ins_v2{suffix}.json"), num_products=None if args.full else 1000, human_goals=False, observation_mode="text")
    settings = dict(num_cpus=4, startup_timeout_s=1800, request_timeout_s=120, seed_view_cache_size=max(2, args.groups), max_pending_requests=256, shuffle_goals=not args.skip_shuffling, measure_seed_view_bytes=not args.skip_serialization)
    actor = start_catalog_service(kwargs, settings)
    envs = None
    try:
        client = CatalogClient(actor, timeout=1800)
        after_load = client.call("health")
        # Warm all group seeds explicitly to distinguish cache memory from worker RSS.
        for seed in range(args.groups):
            client.call("get_seed_view", seed=seed)
        after_views = client.call("health")
        envs = build_webshop_envs(seed=0, env_num=args.groups, group_n=args.replicas, resources_per_worker={"num_cpus": 0.1}, env_kwargs={**kwargs, "backend": "centralized", "catalog_service": actor, "catalog_settings": settings}, worker_class=TreeHCAWebshopWorker)
        manager = TreeHCAWebshopEnvironmentManager(envs, webshop_projection, SimpleNamespace(env=SimpleNamespace(history_length=0)))
        manager.reset({})
        worker_reset = ray.get([worker.diagnostics.remote() for worker in envs._workers])
        count = envs.num_processes
        obs, _, dones, _ = manager.step(["<think>acceptance</think><action>search[shoes]</action>"] * count)
        payloads = manager.scoring_payloads(obs["text"], [""] * count, [True] * count, dones)
        choices = [next(iter(payload["products"])) for payload in payloads]
        obs, _, dones, _ = manager.step([f"<think>acceptance</think><action>click[{asin}]</action>" for asin in choices])
        item_payloads = manager.scoring_payloads(obs["text"], ["search_results"] * count, [True] * count, dones)
        option_actions = []
        for asin, payload in zip(choices, item_payloads):
            options = payload["products"][asin]["options"]
            value = next((values[0] for values in options.values() if values), None)
            option_actions.append(f"<think>acceptance</think><action>click[{value or 'Description'}]</action>")
        manager.step(option_actions)
        if count > 1:
            manager.fork_from(count - 1, 0)
        # Description -> item -> purchase, regardless of whether an option existed.
        manager.step(["<think>acceptance</think><action>click[Description]</action>"] * count)
        manager.step(["<think>acceptance</think><action>click[< Prev]</action>"] * count)
        _, _, dones, infos = manager.step(["<think>acceptance</think><action>click[Buy Now]</action>"] * count)
        assert all(dones)
        assert all(info["page_type"] == "done" for info in infos)
        final = client.call("health")
        worker_final = ray.get([worker.diagnostics.remote() for worker in envs._workers])
        assert final["catalog_load_count"] == 1
        assert final["pid"] == after_load["pid"]
        assert final["connected_workers"] == count
        assert all(not w["owns_catalog"] and not w["has_lucene"] for w in worker_final)
        assert all(w["episode_bytes"] < 100000 for w in worker_final)
        memory = psutil.virtual_memory()
        report = dict(
            full=args.full,
            skip_serialization=args.skip_serialization,
            skip_shuffling=args.skip_shuffling,
            groups=args.groups,
            replicas=args.replicas,
            service_after_load=after_load,
            service_after_seed_views=after_views,
            service_final=final,
            workers_after_reset=worker_reset,
            workers_final=worker_final,
            scoring_payload_bytes=[len(pickle.dumps(p)) for p in payloads],
            node_memory_used_fraction=1 - memory.available / memory.total,
        )
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        print(text)
        assert report["node_memory_used_fraction"] <= args.memory_target
    finally:
        try:
            if envs is not None:
                envs.close()
        finally:
            ray.kill(actor, no_restart=True)
            ray.shutdown()


if __name__ == "__main__":
    main()
