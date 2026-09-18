"""TreeHCA-only WebShop snapshot transport, episode copying, and metrics."""

from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import ray

from agent_system.environments.env_manager import WebshopEnvironmentManager
from agent_system.environments.env_package.webshop.envs import WebshopWorker
from treehca.webshop_probability_snapshot import WebshopSnapshotSource


class TreeHCAWebshopWorker(WebshopWorker):
    def __init__(self, seed, env_kwargs):
        super().__init__(seed, env_kwargs)
        self._initial_prices = self.env.unwrapped.server.product_prices

    def reset(self, idx):
        # A previous branch may have brought prices from another rollout group.
        self.env.unwrapped.server.product_prices = self._initial_prices
        self.__dict__.pop("_probability_source", None)
        return super().reset(idx)

    def _source(self):
        if not hasattr(self, "_probability_source"):
            server = self.env.unwrapped.server
            self._probability_source = WebshopSnapshotSource(server)
            # Prices depend on the worker seed. Never merge caches across them.
            contents = (server.product_item_dict, server.product_prices, server.show_attrs)
            self._probability_source.catalog_key = hashlib.sha256(pickle.dumps(contents)).hexdigest()
            self._max_choices = max((len(values) + 1 for item in server.product_item_dict.values() for values in item.get("options", {}).values()), default=0)
        return self._probability_source

    def page_type(self):
        env = self.env.unwrapped
        return env.server.get_page_name(env.browser.current_url)

    def scoring_payload(self, prompt, task, history, completed_steps, history_limit, previous_page_type):
        source = self._source()
        manager = SimpleNamespace(tasks=[task], memory=[history], config=SimpleNamespace(env=SimpleNamespace(history_length=history_limit)))
        snapshot = source.capture(self.env.unwrapped, manager, prompt, previous_page_type=previous_page_type)
        snapshot = replace(snapshot, completed_steps=completed_steps)
        asins = set(snapshot.visible_asins)
        if snapshot.asin is not None:
            asins.add(snapshot.asin)
        # Transfer only products needed by this turn, never the search index.
        return dict(
            snapshot=snapshot,
            products={asin: source.server.product_item_dict[asin] for asin in asins},
            prices={asin: source.server.product_prices[asin] for asin in asins},
            show_attrs=source.server.show_attrs,
            max_choices=self._max_choices,
        )

    def export_episode(self):
        env = self.env.unwrapped
        return copy.deepcopy(
            dict(
                session=env.session,
                session_data=env.server.user_sessions[env.session],
                browser={key: value for key, value in vars(env.browser).items() if key != "server"},
                instruction_text=env.instruction_text,
                prev_obs=env.prev_obs,
                prev_actions=env.prev_actions,
                product_prices=env.server.product_prices,
            )
        )

    def import_episode(self, state):
        env = self.env.unwrapped
        state = copy.deepcopy(state)
        env.session = state["session"]
        env.server.user_sessions[env.session] = state["session_data"]
        env.server.product_prices = state["product_prices"]
        self.__dict__.pop("_probability_source", None)
        vars(env.browser).update(state["browser"])
        for key in ("instruction_text", "prev_obs", "prev_actions"):
            setattr(env, key, state[key])
        env.text_to_clickable = None  # Native step rebuilds the DOM cache.


class TreeHCAWebshopEnvironmentManager(WebshopEnvironmentManager):
    def scoring_page_types(self):
        return ray.get([worker.page_type.remote() for worker in self.envs._workers])

    def scoring_payloads(self, prompts, previous_page_types, active, dones):
        indices, futures = [], []
        limit = max(0, int(self.config.env.history_length))
        for index, worker in enumerate(self.envs._workers):
            # Native purchase auto-resets. Its new session must not be scored.
            if not active[index] or dones[index]:
                continue
            history = self.memory[index][-limit:] if limit else []
            indices.append(index)
            futures.append(worker.scoring_payload.remote(prompts[index], self.tasks[index], history, len(self.memory[index]), limit, previous_page_types[index]))
        payloads = np.empty(len(prompts), dtype=object)
        payloads[:] = None
        for index, payload in zip(indices, ray.get(futures)):
            payloads[index] = payload
        return payloads

    def fork_from(self, dest_index, src_index):
        # A freed slot may belong to another original group; copy its prices too.
        state = ray.get(self.envs._workers[src_index].export_episode.remote())
        ray.get(self.envs._workers[dest_index].import_episode.remote(state))
        self.memory._data[dest_index] = copy.deepcopy(self.memory[src_index])
        self.tasks[dest_index] = self.tasks[src_index]
        self.pre_text_obs[dest_index] = self.pre_text_obs[src_index]

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success, tree_structure=False):
        if not tree_structure:
            return super()._process_batch(batch_idx, total_batch_list, total_infos, success)
        for row, info in zip(total_batch_list, total_infos):
            if row["is_terminal"]:
                success["success_rate"].append(float(info["won"]))
                success["webshop_task_score (not success_rate)"].append(float(info["task_score"]))
