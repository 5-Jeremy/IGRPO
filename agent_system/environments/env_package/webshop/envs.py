# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import gym
import numpy as np
import ray


_ACTION_PATTERN = re.compile(r"(search|click)\[(.*)\]", re.DOTALL)


def is_admissible_webshop_action(action, available_actions):
    """Return whether *action* can execute in the current WebShop state.

    Projection checks response formatting, while this check validates the
    projected command against the action set from the page before the
    transition. Search queries are free-form; clicks must name an exact
    currently clickable target (case-insensitively).
    """
    if not isinstance(action, str) or not isinstance(available_actions, dict):
        return False, "invalid_command"

    match = _ACTION_PATTERN.fullmatch(action)
    if match is None:
        return False, "invalid_command"

    action_name, action_arg = match.groups()
    action_arg = action_arg.lower()
    if action_name == "search":
        valid = bool(available_actions.get("has_search_bar")) and bool(action_arg.strip())
        return valid, None if valid else "search_unavailable"

    clickables = {str(value).lower() for value in available_actions.get("clickables", ())}
    # The native environment deliberately ignores click[search], even when the
    # search button is rendered as a clickable. Do not label that no-op valid.
    valid = action_arg != "search" and action_arg in clickables
    return valid, None if valid else "unavailable_click"

# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        env_kwargs = dict(env_kwargs)
        backend = env_kwargs.pop('backend', 'legacy')
        env_kwargs.pop('catalog_settings', None)
        env_kwargs['seed'] = seed
        if backend == 'centralized':
            from .remote_text_env import RemoteWebAgentTextEnv
            self.env = RemoteWebAgentTextEnv(**env_kwargs)
        elif backend == 'legacy':
            from .catalog_data import native_imports
            native_imports()
            from web_agent_site.envs import WebAgentTextEnv
            self.env = WebAgentTextEnv(**env_kwargs)
        else:
            raise ValueError(f'Unknown WebShop backend: {backend}')

    def page_type(self):
        env = self.env.unwrapped
        if hasattr(env, 'page_type'):
            return env.page_type
        return env.server.get_page_name(env.browser.current_url)

    def step(self, action):
        """Execute a step in the environment"""
        session = self.env.unwrapped.session
        available_actions = self.env.get_available_actions()
        action_admissible, invalid_action_reason = is_admissible_webshop_action(action, available_actions)
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info["is_action_admissible"] = action_admissible
        info["invalid_action_reason"] = invalid_action_reason
        info.setdefault('available_actions', self.env.get_available_actions())
        info['webshop_session_id'] = session
        info['webshop_task_id'] = getattr(self, '_rollout_task_id', None)
        info['task_score'] = reward
        info.setdefault('page_type', self.page_type())

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        else:
            info['won'] = False
            reward = 0

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        self._rollout_task_id = idx
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info.setdefault('available_actions', self.env.get_available_actions())
        info.setdefault('page_type', self.page_type())
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def get_goal_count(self):
        env = self.env.unwrapped
        return env.goal_count if hasattr(env, "goal_count") else len(env.server.goals)
    
    def diagnostics(self):
        import os
        import pickle
        import sys
        import psutil
        env = self.env.unwrapped
        return dict(pid=os.getpid(), rss_bytes=psutil.Process().memory_info().rss,
                    owns_catalog=hasattr(env, 'server'), has_lucene='jnius' in sys.modules,
                    episode_bytes=len(pickle.dumps(env.export_episode())) if hasattr(env, 'export_episode') else None)

    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        worker_class=None,
        owns_catalog_service=False,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self._closed = False
        self._workers = []
        self._owns_catalog_service = owns_catalog_service
        self._catalog_service = (env_kwargs or {}).get("catalog_service")
        settings = (env_kwargs or {}).get("catalog_settings", {})
        self._batch_requests = settings.get("batch_requests", True)
        self._catalog_batch_size = settings.get("max_batch_size", 256)
        self._catalog_timeout = settings.get("request_timeout_s", 120)
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # -------------------------- Ray actors setup --------------------------
        env_worker = ray.remote(**resources_per_worker)(worker_class or WebshopWorker)
        self._workers = []
        try:
            for i in range(self.num_processes):
                worker_seed = self._env_kwargs["validation"]["seed"] if self._env_kwargs.get("goal_split") == "validation" else seed + (i // self.group_n)
                worker = env_worker.remote(worker_seed, self._env_kwargs)
                self._workers.append(worker)

            # Fetch a scalar only; never transfer the complete goal set.
            counts = ray.get([worker.get_goal_count.remote() for worker in self._workers], timeout=settings.get("startup_timeout_s", 1800))
            goal_count = counts[0]

        except BaseException:
            # Cleanup is best-effort during failed startup. A busy worker may not
            # answer ``close`` before the timeout; do not hide the root failure
            # with a secondary GetTimeoutError from cleanup.
            try:
                self.close()
            except BaseException:
                pass
            raise

        if self._env_kwargs.get("validation"):
            self.goal_idxs = range(goal_count)
        elif not self.is_train:
            self.goal_idxs = range(min(500, goal_count))
        else:
            self.goal_idxs = range(500, goal_count)
            
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        self._closed = True
        try:
            if self._workers:
                ray.get([worker.close.remote() for worker in self._workers], timeout=30)
        finally:
            for worker in self._workers:
                ray.kill(worker, no_restart=True)
            if self._owns_catalog_service and self._catalog_service is not None:
                try:
                    from .catalog_service import CatalogRequest
                    ray.get(self._catalog_service.call.remote(CatalogRequest("flush_metrics")), timeout=10)
                finally:
                    ray.kill(self._catalog_service, no_restart=True)

    # No destructor RPC: trainer owns explicit, idempotent cleanup.


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    worker_class=None,
    owns_catalog_service=False,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        worker_class=worker_class,
        owns_catalog_service=owns_catalog_service,
    )
