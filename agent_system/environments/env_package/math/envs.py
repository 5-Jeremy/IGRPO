"""Vectorized math environment with a Python calculation tool."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import re

from agent_system.environments.env_package.math.python_tool import PythonTool
from agent_system.environments.env_package.math.reward import score_answer


PYTHON = re.compile(r"^<python>(.*?)</python>$", re.DOTALL | re.IGNORECASE)
ANSWER = re.compile(r"^<answer>(.*?)</answer>$", re.DOTALL | re.IGNORECASE)


class MathEnv:
    def __init__(self, tool, max_steps):
        self.tool = tool
        self.max_steps = max_steps
        self.reset({"question": "", "ground_truth": {"target": ""}, "data_source": "math"})

    def reset(self, kwargs):
        self.question = kwargs["question"]
        self.ground_truth = kwargs["ground_truth"]
        self.data_source = kwargs.get("data_source", "math")
        self.turns = 0
        self.done = False
        self.won = False
        self.used_python = False
        return self.question, {"data_source": self.data_source}

    def step(self, action):
        if self.done:
            return "", 0.0, True, {
                "data_source": self.data_source,
                "won": self.won,
                "used_python": self.used_python,
                "tool_calling": False,
                "postprocessed_action": action,
            }
        self.turns += 1
        answer = ANSWER.fullmatch(action)
        python = PYTHON.fullmatch(action)
        self.done = self.done or answer is not None or self.turns >= self.max_steps
        observation = ""
        tool_calling = False
        if not self.done and python:
            observation = f"<result>{self.tool.execute(python.group(1))}</result>"
            tool_calling = True
            self.used_python = True
        reward = score_answer(action, self.ground_truth, self.used_python) if self.done else 0.0
        self.won = bool(self.done and reward >= 1.0)
        return observation, reward, self.done, {
            "data_source": self.data_source,
            "won": self.won,
            "used_python": self.used_python,
            "tool_calling": tool_calling,
            "tool_group": "PythonTool" if tool_calling else None,
            "tool_name": "python" if tool_calling else None,
            "postprocessed_action": action,
        }

    def fork_from(self, other):
        self.question = other.question
        self.ground_truth = deepcopy(other.ground_truth)
        self.data_source = other.data_source
        self.turns = other.turns
        self.done = other.done
        self.won = other.won
        self.used_python = other.used_python


class MathVectorEnv:
    def __init__(self, env_num, group_n, env_config):
        self.batch_size = env_num * group_n
        cfg = env_config.python
        self.tool = PythonTool(
            timeout=cfg.timeout,
            max_output_chars=cfg.max_output_chars,
            max_code_chars=cfg.max_code_chars,
            memory_mb=cfg.memory_mb,
        )
        self.envs = [MathEnv(self.tool, env_config.max_steps) for _ in range(self.batch_size)]
        self.executor = ThreadPoolExecutor(max_workers=min(self.batch_size, 64))

    def reset(self, kwargs):
        if len(kwargs) > self.batch_size:
            raise ValueError("More math examples than environments")
        results = list(self.executor.map(lambda pair: pair[0].reset(pair[1]), zip(self.envs, kwargs)))
        observations, infos = zip(*results)
        return list(observations), list(infos)

    def step(self, actions):
        if len(actions) > self.batch_size:
            raise ValueError("More math actions than environments")
        results = list(self.executor.map(lambda pair: pair[0].step(pair[1]), zip(self.envs, actions)))
        observations, rewards, dones, infos = zip(*results)
        return list(observations), list(rewards), list(dones), list(infos)

    def fork_from(self, dest_index, src_index):
        self.envs[dest_index].fork_from(self.envs[src_index])

    def close(self):
        self.executor.shutdown(wait=True)


def build_math_envs(seed=0, env_num=1, group_n=1, is_train=True, env_config=None):
    return MathVectorEnv(env_num, group_n, env_config)
