from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from copy import deepcopy
import numpy as np
import uuid
from agent_system.environments import EnvironmentManagerBase

def _to_log_prob(values: np.ndarray, prob_diff_mode: bool, prob_floor: float) -> np.ndarray:
    """Ground-truth answer log-probability of a node, clamped away from -inf.

    The rollout stores ``exp(avg answer log-prob)`` in ``info_gain_sum`` when
    ``algorithm.igrpo.prob_diff_mode`` is on, and the raw average log-prob otherwise.
    """
    log_floor = float(np.log(prob_floor))
    values = np.asarray(values, dtype=np.float64)
    if prob_diff_mode:
        values = np.nan_to_num(values, nan=prob_floor, posinf=1.0, neginf=prob_floor)
        return np.log(np.clip(values, prob_floor, 1.0))
    values = np.nan_to_num(values, nan=log_floor, posinf=0.0, neginf=log_floor)
    return np.clip(values, log_floor, 0.0)


def compute_log_ratio_expand_val(info_gain_sum: np.ndarray,
                                 parent_info_gain_sum: np.ndarray,
                                 is_root_parent: np.ndarray,
                                 prob_diff_mode: bool = True,
                                 prob_floor: float = 1e-6,
                                 dtype=np.float32) -> np.ndarray:
    """Branching score ``log p_gt(child) - log p_gt(parent)``.

    Fed to :meth:`TrajectoryNodeStateManagement.compute_expand_prob`, whose softmax at
    gamma=1 then draws a frontier node in proportion to the hindsight ratio
    p_gt(child) / p_gt(parent). The default score ``(p_gt + info_gain) / 2`` instead
    exponentiates a quantity already in [0, 1], so sibling logits differ by hundredths
    and the draw comes out near-uniform; taking the log first is what lets the softmax
    separate them.

    Every value is clamped into ``[prob_floor, 1]`` before the log. Without that a
    hopeless branch whose p_gt underflows to 0, or a root whose parent is recorded as
    0.0, gives -inf and the softmax returns nan. A root's parent is treated as
    ``prob_floor`` ("nothing retrieved yet"), which is identical for every node of a
    group at step 0 and therefore cancels in the softmax, reducing that step to a draw
    in proportion to p_gt.
    """
    if not 0.0 < prob_floor <= 1.0:
        raise ValueError(f"expand_prob_floor must be in (0, 1], got {prob_floor}")

    log_child = _to_log_prob(info_gain_sum, prob_diff_mode, prob_floor)
    log_parent = _to_log_prob(parent_info_gain_sum, prob_diff_mode, prob_floor)
    log_parent = np.where(np.asarray(is_root_parent, dtype=bool), float(np.log(prob_floor)), log_parent)

    val = log_child - log_parent
    assert np.all(np.isfinite(val)), "log_ratio expand score is not finite"
    return val.astype(dtype)


@dataclass
class TrajectoryNodeStateManagement:
    batch_size: int
    group_size: int

    # information of parents
    envs: EnvironmentManagerBase
    
    obs: Dict[str, List[Any]] = field(default_factory=dict)

    uid_batch: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    node_uid: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    tool_callings: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # those parents who will expand in the next round
    active_nodes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    # which gen_batch_index does the node refer to
    original_gen_batch_index: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    
    def __init__(self, batch_size: int, **kwargs):
        self.batch_size = batch_size

        self.envs = kwargs.get("envs")
        
        self.obs = kwargs.get("obs", {})
        
        self.node_uid = np.array(["root" for _ in range(batch_size)], dtype=object)
        self.tool_callings = np.zeros(batch_size, dtype=np.float32)
        
        self.active_nodes = np.ones(batch_size, dtype=np.bool_)
        self.original_gen_batch_index = np.arange(self.batch_size, dtype=np.int32)

    def assign_group_uids(self, group_size: int) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {group_size}")
        self.group_size = group_size

        uid_batch = []
        for i in range(self.batch_size):
            if i % group_size == 0:
                uid = str(uuid.uuid4())
            uid_batch.append(uid)
        self.uid_batch = np.array(uid_batch, dtype=object)
    
    def fork_from(self, dest_index: int, src_index: int):
        # we only allow an inactive node to fork from an active node
        assert src_index < self.batch_size and 0 <= src_index
        assert self.active_nodes[src_index]
        assert dest_index < self.batch_size and 0 <= dest_index
        assert not self.active_nodes[dest_index]
        
        self.envs.fork_from(dest_index, src_index)        
        for k, v in self.obs.items():
            if v is not None:
                v[dest_index] = v[src_index]
        self.uid_batch[dest_index] = self.uid_batch[src_index]
        self.node_uid[dest_index] = self.node_uid[src_index]
        self.tool_callings[dest_index] = self.tool_callings[src_index]
        
        self.active_nodes[dest_index] = True
        self.original_gen_batch_index[dest_index] = self.original_gen_batch_index[src_index]
        
    def update_node_uid(self) -> np.ndarray:
        self.node_uid = np.array([str(uuid.uuid4()) for _ in range(self.batch_size)], dtype=object)
        return self.node_uid
    
    def deactivate(self, mask: np.ndarray) -> None:
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (self.batch_size,)

        self.active_nodes[mask] = False
    
    def compute_expand_prob(self, val: np.ndarray, gamma: float) -> np.ndarray:
        prob = np.zeros(self.batch_size, dtype=val.dtype)
        active_group_uid = np.unique(self.uid_batch[self.active_nodes])
        for group_uid in active_group_uid:
            group_mask = self.active_nodes & (self.uid_batch == group_uid)
            index = np.where(group_mask)[0]
            assert len(index) > 0, f"there should be at least one active node for group {group_uid}, but got none"
            group_logits = gamma * val[index]
            group_logits = group_logits - np.max(group_logits) # more stable
            exp_logits = np.exp(group_logits)
            prob[index] = exp_logits / np.sum(exp_logits)
        
        return prob

    def _sample_expand_num_with_cap(
        self,
        probs: np.ndarray,
        total_expand: int,
        max_per_node: int,
    ) -> np.ndarray:
        """
        probs: shape (n,), sum to 1
        total_expand: total number of expansions to allocate
        max_per_node: cap for each position

        return:
            expand_num: shape (n,), integer allocation
        """
        probs = np.asarray(probs, dtype=np.float64)
        n = len(probs)

        assert probs.ndim == 1
        assert n > 0
        assert total_expand > 0
        assert max_per_node > 0
        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-5, "expand_prob for a group does not sum to 1.0"

        expand_num = np.zeros(n, dtype=np.int32)

        max_total_capacity = n * max_per_node
        actual_expand = min(total_expand, max_total_capacity)

        for _ in range(actual_expand):
            available_mask = expand_num < max_per_node
            if not np.any(available_mask):
                break

            current_probs = probs.copy()
            current_probs[~available_mask] = 0.0

            prob_sum = current_probs.sum()
            if prob_sum <= 0.0:
                # every node still under the cap has underflowed to zero probability,
                # so renormalising would be 0/0. Fall back to a uniform draw over them.
                current_probs = available_mask.astype(np.float64)
                prob_sum = current_probs.sum()
            current_probs /= prob_sum
            chosen = np.random.choice(n, p=current_probs)

            expand_num[chosen] += 1

        return expand_num

    def get_expand_num(self, expand_prob: np.ndarray, max_traj_to_expand_per_node: int, expand_mode: str = 'full') -> np.ndarray:
        expand_num = np.zeros(self.batch_size, dtype=np.int32)

        max_traj_to_expand_per_node = self.group_size if max_traj_to_expand_per_node <= 0 else max_traj_to_expand_per_node
        active_group_uid = np.unique(self.uid_batch[self.active_nodes])
        for group_uid in active_group_uid:
            group_mask = self.uid_batch == group_uid
            group_mask_active = self.active_nodes & group_mask
            active_index = np.where(group_mask_active)[0]
            assert len(active_index) > 0
            if expand_mode == 'full':
                expand_num_total = self.group_size
            elif expand_mode == 'low':
                expand_num_total = len(active_index)
            elif expand_mode == 'mid':
                expand_num_total = (self.group_size + len(active_index)) // 2
            else:
                raise ValueError(f"Invalid expand_mode: {expand_mode}, expected one of ['full', 'low', 'mid']")
            group_probs = expand_prob[active_index]
            expand_num_group = self._sample_expand_num_with_cap(group_probs, expand_num_total, max_traj_to_expand_per_node)
            expand_num[active_index] = expand_num_group
        
        return expand_num

    def expand(self, expand_num: np.ndarray):
            assert np.array_equal(expand_num > 0, self.active_nodes)
            active_index = np.where(self.active_nodes)[0]
            expand_num_active = expand_num[active_index]
            dest_index = 0
            for src_index, num in zip(active_index, expand_num_active):
                while num > 1:
                    # let other inactive node fork from this node
                    assert dest_index < self.batch_size, "cannot find enough inactive nodes to fork"
                    if not self.active_nodes[dest_index]:
                        self.fork_from(dest_index, src_index)
                        num -= 1
                    else:
                        dest_index += 1