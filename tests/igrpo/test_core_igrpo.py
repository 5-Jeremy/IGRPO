import numpy as np
import pytest

from igrpo.core_igrpo import TrajectoryNodeStateManagement


def manager(*, group_size=4, active=(True, True, False, False)):
    state = TrajectoryNodeStateManagement(len(active))
    state.assign_group_uids(group_size)
    state.active_nodes = np.asarray(active, dtype=np.bool_)
    return state


def test_non_branchable_node_gets_one_continuation_and_no_siblings():
    state = manager()
    probabilities = np.asarray([0.99, 0.01, 0.0, 0.0])

    counts = state.get_expand_num(
        probabilities,
        max_traj_to_expand_per_node=4,
        expand_mode="full",
        branchable_mask=np.asarray([False, True, True, True]),
    )

    assert counts.tolist() == [1, 3, 0, 0]


def test_all_non_branchable_nodes_continue_without_filling_free_slots():
    state = manager()
    probabilities = np.asarray([0.5, 0.5, 0.0, 0.0])

    counts = state.get_expand_num(
        probabilities,
        max_traj_to_expand_per_node=4,
        expand_mode="full",
        branchable_mask=np.zeros(4, dtype=np.bool_),
    )

    assert counts.tolist() == [1, 1, 0, 0]


def test_zero_probability_branchable_nodes_still_receive_remaining_budget():
    state = manager()
    counts = state.get_expand_num(
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        max_traj_to_expand_per_node=4,
        expand_mode="full",
        branchable_mask=np.asarray([False, True, True, True]),
    )

    assert counts.tolist() == [1, 3, 0, 0]


def test_branchable_mask_must_match_batch_size():
    state = manager()
    with pytest.raises(ValueError, match="branchable_mask must have shape"):
        state.get_expand_num(
            np.asarray([0.5, 0.5, 0.0, 0.0]),
            max_traj_to_expand_per_node=4,
            branchable_mask=np.ones(2, dtype=np.bool_),
        )
