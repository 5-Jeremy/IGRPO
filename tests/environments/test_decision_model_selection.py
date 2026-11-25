"""Selective questions preserve step identity and cap contributing trajectory turns."""

from types import SimpleNamespace

import pytest

from agent_system.decision_model_requests import DecisionModelRequestBuilder


def test_selective_questions_preserve_full_context_and_original_step_ids():
    builder = DecisionModelRequestBuilder(selective_requests=True)
    assert list(builder.build_request(["First."], "look")["questions"]) == ["step_1"]
    request = builder.build_request(["First.", "Second.", "Third."], "look")
    assert list(request["questions"]) == ["step_1", "step_2", "step_3"]
    assert "[1] First." in request["state"]
    assert "[1]: First." in request["questions"]["step_1"]["instructions"]
    assert "[2]: Second." in request["questions"]["step_2"]["instructions"]
    assert "[3]: Third." in request["questions"]["step_3"]["instructions"]


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("action", ["look", "Lima, Peru"])
@pytest.mark.parametrize("cap", [0, 1, 3, 10, 20])
def test_submission_quota_spans_rounds_and_counts_only_eligible_turns(selective, action, cap):
    builder = DecisionModelRequestBuilder(selective_requests=selective, max_turns_per_trajectory=cap)
    submitted = {}
    pending = SimpleNamespace(submit=lambda key, value: submitted.update({key: value}))
    rows = []
    expected = set()
    eligible_turns = [list(range(16)), [0, *range(4, 16)]]
    if selective:
        eligible_turns = [turns[:cap] for turns in eligible_turns]
    for turn in range(16):
        # B starts with one reasoning step, then has three ineligible turns.
        outputs = [f"<think>First. However, second. Wait, third.</think><action>{action}</action>"] * 2
        active, valid = [True, True], [True, True]
        if turn == 0:
            outputs[1] = f"<think>Only one.</think><action>{action}</action>"
        elif turn == 1:
            outputs[1] = f"<action>{action}</action>"
        elif turn == 2:
            valid[1] = False
        elif turn == 3:
            active[1] = False
        batch = SimpleNamespace(batch={}, non_tensor_batch={
            "traj_uid": ["a", "b"], "uid": ["shared", "shared"], "is_action_valid": valid,
        })
        builder.submit_turns(batch, outputs, active, None, pending)
        for index, trajectory in enumerate(["a", "b"]):
            turn_id = batch.non_tensor_batch["id"][index]
            if turn in eligible_turns[index]:
                expected.add(turn_id)
                questions = submitted[turn_id]["questions"]
                assert list(questions) == (["step_1"] if index == 1 and turn == 0 else ["step_1", "step_2", "step_3"])
            rows.append((turn_id, batch.non_tensor_batch["reasoning_trace"][index], trajectory, active[index]))
        assert set(submitted) == expected
    # Reconstruction has independent quotas, deduplicates padding, and is repeatable.
    padded_rows = [row for row in rows for _ in range(2)]
    batch = SimpleNamespace(non_tensor_batch=dict(zip(
        ["id", "reasoning_trace", "traj_uid", "active_masks"], zip(*padded_rows), strict=True)))
    assert builder.build_rollout_requests(batch) == submitted
    assert builder.build_rollout_requests(batch) == submitted
    # A fresh builder starts a fresh batch even if a standalone caller reuses IDs.
    assert DecisionModelRequestBuilder(
        selective_requests=selective, max_turns_per_trajectory=cap).build_rollout_requests(batch) == submitted


def test_default_cap_is_ten_turns():
    trace = {"steps": ["First.", "Second."], "action": "look"}
    batch = SimpleNamespace(non_tensor_batch={
        "id": list(range(11)), "traj_uid": ["trajectory"] * 11, "reasoning_trace": [trace] * 11,
    })
    assert list(DecisionModelRequestBuilder(selective_requests=True).build_rollout_requests(batch)) == list(range(10))


@pytest.mark.parametrize("cap", [-1, 1.5, "10", None, True, False])
def test_invalid_caps_are_rejected(cap):
    with pytest.raises(ValueError, match="max_turns_per_trajectory must be a nonnegative integer"):
        DecisionModelRequestBuilder(selective_requests=True, max_turns_per_trajectory=cap)


def test_selective_mode_requires_trajectory_identity_instead_of_using_group_id():
    batch = SimpleNamespace(non_tensor_batch={
        "id": ["turn"], "uid": ["group"],
        "reasoning_trace": [{"steps": ["First.", "Second."], "action": "look"}],
    })
    with pytest.raises(KeyError, match="traj_uid"):
        DecisionModelRequestBuilder(selective_requests=True).build_rollout_requests(batch)
