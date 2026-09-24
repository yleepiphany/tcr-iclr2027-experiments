from pathlib import Path

import pytest

import capture_contract as contract
from preflight_reset_bank import amend_protocol, candidate


def fixture():
    replacement = contract.uint31(contract.SEED_NAMESPACE, "reset", 1, "A", 8, 1)
    protocol = {"jobs": [{"id": "r01-A-coordination", "repeat": 1, "pool": "A",
                          "output": "/old/job", "tasks": [
                              {"task": "first", "task_index": 8, "seed": 11, "derivation_counter": 0},
                              {"task": "second", "task_index": 9, "seed": 22, "derivation_counter": 0},
                          ]}]}
    checks = [
        {"job": "r01-A-coordination", "task": "first", "task_index": 8,
         "old_seed": 11, "seed": replacement, "counter": 1, "status": "reset_valid"},
        {"job": "r01-A-coordination", "task": "second", "task_index": 9,
         "old_seed": 22, "seed": 22, "counter": 0, "status": "reset_valid"},
    ]
    return protocol, checks


def test_amendment_changes_only_reset_proven_invalid_and_rebases_outputs():
    protocol, checks = fixture()
    amended = amend_protocol(protocol, checks, {(8, 44)}, Path("/new/capture-v2"))
    assert [row["seed"] for row in amended["jobs"][0]["tasks"]] == [checks[0]["seed"], 22]
    assert [row["seed"] for row in protocol["jobs"][0]["tasks"]] == [11, 22]
    assert amended["jobs"][0]["output"] == "/new/capture-v2/jobs/r01-A-coordination"
    assert amended["reset_selection_uses_only_simulator_validity"] is True


def test_amendment_rejects_missing_or_colliding_reset_evidence():
    protocol, checks = fixture()
    with pytest.raises(ValueError, match="exactly cover"):
        amend_protocol(protocol, checks[:1], set(), Path("/new"))
    with pytest.raises(ValueError, match="collision"):
        amend_protocol(protocol, checks, {(8, checks[0]["seed"])}, Path("/new"))
    checks[0]["status"] = "simulator_unstable"
    with pytest.raises(ValueError, match="did not pass"):
        amend_protocol(protocol, checks, set(), Path("/new"))


def test_amendment_rejects_an_unproven_hand_chosen_seed():
    protocol, checks = fixture()
    checks[0]["seed"] += 1
    with pytest.raises(ValueError, match="frozen namespace"):
        amend_protocol(protocol, checks, set(), Path("/new"))


def test_candidate_derivation_preserves_frozen_seed_then_advances_namespace():
    row = {"task_index": 28, "seed": 407709750, "derivation_counter": 0}
    assert candidate(row, 1, "A", 0) == 407709750
    assert candidate(row, 1, "A", 1) == contract.uint31(
        contract.SEED_NAMESPACE, "reset", 1, "A", 28, 1,
    )
