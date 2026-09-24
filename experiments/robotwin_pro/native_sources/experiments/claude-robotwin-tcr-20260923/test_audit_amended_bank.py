import json

import pytest

import audit_capture as audit
import capture_contract as contract


def bank(tmp_path):
    protocol = contract.read(audit.PROTOCOL)
    for job in protocol["jobs"]:
        job["output"] = str(contract.CAPTURE_ROOT / "capture-v2/jobs" / job["id"])
    receipts = []
    for job in protocol["jobs"]:
        for task in job["tasks"]:
            receipts.append({"job": job["id"], "task": task["task"],
                             "task_index": task["task_index"], "old_seed": task["seed"],
                             "seed": task["seed"], "counter": task["derivation_counter"],
                             "status": "reset_valid", "observation_sha256": "a" * 64})
    path = tmp_path / "probes.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in receipts))
    protocol["reset_preflight"] = {
        "status": "all_180_simulator_resets_valid_before_policy_inference",
        "source_protocol": str(audit.PROTOCOL),
        "source_protocol_sha256": contract.sha256_file(audit.PROTOCOL),
        "probe_receipts": contract.bind(path),
        "success_values_read": False, "policy_inference": False,
    }
    return protocol, path


def test_independent_auditor_checks_full_reset_bank(tmp_path):
    protocol, _ = bank(tmp_path)
    audit.audit_amended_bank(protocol, tmp_path / "candidate-protocol.json")


def test_independent_auditor_rejects_a_hash_bound_partial_bank(tmp_path):
    protocol, path = bank(tmp_path)
    path.write_text(path.read_text().splitlines()[0] + "\n")
    protocol["reset_preflight"]["probe_receipts"] = contract.bind(path)
    with pytest.raises(ValueError, match="all 180"):
        audit.audit_amended_bank(protocol, tmp_path / "candidate-protocol.json")


def test_independent_auditor_rejects_one_timeout_as_a_seed_replacement(tmp_path):
    protocol, path = bank(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    job = protocol["jobs"][0]
    task = job["tasks"][0]
    old_seed = task["seed"]
    new_seed = contract.uint31(contract.SEED_NAMESPACE, "reset", job["repeat"],
                               job["pool"], task["task_index"], 1)
    task["seed"] = new_seed
    task["derivation_counter"] = 1
    rows[0]["seed"] = new_seed
    rows[0]["counter"] = 1
    timeout = {"job": job["id"], "task_index": task["task_index"],
               "seed": old_seed, "counter": 0, "status": "simulator_reset_timeout"}
    path.write_text(json.dumps(timeout) + "\n" + "".join(json.dumps(row) + "\n" for row in rows))
    protocol["reset_preflight"]["probe_receipts"] = contract.bind(path)
    protocol["reset_preflight"]["timeout_confirmed_candidates"] = 1
    with pytest.raises(ValueError, match="without two confirmations"):
        audit.audit_amended_bank(protocol, tmp_path / "candidate-protocol.json")
