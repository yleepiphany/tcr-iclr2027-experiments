import json
import pytest

import capture_contract as contract
import run_capture_queue as queue


def amended(tmp_path):
    protocol = contract.read(queue.PROTOCOL)
    for job in protocol["jobs"]:
        job["output"] = str(contract.CAPTURE_ROOT / "capture-v2/jobs" / job["id"])
    receipt = tmp_path / "probes.jsonl"
    receipts = []
    for job in protocol["jobs"]:
        for row in job["tasks"]:
            receipts.append({
                "job": job["id"], "task": row["task"], "task_index": row["task_index"],
                "old_seed": row["seed"], "seed": row["seed"],
                "counter": row["derivation_counter"], "status": "reset_valid",
                "observation_sha256": "0" * 64, "checked_at": "2026-09-23T00:00:00Z",
                "phase": "simulator_reset_complete",
            })
    receipt.write_text("".join(json.dumps(row) + "\n" for row in receipts))
    instruction_path = tmp_path / "instruction-contract.json"
    contract.save(instruction_path, {
        "schema": "robotwin_exact_lazy_official_instruction_v1",
        "upstream_max_descriptions": 1_000_000,
        "upstream_wrapper": contract.bind(contract.WORK / "pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/envs/robotwin.py"),
        "upstream_generator": contract.bind(contract.WORK / "vla-merge-runtime/environments/robotwin2-0aee-open3d-optional-v2/description/utils/generate_episode_instructions.py"),
        "lazy_implementation": contract.bind(queue.HERE / "fast_official_instruction.py"),
    })
    protocol["reset_preflight"] = {
        "status": "all_180_simulator_resets_valid_before_policy_inference",
        "source_protocol": str(queue.PROTOCOL),
        "source_protocol_sha256": contract.sha256_file(queue.PROTOCOL),
        "probe_receipts": contract.bind(receipt),
        "instruction_contract": contract.bind(instruction_path),
    }
    path = tmp_path / "candidate-protocol.json"
    contract.save(path, protocol)
    return path, protocol


def test_amended_queue_uses_frozen_v2_protocol_and_isolated_outputs(tmp_path):
    path, protocol = amended(tmp_path)
    plan = queue.build_plan(tmp_path / "attempt-02", "formal", path)
    assert len(plan["jobs"]) == 18
    assert plan["protocol"] == str(path)
    assert plan["protocol_sha256"] == contract.sha256_file(path)
    assert plan["jobs"][0]["output"] == protocol["jobs"][0]["output"]
    assert plan["jobs"][0]["command"][5] == str(path)


def test_amended_queue_refuses_missing_preflight_and_old_outputs(tmp_path):
    path, protocol = amended(tmp_path)
    protocol.pop("reset_preflight")
    contract.save(tmp_path / "missing.json", protocol)
    with pytest.raises(ValueError, match="reset-only preflight"):
        queue.build_plan(tmp_path / "bad-attempt", "formal", tmp_path / "missing.json")
    protocol["reset_preflight"] = contract.read(path)["reset_preflight"]
    protocol["jobs"][0]["output"] = str(contract.CAPTURE_ROOT / "capture-v1/jobs" / protocol["jobs"][0]["id"])
    contract.save(tmp_path / "old-output.json", protocol)
    with pytest.raises(ValueError, match="new and isolated"):
        queue.build_plan(tmp_path / "bad-output", "formal", tmp_path / "old-output.json")


def test_amended_queue_rejects_a_hash_bound_but_incomplete_reset_scan(tmp_path):
    path, protocol = amended(tmp_path)
    receipt = tmp_path / "probes.jsonl"
    receipt.write_text(receipt.read_text().splitlines()[0] + "\n")
    protocol["reset_preflight"]["probe_receipts"] = contract.bind(receipt)
    contract.save(tmp_path / "short.json", protocol)
    with pytest.raises(ValueError, match="all 180"):
        queue.build_plan(tmp_path / "short-attempt", "formal", tmp_path / "short.json")


def test_timeout_seed_requires_two_reset_only_confirmations(tmp_path):
    _, protocol = amended(tmp_path)
    receipt = tmp_path / "probes.jsonl"
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    job = protocol["jobs"][0]
    task = job["tasks"][0]
    old_seed = task["seed"]
    replacement = contract.uint31(contract.SEED_NAMESPACE, "reset", job["repeat"],
                                   job["pool"], task["task_index"], 1)
    task["seed"] = replacement
    task["derivation_counter"] = 1
    rows[0]["seed"] = replacement
    rows[0]["counter"] = 1
    timeout = {"job": job["id"], "task_index": task["task_index"],
               "seed": old_seed, "counter": 0, "status": "simulator_reset_timeout",
               "phase": "simulator_reset_running"}
    protocol["reset_preflight"]["timeout_confirmed_candidates"] = 1
    receipt.write_text(json.dumps(timeout) + "\n" + "".join(json.dumps(row) + "\n" for row in rows))
    protocol["reset_preflight"]["probe_receipts"] = contract.bind(receipt)
    contract.save(tmp_path / "one-timeout.json", protocol)
    with pytest.raises(ValueError, match="without two confirmations"):
        queue.build_plan(tmp_path / "one-timeout-run", "formal", tmp_path / "one-timeout.json")
    receipt.write_text(json.dumps(timeout) + "\n" + receipt.read_text())
    protocol["reset_preflight"]["probe_receipts"] = contract.bind(receipt)
    contract.save(tmp_path / "two-timeouts.json", protocol)
    plan = queue.build_plan(tmp_path / "two-timeout-run", "formal", tmp_path / "two-timeouts.json")
    assert len(plan["jobs"]) == 18


def test_instruction_generation_timeout_cannot_advance_a_seed(tmp_path):
    _, protocol = amended(tmp_path)
    receipt = tmp_path / "probes.jsonl"
    rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    job = protocol["jobs"][0]
    task = job["tasks"][0]
    old_seed = task["seed"]
    replacement = contract.uint31(contract.SEED_NAMESPACE, "reset", job["repeat"],
                                   job["pool"], task["task_index"], 1)
    task["seed"], task["derivation_counter"] = replacement, 1
    rows[0]["seed"], rows[0]["counter"] = replacement, 1
    timeout = {"job": job["id"], "task_index": task["task_index"],
               "seed": old_seed, "counter": 0, "status": "simulator_reset_timeout",
               "phase": "instruction_entered_after_simulator_setup"}
    receipt.write_text(json.dumps(timeout) + "\n" + json.dumps(timeout) + "\n"
                       + "".join(json.dumps(row) + "\n" for row in rows))
    protocol["reset_preflight"]["probe_receipts"] = contract.bind(receipt)
    protocol["reset_preflight"]["timeout_confirmed_candidates"] = 1
    path = tmp_path / "bad-instruction-timeout.json"
    contract.save(path, protocol)
    with pytest.raises(ValueError, match="Instruction-generation timeout"):
        queue.build_plan(tmp_path / "bad-instruction-run", "formal", path)


def test_amended_capture_rejects_unbound_instruction_implementation(tmp_path):
    path, protocol = amended(tmp_path)
    instruction = tmp_path / "instruction-contract.json"
    value = contract.read(instruction)
    value["upstream_max_descriptions"] = 100
    contract.save(instruction, value)
    protocol["reset_preflight"]["instruction_contract"] = contract.bind(instruction)
    changed = tmp_path / "changed-instruction.json"
    contract.save(changed, protocol)
    with pytest.raises(ValueError, match="Exact lazy instruction implementation"):
        queue.build_plan(tmp_path / "changed-instruction-run", "formal", changed)
