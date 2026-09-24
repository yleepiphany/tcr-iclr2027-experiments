#!/usr/bin/env python3
"""Strict partial/final audit for the frozen RoboTwin TCR A/B capture."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from safetensors import safe_open


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
PROTOCOL = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
RUNNER = HERE / "run_capture.py"
PYTHON = RUNTIME / "envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python"
sys.path.insert(0, str(HERE))
import capture_contract as contract  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_amended_bank(protocol: dict[str, Any], protocol_path: Path) -> None:
    if protocol_path == PROTOCOL.resolve():
        return
    gate = protocol.get("reset_preflight") or {}
    require(gate.get("status") == "all_180_simulator_resets_valid_before_policy_inference",
            "Amended bank lacks completed reset-only gate")
    require(gate.get("success_values_read") is False and gate.get("policy_inference") is False,
            "Amended bank was selected using policy outcomes")
    require(gate.get("source_protocol_sha256") == contract.sha256_file(PROTOCOL)
            and gate.get("source_protocol") == str(PROTOCOL), "Amended bank source differs")
    binding = gate.get("probe_receipts") or {}
    require(contract.bind(Path(binding.get("path", "/nonexistent"))) == binding,
            "Amended reset probe receipts changed")
    old = contract.read(PROTOCOL)
    old_rows = {(job["id"], row["task_index"]): (job, row)
                for job in old["jobs"] for row in job["tasks"]}
    new_rows = {(job["id"], row["task_index"]): (job, row)
                for job in protocol["jobs"] for row in job["tasks"]}
    require(len(old_rows) == 180 and set(new_rows) == set(old_rows),
            "Amended bank task partition differs")
    valid = {}
    timeouts: dict[tuple[str, int, int, int], int] = {}
    for line in Path(binding["path"]).read_text().splitlines():
        event = json.loads(line)
        require(not any(key in event for key in ("success", "reward", "action")),
                "Reset probe recorded policy outcomes")
        if event.get("status") != "reset_valid":
            require(event.get("status") in {"simulator_unstable", "simulator_reset_timeout"},
                    "Unknown reset probe event")
            if event["status"] == "simulator_reset_timeout":
                key = (event.get("job"), event.get("task_index"),
                       event.get("counter"), event.get("seed"))
                require(key[:2] in old_rows, "Timeout reset identity is unknown")
                timeouts[key] = timeouts.get(key, 0) + 1
            continue
        key = (event.get("job"), event.get("task_index"))
        require(key in old_rows and key not in valid, "Duplicate or unknown valid reset")
        previous_job, previous = old_rows[key]
        new_job, current = new_rows[key]
        require(event.get("task") == previous["task"] == current["task"],
                "Reset probe task differs")
        require(event.get("old_seed") == previous["seed"] and event.get("seed") == current["seed"]
                and event.get("counter") == current["derivation_counter"],
                "Reset probe old/new seed differs")
        counter = event["counter"]
        require(type(counter) is int and counter >= previous["derivation_counter"],
                "Reset derivation counter differs")
        expected_seed = (previous["seed"] if counter == previous["derivation_counter"]
                         else contract.uint31(contract.SEED_NAMESPACE, "reset", new_job["repeat"],
                                              new_job["pool"], current["task_index"], counter))
        require(event["seed"] == expected_seed, "Reset seed was not deterministically derived")
        digest = event.get("observation_sha256")
        require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Reset-valid observation hash is missing")
        valid[key] = event
    require(len(valid) == 180, "Reset-only gate lacks all 180 valid identities")
    confirmed = 0
    for (job_id, task_index, counter, seed), count in timeouts.items():
        accepted = valid[(job_id, task_index)]
        require(type(counter) is int and counter <= accepted["counter"]
                and (counter == accepted["counter"] or count >= 2),
                "Timeout seed was skipped without two confirmations")
        previous_job, previous = old_rows[(job_id, task_index)]
        expected = (previous["seed"] if counter == previous["derivation_counter"]
                    else contract.uint31(contract.SEED_NAMESPACE, "reset", previous_job["repeat"],
                                         previous_job["pool"], task_index, counter))
        require(seed == expected, "Timeout seed differs from namespace")
        confirmed += int(counter < accepted["counter"])
    require(gate.get("timeout_confirmed_candidates", 0) == confirmed,
            "Confirmed timeout count differs")
    require(len({(row["task_index"], row["seed"]) for job in protocol["jobs"] for row in job["tasks"]}) == 180,
            "Amended reset bank has duplicate task/seed identities")
    output_root = contract.CAPTURE_ROOT / "capture-v2/jobs"
    require(all(Path(job["output"]) == output_root / job["id"] for job in protocol["jobs"]),
            "Amended capture outputs are not isolated")


def audit_plan(run: Path, protocol: dict[str, Any], protocol_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    plan = contract.read(run / "plan.json")
    expected = {row["id"]: row for row in protocol["jobs"]}
    require(plan.get("schema") == "robotwin_tcr_capture_queue_v1", "Wrong queue schema")
    require(plan.get("mode") == "formal", "Capture queue is not formal mode")
    require(plan.get("protocol") == str(protocol_path.resolve()), "Queue protocol path differs")
    require(plan.get("protocol_sha256") == contract.sha256_file(protocol_path), "Protocol hash differs")
    require(plan.get("runner") == str(RUNNER.resolve()), "Queue runner path differs")
    require(plan.get("runner_sha256") == contract.sha256_file(RUNNER), "Runner changed after freeze")
    require(plan.get("no_retry") is True, "Queue permits retry")
    require(plan.get("no_score_based_scheduling_or_stopping") is True, "Queue permits score scheduling")
    require(plan.get("openvla_formal_remains_cancelled") is True, "OpenVLA cancellation guard missing")
    require(plan.get("max_workers") == 2 and plan.get("reserve_mib") == 8192, "Worker/resource contract differs")
    require(plan.get("min_free_mib") == 32768, "Admission threshold differs")
    jobs = plan.get("jobs") or []
    require(len(jobs) == 18 and {row.get("id") for row in jobs} == set(expected), "Job partition differs")
    for row in jobs:
        frozen = expected[row["id"]]
        require(row.get("output") == frozen["output"], f"{row['id']} output differs")
        require(row.get("expected_episodes") == 10, f"{row['id']} episode coverage differs")
        require(row.get("expected_requests") == 50, f"{row['id']} request coverage differs")
        require(row.get("expected_flow_rows") == 150, f"{row['id']} flow coverage differs")
        command = row.get("command")
        expected_command = [str(PYTHON), "-u", str(RUNNER), "run", "--protocol", str(protocol_path),
                            "--job-id", row["id"], "--output", frozen["output"]]
        require(command == expected_command, f"{row['id']} command differs")
    return plan, expected


def audit_artifact(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output"])
    exit_path = output.with_suffix(".exit.json")
    receipt = contract.read(exit_path)
    require(receipt.get("outcome") == "success" and receipt.get("returncode") == 0,
            f"{job['id']} did not exit successfully")
    complete_path = output / "complete.json"
    complete = contract.read(complete_path)
    expected_complete = {
        "status": "capture_job_complete", "formal_result": False, "job_id": job["id"],
        "smoke": False, "episodes": 10, "requests": 50, "flow_rows": 150,
        "success_values_not_used_for_acceptance_or_scheduling": True,
    }
    for key, value in expected_complete.items():
        require(complete.get(key) == value, f"{job['id']} completion differs at {key}")
    require(receipt.get("artifacts", {}).get("complete_sha256") == contract.sha256_file(complete_path),
            f"{job['id']} complete hash differs")
    manifest_path = output / "replay.json"
    tensor_path = output / "replay.safetensors"
    require(contract.bind(manifest_path) == complete.get("replay_manifest"), f"{job['id']} manifest binding differs")
    require(contract.bind(tensor_path) == complete.get("replay_tensor"), f"{job['id']} tensor binding differs")
    manifest = contract.read(manifest_path)
    require(manifest.get("sample_count") == 150 and len(manifest.get("samples") or []) == 150,
            f"{job['id']} sample coverage differs")
    fixed = {
        "robotwin_capture_version": 1, "source_kind": "physical_simulator_expert_execution",
        "benchmark": "RoboTwin 2.0", "group": job["group"], "repeat": job["repeat"],
        "pool": job["pool"], "flow_seed": job["flow_seed"], "success_filtering": False,
        "demonstration_actions_used": False, "closed_loop_expert_execution": True,
        "request_selection": "full-trajectory quantiles 0,25,50,75,100 percent",
    }
    for key, value in fixed.items():
        require(manifest.get(key) == value, f"{job['id']} manifest differs at {key}")
    reset_keys = [{"task": row["task"], "task_index": row["task_index"], "seed": row["seed"]}
                  for row in job["tasks"]]
    require(manifest.get("task_reset_keys") == reset_keys, f"{job['id']} reset identities differ")
    provenance = manifest.get("selected_requests") or {}
    require(len(provenance) == 10, f"{job['id']} request provenance coverage differs")
    expected_tasks = {(row["task"], row["task_index"], row["seed"]) for row in job["tasks"]}
    observed_tasks = set()
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(manifest["samples"]):
        require(row.get("index") == index, f"{job['id']} sample indices are not contiguous")
        key = (row.get("task"), row.get("task_index"), row.get("seed"))
        observed_tasks.add(key)
        grouped[key].append(row)
        require(row.get("group") == job["group"] and row.get("repeat") == job["repeat"]
                and row.get("pool") == job["pool"], f"{job['id']} sample arm identity differs")
    require(observed_tasks == expected_tasks, f"{job['id']} task/reset sample identities differ")
    for key, rows in grouped.items():
        require(len(rows) == 15, f"{job['id']} task {key} has {len(rows)} rows")
        by_slot = defaultdict(list)
        for row in rows:
            by_slot[row.get("selected_request_slot")].append(row)
        require(set(by_slot) == set(range(5)), f"{job['id']} task {key} slots differ")
        selected = []
        for slot in range(5):
            slot_rows = sorted(by_slot[slot], key=lambda row: row["flow_index"])
            require([row["flow_index"] for row in slot_rows] == [0, 5, 9],
                    f"{job['id']} task {key} flow rows differ")
            require(len({row["request_index"] for row in slot_rows}) == 1,
                    f"{job['id']} task {key} request identity differs within slot")
            selected.append(slot_rows[0]["request_index"])
        matches = [value for value in provenance.values()
                   if (value.get("task"), value.get("task_index"), value.get("seed")) == key]
        require(len(matches) == 1, f"{job['id']} task {key} provenance differs")
        available = matches[0].get("available_request_count")
        require(selected == contract.quantile_request_indices(available),
                f"{job['id']} task {key} quantile selection differs")
        require(matches[0].get("selected_request_indices") == selected,
                f"{job['id']} task {key} recorded quantiles differ")
    require(len(complete.get("closed_loop_rows") or []) == 10, f"{job['id']} closed-loop row coverage differs")
    for binding in complete["closed_loop_rows"]:
        require(contract.bind(Path(binding["path"])) == binding, f"{job['id']} closed-loop binding differs")
    noise = manifest.get("native_noise_sha256") or []
    require(noise and all(isinstance(value, str) and len(value) == 64 for value in noise),
            f"{job['id']} native noise hashes missing")
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        names = list(handle.keys())
    suffix_counts = Counter(name.split(".", 1)[0] for name in names)
    require(set(suffix_counts) == {f"sample_{index:03d}" for index in range(150)},
            f"{job['id']} tensor sample keys differ")
    require(len(set(suffix_counts.values())) == 1, f"{job['id']} tensor field coverage differs across samples")
    return {
        "id": job["id"], "episodes": 10, "requests": 50, "flow_rows": 150,
        "manifest_sha256": contract.sha256_file(manifest_path),
        "tensor_sha256": contract.sha256_file(tensor_path), "tensor_keys": len(names),
    }


def audit(run: Path, require_complete: bool, protocol_path: Path = PROTOCOL) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol = contract.read(protocol_path)
    require(protocol.get("schema") == "robotwin_tcr_native_ab_capture_v1", "Wrong protocol schema")
    require(protocol.get("episodes") == 180 and protocol.get("requests") == 900
            and protocol.get("flow_rows") == 2700, "Protocol coverage differs")
    audit_amended_bank(protocol, protocol_path)
    plan, jobs = audit_plan(run, protocol, protocol_path)
    accepted = []
    for job_id in [row["id"] for row in plan["jobs"]]:
        job = jobs[job_id]
        exit_path = Path(job["output"]).with_suffix(".exit.json")
        if exit_path.exists():
            accepted.append(audit_artifact(job))
    ended_path = run / "queue-ended.json"
    if require_complete:
        require(ended_path.exists(), "Final audit requires queue-ended.json")
        ended = contract.read(ended_path)
        require(ended.get("status") == "complete" and ended.get("stopped") is False,
                "Capture queue did not end cleanly")
        require(ended.get("completed_jobs") == 18 and len(accepted) == 18,
                "Final capture coverage differs")
    report = {
        "schema": "robotwin_tcr_capture_audit_v1", "accepted": True,
        "scope": "final" if require_complete else "partial", "formal_result": False,
        "run": str(run.resolve()), "protocol": str(protocol_path),
        "protocol_sha256": contract.sha256_file(protocol_path),
        "runner_sha256": contract.sha256_file(RUNNER), "accepted_jobs": len(accepted),
        "expected_jobs": 18, "accepted_episodes": 10 * len(accepted),
        "accepted_requests": 50 * len(accepted), "accepted_flow_rows": 150 * len(accepted),
        "success_values_read": False, "jobs": accepted,
        "audited_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    report = audit(args.run.resolve(), args.final, args.protocol)
    contract.save(args.output.resolve(), report)
    print(json.dumps({key: report[key] for key in ("accepted", "scope", "accepted_jobs",
                                                    "accepted_episodes", "accepted_flow_rows")}, indent=2))


if __name__ == "__main__":
    main()
