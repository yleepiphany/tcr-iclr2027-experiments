#!/usr/bin/env python3
"""Dynamic no-retry queue for RoboTwin TCR native A/B collection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
PYTHON = RUNTIME / "envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python"
RUNNER = HERE / "run_capture.py"
PROTOCOL = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
FLOCK_DIR = VLA / "experiments/claude-firstpass-cause-20260920"
SAFETY_DIR = VLA / "experiments/claude-balanced-pass2-20260919"
sys.path[:0] = [str(HERE), str(FLOCK_DIR), str(SAFETY_DIR)]
import capture_contract as contract  # noqa: E402
import card_flock  # noqa: E402
from batch_safety import BatchStop, Child, classify_exit, start_time  # noqa: E402


GPUS = tuple(range(8))
MIN_FREE_MIB = 32 * 1024
RESERVE_MIB = 8 * 1024
POLL_SECONDS = 20


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def gpu_row(gpu: int) -> dict[str, object]:
    raw = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    index, uuid, total, used, free, util = [part.strip() for part in raw]
    return {"index": int(index), "uuid": uuid, "total_mib": int(total), "used_mib": int(used),
            "free_mib": int(free), "utilization_percent": int(util)}


def validate_reset_preflight(protocol: dict, gate: dict) -> None:
    """Independently bind all 180 amended identities to reset-only receipts."""
    instruction = gate.get("instruction_contract") or {}
    instruction_path = Path(instruction.get("path", "/nonexistent"))
    if not instruction_path.is_file() or contract.bind(instruction_path) != instruction:
        raise ValueError("Amended protocol lacks the bound instruction contract")
    instruction_row = contract.read(instruction_path)
    if (instruction_row.get("schema") != "robotwin_exact_lazy_official_instruction_v1"
            or instruction_row.get("upstream_max_descriptions") != 1_000_000
            or instruction_row.get("lazy_implementation") != contract.bind(HERE / "fast_official_instruction.py")):
        raise ValueError("Exact lazy instruction implementation differs")
    for field in ("upstream_wrapper", "upstream_generator"):
        source = instruction_row.get(field) or {}
        source_path = Path(source.get("path", "/nonexistent"))
        if not source_path.is_file() or contract.bind(source_path) != source:
            raise ValueError(f"Official instruction source differs: {field}")
    source = contract.read(PROTOCOL)
    old = {(job["id"], row["task_index"]): row
           for job in source["jobs"] for row in job["tasks"]}
    current = {(job["id"], row["task_index"]): (job, row)
               for job in protocol["jobs"] for row in job["tasks"]}
    if len(old) != 180 or set(current) != set(old):
        raise ValueError("Amended task partition differs from frozen capture-v1")
    receipt_path = Path(gate["probe_receipts"]["path"])
    valid: dict[tuple[str, int], dict] = {}
    timeouts: dict[tuple[str, int, int, int], int] = {}
    allowed = {"job", "task", "task_index", "old_seed", "seed", "counter",
               "status", "observation_sha256", "checked_at", "phase"}
    for line in receipt_path.read_text().splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("status") not in {
                "reset_valid", "simulator_unstable", "simulator_reset_timeout"}:
            raise ValueError("Preflight receipt has an unknown event")
        if row["status"] != "reset_valid":
            if any(key in row for key in ("success", "reward", "action")):
                raise ValueError("Preflight receipt contains policy outcome data")
            if row["status"] == "simulator_reset_timeout":
                if row.get("phase") != "simulator_reset_running":
                    raise ValueError("Instruction-generation timeout cannot advance a reset seed")
                key = (row.get("job"), row.get("task_index"), row.get("counter"), row.get("seed"))
                if key[:2] not in old:
                    raise ValueError("Timeout receipt has an unknown task")
                timeouts[key] = timeouts.get(key, 0) + 1
            continue
        if set(row) != allowed:
            raise ValueError("Reset-valid receipt fields differ from reset-only contract")
        if row["phase"] != "simulator_reset_complete":
            raise ValueError("Reset-valid receipt lacks completed simulator phase")
        key = (row["job"], row["task_index"])
        if key in valid or key not in old:
            raise ValueError("Duplicate or unknown reset-valid receipt")
        observation_sha = row["observation_sha256"]
        if not isinstance(observation_sha, str) or len(observation_sha) != 64:
            raise ValueError("Reset-valid observation hash is missing")
        try:
            int(observation_sha, 16)
        except ValueError as error:
            raise ValueError("Reset-valid observation hash is malformed") from error
        job, amended = current[key]
        previous = old[key]
        if (row["task"], row["old_seed"], row["seed"], row["counter"]) != (
                previous["task"], previous["seed"], amended["seed"], amended["derivation_counter"]):
            raise ValueError("Reset-valid receipt differs from old/amended task identity")
        counter = row["counter"]
        if type(counter) is not int or counter < previous["derivation_counter"]:
            raise ValueError("Reset-valid counter is invalid")
        expected = (previous["seed"] if counter == previous["derivation_counter"]
                    else contract.uint31(contract.SEED_NAMESPACE, "reset", job["repeat"],
                                         job["pool"], row["task_index"], counter))
        if row["seed"] != expected:
            raise ValueError("Reset-valid seed differs from deterministic namespace")
        valid[key] = row
    if len(valid) != 180:
        raise ValueError("Reset-only preflight does not cover all 180 identities")
    confirmed = 0
    for (job_id, task_index, counter, seed), count in timeouts.items():
        accepted = valid[(job_id, task_index)]
        if counter > accepted["counter"] or (counter < accepted["counter"] and count < 2):
            raise ValueError("A reset timeout was used to skip a seed without two confirmations")
        old_job = next(job for job in source["jobs"] if job["id"] == job_id)
        old_row = old[(job_id, task_index)]
        expected = (old_row["seed"] if counter == old_row["derivation_counter"]
                    else contract.uint31(contract.SEED_NAMESPACE, "reset", old_job["repeat"],
                                         old_job["pool"], task_index, counter))
        if seed != expected:
            raise ValueError("Timeout seed differs from deterministic namespace")
        confirmed += int(counter < accepted["counter"])
    if gate.get("timeout_confirmed_candidates", 0) != confirmed:
        raise ValueError("Confirmed reset timeout count differs")


def build_plan(run: Path, mode: str, protocol_path: Path = PROTOCOL) -> dict:
    if run.exists():
        raise FileExistsError(run)
    protocol_path = protocol_path.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema") != "robotwin_tcr_native_ab_capture_v1" or len(protocol.get("jobs", [])) != 18:
        raise ValueError("Capture protocol schema or 18-job coverage differs")
    if (protocol.get("episodes"), protocol.get("requests"), protocol.get("flow_rows")) != (180, 900, 2700):
        raise ValueError("Capture protocol episode/request/flow coverage differs")
    if protocol_path != PROTOCOL.resolve():
        gate = protocol.get("reset_preflight") or {}
        if gate.get("status") != "all_180_simulator_resets_valid_before_policy_inference":
            raise ValueError("Amended protocol lacks complete reset-only preflight")
        source = Path(gate.get("source_protocol", ""))
        if source.resolve() != PROTOCOL.resolve() or gate.get("source_protocol_sha256") != contract.sha256_file(PROTOCOL):
            raise ValueError("Amended protocol source differs from frozen capture-v1")
        receipts = gate.get("probe_receipts") or {}
        if contract.bind(Path(receipts.get("path", "/nonexistent"))) != receipts:
            raise ValueError("Reset-only preflight receipt binding differs")
        validate_reset_preflight(protocol, gate)
        expected_output_root = contract.CAPTURE_ROOT / "capture-v2/jobs"
        for job in protocol["jobs"]:
            output = Path(job.get("output", ""))
            if output != expected_output_root / job["id"] or output.exists():
                raise ValueError(f"Amended capture output is not new and isolated: {job['id']}")
    jobs = []
    selected = protocol["jobs"][:1] if mode == "smoke" else protocol["jobs"]
    for row in selected:
        output = (run / "jobs" / row["id"]) if mode == "smoke" else Path(row["output"])
        command = [str(PYTHON), "-u", str(RUNNER), "run", "--protocol", str(protocol_path),
                   "--job-id", row["id"], "--output", str(output)]
        if mode == "smoke":
            command.append("--smoke")
        jobs.append({"id": row["id"], "mode": mode, "output": str(output), "command": command,
                     "expected_episodes": 1 if mode == "smoke" else 10,
                     "expected_requests": 5 if mode == "smoke" else 50,
                     "expected_flow_rows": 15 if mode == "smoke" else 150})
    plan = {
        "schema": "robotwin_tcr_capture_queue_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(), "run": str(run.resolve()), "mode": mode,
        "protocol": str(protocol_path), "protocol_sha256": contract.sha256_file(protocol_path),
        "runner": str(RUNNER.resolve()), "runner_sha256": contract.sha256_file(RUNNER),
        "jobs": jobs, "gpus": list(GPUS), "max_workers": 1 if mode == "smoke" else 2,
        "min_free_mib": MIN_FREE_MIB, "reserve_mib": RESERVE_MIB,
        "no_retry": True, "no_score_based_scheduling_or_stopping": True,
        "openvla_formal_remains_cancelled": True,
    }
    run.mkdir(parents=True, exist_ok=False)
    save(run / "plan.json", plan)
    return plan


def verify(job: dict) -> dict[str, object]:
    complete_path = Path(job["output"]) / "complete.json"
    complete = json.loads(complete_path.read_text())
    expected_status = "technical_smoke_complete" if job["mode"] == "smoke" else "capture_job_complete"
    for key, expected in (("status", expected_status), ("formal_result", False),
                          ("job_id", job["id"]), ("episodes", job["expected_episodes"]),
                          ("requests", job["expected_requests"]), ("flow_rows", job["expected_flow_rows"])):
        if complete.get(key) != expected:
            raise ValueError(f"{job['id']} completion differs at {key}")
    if complete.get("success_values_not_used_for_acceptance_or_scheduling") is not True:
        raise ValueError("Capture receipt does not exclude success-based acceptance")
    for name in ("replay_manifest", "replay_tensor"):
        binding = complete[name]
        if contract.bind(Path(binding["path"])) != binding:
            raise ValueError(f"Capture artifact binding changed: {name}")
    manifest = json.loads(Path(complete["replay_manifest"]["path"]).read_text())
    if manifest.get("sample_count") != job["expected_flow_rows"] or manifest.get("success_filtering") is not False:
        raise ValueError("Capture replay manifest scope/coverage differs")
    return {"complete": str(complete_path), "complete_sha256": contract.sha256_file(complete_path),
            "replay_manifest_sha256": complete["replay_manifest"]["sha256"],
            "replay_tensor_sha256": complete["replay_tensor"]["sha256"],
            "episodes": job["expected_episodes"], "requests": job["expected_requests"],
            "flow_rows": job["expected_flow_rows"]}


def environment(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "TORCHINDUCTOR_COMPILE_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def run_queue(plan: dict) -> None:
    run = Path(plan["run"])
    if (run / "started.json").exists() or plan["host"] != socket.gethostname():
        raise ValueError("Wrong host or attempted automatic restart")
    protocol_path = Path(plan["protocol"])
    if contract.sha256_file(protocol_path) != plan["protocol_sha256"] or contract.sha256_file(RUNNER) != plan["runner_sha256"]:
        raise ValueError("Frozen capture protocol/runner changed")
    stop = BatchStop(report_path=run / "batch-stop-report.json")
    stop.install_signal_handlers()
    pending = list(plan["jobs"])
    active: list[dict] = []
    finished: list[dict] = []
    uuids = card_flock.gpu_uuids()
    save(run / "started.json", {"pid": os.getpid(), "kernel_starttime": start_time(os.getpid()),
                                 "host": socket.gethostname(), "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        while (pending or active) and not stop.stopped:
            for item in list(active):
                child, job = item["child"], item["job"]
                child._snapshot = child.group_members()
                row = gpu_row(item["gpu"])
                item["minimum_free_mib"] = min(item["minimum_free_mib"], row["free_mib"])
                item["peak_total_used_mib"] = max(item["peak_total_used_mib"], row["used_mib"])
                code = child.process.poll()
                if code is None:
                    continue
                code = child.wait()
                receipt = {**classify_exit(code), "gpu": item["gpu"], "gpu_uuid": item["uuid"],
                           "child": child.identity(), "elapsed_seconds": time.time() - item["started_unix"],
                           "minimum_free_mib": item["minimum_free_mib"],
                           "peak_total_used_mib": item["peak_total_used_mib"],
                           "finished_at": datetime.now(timezone.utc).isoformat()}
                try:
                    if code == 0:
                        receipt["artifacts"] = verify(job)
                except Exception as error:
                    receipt["outcome"] = "audit_failed"; receipt["audit_error"] = repr(error); code = 1
                save(Path(job["output"]).with_suffix(".exit.json"), receipt)
                item["legacy"].release(); item["lease"].release(); active.remove(item)
                finished.append({"id": job["id"], **receipt})
                if code != 0:
                    stop.trip(f"{job['id']} exited or audited unsuccessfully; no retry")
                    break
            if stop.stopped:
                break
            busy = {item["gpu"] for item in active}
            for gpu in sorted(plan["gpus"], key=lambda value: gpu_row(value)["free_mib"], reverse=True):
                if not pending or len(active) >= plan["max_workers"]:
                    break
                if gpu in busy:
                    continue
                row = gpu_row(gpu)
                if row["free_mib"] < plan["min_free_mib"]:
                    continue
                job = pending[0]
                lease = card_flock.take_card(gpu, row["uuid"], job["id"], "robotwin-tcr-capture")
                if lease is None:
                    continue
                legacy_path = RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock"
                legacy = card_flock._try_lock(legacy_path, job["id"], {"job": job["id"], "gpu": gpu,
                                                                      "stage": "robotwin-tcr-capture"})
                if legacy is None:
                    lease.release(); continue
                row = gpu_row(gpu)
                if row["uuid"] != uuids[gpu] or row["free_mib"] < plan["min_free_mib"]:
                    legacy.release(); lease.release(); continue
                pending.pop(0)
                child = Child(job["command"], environment(gpu), Path(job["output"]).with_suffix(".worker.log"), cwd=WORK)
                if not stop.register(child):
                    pending.insert(0, job); legacy.release(); lease.release(); break
                save(Path(job["output"]).with_suffix(".launch.json"), {"job": job, "gpu": gpu,
                     "gpu_uuid": row["uuid"], "resource_at_admission": row, "child": child.identity(),
                     "started_at": datetime.now(timezone.utc).isoformat()})
                active.append({"job": job, "gpu": gpu, "uuid": row["uuid"], "lease": lease,
                               "legacy": legacy, "child": child, "started_unix": time.time(),
                               "minimum_free_mib": row["free_mib"], "peak_total_used_mib": row["used_mib"]})
                busy.add(gpu)
                print(json.dumps({"started": job["id"], "gpu": gpu, "pid": child.pid}), flush=True)
            save(run / "state.json", {"updated_at": datetime.now(timezone.utc).isoformat(),
                "active": [{"id": item["job"]["id"], "gpu": item["gpu"], "pid": item["child"].pid} for item in active],
                "finished_count": len(finished), "pending_count": len(pending), "stopped": stop.stopped})
            if pending or active:
                time.sleep(POLL_SECONDS)
    except BaseException as error:
        stop.trip(f"supervisor exception: {type(error).__name__}: {error}")
        raise
    finally:
        for item in active:
            code = item["child"].wait()
            path = Path(item["job"]["output"]).with_suffix(".exit.json")
            if not path.exists():
                save(path, {**classify_exit(code), "gpu": item["gpu"], "child": item["child"].identity(),
                            "finished_at": datetime.now(timezone.utc).isoformat()})
            item["legacy"].release(); item["lease"].release()
        complete = len(finished) == len(plan["jobs"]) and not stop.stopped
        save(run / "queue-ended.json", {"status": "complete" if complete else "incomplete",
             "completed_jobs": len(finished), "expected_jobs": len(plan["jobs"]), "stopped": stop.stopped,
             "reason": stop.reason, "restarted": False, "finished": finished,
             "ended_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--protocol", type=Path,
                        help="Frozen capture protocol; defaults to capture-v1 only when preparing")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.prepare:
        plan = build_plan(run, args.mode, args.protocol or PROTOCOL)
        print(json.dumps({"prepared": str(run), "mode": args.mode, "jobs": len(plan["jobs"])}))
        return
    plan = json.loads((run / "plan.json").read_text())
    if plan.get("mode") != args.mode:
        raise ValueError("Queue mode differs from frozen plan")
    if args.protocol is not None and str(args.protocol.resolve()) != plan.get("protocol"):
        raise ValueError("Queue protocol differs from frozen plan")
    run_queue(plan)


if __name__ == "__main__":
    main()
