#!/usr/bin/env python3
"""Frozen native RoboTwin TCR formal panel; corrected Experts receipt binding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUNTIME = WORK / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-robotwin-tcr-20260923"
BUILD = BASE / "tcr-build-attempt-01"
MODELS = BASE / "tcr-checkpoints-v1"
RUN = BASE / "tcr-formal-attempt-02"
EXPERTS = WORK / "vla-merge/scripts/watch_robotwin_experts_formal_v2.py"
BASE_EVALUATOR = WORK / "vla-merge/scripts/run_iclr2027_robotwin_checkpoint_development.py"
SLICER = WORK / "vla-merge/scripts/run_iclr2027_robotwin_development_slices.py"
NATIVE_LOOP = WORK / "vla-merge/scripts/run_iclr2027_robotwin_native_pi05_smoke_v2.py"
EXPERT_PROTOCOL = (RUNTIME / "experiments/iclr2027-table5-20260910/evaluation-queues/"
                   "robotwin-three-expert-formal-v2/protocol.json")
PYTHON = RUNTIME / "envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python"
GROUPS = ("coordination", "receptacle", "precision")
REPEATS = (1, 2, 3)
GPUS = tuple(range(8))
MIN_FREE_MIB = 40_000
FLOOR_MIB = 12 * 1024
MAX_ACTIVE = 2
POLL_SECONDS = 60

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WORK / "vla-merge/experiments/claude-firstpass-cause-20260920"))
import card_flock  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(32 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def experts_module():
    spec = importlib.util.spec_from_file_location("robotwin_tcr_formal_experts", EXPERTS)
    require(spec is not None and spec.loader is not None, "Native Experts evaluator missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint(repeat: int) -> Path:
    return MODELS / f"r{repeat:02d}-passB" / "pretrained_model"


def native_formal_runtime(source, group: str, repeat: int) -> tuple[dict, set]:
    manifest, runtime, expected = source.validate_job(group, repeat)
    require(len(expected) == 60 and sum(len(task["seeds"]) for task in manifest["tasks"]) == 60,
            "Native formal reset panel differs")
    # validate_job returns the historical development parent as runtime.
    # Replace its twenty-seed tasks with the six-seed formal bank.
    runtime["tasks"] = manifest["tasks"]
    return runtime, expected


def expert_reset_hashes(source, group: str, repeat: int) -> tuple[dict, dict]:
    """Bind all 60 previously accepted native Experts formal reset observations."""
    root = source.QUEUE / "runs" / group / f"repeat-{repeat:02d}"
    terminal_path = root / "complete.json"
    terminal = read(terminal_path)
    require(terminal.get("status") == "robotwin_experts_formal_job_complete"
            and terminal.get("group") == group and terminal.get("repeat") == repeat
            and terminal.get("formal_result") is True,
            "Matched Experts formal job has no accepted terminal receipt")
    receipt_binding = terminal["receipt"]
    receipt_path = Path(receipt_binding["path"])
    require(sha(receipt_path) == receipt_binding["sha256"],
            "Matched Experts attempt receipt changed")
    receipt = read(receipt_path)
    require(receipt.get("episodes") == 60 and receipt.get("method") == "Experts"
            and len(receipt.get("rows", [])) == 60,
            "Matched Experts attempt has incomplete formal rows")
    hashes = {}
    for row_binding in receipt["rows"]:
        path = Path(row_binding["path"])
        require(sha(path) == row_binding["sha256"], "Matched Experts raw row changed")
        row = read(path)
        key = (row["task_index"], row["seed"])
        require(key not in hashes and type(row.get("success")) is bool
                and isinstance(row.get("initial_observation_sha256"), str)
                and len(row["initial_observation_sha256"]) == 64,
                "Matched Experts reset evidence differs")
        hashes[key] = row["initial_observation_sha256"]
    require(len(hashes) == 60, "Matched Experts reset coverage differs")
    return hashes, {"terminal_path": str(terminal_path), "terminal_sha256": sha(terminal_path),
                    "receipt_path": str(receipt_path),
                    "receipt_sha256": receipt_binding["sha256"]}


def gpu_rows() -> dict[int, dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in output.strip().splitlines():
        gpu, uuid, free, used = [item.strip() for item in line.split(",")]
        result[int(gpu)] = {"gpu": int(gpu), "uuid": uuid,
                            "free_mib": int(free), "used_mib": int(used)}
    return result


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(RUN)
    build = read(BUILD / "plan.json")
    require(build.get("schema") == "robotwin_three_expert_two_pass_build_v1"
            and build.get("host") == socket.gethostname()
            and len(build.get("jobs", [])) == 6
            and build.get("training") is False, "Frozen three-repeat TCR build plan missing")
    source = experts_module()
    require(read(EXPERT_PROTOCOL).get("episodes_total") == 540,
            "Experts formal-v2 protocol differs")
    jobs = []
    for repeat in REPEATS:
        for group in GROUPS:
            manifest, _, expected = source.validate_job(group, repeat)
            require(len(expected) == 60 and manifest["repeat"] == repeat
                    and manifest["group"] == group,
                    "Native Experts reset bank differs")
            manifest_path = source.QUEUE / f"manifests/{group}/repeat-{repeat:02d}.json"
            expert_hashes, expert_binding = expert_reset_hashes(source, group, repeat)
            require(set(expert_hashes) == expected,
                    "Matched Experts result differs from frozen formal reset bank")
            job_id = f"r{repeat:02d}-{group}"
            jobs.append({"id": job_id, "repeat": repeat, "group": group,
                         "episodes": 60, "source_manifest": str(manifest_path),
                         "source_manifest_sha256": sha(manifest_path),
                         "expert_reference": expert_binding,
                         "checkpoint": str(checkpoint(repeat)),
                         "model_job": f"r{repeat:02d}-passB",
                         "output": str(RUN / "jobs" / job_id)})
    require(len(jobs) == 9 and len({j["id"] for j in jobs}) == 9,
            "TCR formal panel partition differs")
    plan = {"schema": "robotwin_tcr_three_repeat_formal_v2",
            "created_at": now(), "host": socket.gethostname(),
            "paper_cell": "RoboTwin Table5 TCR; mean and sample std over three repeats",
            "formal": True, "training": False, "no_retry": True,
            "no_score_based_scheduling_or_stopping": True,
            "build_plan_sha256": sha(BUILD / "plan.json"),
            "experts_protocol_sha256": sha(EXPERT_PROTOCOL),
            "native_evaluator_sha256": sha(EXPERTS),
            "base_evaluator_sha256": sha(BASE_EVALUATOR),
            "slicer_sha256": sha(SLICER),
            "native_loop_sha256": sha(NATIVE_LOOP),
            "source_sha256": sha(Path(__file__)),
            "episodes": 540, "jobs": jobs,
            "min_free_mib": MIN_FREE_MIB, "runtime_floor_mib": FLOOR_MIB,
            "max_active": MAX_ACTIVE, "permitted_gpus": list(GPUS)}
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / "plan.json", plan)
    return {"plan": str(RUN / "plan.json"), "jobs": 9, "episodes": 540}


def validate_plan(plan: dict) -> None:
    require(plan.get("schema") == "robotwin_tcr_three_repeat_formal_v2"
            and plan.get("host") == socket.gethostname()
            and plan.get("formal") is True and plan.get("training") is False
            and plan.get("no_retry") is True
            and plan.get("no_score_based_scheduling_or_stopping") is True
            and plan.get("episodes") == 540 and len(plan.get("jobs", [])) == 9,
            "Frozen RoboTwin TCR formal plan differs")
    for path, key in ((BUILD / "plan.json", "build_plan_sha256"),
                      (EXPERT_PROTOCOL, "experts_protocol_sha256"),
                      (EXPERTS, "native_evaluator_sha256"),
                      (BASE_EVALUATOR, "base_evaluator_sha256"),
                      (SLICER, "slicer_sha256"),
                      (NATIVE_LOOP, "native_loop_sha256"),
                      (Path(__file__), "source_sha256")):
        require(sha(path) == plan[key], f"Frozen source changed: {path}")
    for job in plan["jobs"]:
        require(sha(Path(job["source_manifest"])) == job["source_manifest_sha256"],
                f"Formal reset manifest changed: {job['id']}")
        for name in ("terminal", "receipt"):
            reference = job["expert_reference"]
            require(sha(Path(reference[name + "_path"])) == reference[name + "_sha256"],
                    f"Matched Experts formal receipt changed: {job['id']}")


def accepted_build(repeat: int) -> dict | None:
    state_path = BUILD / "state.json"
    if not state_path.exists():
        return None
    state = read(state_path)
    return (state.get("accepted") or {}).get(f"r{repeat:02d}-passB")


def audit_job_rows(job: dict) -> dict:
    source = experts_module()
    _, _, expected = source.validate_job(job["group"], job["repeat"])
    expert_hashes, _ = expert_reset_hashes(source, job["group"], job["repeat"])
    require(set(expert_hashes) == expected, "Matched Experts reset panel differs")
    output = Path(job["output"])
    complete = read(output / "complete.json")
    require(complete.get("status") == "robotwin_tcr_formal_job_complete"
            and complete.get("id") == job["id"] and complete.get("episodes") == 60,
            f"Formal completion receipt differs: {job['id']}")
    seen = {}
    for row_path in sorted(output.glob("*.json")):
        if row_path.name in {"started.json", "complete.json", "simulator.json", "failure.json"}:
            continue
        row = read(row_path)
        key = (row.get("task_index"), row.get("seed"))
        require(key in expected and key not in seen and type(row.get("success")) is bool
                and isinstance(row.get("initial_observation_sha256"), str)
                and row["initial_observation_sha256"] == expert_hashes[key],
                f"Invalid or duplicate formal episode: {row_path}")
        seen[key] = row_path
    require(set(seen) == expected and len(complete.get("rows", [])) == 60,
            f"Incomplete formal episode panel: {job['id']}")
    bound_paths = set()
    for bound in complete["rows"]:
        row_path = Path(bound["path"])
        require(row_path in seen.values() and sha(row_path) == bound["sha256"],
                f"Formal raw row changed: {row_path}")
        bound_paths.add(row_path)
    require(bound_paths == set(seen.values()), "Formal receipt omitted raw rows")
    return {"id": job["id"], "episodes": 60,
            "complete_sha256": sha(output / "complete.json"),
            "model_sha256": complete["model_sha256"]}


def worker(job_id: str) -> None:
    plan = read(RUN / "plan.json")
    validate_plan(plan)
    job = next((row for row in plan["jobs"] if row["id"] == job_id), None)
    require(job is not None, "Unknown formal job")
    built = accepted_build(job["repeat"])
    require(built is not None and built["model_sha256"] == sha(Path(job["checkpoint"]) / "model.safetensors"),
            "Formal TCR checkpoint is not accepted by its build queue")
    source = experts_module()
    runtime, expected = native_formal_runtime(source, job["group"], job["repeat"])
    expert_hashes, _ = expert_reset_hashes(source, job["group"], job["repeat"])
    require(set(expert_hashes) == expected, "Matched Experts reset panel differs")
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "started.json", {"job": job_id, "model_sha256": built["model_sha256"],
                                 "source_manifest_sha256": job["source_manifest_sha256"],
                                 "pid": os.getpid(), "started_at": now()})
    try:
        runtime["checkpoint"] = str(Path(job["checkpoint"]).parent)
        runtime["formal_result"] = True
        runtime["purpose"] = "robotwin_tcr_formal_evaluation"
        runtime["plateau_eligible"] = False
        source.slicer.runtime_setup(runtime)
        import torch
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        source.base.simulator_audit(runtime, output, False)
        task_names = {int(t["task_index"]): t["task"] for t in runtime["tasks"]}
        rows = []
        for task_index, seed in sorted(expected):
            path = output / f"{task_names[task_index]}-{seed}.json"
            row = read(path)
            require(row.get("task_index") == task_index and row.get("seed") == seed
                    and type(row.get("success")) is bool
                    and row.get("initial_observation_sha256") == expert_hashes[(task_index, seed)],
                    f"Native formal episode differs: {path}")
            rows.append({"path": str(path), "sha256": sha(path)})
        save(output / "complete.json", {"status": "robotwin_tcr_formal_job_complete",
             "id": job_id, "formal": True, "episodes": 60,
             "model_sha256": built["model_sha256"], "rows": rows,
             "seconds": time.monotonic() - start,
             "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
             "finished_at": now()})
    except BaseException as error:
        save(output / "failure.json", {"type": type(error).__name__,
              "error": str(error), "failed_at": now()})
        raise


def run() -> None:
    plan = read(RUN / "plan.json")
    require(not (RUN / "started.json").exists(), "No restart within frozen formal attempt")
    validate_plan(plan)
    save(RUN / "started.json", {"pid": os.getpid(), "host": socket.gethostname(),
         "kernel_starttime": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
         "started_at": now()})
    pending = {job["id"]: job for job in plan["jobs"]}
    accepted: dict[str, dict] = {}
    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    active: dict[str, dict] = {}
    stopping = False

    def on_signal(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    while pending or active:
        if stopping:
            for item in active.values():
                if item["process"].poll() is None:
                    os.killpg(item["process"].pid, signal.SIGTERM)
        for job_id, item in list(active.items()):
            child = item["process"]
            code = child.poll()
            if code is None and not stopping and gpu_rows()[item["gpu"]]["free_mib"] < FLOOR_MIB:
                os.killpg(child.pid, signal.SIGTERM)
                item["resource_floor"] = True
            if code is None:
                continue
            child.wait()
            item["log"].close()
            item["legacy"].release(); item["card"].release()
            active.pop(job_id)
            try:
                result = audit_job_rows(item["job"]) if code == 0 and not item.get("resource_floor") else None
                error = None if result else f"child exit {code}; resource_floor={item.get('resource_floor', False)}"
            except Exception as exc:
                result, error = None, f"artifact gate: {type(exc).__name__}: {exc}"
            save(RUN / "exits" / f"{job_id}.json", {"id": job_id,
                 "returncode": code, "accepted": result is not None,
                 "artifact": result, "error": error, "finished_at": now()})
            if result:
                accepted[job_id] = result
            else:
                failed[job_id] = error
        build_end = BUILD / "queue-ended.json"
        if build_end.exists() and read(build_end).get("complete") is not True:
            for job_id, job in list(pending.items()):
                if accepted_build(job["repeat"]) is None:
                    skipped[job_id] = "Matching TCR pass B did not complete"
                    pending.pop(job_id)
        if not stopping and len(active) < MAX_ACTIVE:
            ready = [job for job in pending.values() if accepted_build(job["repeat"]) is not None]
            if ready:
                rows = gpu_rows()
                for gpu in sorted(GPUS, key=lambda index: rows[index]["free_mib"], reverse=True):
                    if not ready or len(active) >= MAX_ACTIVE:
                        break
                    row = rows[gpu]
                    if (gpu in {item["gpu"] for item in active.values()}
                            or row["free_mib"] < MIN_FREE_MIB):
                        continue
                    job = ready[0]
                    card = card_flock.take_card(gpu, row["uuid"], job["id"], "robotwin-tcr-formal")
                    if card is None:
                        continue
                    legacy = card_flock._try_lock(
                        RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                        job["id"], {"gpu": gpu, "job": job["id"], "stage": "robotwin-tcr-formal"})
                    if legacy is None:
                        card.release()
                        continue
                    again = gpu_rows()[gpu]
                    if again["uuid"] != row["uuid"] or again["free_mib"] < MIN_FREE_MIB:
                        legacy.release(); card.release()
                        continue
                    try:
                        require(not Path(job["output"]).exists(), "Formal job output already exists")
                        log_path = RUN / "logs" / f"{job['id']}.log"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("x")
                        command = [str(PYTHON), "-u", str(Path(__file__)), "worker", "--job-id", job["id"]]
                        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                               "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
                        child = subprocess.Popen(command, cwd=WORK, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, env=env,
                                                 pass_fds=(card.handle, legacy.handle))
                        save(RUN / "launches" / f"{job['id']}.json", {"id": job["id"],
                             "gpu": gpu, "uuid": row["uuid"], "pid": child.pid,
                             "command": command, "admission": again,
                             "model_sha256": accepted_build(job["repeat"])["model_sha256"],
                             "started_at": now()})
                        active[job["id"]] = {"job": job, "gpu": gpu, "card": card,
                                              "legacy": legacy, "process": child, "log": log}
                        pending.pop(job["id"])
                        ready.pop(0)
                    except Exception as exc:
                        legacy.release(); card.release()
                        failed[job["id"]] = f"prelaunch: {type(exc).__name__}: {exc}"
                        pending.pop(job["id"])
        save(RUN / "state.json", {"accepted": accepted, "failed": failed,
             "skipped": skipped, "active": {key: {"gpu": item["gpu"], "pid": item["process"].pid}
                                                for key, item in active.items()},
             "pending": list(pending), "stopped": stopping, "updated_at": now()})
        if stopping and not active:
            break
        if pending or active:
            time.sleep(POLL_SECONDS)
    save(RUN / "queue-ended.json", {"complete": len(accepted) == 9 and not failed and not skipped,
         "accepted": accepted, "failed": failed, "skipped": skipped,
         "pending": list(pending), "stopped": stopping, "ended_at": now()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "worker"))
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), sort_keys=True))
    elif args.action == "worker":
        require(args.job_id is not None, "worker requires --job-id")
        worker(args.job_id)
    else:
        run()


if __name__ == "__main__":
    main()
