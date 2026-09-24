#!/usr/bin/env python3
"""Formal repeat-01 400 episodes for each of two fixed appendix scopes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
VLA = WORK / "vla-merge"
RUNTIME = WORK / "vla-merge-runtime"
ROOT = RUNTIME / "experiments/claude-pi05-fixed-appendix-20260923"
BUILDS = ROOT / "scope-build-attempt-02"
RUN = ROOT / "scope-formal-eval-attempt-03"
EVALUATOR = VLA / "scripts/eval_tcr_10k_formal_bounded.py"
PYTHON = WORK / "pi05_lora_finetune_v2_20260826/.venv/bin/python"
FORMAL = VLA / "experiments/claude-formal-mainline-20260919"
FLOCKS = VLA / "experiments/claude-firstpass-cause-20260920"
SUITE_NAMES = {"spatial": "libero_spatial", "object": "libero_object",
               "goal": "libero_goal", "long": "libero_10"}
MIN_FREE_MIB = 40 * 1024
RUNTIME_FLOOR_MIB = 12 * 1024  # Existing formal evaluator enforces this floor.
MAX_ACTIVE = 2
POLL_SECONDS = 30
GPUS = tuple(range(8))

sys.path[:0] = [str(HERE), str(FORMAL), str(FLOCKS), str(VLA / "scripts")]
import card_flock  # noqa: E402
import formal_contract  # noqa: E402
import appendix_contract as appendix  # noqa: E402
import appendix_scope_contract_v8 as scope  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def gpu_rows() -> dict[int, dict]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    rows = {}
    for line in output.strip().splitlines():
        index, uuid, free, used = [part.strip() for part in line.split(",")]
        rows[int(index)] = {"gpu": int(index), "uuid": uuid,
                            "free_mib": int(free), "used_mib": int(used)}
    return rows


def checkpoint(arm: str) -> Path:
    return ROOT / "checkpoints-v12" / arm


def make_command(job: dict, selection: dict) -> list[str]:
    output = Path(job["output"])
    return [str(PYTHON), "-u", str(EVALUATOR),
            f"--procedural-bank={selection['bank']}",
            f"--procedural-selection={selection['selection']}",
            f"--output_dir={output / 'eval'}",
            "--env.type=libero", f"--env.task={job['full_suite']}",
            "--env.init_states=true", "--env.hard_reset=true",
            "--eval.batch_size=1", "--eval.n_episodes=10",
            f"--seed={selection['eval_seed']}",
            f"--policy.path={job['checkpoint']}", "--policy.device=cuda",
            "--policy.compile_model=false", "--policy.gradient_checkpointing=false",
            "--policy.n_action_steps=10"]


def prepare() -> dict:
    if RUN.exists():
        raise FileExistsError(RUN)
    builds = json.loads((BUILDS / "plan.json").read_text())
    selection = formal_contract.selection_for(1)
    if (builds.get("schema") != "pi05_fixed_appendix_scope_build_v5"
            or [row.get("id") for row in builds.get("jobs", [])] != list(scope.ARMS)
            or selection["tasks"] != 40 or selection["eval_seed"] != 274001
            or sha(Path(selection["selection"])) != selection["sha256"]):
        raise ValueError("Appendix scope builds or formal reset bank differ")
    audit = json.loads((WORK / "coordination/2026-09-23/appendix-abc-pools-content-audit.json").read_text())
    if audit.get("accepted") is not True or audit.get("heldout_disjoint") is not True:
        raise ValueError("A/B/C calibration and heldout reset identity gate differs")
    jobs = []
    for arm_name in ("action_AB", "conditioning_AB"):
        for name in appendix.SUITES:
            job = {"id": f"{arm_name}-{name}", "arm": arm_name,
                   "suite": name, "full_suite": SUITE_NAMES[name],
                   "checkpoint": str(checkpoint(arm_name)), "needs_build": arm_name,
                   "output": str(RUN / "jobs" / f"{arm_name}-{name}"),
                   "episodes": 100, "repeat": 1}
            job["command"] = make_command(job, selection)
            jobs.append(job)
    if len(jobs) != 8 or len({row["id"] for row in jobs}) != 8:
        raise ValueError("Scope formal evaluation coverage differs")
    frozen = {"schema": "pi05_fixed_appendix_scope_formal_eval_v1",
              "created_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
              "build_plan_sha256": sha(BUILDS / "plan.json"),
              "input_content_audit_sha256": sha(WORK / "coordination/2026-09-23/appendix-abc-pools-content-audit.json"),
              "selection": selection, "selection_sha256": selection["sha256"],
              "bank_manifest_sha256": sha(Path(selection["bank"]) / "manifest.json"),
              "procedural_entrypoint_sha256": sha(VLA / "scripts/eval_pi05_policy_with_procedural_bank.py"),
              "evaluator_sha256": sha(EVALUATOR), "jobs": jobs,
              "formal_repeat": 1, "formal_episodes": 800,
              "per_suite_episodes": 100, "per_task_episodes": 10,
              "max_active": MAX_ACTIVE, "min_free_mib": MIN_FREE_MIB,
              "runtime_floor_mib": RUNTIME_FLOOR_MIB, "permitted_gpus": list(GPUS),
              "no_retry": True, "no_score_based_scheduling": True,
              "paper_scope": "action-only and conditioning+action final two-pass checkpoints, one accepted model and 400 formal episodes each on same repeat-01 selection"}
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / "plan.json", frozen)
    return frozen


def validate_plan(plan: dict) -> None:
    selection = formal_contract.selection_for(1)
    if (plan.get("schema") != "pi05_fixed_appendix_scope_formal_eval_v1"
            or plan.get("host") != socket.gethostname()
            or [job.get("id") for job in plan.get("jobs", [])] != [
                f"{arm}-{suite}" for arm in ("action_AB", "conditioning_AB") for suite in appendix.SUITES]
            or sum(job["episodes"] for job in plan["jobs"]) != 800
            or plan.get("build_plan_sha256") != sha(BUILDS / "plan.json")
            or plan.get("input_content_audit_sha256") != sha(WORK / "coordination/2026-09-23/appendix-abc-pools-content-audit.json")
            or plan.get("selection") != selection
            or plan.get("bank_manifest_sha256") != sha(Path(selection["bank"]) / "manifest.json")
            or plan.get("procedural_entrypoint_sha256") != sha(VLA / "scripts/eval_pi05_policy_with_procedural_bank.py")
            or plan.get("evaluator_sha256") != sha(EVALUATOR)):
        raise ValueError("Frozen appendix scope evaluation plan changed")


def verify_checkpoint(job: dict, builds: dict) -> str:
    if job["needs_build"] not in builds.get("accepted", {}):
        raise ValueError("scope build is not accepted")
    root = Path(job["checkpoint"])
    model = root / "model.safetensors"
    manifest = root / "block_regmeanpp_manifest.json"
    if not model.is_file() or not manifest.is_file():
        raise ValueError("scope checkpoint is missing")
    data = json.loads(manifest.read_text())
    if (data.get("fixed_appendix_arm") != job["arm"]
            or data.get("fixed_appendix_pass") != 2
            or data.get("realized_row_total") != scope.ROWS[(scope.ARMS[job["arm"]]["scope"], 2)]
            or data.get("model_sha256") != builds["accepted"][job["needs_build"]]["model_sha256"]):
        raise ValueError("scope model identity/rows differ")
    if sha(model) != data["model_sha256"]:
        raise ValueError("scope model file changed")
    return data["model_sha256"]


def audit_result(job: dict, model_sha: str, plan: dict) -> dict:
    output = Path(job["output"]) / "eval"
    info = json.loads((output / "eval_info.json").read_text())
    receipt = json.loads((output / "procedural_bank_receipt.json").read_text())
    bounded = json.loads((output / "bounded_resources.json").read_text())
    per_task = info.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != 10:
        raise ValueError("Formal evaluation lacks ten tasks")
    task_ids = [row.get("task_id") for row in per_task]
    if set(task_ids) != set(range(10)):
        raise ValueError("Formal evaluation task IDs repeat")
    successes = []
    for row in per_task:
        values = (row.get("metrics") or {}).get("successes")
        if not isinstance(values, list) or len(values) != 10 or any(type(value) is not bool for value in values):
            raise ValueError("Formal task lacks ten boolean episodes")
        successes.extend(values)
    if (len(successes) != 100 or info.get("overall", {}).get("n_episodes") != 100
            or abs(info["overall"].get("pc_success", -1) - 100 * sum(successes) / 100) > 1e-6):
        raise ValueError("Formal success denominator differs")
    selection = plan["selection"]
    if (receipt.get("selection_sha256") != selection["sha256"]
            or receipt.get("eval_seed") != selection["eval_seed"]
            or receipt.get("repeat_id") != "repeat-01"
            or receipt.get("bank_manifest_sha256") != plan["bank_manifest_sha256"]
            or receipt.get("entrypoint_sha256") != plan["procedural_entrypoint_sha256"]
            or receipt.get("eval_info_sha256") != sha(output / "eval_info.json")
            or bounded.get("evaluation_completed") is not True):
        raise ValueError("Formal procedural bank or bounded evaluator receipt differs")
    entries = receipt.get("tasks") or {}
    selected = json.loads(Path(selection["selection"]).read_text())["tasks"]
    manifest = json.loads((Path(selection["bank"]) / "manifest.json").read_text())["tasks"]
    expected_keys = {f"{job['full_suite']}/{task_id:02d}" for task_id in range(10)}
    if set(entries) != expected_keys:
        raise ValueError("Formal actual reset-state coverage differs")
    for key in expected_keys:
        indices = selected[key]
        expected_hashes = [manifest[key]["states"][index]["raw_sha256"] for index in indices]
        if (len(indices) != 10 or entries[key].get("state_indices") != indices
                or entries[key].get("raw_state_sha256") != expected_hashes
                or entries[key].get("suite") != job["full_suite"]
                or entries[key].get("task_id") != int(key.rsplit("/", 1)[1])):
            raise ValueError(f"Actual procedural reset identity differs: {key}")
    return {"id": job["id"], "episodes": 100, "successes": sum(successes),
            "model_sha256": model_sha, "eval_info_sha256": sha(output / "eval_info.json"),
            "procedural_receipt_sha256": sha(output / "procedural_bank_receipt.json")}


def run() -> None:
    plan = json.loads((RUN / "plan.json").read_text())
    if (RUN / "started.json").exists():
        raise ValueError("No restart within this formal scope attempt")
    validate_plan(plan)
    save(RUN / "started.json", {"pid": os.getpid(), "host": socket.gethostname(),
                                "started_at": datetime.now(timezone.utc).isoformat()})
    pending = {job["id"]: job for job in plan["jobs"]}
    accepted: dict[str, dict] = {}
    failed: dict[str, str] = {}
    blocked: dict[str, str] = {}
    active: dict[str, dict] = {}
    stop = False

    def stop_handler(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        while pending or active:
            if stop:
                for item in active.values():
                    if item["process"].poll() is None:
                        os.killpg(item["process"].pid, signal.SIGTERM)
            build_state = BUILDS / "state.json"
            if not build_state.exists():
                # The build supervisor validates large replay files before its first
                # state write. Waiting here preserves the frozen formal jobs.
                save(RUN / "state.json", {"accepted": accepted, "failed": failed,
                     "blocked": blocked, "active": {}, "pending": list(pending),
                     "stopped": stop,
                     "updated_at": datetime.now(timezone.utc).isoformat()})
                if stop:
                    break
                time.sleep(POLL_SECONDS)
                continue
            builds = json.loads(build_state.read_text())
            for job_id, item in list(active.items()):
                child = item["process"]
                code = child.poll()
                if code is None and not stop and gpu_rows()[item["gpu"]]["free_mib"] < RUNTIME_FLOOR_MIB:
                    os.killpg(child.pid, signal.SIGTERM)
                    item["resource_floor"] = True
                if code is None:
                    continue
                child.wait(); item["log"].close()
                item["legacy"].release(); item["card"].release()
                active.pop(job_id)
                try:
                    result = audit_result(item["job"], item["model_sha"], plan) if code == 0 and not item.get("resource_floor") else None
                    error = None if result else f"child exit {code}; resource_floor={item.get('resource_floor', False)}"
                except Exception as exc:
                    result, error = None, f"strict audit: {type(exc).__name__}: {exc}"
                save(RUN / "exits" / f"{job_id}.json", {"job": job_id,
                     "returncode": code, "accepted": result is not None,
                     "artifact": result, "error": error,
                     "finished_at": datetime.now(timezone.utc).isoformat()})
                if result:
                    accepted[job_id] = result
                else:
                    failed[job_id] = error
            for job_id, job in list(pending.items()):
                need = job["needs_build"]
                if need in builds.get("failed", {}) or need in builds.get("skipped", {}):
                    blocked[job_id] = "Required scope build failed"
                    pending.pop(job_id)
            if not stop:
                ready = [job for job in pending.values() if job["needs_build"] in builds.get("accepted", {})]
                if ready and len(active) < MAX_ACTIVE:
                    rows = gpu_rows()
                    for gpu in sorted(GPUS, key=lambda idx: rows[idx]["free_mib"], reverse=True):
                        if not ready or len(active) >= MAX_ACTIVE:
                            break
                        row = rows[gpu]
                        if gpu in {item["gpu"] for item in active.values()} or row["free_mib"] < MIN_FREE_MIB:
                            continue
                        job = ready[0]
                        card = card_flock.take_card(gpu, row["uuid"], job["id"], "pi05-appendix-scope-formal-eval")
                        if card is None:
                            continue
                        legacy = card_flock._try_lock(
                            RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                            job["id"], {"gpu": gpu, "job": job["id"], "stage": "pi05-appendix-scope-formal-eval"})
                        again = gpu_rows()[gpu]
                        if legacy is None or again["uuid"] != row["uuid"] or again["free_mib"] < MIN_FREE_MIB:
                            if legacy: legacy.release()
                            card.release(); continue
                        try:
                            model_sha = verify_checkpoint(job, builds)
                            if Path(job["output"]).exists():
                                raise FileExistsError(job["output"])
                            log_path = RUN / "logs" / f"{job['id']}.log"
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            log = log_path.open("x")
                            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                   "ITERATION_PHYSICAL_GPU": str(gpu), "PYTHONUNBUFFERED": "1",
                                   "PYTHONDONTWRITEBYTECODE": "1",
                                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
                            for name in formal_contract.FORBIDDEN_EVAL_ENV:
                                env.pop(name, None)
                            env["PYTHONPATH"] = ":".join([str(VLA / 'scripts'),
                                str(WORK / 'pi05_lora_finetune_v2_20260826/src'), env.get('PYTHONPATH', '')])
                            formal_contract.check_eval_command(job["command"], env)
                            child = subprocess.Popen(job["command"], cwd=WORK, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True, env=env,
                                                     pass_fds=(card.handle, legacy.handle))
                            save(RUN / "launches" / f"{job['id']}.json", {"job": job["id"],
                                 "gpu": gpu, "uuid": row["uuid"], "pid": child.pid,
                                 "command": job["command"], "model_sha256": model_sha,
                                 "admission": again, "started_at": datetime.now(timezone.utc).isoformat()})
                            active[job["id"]] = {"job": job, "gpu": gpu,
                                                  "card": card, "legacy": legacy,
                                                  "process": child, "log": log,
                                                  "model_sha": model_sha}
                            pending.pop(job["id"]); ready.pop(0)
                        except Exception as exc:
                            legacy.release(); card.release()
                            failed[job["id"]] = f"prelaunch: {type(exc).__name__}: {exc}"
                            pending.pop(job["id"])
            save(RUN / "state.json", {"accepted": accepted, "failed": failed,
                 "blocked": blocked, "active": {key: {"gpu": item["gpu"], "pid": item["process"].pid}
                                            for key, item in active.items()},
                 "pending": list(pending), "stopped": stop,
                 "updated_at": datetime.now(timezone.utc).isoformat()})
            if stop and not active:
                break
            if pending or active:
                time.sleep(POLL_SECONDS)
    finally:
        if not active:
            save(RUN / "queue-ended.json", {"complete": len(accepted) == 8 and not failed and not blocked,
                 "accepted": accepted, "failed": failed, "blocked": blocked,
                 "pending": list(pending), "stopped": stop,
                 "ended_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    if action == "prepare":
        plan = prepare()
        print(json.dumps({"jobs": len(plan["jobs"]), "episodes": plan["formal_episodes"]}, sort_keys=True))
    else:
        run()


if __name__ == "__main__":
    main()
