#!/usr/bin/env python3
"""No-retry A∪B build with corrected ridge-reference manifest metadata."""
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
RUN = ROOT / "union-build-attempt-03"
SOLVER = HERE / "materialize_union_v9.py"
PYTHON = WORK / "pi05_lora_finetune_v2_20260826/.venv/bin/python"
THREE_LEVEL = VLA / "experiments/claude-three-level-main-20260921"
FORMAL = VLA / "experiments/claude-firstpass-cause-20260920"
SOURCE_AUDIT = WORK / "coordination/2026-09-23/appendix-abc-pools-content-audit.json"
ROW_BUDGET = WORK / "coordination/2026-09-23/appendix-row-budget-contract.json"
MIN_FREE_MIB = 78_000
RUNTIME_FLOOR_MIB = 8 * 1024
GPUS = tuple(range(8))
POLL_SECONDS = 30

sys.path[:0] = [str(HERE), str(THREE_LEVEL), str(VLA / "scripts"), str(FORMAL)]
import appendix_contract as contract  # noqa: E402
import appendix_union_contract as union  # noqa: E402
import card_flock  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
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


def output_path() -> Path:
    return ROOT / "checkpoints-v14/A_union_B"


def source_gate() -> tuple[dict[str, dict], dict[str, dict]]:
    audit = json.loads(SOURCE_AUDIT.read_text())
    budget = json.loads(ROW_BUDGET.read_text())
    if (audit.get("accepted") is not True or audit.get("jobs") != 12
            or audit.get("samples") != 1800 or audit.get("heldout_disjoint") is not True
            or budget.get("B_second_pass_rows", {}).get("base") != contract.ROWS[2]
            or budget.get("A_first_pass_rows") != contract.ROWS[1]):
        raise ValueError("A/B/C content or row-budget gate differs")
    bank = json.loads(contract.DENSE_BANK.read_text())
    if bank.get("status") != "passed_parameter_exact" or len(bank.get("experts") or []) != 4:
        raise ValueError("dense expert bank is not fully accepted")
    experts = {row["name"]: row for row in bank["experts"]}
    if set(experts) != set(contract.SUITES):
        raise ValueError("expert bank suite set differs")
    bindings = {}
    for pass_index, pool in contract.POOLS.items():
        for name in contract.SUITES:
            cache = contract.INPUTS / pool / "inputs" / name / "across"
            manifest, replay = cache / "replay.json", cache / "replay.safetensors"
            dense_path = Path(experts[name]["dense_checkpoint"]["path"])
            contract.validate_trace(json.loads(manifest.read_text()), dense_path, name, pass_index)
            bindings[f"{pool}/{name}"] = {
                "manifest": str(manifest), "manifest_sha256": sha(manifest),
                "replay": str(replay), "replay_sha256": sha(replay),
                "dense_model_sha256": experts[name]["dense_checkpoint"]["model_sha256"],
            }
    return experts, bindings


def command_for(experts: dict[str, dict]) -> list[str]:
    config = RUN / "configs" / "union.json"
    command = [str(PYTHON), "-u", str(SOLVER), f"--ablation-config={config}",
               f"--dense-expert-bank={contract.DENSE_BANK}",
               f"--base-model={WORK / 'pi05_lora_finetune_v2_20260826/ckpt/theta0/checkpoints/000200/pretrained_model'}",
               f"--prior-model={contract.SOUP}", f"--output={output_path()}",
               "--ridge-ratio=.05", "--ridge-scale=feature_energy",
               "--max-correction-ratio=3", "--max-rows-per-sample=16",
               "--device=cuda", "--allow-mixed-calibration-policies",
               "--allow-prior-calibration-mismatch",
               "--expert-loss-normalization=prior", "--replay-prefix=merged"]
    for name in contract.SUITES:
        command.append(f"--expert={name}={experts[name]['source_adapter']['path']}")
        for pool in union.POOLS:
            cache = contract.INPUTS / pool / "inputs" / name / "across"
            command.extend([f"--calibration={name}={cache / 'replay.safetensors'}",
                            f"--manifest={name}={cache / 'replay.json'}"])
    return command


def priority_dependencies() -> list[Path]:
    return [ROOT / stem / "queue-ended.json" for stem in (
        "ab-reference-build-attempt-01", "a-followons-build-attempt-01",
        "uniform-build-attempt-01", "scope-build-attempt-01",
        "cache-followons-build-attempt-01")]


def prepare() -> dict:
    if RUN.exists() or output_path().exists():
        raise FileExistsError("frozen one-pass union queue/checkpoint already exists")
    experts, bindings = source_gate()
    job = {"id": union.ARM, "pass_index": 1, "needs": [],
           "output": str(output_path()), "config": str(RUN / "configs/union.json"),
           "command": command_for(experts)}
    plan = {"schema": "pi05_fixed_appendix_union_build_queue_v1",
            "created_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
            "jobs": [job], "trace_bindings": bindings,
            "source_audit_sha256": sha(SOURCE_AUDIT), "row_budget_sha256": sha(ROW_BUDGET),
            "dense_bank_sha256": sha(contract.DENSE_BANK),
            "solver_sha256": sha(SOLVER), "contract_sha256": sha(HERE / "appendix_union_contract.py"),
            "priority_dependencies": [str(path) for path in priority_dependencies()],
            "min_free_mib": MIN_FREE_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
            "max_active": 1, "permitted_gpus": list(GPUS), "no_retry": True,
            "training": False, "formal_evaluation_episodes": 0,
            "paper_scope": "single Soup->(A_new union B_new) solve; cap10 A and cap16 B together; one 400-episode formal repeat"}
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / "plan.json", plan)
    return plan


def validate_plan(plan: dict) -> None:
    if (plan.get("schema") != "pi05_fixed_appendix_union_build_queue_v1"
            or plan.get("host") != socket.gethostname()
            or [job.get("id") for job in plan.get("jobs", [])] != [union.ARM]
            or plan.get("source_audit_sha256") != sha(SOURCE_AUDIT)
            or plan.get("row_budget_sha256") != sha(ROW_BUDGET)
            or plan.get("dense_bank_sha256") != sha(contract.DENSE_BANK)
            or plan.get("solver_sha256") != sha(SOLVER)
            or plan.get("contract_sha256") != sha(HERE / "appendix_union_contract.py")
            or plan.get("priority_dependencies") != [str(path) for path in priority_dependencies()]):
        raise ValueError("one-pass union build plan or source changed")
    for key, row in plan["trace_bindings"].items():
        if row["manifest_sha256"] != sha(Path(row["manifest"])) or \
                row["replay_sha256"] != sha(Path(row["replay"])):
            raise ValueError(f"one-pass union trace changed: {key}")


def config_for(job: dict, accepted: dict[str, dict]) -> dict:
    config = {"schema": union.SCHEMA, "arm": union.ARM, "variant": "full",
              "pass_index": 1, "pools": list(union.POOLS),
              "row_caps": {"A_new": 10, "B_new": 16},
              "expected_realized_rows": union.ROW_TOTAL,
              "mass_rule": "relative_prior_error", "expert_loss_normalization": "prior",
              "replay_prefix": "merged", "module_count": 418,
              "dense_expert_bank_sha256": sha(contract.DENSE_BANK),
              "historical_table1_full": False, "evaluation_repeat": 1,
              "single_pass": True, "parent_arm": "Soup",
              "start_point": str(contract.SOUP),
              "start_point_sha256": sha(contract.SOUP / "model.safetensors"),
              "ridge_source_manifest": str(contract.RIDGE),
              "ridge_source_sha256": sha(contract.RIDGE)}
    union.validate_config(config)
    path = Path(job["config"])
    if path.exists():
        raise FileExistsError(path)
    save(path, config)
    return config


def verify(job: dict) -> dict:
    output = Path(job["output"])
    model, manifest = output / "model.safetensors", output / "block_regmeanpp_manifest.json"
    if not model.is_file() or not manifest.is_file():
        raise ValueError("one-pass union solve produced no model and manifest")
    data = json.loads(manifest.read_text())
    modules = data.get("modules") or {}
    rows = union.validate_rows(modules, list(contract.SUITES))
    if (data.get("fixed_appendix_arm") != union.ARM
            or data.get("fixed_appendix_pass") != 1
            or data.get("realized_row_total") != rows
            or data.get("modified_tensor_count") != 422
            or data.get("model_sha256") != sha(model)
            or data.get("ablation_config_sha256") != sha(Path(job["config"]))
            or data.get("fixed_ridge_reference") != str(contract.RIDGE)
            or data.get("calibration_source_counts") != {name: 2 for name in contract.SUITES}):
        raise ValueError("one-pass union model identity, coverage or hash differs")
    union.validate_config(json.loads(Path(job["config"]).read_text()))
    return {"id": union.ARM, "model_sha256": data["model_sha256"],
            "manifest_sha256": sha(manifest), "modules": 418, "rows": rows}


def run() -> None:
    plan = json.loads((RUN / "plan.json").read_text())
    if (RUN / "started.json").exists():
        raise ValueError("no restart within this attempt")
    validate_plan(plan)
    save(RUN / "started.json", {"pid": os.getpid(), "host": socket.gethostname(),
         "kernel_starttime": int(Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]),
         "started_at": datetime.now(timezone.utc).isoformat()})
    pending = {job["id"]: job for job in plan["jobs"]}
    accepted: dict[str, dict] = {}
    failed: dict[str, str] = {}
    skipped: dict[str, str] = {}
    active: dict[str, dict] = {}
    stop = False

    def on_signal(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        while pending or active:
            if stop:
                for item in active.values():
                    if item["process"].poll() is None:
                        os.killpg(item["process"].pid, signal.SIGTERM)
            for job_id, item in list(active.items()):
                child = item["process"]
                code = child.poll()
                if code is None and not stop and gpu_rows()[item["gpu"]]["free_mib"] < RUNTIME_FLOOR_MIB:
                    os.killpg(child.pid, signal.SIGTERM)
                    item["resource_floor"] = True
                if code is None:
                    continue
                child.wait()
                item["log"].close()
                item["slot"].release(); item["legacy"].release(); item["card"].release()
                active.pop(job_id)
                try:
                    result = verify(item["job"]) if code == 0 and not item.get("resource_floor") else None
                    error = None if result else f"child exit {code}; resource_floor={item.get('resource_floor', False)}"
                except Exception as exc:
                    result, error = None, f"artifact gate: {type(exc).__name__}: {exc}"
                save(RUN / "exits" / f"{job_id}.json", {"job": job_id,
                     "returncode": code, "accepted": result is not None,
                     "artifact": result, "error": error,
                     "finished_at": datetime.now(timezone.utc).isoformat()})
                if result:
                    accepted[job_id] = result
                else:
                    failed[job_id] = error
            for job_id, job in list(pending.items()):
                if any(dep in failed or dep in skipped for dep in job["needs"]):
                    skipped[job_id] = "required A build failed"
                    pending.pop(job_id)
            if not stop and not failed and not active:
                ready = [job for job in pending.values()
                         if all(dep in accepted for dep in job["needs"])]
                if not all(Path(path).is_file() for path in plan["priority_dependencies"]):
                    ready = []
                if ready:
                    rows = gpu_rows()
                    for gpu in sorted(GPUS, key=lambda idx: rows[idx]["free_mib"], reverse=True):
                        row = rows[gpu]
                        if row["free_mib"] < MIN_FREE_MIB:
                            continue
                        job = ready[0]
                        card = card_flock.take_card(gpu, row["uuid"], job["id"], "pi05-fixed-appendix")
                        if card is None:
                            continue
                        legacy = card_flock._try_lock(
                            RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
                            job["id"], {"gpu": gpu, "job": job["id"], "stage": "pi05-fixed-appendix"})
                        if legacy is None:
                            card.release(); continue
                        slot = card_flock.take_build_slot(job["id"])
                        again = gpu_rows()[gpu]
                        if slot is None or again["uuid"] != row["uuid"] or again["free_mib"] < MIN_FREE_MIB:
                            if slot: slot.release()
                            legacy.release(); card.release(); continue
                        try:
                            config_for(job, accepted)
                            if Path(job["output"]).exists():
                                raise FileExistsError(job["output"])
                            log_path = RUN / "logs" / f"{job['id']}.log"
                            log_path.parent.mkdir(parents=True, exist_ok=True)
                            log = log_path.open("x")
                            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                   "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                   "PYTHONPATH": ":".join([str(HERE), str(THREE_LEVEL), str(VLA / "scripts"),
                                                         str(WORK / "pi05_lora_finetune_v2_20260826/src"),
                                                         os.environ.get("PYTHONPATH", "")])}
                            child = subprocess.Popen(job["command"], cwd=WORK, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True, env=env,
                                                     pass_fds=(card.handle, legacy.handle, slot.handle))
                            save(RUN / "launches" / f"{job['id']}.json", {"job": job["id"],
                                 "gpu": gpu, "uuid": row["uuid"], "pid": child.pid,
                                 "command": job["command"], "config_sha256": sha(Path(job["config"])),
                                 "admission": again, "started_at": datetime.now(timezone.utc).isoformat()})
                            active[job["id"]] = {"job": job, "gpu": gpu, "card": card,
                                                 "legacy": legacy, "slot": slot,
                                                 "process": child, "log": log}
                            pending.pop(job["id"])
                        except Exception as exc:
                            slot.release(); legacy.release(); card.release()
                            failed[job["id"]] = f"prelaunch: {type(exc).__name__}: {exc}"
                            pending.pop(job["id"])
                        break
            save(RUN / "state.json", {"accepted": accepted, "failed": failed,
                  "skipped": skipped,
                  "active": {key: {"gpu": item["gpu"], "pid": item["process"].pid}
                             for key, item in active.items()},
                  "pending": list(pending), "stopped": stop,
                  "updated_at": datetime.now(timezone.utc).isoformat()})
            if stop and not active:
                break
            if pending or active:
                time.sleep(POLL_SECONDS)
    finally:
        if not active:
            save(RUN / "queue-ended.json", {"complete": len(accepted) == 1 and not failed and not skipped,
                  "accepted": accepted, "failed": failed, "skipped": skipped,
                  "pending": list(pending), "stopped": stop,
                  "ended_at": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    action = parser.parse_args().action
    if action == "prepare":
        plan = prepare()
        print(json.dumps({"jobs": len(plan["jobs"]), "trace_bindings": len(plan["trace_bindings"])}))
    else:
        run()


if __name__ == "__main__":
    main()
