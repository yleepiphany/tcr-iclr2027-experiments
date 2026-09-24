#!/usr/bin/env python3
"""Paired 400-episode development check of the ordered-auxiliary R2 build."""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path

import evaluate_development as evaluator
import run_development400 as previous
import run_development40 as screen
import run_ordered_candidate as build


HERE = Path(__file__).resolve().parent
base, formal = screen.base, screen.formal
WORK, BANK = screen.WORK, screen.BANK
SELECTION = BANK / "selection-400.json"
MODEL = build.CANDIDATE
BUILD = (WORK / "vla-merge-runtime/experiments/"
         "claude-openvla-tcr-repair-20260922/ordered-aux-build-attempt-01")
CHECKPOINT = BUILD / "jobs" / MODEL / "build/pass-B/export/checkpoint"
MANIFEST = BUILD / "jobs" / MODEL / "build/pass-B/manifest.json"
BASELINE_AUDIT = WORK / "coordination/2026-09-22/openvla-repair-development400-audit.json"
ADMISSION_MIB = 56 * 1024
RUNTIME_FLOOR_MIB = 8 * 1024
MAX_WORKERS = 1


def checkpoint_identity() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if (manifest.get("complete") is not True
            or Path(manifest["checkpoint"]).resolve() != CHECKPOINT.resolve()
            or manifest.get("success_evaluated") is not False
            or manifest.get("auxiliary_replay") != "ordered_current_linear_expert_only"):
        raise ValueError("Ordered-auxiliary checkpoint manifest differs")
    expected = manifest["files_sha256"]
    if set(expected) != {path.name for path in CHECKPOINT.iterdir() if path.is_file()}:
        raise ValueError("Checkpoint file set differs")
    files = {}
    for name, digest in expected.items():
        path = (CHECKPOINT / name).resolve()
        receipt = base.file_identity(path)
        if receipt["sha256"] != digest:
            raise ValueError("Checkpoint file digest differs")
        files[str(path)] = receipt
    files[str(MANIFEST.resolve())] = base.file_identity(MANIFEST.resolve())
    return files


def jobs_for(run: Path) -> list[dict]:
    jobs = []
    for suite in screen.SUITES:
        output = run / "jobs" / f"{MODEL}-{screen.SHORT[suite]}"
        command = [str(base.PYTHON), str(HERE / "evaluate_development.py"),
            "--checkpoint", str(CHECKPOINT), "--suite", suite,
            "--bank", str(BANK), "--selection", str(SELECTION),
            "--output", str(output / "eval")]
        jobs.append({"id": output.name, "model": MODEL, "suite": suite,
                     "checkpoint": str(CHECKPOINT.resolve()), "output": str(output),
                     "identity": str(run / "identities.json"), "command": command})
    return jobs


def verify(job: dict) -> dict:
    result = previous.verify(job)
    contract = json.loads((Path(job["output"]) / "eval/contract.json").read_text())
    if (contract.get("hard_reset") is not True
            or contract.get("native_chunk") != 8
            or contract.get("execute_actions") != 8
            or result.get("episodes") != 100):
        raise ValueError("Native paired development evaluation contract differs")
    return result


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    selection = evaluator.load_development_selection(BANK, SELECTION)
    terminal = json.loads((BUILD / "queue-ended.json").read_text())
    if (terminal.get("status") != "complete" or terminal.get("accepted_jobs") != 1
            or terminal.get("stopped") is not False):
        raise ValueError("Ordered build has not passed terminal audit")
    baseline = json.loads(BASELINE_AUDIT.read_text())
    if (baseline.get("accepted") is not True
            or baseline.get("selection_id") != "development-400"):
        raise ValueError("Existing paired R2 development bank is not accepted")
    if ADMISSION_MIB < screen.MEASURED_EVAL_REQUIREMENT_MIB + screen.RESERVE_MIB:
        raise ValueError("Evaluation admission lacks 8 GiB reserve")
    model_files = checkpoint_identity()
    run.mkdir(parents=True)
    for suite in screen.SUITES:
        output = run / "preflight" / suite
        command = [str(base.PYTHON), str(HERE / "evaluate_development.py"),
                   "--checkpoint", str(CHECKPOINT), "--suite", suite,
                   "--bank", str(BANK), "--selection", str(SELECTION),
                   "--output", str(output), "--preflight-only"]
        subprocess.run(command, env=screen.environment(0), check=True,
                       cwd=str(WORK / "vla-merge"), capture_output=True, text=True)
        contract = json.loads((output / "contract.json").read_text())
        if (contract.get("mode") != "preflight" or contract.get("episodes") != 100
                or contract.get("selection_id") != "development-400"):
            raise ValueError("Frozen development bank child preflight differs")
    sources = [Path(__file__).resolve(), HERE / "evaluate_development.py",
               HERE / "dev_bank.py", HERE / "run_development400.py",
               screen.COMMON / "native_oft.py", screen.COMMON / "evaluate.py",
               BANK / "manifest.json", SELECTION, BASELINE_AUDIT,
               BUILD / "queue-ended.json"]
    files = {str(path.resolve()): base.file_identity(path.resolve()) for path in sources}
    files.update(model_files)
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {"complete": True, "files": files,
        "model": MODEL, "checkpoint": str(CHECKPOINT.resolve()),
        "selection": str(SELECTION.resolve()), "selection_id": selection.selection_id,
        "paired_baseline": str(BASELINE_AUDIT.resolve()),
        "formal_outcomes_read": False})
    uuids = {gpu: formal.gpu_row(gpu)["uuid"] for gpu in screen.GPUS}
    frozen_env = screen.environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {"schema": "openvla_r2_ordered_aux_development400_v1",
        "host": socket.gethostname(), "run": str(run), "gpus": list(screen.GPUS),
        "gpu_uuids": uuids, "max_workers": MAX_WORKERS,
        "min_free_mib": ADMISSION_MIB,
        "admission_reserve_mib": screen.RESERVE_MIB,
        "measured_eval_requirement_mib": screen.MEASURED_EVAL_REQUIREMENT_MIB,
        "runtime_floor_mib": RUNTIME_FLOOR_MIB,
        "jobs": jobs_for(run), "episodes": 400, "models": [MODEL],
        "identity_audit": str(identity), "identity_audit_sha256": formal.sha(identity),
        "evaluator_sha256": formal.sha(HERE / "evaluate_development.py"),
        "environment_contract": environment_contract, "no_retry": True,
        "selection": str(SELECTION), "selection_id": selection.selection_id,
        "eval_seed": selection.eval_seed, "formal_evaluation": False,
        "paired_R2_episodes_reused": 400,
        "authorization": "User explicitly requested 400 development episodes for corrected OpenVLA candidate"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "scope": "one ordered-auxiliary R2 candidate on frozen development-400 bank",
        "episodes": 400, "formal_evaluation": False, "no_retry": True})


def run(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_r2_ordered_aux_development400_v1"
            or plan.get("host") != socket.gethostname() or (run / "started.json").exists()):
        raise RuntimeError("Wrong host, wrong plan or already-started attempt")
    for raw_gpu, expected in plan["gpu_uuids"].items():
        if formal.gpu_row(int(raw_gpu))["uuid"] != expected:
            raise ValueError("GPU UUID changed")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.GPUS = screen.DynamicCards(plan["gpus"])
    formal.FLOOR_MIB = RUNTIME_FLOOR_MIB
    formal.EVALUATOR = HERE / "evaluate_development.py"
    formal.IDENTITIES = run / "identities.json"
    formal.scientific_environment = screen.environment
    formal.verify_job = verify
    formal.save = previous.save_with_actual_counts
    formal.run_queue(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    target = args.run.resolve()
    prepare(target) if args.prepare else run(target)


if __name__ == "__main__":
    main()
