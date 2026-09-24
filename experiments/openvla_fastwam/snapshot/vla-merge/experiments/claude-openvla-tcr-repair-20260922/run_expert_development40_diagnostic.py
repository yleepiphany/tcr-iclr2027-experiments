#!/usr/bin/env python3
"""Matched 40-reset expert diagnostic for the OpenVLA development bank.

This is a control for reset/evaluator difficulty, not a TCR selection or a
formal paper result. It uses released expert checkpoints and the exact
frozen development-40 initial states; no score controls job scheduling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess

import evaluate_development as evaluator
import run_development40 as screen
import run_expert_formal_repeats23 as experts
import run_local as base


HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BANK, SELECTION = screen.BANK, screen.SELECTION
SUITES, SHORT = screen.SUITES, screen.SHORT
ADMISSION_MIB = 56 * 1024
RUNTIME_FLOOR_MIB = 8 * 1024


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    prior = experts.prior_gate()
    selection = evaluator.load_development_selection(BANK, SELECTION)
    if selection.selection_id != "development-40" or len(selection.tasks) != 40:
        raise ValueError("Frozen development-40 selection differs")
    by_suite = {row["suite"]: Path(row["local_path"]).resolve()
                for row in prior["experts"].values()}
    if set(by_suite) != set(SUITES):
        raise ValueError("Released expert suite coverage differs")
    run.mkdir(parents=True, exist_ok=False)
    for suite in SUITES:
        output = run / "preflight" / suite
        command = [str(base.PYTHON), str(HERE / "evaluate_development.py"),
                   "--checkpoint", str(by_suite[suite]), "--suite", suite,
                   "--bank", str(BANK), "--selection", str(SELECTION),
                   "--output", str(output), "--preflight-only"]
        subprocess.run(command, cwd=WORK / "vla-merge", env=screen.environment(0),
                       capture_output=True, text=True, check=True)
        contract = json.loads((output / "contract.json").read_text())
        if (contract.get("mode") != "preflight" or contract.get("suite") != suite
                or contract.get("episodes") != 10
                or contract.get("selection_id") != "development-40"):
            raise ValueError("Expert development child preflight differs")
    files = dict(prior["files"])
    for path in (Path(__file__).resolve(), HERE / "evaluate_development.py",
                 HERE / "dev_bank.py", SELECTION, BANK / "manifest.json",
                 experts.PRIOR / "identities.json"):
        files[str(path.resolve())] = base.file_identity(path.resolve())
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {"complete": True, "files": files,
        "experts": {suite: str(path) for suite, path in by_suite.items()},
        "selection": str(SELECTION.resolve()), "selection_id": "development-40",
        "episodes": 40, "diagnostic_only": True, "formal_outcomes_read": False})
    jobs = []
    for suite in SUITES:
        output = run / "jobs" / f"expert-{SHORT[suite]}"
        command = [str(base.PYTHON), str(HERE / "evaluate_development.py"),
                   "--checkpoint", str(by_suite[suite]), "--suite", suite,
                   "--bank", str(BANK), "--selection", str(SELECTION),
                   "--output", str(output / "eval")]
        jobs.append({"id": output.name, "model": "expert", "suite": suite,
                     "checkpoint": str(by_suite[suite]), "output": str(output),
                     "identity": str(identity), "command": command})
    uuids = {gpu: screen.formal.gpu_row(gpu)["uuid"] for gpu in screen.GPUS}
    frozen_env = screen.environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {"schema": "openvla_expert_development40_diagnostic_v1",
            "host": socket.gethostname(), "run": str(run), "gpus": list(screen.GPUS),
            "gpu_uuids": uuids, "max_workers": 1, "min_free_mib": ADMISSION_MIB,
            "admission_reserve_mib": RUNTIME_FLOOR_MIB,
            "runtime_floor_mib": RUNTIME_FLOOR_MIB, "jobs": jobs,
            "episodes": 40, "identity_audit": str(identity),
            "identity_audit_sha256": screen.formal.sha(identity),
            "evaluator_sha256": screen.formal.sha(HERE / "evaluate_development.py"),
            "environment_contract": environment_contract, "no_retry": True,
            "no_score_based_scheduling": True, "formal_evaluation": False,
            "selection": str(SELECTION), "selection_id": "development-40",
            "authorization": "User requested explanation of very low OpenVLA merge rate versus strong experts"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "scope": "matched expert reset/evaluator diagnostic, 4 suites x 10 episodes",
        "episodes": 40, "formal_evaluation": False, "no_retry": True})


def run_queue(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_expert_development40_diagnostic_v1"
            or plan.get("host") != socket.gethostname() or (run / "started.json").exists()):
        raise ValueError("Wrong host/plan or already started")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    for gpu, uuid in plan["gpu_uuids"].items():
        if screen.formal.gpu_row(int(gpu))["uuid"] != uuid:
            raise ValueError("GPU identity changed")
    screen.formal.GPUS = screen.DynamicCards(plan["gpus"])
    screen.formal.FLOOR_MIB = RUNTIME_FLOOR_MIB
    screen.formal.EVALUATOR = HERE / "evaluate_development.py"
    screen.formal.IDENTITIES = run / "identities.json"
    screen.formal.scientific_environment = screen.environment
    screen.formal.verify_job = screen.verify
    screen.formal.save = screen.save_with_actual_counts
    screen.formal.run_queue(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare(args.run.resolve()) if args.prepare else run_queue(args.run.resolve())


if __name__ == "__main__":
    main()
