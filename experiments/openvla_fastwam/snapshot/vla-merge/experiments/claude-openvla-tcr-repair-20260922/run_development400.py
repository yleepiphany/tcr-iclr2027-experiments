#!/usr/bin/env python3
"""Frozen 400-episode OpenVLA repair confirmation for old B and positive R2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_development as evaluator  # noqa: E402
import run_development40 as screen  # noqa: E402

formal, base = screen.formal, screen.base
WORK, BANK = screen.WORK, screen.BANK
SELECTION = BANK / "selection-400.json"
GPUS, SUITES, SHORT = screen.GPUS, screen.SUITES, screen.SHORT
MODELS = {name: screen.MODELS[name] for name in ("old_B", "R2")}
ADMISSION_MIB = screen.ADMISSION_MIB
RESERVE_MIB = screen.RESERVE_MIB
MEASURED_EVAL_REQUIREMENT_MIB = screen.MEASURED_EVAL_REQUIREMENT_MIB
RUNTIME_FLOOR_MIB = screen.RUNTIME_FLOOR_MIB
DEVELOPMENT40_AUDIT = WORK / "coordination/2026-09-22/openvla-repair-development40-audit.json"

environment = screen.environment
DynamicCards = screen.DynamicCards


def jobs_for(run: Path) -> list[dict]:
    jobs = []
    for model, checkpoint in MODELS.items():
        for suite in SUITES:
            output = run / "jobs" / f"{model}-{SHORT[suite]}"
            command = [str(base.PYTHON), str(HERE / "evaluate_development.py"),
                       "--checkpoint", str(checkpoint), "--suite", suite,
                       "--bank", str(BANK), "--selection", str(SELECTION),
                       "--output", str(output / "eval")]
            jobs.append({"id": output.name, "model": model, "suite": suite,
                         "checkpoint": str(checkpoint.resolve()), "output": str(output),
                         "identity": str(run / "identities.json"), "command": command})
    return jobs


def verify(job: dict) -> dict:
    folder = Path(job["output"]) / "eval"
    selection = evaluator.load_development_selection(BANK, SELECTION)
    contract = json.loads((folder / "contract.json").read_text())
    if (contract.get("checkpoint") != job["checkpoint"]
            or contract.get("suite") != job["suite"]
            or contract.get("selection_id") != "development-400"
            or contract.get("selection_sha256") != selection.selection_sha256
            or contract.get("bank_manifest_sha256") != selection.manifest_sha256
            or contract.get("eval_seed") != selection.eval_seed
            or contract.get("episodes") != 100
            or contract.get("mode") != "development_eval"
            or contract.get("formal_evaluation") is not False
            or contract.get("used_for_candidate_selection") is not True):
        raise ValueError("Development-400 contract differs")
    base.assert_unchanged(json.loads(Path(job["identity"]).read_text())["files"])
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    successes = evaluator.validate_rows(rows, selection, job["suite"])
    summary = json.loads((folder / "summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("episodes") != 100
            or summary.get("successes") != successes
            or summary.get("episodes_sha256") != formal.sha(folder / "episodes.jsonl")):
        raise ValueError("Development-400 summary differs")
    return {"episodes": 100, "successes": successes, "pc_success": float(successes),
            "episodes_sha256": summary["episodes_sha256"],
            "selection_sha256": selection.selection_sha256,
            "bank_manifest_sha256": selection.manifest_sha256}


def save_with_actual_counts(path, value):
    if Path(path).name == "queue-ended.json":
        value = dict(value)
        accepted = [row for row in value.get("finished", [])
                    if row.get("outcome") == "success" and "artifacts" in row]
        value["episodes"] = sum(row["artifacts"]["episodes"] for row in accepted)
        value["accepted_jobs"] = len(accepted)
        value["development_only"] = True
    base.BASE_SAVE(path, value)


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    selection = evaluator.load_development_selection(BANK, SELECTION)
    audit = json.loads(DEVELOPMENT40_AUDIT.read_text())
    signal = audit.get("paired", {}).get("R2_minus_old_B", {})
    if (audit.get("accepted") is not True or audit.get("selection_id") != "development-40"
            or audit.get("summaries", {}).get("R2", {}).get("episodes") != 40
            or signal.get("difference_pp", 0) <= 0 or signal.get("wins", 0) <= signal.get("losses", 0)):
        raise ValueError("R2 lacks the pre-specified positive development-40 signal")
    run.mkdir(parents=True)
    screen.run_preflights(run)
    # Replace the preflight contracts with the actual 400-selection child preflights.
    for suite in SUITES:
        output = run / "preflight400" / suite
        command = [str(base.PYTHON), str(HERE / "evaluate_development.py"),
                   "--checkpoint", str(MODELS["old_B"]), "--suite", suite,
                   "--bank", str(BANK), "--selection", str(SELECTION),
                   "--output", str(output), "--preflight-only"]
        import subprocess
        subprocess.run(command, env=environment(0), check=True,
                       cwd=str(WORK / "vla-merge"), capture_output=True, text=True)
        contract = json.loads((output / "contract.json").read_text())
        if contract.get("selection_id") != "development-400" or contract.get("episodes") != 100:
            raise ValueError(f"Development-400 child preflight differs: {suite}")
    sources = [Path(__file__).resolve(), HERE / "run_development40.py",
               HERE / "evaluate_development.py",
               HERE / "dev_bank.py", HERE / "audit_development.py",
               screen.COMMON / "native_oft.py", screen.COMMON / "evaluate.py",
               BANK / "manifest.json", SELECTION, DEVELOPMENT40_AUDIT, *screen.BUILD_TERMINALS]
    files = {str(path.resolve()): base.file_identity(path.resolve()) for path in sources}
    for model in MODELS:
        files.update(screen._checkpoint_identity(model))
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {"complete": True, "files": files,
        "models": {name: str(path.resolve()) for name, path in MODELS.items()},
        "selection": str(SELECTION.resolve()), "selection_id": "development-400",
        "development40_audit": str(DEVELOPMENT40_AUDIT.resolve()),
        "positive_signal": signal, "formal_outcomes_read": False})
    jobs = jobs_for(run)
    uuids = {gpu: formal.gpu_row(gpu)["uuid"] for gpu in GPUS}
    frozen_env = environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {"schema": "openvla_tcr_repair_development400_v1",
        "host": socket.gethostname(), "run": str(run), "gpus": list(GPUS),
        "gpu_uuids": uuids, "max_workers": 4, "min_free_mib": ADMISSION_MIB,
        "measured_eval_requirement_mib": MEASURED_EVAL_REQUIREMENT_MIB,
        "admission_reserve_mib": RESERVE_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
        "jobs": jobs, "episodes": 800, "models": list(MODELS),
        "identity_audit": str(identity), "identity_audit_sha256": formal.sha(identity),
        "evaluator_sha256": formal.sha(HERE / "evaluate_development.py"),
        "environment_contract": environment_contract, "no_retry": True,
        "selection": str(SELECTION), "selection_id": selection.selection_id,
        "eval_seed": selection.eval_seed, "formal_evaluation": False,
        "development40_audit": str(DEVELOPMENT40_AUDIT), "positive_signal": signal,
        "authorization": "Codex C179: expand positive candidate and old B comparator to frozen development-400"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "gpus": list(GPUS), "gpu_uuids": uuids,
        "scope": "development-400 only: positive R2 and old-B paired comparator",
        "episodes": 800, "formal_evaluation": False, "no_retry": True})


def run(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_tcr_repair_development400_v1"
            or plan.get("host") != socket.gethostname() or (run / "started.json").exists()):
        raise RuntimeError("Wrong plan, wrong host or already-started attempt")
    for raw_gpu, expected in plan["gpu_uuids"].items():
        if formal.gpu_row(int(raw_gpu))["uuid"] != expected:
            raise ValueError("GPU UUID changed")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.GPUS = DynamicCards(plan["gpus"])
    formal.FLOOR_MIB = RUNTIME_FLOOR_MIB
    formal.EVALUATOR = HERE / "evaluate_development.py"
    formal.IDENTITIES = run / "identities.json"
    formal.scientific_environment = environment
    formal.verify_job = verify
    formal.save = save_with_actual_counts
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
