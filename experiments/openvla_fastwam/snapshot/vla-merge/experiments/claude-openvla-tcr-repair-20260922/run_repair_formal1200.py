#!/usr/bin/env python3
"""Formal three-repeat evaluation of the frozen, development-selected OpenVLA R2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "openvla-tcr-20260921"
sys.path[:0] = [str(HERE), str(COMMON)]

import run_local as base  # noqa: E402
import run_development40 as screen  # noqa: E402

formal = base.formal
WORK = HERE.parents[2]
CHECKPOINT = screen.MODELS["R2"]
CHECKPOINT_MANIFEST = screen.MANIFESTS["R2"]
REPAIR_TERMINAL = screen.REPAIR / "queue-ended.json"
REPAIR_FINAL = screen.REPAIR / "jobs/R2/build/final-manifest.json"
DEV400_RUN = (WORK / "vla-merge-runtime/experiments/"
              "claude-openvla-tcr-repair-20260922/development400-attempt-01")
DEV400_IDENTITY = DEV400_RUN / "identities.json"
DEV400_AUDIT = WORK / "coordination/2026-09-22/openvla-repair-development400-audit.json"
GPUS = tuple(range(8))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHORT = {"libero_spatial": "spatial", "libero_object": "object",
         "libero_goal": "goal", "libero_10": "long"}
REPEATS = (1, 2, 3)
ADMISSION_MIB = 32 * 1024
RESERVE_MIB = 8 * 1024
MEASURED_EVAL_REQUIREMENT_MIB = 24 * 1024
RUNTIME_FLOOR_MIB = 0


def repeat_contract(repeat: int) -> tuple[str, int, Path]:
    if repeat not in REPEATS:
        raise ValueError("Only formal repeat-01/02/03 are authorized")
    repeat_id = f"repeat-{repeat:02d}"
    return repeat_id, 274000 + repeat, formal.BANK / f"selections/{repeat_id}.json"


def environment(gpu: int) -> dict[str, str]:
    if gpu not in GPUS:
        raise ValueError("GPU outside authorized remote set")
    env = screen.environment(gpu)
    return env


class DynamicCards(screen.DynamicCards):
    pass


def command(suite: str, repeat: int, output: Path) -> list[str]:
    _, _, selection = repeat_contract(repeat)
    return [str(base.PYTHON), str(COMMON / "evaluate.py"),
            "--checkpoint", str(CHECKPOINT), "--suite", suite,
            "--bank", str(formal.BANK), "--selection", str(selection),
            "--output", str(output / "eval")]


def jobs_for(run: Path) -> list[dict]:
    jobs = []
    for repeat in REPEATS:
        repeat_id, eval_seed, selection = repeat_contract(repeat)
        for suite in SUITES:
            output = run / "jobs" / f"R2-{SHORT[suite]}-r{repeat:02d}"
            jobs.append({"id": output.name, "output": str(output), "suite": suite,
                         "repeat": repeat, "repeat_id": repeat_id, "eval_seed": eval_seed,
                         "selection": str(selection), "model": "R2", "expert": "tcr",
                         "smoke": False, "identity": str(run / "identities.json"),
                         "command": command(suite, repeat, output)})
    return jobs


def verify(job: dict) -> dict:
    from evaluate import load_selection, validate_rows
    repeat_id, eval_seed, selection_path = repeat_contract(int(job["repeat"]))
    folder = Path(job["output"]) / "eval"
    selection = load_selection(formal.BANK, selection_path, verify_source_files=False)
    contract = json.loads((folder / "contract.json").read_text())
    if ((selection.repeat_id, selection.eval_seed) != (repeat_id, eval_seed)
            or contract.get("checkpoint") != str(CHECKPOINT.resolve())
            or contract.get("selection_sha256") != selection.selection_sha256
            or contract.get("bank_manifest_sha256") != selection.bank_manifest_sha256
            or contract.get("suite") != job["suite"]
            or contract.get("repeat_id") != repeat_id or contract.get("eval_seed") != eval_seed
            or contract.get("native_chunk") != 8 or contract.get("execute_actions") != 8
            or contract.get("hard_reset") is not True or contract.get("mode") != "eval"):
        raise ValueError("Formal R2 evaluation contract changed")
    base.assert_unchanged(json.loads(Path(job["identity"]).read_text())["files"])
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    successes = validate_rows(rows, selection, job["suite"])
    summary = json.loads((folder / "summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("episodes") != 100
            or summary.get("successes") != successes
            or summary.get("episodes_sha256") != formal.sha(folder / "episodes.jsonl")):
        raise ValueError("Formal R2 summary differs from raw episodes")
    return {"episodes": 100, "successes": successes, "pc_success": float(successes),
            "episodes_sha256": summary["episodes_sha256"],
            "selection_sha256": selection.selection_sha256,
            "bank_manifest_sha256": selection.bank_manifest_sha256}


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    terminal = json.loads(REPAIR_TERMINAL.read_text())
    final = json.loads(REPAIR_FINAL.read_text())
    development = json.loads(DEV400_AUDIT.read_text())
    signal = development.get("paired", {}).get("R2_minus_old_B", {})
    if (terminal.get("status") != "complete" or terminal.get("accepted_jobs") != 2
            or final.get("complete") is not True or final.get("candidate") != "R2"
            or final.get("native_reload_verified") is not True
            or final.get("native_actions_reload_exact") is not True
            or Path(final["checkpoint"]).resolve() != CHECKPOINT.resolve()
            or development.get("accepted") is not True
            or development.get("selection_id") != "development-400"
            or development.get("summaries", {}).get("R2", {}).get("episodes") != 400
            or signal.get("difference_pp", 0) <= 0 or signal.get("wins", 0) <= signal.get("losses", 0)):
        raise ValueError("R2 construction or development selection is not accepted")
    dev_manifest = json.loads((screen.BANK / "manifest.json").read_text())
    if dev_manifest.get("formal_hash_overlap") != 0 or dev_manifest.get("policy_outcomes_read") is not False:
        raise ValueError("Development/formal separation differs")
    if ADMISSION_MIB < MEASURED_EVAL_REQUIREMENT_MIB + RESERVE_MIB:
        raise ValueError("Formal admission does not preserve 8 GiB")
    for repeat in REPEATS:
        repeat_id, eval_seed, selection = repeat_contract(repeat)
        loaded = __import__("evaluate").load_selection(formal.BANK, selection, verify_source_files=True)
        if (loaded.repeat_id, loaded.eval_seed) != (repeat_id, eval_seed):
            raise ValueError("Formal repeat selection differs")

    run.mkdir(parents=True)
    sources = [Path(__file__).resolve(), HERE / "run_development40.py",
               COMMON / "evaluate.py", COMMON / "native_oft.py", CHECKPOINT_MANIFEST,
               REPAIR_TERMINAL, REPAIR_FINAL, DEV400_IDENTITY, DEV400_AUDIT,
               screen.BANK / "manifest.json", formal.BANK / "manifest.json",
               *(repeat_contract(repeat)[2] for repeat in REPEATS)]
    files = {str(path.resolve()): base.file_identity(path.resolve()) for path in sources}
    dev_identity = json.loads(DEV400_IDENTITY.read_text())
    checkpoint_prefix = str(CHECKPOINT.resolve()) + "/"
    checkpoint_files = {path: record for path, record in dev_identity["files"].items()
                        if path.startswith(checkpoint_prefix)}
    expected_names = set(json.loads(CHECKPOINT_MANIFEST.read_text())["files_sha256"])
    if {Path(path).name for path in checkpoint_files} != expected_names:
        raise ValueError("Development identity does not bind the full R2 checkpoint")
    files.update(checkpoint_files)
    base.assert_unchanged(files)
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {"complete": True, "files": files,
        "checkpoint": str(CHECKPOINT.resolve()), "model": "R2",
        "construction_terminal": str(REPAIR_TERMINAL.resolve()),
        "development400_audit": str(DEV400_AUDIT.resolve()),
        "development_signal": signal, "formal_outcomes_read": False,
        "scope": "one frozen R2 build across formal repeat-01/02/03"})

    jobs = jobs_for(run)
    uuids = {gpu: formal.gpu_row(gpu)["uuid"] for gpu in GPUS}
    frozen_env = environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {"schema": "openvla_tcr_repair_R2_formal1200_v1",
        "host": socket.gethostname(), "run": str(run), "gpus": list(GPUS),
        "gpu_uuids": uuids, "max_workers": 4, "min_free_mib": ADMISSION_MIB,
        "measured_eval_requirement_mib": MEASURED_EVAL_REQUIREMENT_MIB,
        "admission_reserve_mib": RESERVE_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
        "jobs": jobs, "episodes": 1200, "repeats": list(REPEATS), "model": "R2",
        "identity_audit": str(identity), "identity_audit_sha256": formal.sha(identity),
        "evaluator_sha256": formal.sha(COMMON / "evaluate.py"),
        "environment_contract": environment_contract, "no_retry": True,
        "selection": {str(r): str(repeat_contract(r)[2]) for r in REPEATS},
        "eval_seed": {str(r): repeat_contract(r)[1] for r in REPEATS},
        "native_actions": 8, "execute_actions": 8, "formal_evaluation": True,
        "development400_audit": str(DEV400_AUDIT),
        "authorization": "Codex C179: freeze a formal protocol after development selection"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "gpus": list(GPUS), "gpu_uuids": uuids,
        "scope": "selected OpenVLA R2 formal repeat-01/02/03 only",
        "checkpoint": str(CHECKPOINT.resolve()), "episodes": 1200, "no_retry": True})


def run(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_tcr_repair_R2_formal1200_v1"
            or plan.get("host") != socket.gethostname() or (run / "started.json").exists()):
        raise RuntimeError("Wrong plan, wrong host or already-started attempt")
    for raw_gpu, expected in plan["gpu_uuids"].items():
        if formal.gpu_row(int(raw_gpu))["uuid"] != expected:
            raise ValueError("GPU UUID changed")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.GPUS = DynamicCards(plan["gpus"])
    formal.FLOOR_MIB = RUNTIME_FLOOR_MIB
    formal.EVALUATOR = COMMON / "evaluate.py"
    formal.IDENTITIES = run / "identities.json"
    formal.scientific_environment = environment
    formal.verify_job = verify
    formal.save = base.save_with_actual_counts
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
