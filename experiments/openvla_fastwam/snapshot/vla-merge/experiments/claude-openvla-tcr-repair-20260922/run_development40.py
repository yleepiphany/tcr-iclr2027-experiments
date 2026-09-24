#!/usr/bin/env python3
"""Frozen 40-episode OpenVLA repair screen for Soup, old A/B, R1 and R2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "openvla-tcr-20260921"
sys.path[:0] = [str(HERE), str(COMMON)]

import evaluate_development as evaluator  # noqa: E402
import run_local as base  # noqa: E402

formal = base.formal
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge-runtime/experiments/openvla-tcr-20260921"
REPAIR = (WORK / "vla-merge-runtime/experiments/"
          "claude-openvla-tcr-repair-20260922/candidate-build-attempt-02")
VLA_MERGE_SRC = WORK / "vla-merge/src"
BANK = (WORK / "vla-merge-runtime/experiments/"
        "claude-openvla-tcr-repair-20260922/dev-bank-v1")
SELECTION = BANK / "selection-40.json"
GPUS = tuple(range(8))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHORT = {"libero_spatial": "spatial", "libero_object": "object",
         "libero_goal": "goal", "libero_10": "long"}
ADMISSION_MIB = 32 * 1024
RESERVE_MIB = 8 * 1024
MEASURED_EVAL_REQUIREMENT_MIB = 24 * 1024
RUNTIME_FLOOR_MIB = 0

MODELS = {
    "soup": ROOT / "soup-prior-attempt-02/checkpoint",
    "old_A": (ROOT / "two-pass-build-attempt-03-remote/jobs/openvla-tcr-two-pass/"
              "build/pass-A/export/checkpoint"),
    "old_B": (ROOT / "two-pass-build-attempt-03-remote/jobs/openvla-tcr-two-pass/"
              "build/pass-B/export/checkpoint"),
    "R1": REPAIR / "jobs/R1/build/pass-B/export/checkpoint",
    "R2": REPAIR / "jobs/R2/build/pass-B/export/checkpoint",
}
MANIFESTS = {
    "soup": ROOT / "soup-prior-attempt-02/manifest.json",
    "old_A": (ROOT / "two-pass-build-attempt-03-remote/jobs/openvla-tcr-two-pass/"
              "build/pass-A/export/manifest.json"),
    "old_B": (ROOT / "two-pass-build-attempt-03-remote/jobs/openvla-tcr-two-pass/"
              "build/pass-B/export/manifest.json"),
    "R1": REPAIR / "jobs/R1/build/pass-B/export/manifest.json",
    "R2": REPAIR / "jobs/R2/build/pass-B/export/manifest.json",
}
BUILD_TERMINALS = (
    ROOT / "two-pass-build-attempt-03-remote/queue-ended.json",
    REPAIR / "queue-ended.json",
)


def environment(gpu: int) -> dict[str, str]:
    if gpu not in GPUS:
        raise ValueError("GPU outside authorized remote set")
    env = base.environment(3)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ITERATION_PHYSICAL_GPU"] = str(gpu)
    env["PYTHONPATH"] = ":".join(
        (str(HERE), str(COMMON), str(VLA_MERGE_SRC), env["PYTHONPATH"]))
    return env


class DynamicCards:
    def __init__(self, candidates=GPUS):
        self.candidates = tuple(candidates)

    def __iter__(self):
        rows = [formal.gpu_row(gpu) for gpu in self.candidates]
        rows.sort(key=lambda row: (-row["free_mib"], row["index"]))
        return iter(row["index"] for row in rows)


def _checkpoint_identity(model: str) -> dict[str, dict]:
    checkpoint = MODELS[model].resolve()
    manifest_path = MANIFESTS[model].resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("complete") is not True
            or Path(manifest["checkpoint"]).resolve() != checkpoint
            or manifest.get("success_evaluated") is not False):
        raise ValueError(f"Unaccepted checkpoint manifest: {model}")
    expected = manifest.get("files_sha256") or {}
    actual_names = {path.name for path in checkpoint.iterdir() if path.is_file()}
    if set(expected) != actual_names:
        raise ValueError(f"Checkpoint file set differs: {model}")
    identities = {}
    for name, digest in sorted(expected.items()):
        path = (checkpoint / name).resolve()
        identity = base.file_identity(path)
        if identity["sha256"] != digest:
            raise ValueError(f"Checkpoint digest differs: {model}/{name}")
        identities[str(path)] = identity
    identities[str(manifest_path)] = base.file_identity(manifest_path)
    return identities


def jobs_for(run: Path) -> list[dict]:
    jobs = []
    for model, checkpoint in MODELS.items():
        for suite in SUITES:
            output = run / "jobs" / f"{model}-{SHORT[suite]}"
            command = [
                str(base.PYTHON), str(HERE / "evaluate_development.py"),
                "--checkpoint", str(checkpoint), "--suite", suite,
                "--bank", str(BANK), "--selection", str(SELECTION),
                "--output", str(output / "eval"),
            ]
            jobs.append({"id": output.name, "model": model, "suite": suite,
                         "checkpoint": str(checkpoint.resolve()),
                         "output": str(output), "identity": str(run / "identities.json"),
                         "command": command})
    return jobs


def verify(job: dict) -> dict:
    folder = Path(job["output"]) / "eval"
    selection = evaluator.load_development_selection(BANK, SELECTION)
    contract = json.loads((folder / "contract.json").read_text())
    if (contract.get("checkpoint") != job["checkpoint"]
            or contract.get("suite") != job["suite"]
            or contract.get("selection_id") != "development-40"
            or contract.get("selection_sha256") != selection.selection_sha256
            or contract.get("bank_manifest_sha256") != selection.manifest_sha256
            or contract.get("eval_seed") != selection.eval_seed
            or contract.get("episodes") != 10
            or contract.get("mode") != "development_eval"
            or contract.get("formal_evaluation") is not False
            or contract.get("used_for_candidate_selection") is not True
            or contract.get("hard_reset") is not True
            or contract.get("native_chunk") != 8
            or contract.get("execute_actions") != 8):
        raise ValueError("Development evaluation contract differs")
    base.assert_unchanged(json.loads(Path(job["identity"]).read_text())["files"])
    rows = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines()]
    successes = evaluator.validate_rows(rows, selection, job["suite"])
    summary = json.loads((folder / "summary.json").read_text())
    if (summary.get("complete") is not True or summary.get("episodes") != 10
            or summary.get("successes") != successes
            or summary.get("episodes_sha256") != formal.sha(folder / "episodes.jsonl")
            or summary.get("formal_evaluation") is not False
            or summary.get("used_for_candidate_selection") is not True):
        raise ValueError("Development summary differs from audited episodes")
    return {"episodes": 10, "successes": successes,
            "pc_success": 10.0 * successes,
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


def run_preflights(run: Path) -> None:
    """Exercise the exact child interpreter/environment before any GPU launch."""
    for suite in SUITES:
        output = run / "preflight" / suite
        command = [
            str(base.PYTHON), str(HERE / "evaluate_development.py"),
            "--checkpoint", str(MODELS["soup"]), "--suite", suite,
            "--bank", str(BANK), "--selection", str(SELECTION),
            "--output", str(output), "--preflight-only",
        ]
        subprocess.run(command, env=environment(0), check=True,
                       cwd=str(WORK / "vla-merge"), capture_output=True, text=True)
        contract = json.loads((output / "contract.json").read_text())
        if (contract.get("mode") != "preflight" or contract.get("episodes") != 10
                or contract.get("suite") != suite
                or contract.get("formal_evaluation") is not False
                or contract.get("used_for_candidate_selection") is not True):
            raise ValueError(f"Development child preflight differs: {suite}")


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    evaluator.load_development_selection(BANK, SELECTION)
    old_terminal, repair_terminal = (json.loads(path.read_text()) for path in BUILD_TERMINALS)
    if (old_terminal.get("status") != "complete" or old_terminal.get("accepted_jobs") != 1
            or repair_terminal.get("status") != "complete"
            or repair_terminal.get("accepted_jobs") != 2
            or repair_terminal.get("stopped") is not False):
        raise ValueError("Checkpoint construction terminals are not accepted")
    if ADMISSION_MIB < MEASURED_EVAL_REQUIREMENT_MIB + RESERVE_MIB:
        raise ValueError("Evaluation admission does not preserve 8 GiB")
    run.mkdir(parents=True)
    run_preflights(run)
    sources = [Path(__file__).resolve(), HERE / "evaluate_development.py",
               HERE / "dev_bank.py", COMMON / "native_oft.py", COMMON / "evaluate.py",
               BANK / "manifest.json", SELECTION, *BUILD_TERMINALS]
    files = {str(path.resolve()): base.file_identity(path.resolve()) for path in sources}
    for model in MODELS:
        files.update(_checkpoint_identity(model))
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {"complete": True, "files": files,
        "models": {name: str(path.resolve()) for name, path in MODELS.items()},
        "selection": str(SELECTION.resolve()), "selection_id": "development-40",
        "formal_outcomes_read": False, "scope": "five models x four suites x ten tasks"})

    jobs = jobs_for(run)
    uuids = {gpu: formal.gpu_row(gpu)["uuid"] for gpu in GPUS}
    frozen_env = environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {"schema": "openvla_tcr_repair_development40_v1",
        "host": socket.gethostname(), "run": str(run), "gpus": list(GPUS),
        "gpu_uuids": uuids, "max_workers": 4, "min_free_mib": ADMISSION_MIB,
        "measured_eval_requirement_mib": MEASURED_EVAL_REQUIREMENT_MIB,
        "admission_reserve_mib": RESERVE_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
        "jobs": jobs, "episodes": 200, "models": list(MODELS),
        "identity_audit": str(identity), "identity_audit_sha256": formal.sha(identity),
        "evaluator_sha256": formal.sha(HERE / "evaluate_development.py"),
        "environment_contract": environment_contract, "no_retry": True,
        "selection": str(SELECTION), "selection_id": "development-40",
        "eval_seed": evaluator.dev_bank.EVAL_SEED, "formal_evaluation": False,
        "success_used_only_after_terminal_audit": True,
        "authorization": "Codex C179 and user authorization for autonomous OpenVLA TCR repair"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "gpus": list(GPUS), "gpu_uuids": uuids,
        "scope": "development-40 screen only: Soup, old A/B, R1/R2",
        "episodes": 200, "formal_evaluation": False, "no_retry": True})


def run(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_tcr_repair_development40_v1"
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
