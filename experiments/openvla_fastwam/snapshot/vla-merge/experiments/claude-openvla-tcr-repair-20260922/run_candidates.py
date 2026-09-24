#!/usr/bin/env python3
"""Dynamic one-at-a-time queue for the two frozen OpenVLA repair candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
OLD_CLAUDE = HERE.parent / "claude-openvla-tcr-20260921"
COMMON = HERE.parent / "openvla-tcr-20260921"
sys.path[:0] = [str(HERE), str(OLD_CLAUDE), str(COMMON)]

import run_local as base  # noqa: E402

formal = base.formal
WORK = HERE.parents[2]
OLD_ROOT = WORK / "vla-merge-runtime/experiments/openvla-tcr-20260921"
PRIOR = OLD_ROOT / "soup-prior-attempt-02/checkpoint"
LEDGER = OLD_ROOT / "local-expert-dynamic-01/identities.json"
CAPTURE = OLD_ROOT / "expert-ab-capture-attempt-01"
ACCEPTANCE = WORK / "coordination/2026-09-21/openvla-ab-capture-acceptance.json"
BLOCK_GATE = (OLD_ROOT / "block-interface-smoke-attempt-02-remote/jobs/"
              "native-block-interface/validation/summary.json")
BLOCK_PLAN = WORK / "coordination/2026-09-21/openvla-native-block-capacity.json"
REFERENCE_TERMINAL = OLD_ROOT / "two-pass-build-attempt-03-remote/queue-ended.json"
GPUS = tuple(range(8))
# The accepted same-shape build reserved 31,696,355,328 bytes (30,228 MiB).
# Forty GiB admission therefore leaves more than the user's requested 8 GiB.
MEASURED_REQUIREMENT_MIB = math.ceil(31_696_355_328 / 2**20)
RESERVE_MIB = 8 * 1024
ADMISSION_MIB = 40 * 1024
RUNTIME_FLOOR_MIB = 0


def environment(gpu: int) -> dict[str, str]:
    if gpu not in GPUS:
        raise ValueError("GPU outside authorized remote set")
    env = base.environment(3)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["ITERATION_PHYSICAL_GPU"] = str(gpu)
    env["PYTHONPATH"] = ":".join((str(HERE), str(OLD_CLAUDE), str(COMMON), env["PYTHONPATH"]))
    return env


class DynamicCards:
    def __init__(self, candidates=GPUS):
        self.candidates = tuple(candidates)

    def __iter__(self):
        rows = [formal.gpu_row(gpu) for gpu in self.candidates]
        rows.sort(key=lambda row: (-row["free_mib"], row["index"]))
        return iter(row["index"] for row in rows)


def verify(job: dict) -> dict:
    output = Path(job["output"]) / "build"
    result = json.loads((output / "final-manifest.json").read_text())
    candidate = job["candidate"]
    row_mode = "uniform" if candidate == "R1" else "llm_action_4_context_4"
    if (result.get("complete") is not True or result.get("candidate") != candidate
            or result.get("row_mode") != row_mode or result.get("is_tcr") is not True
            or result.get("passes") != ["A", "B"] or result.get("rows_total") != 1_396_000
            or result.get("native_reload_verified") is not True
            or result.get("native_actions_reload_exact") is not True
            or result.get("pass_B_reuses_exact_pass_A_ridges") is not True
            or result.get("success_evaluation") is not False or result.get("episodes") != 0):
        raise ValueError("Incomplete or invalid repair candidate")
    manifests = {}
    for pass_id in ("A", "B"):
        manifest = json.loads((output / f"pass-{pass_id}/manifest.json").read_text())
        manifests[pass_id] = manifest
        if (manifest.get("complete") is not True or manifest.get("candidate") != candidate
                or manifest.get("row_mode") != row_mode or manifest.get("pass_id") != pass_id
                or manifest.get("rows") != 698_000 or manifest.get("modules") != 438
                or manifest.get("blocks") != 87 or manifest.get("success_evaluated") is not False):
            raise ValueError(f"Invalid repair pass {candidate}/{pass_id}")
    if (manifests["B"].get("ridge_map") != manifests["A"].get("ridge_map")
            or manifests["B"].get("ridge_source") != "fixed_from_same_candidate_pass_A"):
        raise ValueError("Pass B did not freeze this candidate's pass-A ridge map")
    base.assert_unchanged(json.loads(Path(job["identity"]).read_text())["files"])
    return result


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    reference = json.loads(REFERENCE_TERMINAL.read_text())
    if reference.get("status") != "complete" or reference.get("accepted_jobs") != 1:
        raise ValueError("Same-shape resource reference is not accepted")
    reference_peak = reference["finished"][0]["artifacts"]["process_peak_reserved_bytes"]
    if reference_peak != 31_696_355_328 or ADMISSION_MIB < math.ceil(reference_peak / 2**20) + RESERVE_MIB:
        raise ValueError("Repair admission does not preserve the 8-GiB reserve")
    gate = json.loads(BLOCK_GATE.read_text())
    capture_terminal = json.loads((CAPTURE / "queue-ended.json").read_text())
    acceptance = json.loads(ACCEPTANCE.read_text())
    if (gate.get("complete") is not True or gate.get("native_actions_restored_exactly") is not True
            or capture_terminal.get("status") != "complete" or capture_terminal.get("accepted_jobs") != 8
            or acceptance.get("accepted") is not True or acceptance.get("retained_requests") != 400):
        raise ValueError("OpenVLA input gates are not accepted")

    run.mkdir(parents=True)
    sources = [
        Path(__file__).resolve(), HERE / "build_candidate.py", HERE / "pass_engine.py",
        HERE / "block_calibration.py", HERE / "linear_calibration.py",
        COMMON / "ridge.py", COMMON / "expert_bank.py", COMMON / "export_native.py",
        COMMON / "materialize_soup.py", COMMON / "native_oft.py", BLOCK_PLAN, BLOCK_GATE,
        ACCEPTANCE, CAPTURE / "capture-contract.json", CAPTURE / "identities.json",
        OLD_ROOT / "soup-prior-attempt-02/manifest.json", LEDGER, REFERENCE_TERMINAL,
    ]
    for pool in ("A", "B"):
        for expert in ("spatial", "object", "goal", "long"):
            sources.extend(sorted((CAPTURE / "jobs" / f"{pool}-{expert}" / "capture").glob("task-*.pt")))
    files = {str(path.resolve()): base.file_identity(path.resolve()) for path in sources}
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {
        "complete": True, "files": files,
        "scope": "Frozen R1/R2 source, Soup, accepted A/B capture and corrected block gate",
        "large_checkpoint_files": "bound transitively by the reused Soup and expert ledgers",
        "formal_outcomes_read": False,
    })

    jobs = []
    for candidate in ("R1", "R2"):
        output = run / "jobs" / candidate
        command = [
            str(base.PYTHON), str(HERE / "build_candidate.py"),
            "--prior", str(PRIOR), "--candidate", candidate,
            "--ledger", str(LEDGER), "--capture", str(CAPTURE),
            "--capture-acceptance", str(ACCEPTANCE), "--block-gate", str(BLOCK_GATE),
            "--plan", str(BLOCK_PLAN), "--output", str(output / "build"),
        ]
        jobs.append({"id": f"repair-{candidate}", "candidate": candidate,
                     "output": str(output), "identity": str(identity), "command": command})
    uuids = {gpu: formal.gpu_row(gpu)["uuid"] for gpu in GPUS}
    frozen_env = environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {
        "schema": "openvla_tcr_repair_candidates_v1", "host": socket.gethostname(),
        "run": str(run), "gpus": list(GPUS), "gpu_uuids": uuids, "max_workers": 1,
        "min_free_mib": ADMISSION_MIB, "measured_requirement_mib": MEASURED_REQUIREMENT_MIB,
        "admission_reserve_mib": RESERVE_MIB, "runtime_floor_mib": RUNTIME_FLOOR_MIB,
        "episodes": 0, "no_retry": True, "jobs": jobs,
        "identity_audit": str(identity), "identity_audit_sha256": formal.sha(identity),
        "evaluator_sha256": formal.sha(HERE / "build_candidate.py"),
        "environment_contract": environment_contract,
        "candidates": {
            "R1": {"row_mode": "uniform", "ridge": "fixed A->B", "trust": "expert_delta"},
            "R2": {"row_mode": "llm_action_4_context_4", "ridge": "fixed A->B", "trust": "expert_delta"},
        },
        "rows_per_pass": 698_000, "preserve_pass_A": True,
        "requires_native_reload_before_development_evaluation": True,
        "formal_outcomes_read": False,
        "authorization": "User requested OpenVLA TCR repair and parallel use of eligible GPUs; main ablation remains priority",
    }
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {
        "owner": "Claude", "host": socket.gethostname(), "gpus": list(GPUS),
        "gpu_uuids": uuids, "scope": "two frozen R1/R2 repair candidates, one worker",
        "evaluation_episodes": 0, "no_retry": True, "do_not_duplicate": True,
    })


def run(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if plan["host"] != socket.gethostname() or (run / "started.json").exists():
        raise RuntimeError("Wrong host or already-started attempt")
    for raw_gpu, expected in plan["gpu_uuids"].items():
        if formal.gpu_row(int(raw_gpu))["uuid"] != expected:
            raise ValueError("GPU UUID changed")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.GPUS = DynamicCards(plan["gpus"])
    formal.FLOOR_MIB = RUNTIME_FLOOR_MIB
    formal.EVALUATOR = HERE / "build_candidate.py"
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
