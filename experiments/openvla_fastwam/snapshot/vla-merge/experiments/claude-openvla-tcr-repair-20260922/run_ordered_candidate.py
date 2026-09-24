#!/usr/bin/env python3
"""One immutable GPU build of R2 with ordered auxiliary replay; no rollout."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import run_candidates as prior_queue


HERE = Path(__file__).resolve().parent
base, formal = prior_queue.base, prior_queue.formal
CANDIDATE = "R2_aux_ordered"
BASELINE = (HERE.parents[2] / "vla-merge-runtime/experiments/"
            "claude-openvla-tcr-repair-20260922/candidate-build-attempt-02")


def verify(job: dict) -> dict:
    result = prior_queue.verify(job)
    output = Path(job["output"]) / "build"
    if result.get("auxiliary_replay") != "ordered_current_linear_expert_only":
        raise ValueError("Final ordered auxiliary receipt missing")
    plan = json.loads(prior_queue.BLOCK_PLAN.read_text())
    for pass_id in ("A", "B"):
        manifest = json.loads((output / f"pass-{pass_id}/manifest.json").read_text())
        if manifest.get("auxiliary_replay") != "ordered_current_linear_expert_only":
            raise ValueError("Pass was not ordered auxiliary replay")
        for block in ("proprio_projector", "action_head"):
            index = list(plan["blocks"]).index(block)
            report = json.loads((output / f"pass-{pass_id}/block-{index:03d}.json").read_text())
            planned = [row["module"] for row in plan["blocks"][block]]
            expected_rows = sum(row["rows_all_experts"] for row in plan["blocks"][block])
            if (report.get("complete") is not True or report.get("candidate_only") is not True
                    or report.get("all_features_before_solve") is not False
                    or set(report.get("actual_linear_order", ())) != set(planned)
                    or report.get("collected_rows") != expected_rows):
                raise ValueError(f"Ordered auxiliary block receipt differs: {pass_id}/{block}")
    return result


def prepare(run: Path) -> None:
    if run.exists():
        raise FileExistsError(run)
    accepted = json.loads((BASELINE / "queue-ended.json").read_text())
    baseline = json.loads((BASELINE / "jobs/R2/build/final-manifest.json").read_text())
    if (accepted.get("status") != "complete" or accepted.get("accepted_jobs") != 2
            or baseline.get("complete") is not True or baseline.get("candidate") != "R2"
            or baseline.get("native_actions_reload_exact") is not True):
        raise ValueError("Existing R2 build is not accepted")
    capture = json.loads((prior_queue.CAPTURE / "queue-ended.json").read_text())
    acceptance = json.loads(prior_queue.ACCEPTANCE.read_text())
    gate = json.loads(prior_queue.BLOCK_GATE.read_text())
    if (capture.get("status") != "complete" or capture.get("accepted_jobs") != 8
            or acceptance.get("accepted") is not True or acceptance.get("retained_requests") != 400
            or gate.get("complete") is not True
            or gate.get("native_actions_restored_exactly") is not True):
        raise ValueError("A/B capture or native block interface gate differs")
    reference = json.loads(prior_queue.REFERENCE_TERMINAL.read_text())
    reference_peak = reference["finished"][0]["artifacts"]["process_peak_reserved_bytes"]
    if prior_queue.ADMISSION_MIB < (reference_peak + 2**20 - 1) // 2**20 + prior_queue.RESERVE_MIB:
        raise ValueError("Build admission lacks measured peak plus 8 GiB")

    run.mkdir(parents=True)
    sources = [Path(__file__).resolve(), HERE / "build_ordered_candidate.py",
               HERE / "pass_engine_ordered.py", HERE / "refine_auxiliary_block.py",
               HERE / "run_candidates.py", HERE / "block_calibration.py",
               HERE / "linear_calibration.py", prior_queue.COMMON / "ridge.py",
               prior_queue.COMMON / "expert_bank.py",
               prior_queue.COMMON / "export_native.py",
               prior_queue.COMMON / "materialize_soup.py",
               prior_queue.COMMON / "native_oft.py", prior_queue.BLOCK_PLAN,
               prior_queue.BLOCK_GATE, prior_queue.ACCEPTANCE,
               prior_queue.CAPTURE / "capture-contract.json",
               prior_queue.CAPTURE / "identities.json",
               prior_queue.OLD_ROOT / "soup-prior-attempt-02/manifest.json",
               prior_queue.LEDGER, prior_queue.REFERENCE_TERMINAL,
               BASELINE / "queue-ended.json", BASELINE / "jobs/R2/build/final-manifest.json"]
    for pool in ("A", "B"):
        for expert in ("spatial", "object", "goal", "long"):
            sources.extend(sorted((prior_queue.CAPTURE / "jobs" / f"{pool}-{expert}" / "capture")
                                  .glob("task-*.pt")))
    files = {str(path.resolve()): base.file_identity(path.resolve()) for path in sources}
    identity = run / "identities.json"
    base.BASE_SAVE(identity, {"complete": True, "files": files,
        "scope": "R2 recipe and A/B bank frozen; only internal head/proprio replay changes",
        "baseline": str((BASELINE / "jobs/R2/build/final-manifest.json").resolve()),
        "formal_outcomes_read": False})
    output = run / "jobs" / CANDIDATE
    command = [str(base.PYTHON), str(HERE / "build_ordered_candidate.py"),
        "--prior", str(prior_queue.PRIOR), "--candidate", CANDIDATE,
        "--ledger", str(prior_queue.LEDGER), "--capture", str(prior_queue.CAPTURE),
        "--capture-acceptance", str(prior_queue.ACCEPTANCE),
        "--block-gate", str(prior_queue.BLOCK_GATE),
        "--plan", str(prior_queue.BLOCK_PLAN), "--output", str(output / "build")]
    job = {"id": "repair-R2-aux-ordered", "candidate": CANDIDATE,
           "output": str(output), "identity": str(identity), "command": command}
    uuids = {gpu: formal.gpu_row(gpu)["uuid"] for gpu in prior_queue.GPUS}
    frozen_env = prior_queue.environment(0)
    environment_contract = {key: frozen_env[key] for key in (
        "PYTHONPATH", "LIBERO_CONFIG_PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS")}
    environment_contract.update({"CUDA_VISIBLE_DEVICES": "selected GPU 0-7",
                                 "ITERATION_PHYSICAL_GPU": "selected GPU 0-7"})
    plan = {"schema": "openvla_r2_ordered_aux_build_v1", "host": socket.gethostname(),
        "run": str(run), "gpus": list(prior_queue.GPUS), "gpu_uuids": uuids,
        "max_workers": 1, "min_free_mib": prior_queue.ADMISSION_MIB,
        "admission_reserve_mib": prior_queue.RESERVE_MIB,
        "measured_requirement_mib": prior_queue.MEASURED_REQUIREMENT_MIB,
        "runtime_floor_mib": prior_queue.RUNTIME_FLOOR_MIB,
        "jobs": [job], "episodes": 0, "no_retry": True,
        "identity_audit": str(identity), "identity_audit_sha256": formal.sha(identity),
        "evaluator_sha256": formal.sha(HERE / "build_ordered_candidate.py"),
        "environment_contract": environment_contract,
        "only_change_from_R2": "sequential live-prefix replay of action_head/proprio_projector",
        "authorization": "User requested 400 development episodes for this diagnosed correction"}
    base.BASE_SAVE(run / "plan.json", plan)
    base.BASE_SAVE(run / "CLAIM.json", {"owner": "Claude", "host": socket.gethostname(),
        "scope": "one R2 ordered-auxiliary build, then separate 400 development episodes",
        "episodes": 0, "no_retry": True, "do_not_duplicate": True})


def run(run: Path) -> None:
    plan = json.loads((run / "plan.json").read_text())
    if (plan.get("schema") != "openvla_r2_ordered_aux_build_v1"
            or plan.get("host") != socket.gethostname() or (run / "started.json").exists()):
        raise RuntimeError("Wrong host, wrong plan or already-started attempt")
    for raw_gpu, expected in plan["gpu_uuids"].items():
        if formal.gpu_row(int(raw_gpu))["uuid"] != expected:
            raise ValueError("GPU UUID changed")
    base.assert_unchanged(json.loads((run / "identities.json").read_text())["files"])
    formal.GPUS = prior_queue.DynamicCards(plan["gpus"])
    formal.FLOOR_MIB = prior_queue.RUNTIME_FLOOR_MIB
    formal.EVALUATOR = HERE / "build_ordered_candidate.py"
    formal.IDENTITIES = run / "identities.json"
    formal.scientific_environment = prior_queue.environment
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
