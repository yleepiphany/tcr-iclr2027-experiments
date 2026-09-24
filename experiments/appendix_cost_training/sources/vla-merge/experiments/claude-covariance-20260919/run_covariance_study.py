#!/usr/bin/env python3
"""Solve the 2x2 covariance study: S0 / S03 / N0 / N03.

Every arm shares the Soup start point (also the regularisation anchor), the frozen
pure-expert caches, the cap-16 row quota (1,331,600 rows), the 418 numeric ridges taken
from the *fresh tcre-r1* manifest, and the 414-module dual-propagation scope. The only
things that differ are the teacher alpha and the covariance balance flag.

The start point and the ridge source are deliberately **different checkpoints**, which is
why this study carries its own contract schema rather than reusing an older one.

Smoke builds go to a quarantined `smoke/` directory, are stamped `not_an_arm`, and are
deleted with `--clean-smoke` once their receipts have been kept.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
sys.path.insert(0, str(ROOT / "scripts"))

import run_pi05_table3_queue as table3  # noqa: E402  read-only reference

sys.path.insert(0, str(HERE))
from covariance_split_solve import REGISTERED_ARMS  # noqa: E402
from batch_safety import BatchStop, Child, classify_exit, validate_gpus  # noqa: E402

# This study's grant: six idle cards on this host. GPU 0 and 3 carry other people's work.
AUTHORISED_GPUS = (1, 2, 4, 5, 6, 7)
T3 = WORK / "vla-merge-runtime/experiments/table3-ablation-20260916"
# Independent calibration draws. "across" is the original study's cache; "repeat-01" is a
# separate collection of the same four 10k experts under the same expert-execution policy,
# used only for the pre-registered replication of the n0 - s0 contrast.
CALIBRATION_SETS = {
    "across": (T3 / "inputs", "{suite}/across"),
    # A different request slice of the SAME episodes/seeds, with the same capture schema
    # and the same row-selection mechanics. Not an independent re-collection: across and
    # early pick different requests within each episode (quantiles vs first five), so a
    # replication here tests slice-robustness, not independent data.
    "early-slice": (T3 / "inputs", "{suite}/early"),
    # REJECTED: tcr-unified-night traces/repeat-01. Same episodes, but a different capture
    # protocol (reservoir mode, 15 calls/prompt, no selection rule, and no
    # `selected_request_slot`, which the row selection depends on). Using it would change
    # the protocol and the data at once. See blocked-attempts/fresh-repeat-01/.
}
SOUP = WORK / ("vla-merge-runtime/experiments/iclr2027-table1-20260910/libero/model-soups"
               "/repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model")
RIDGE_MANIFEST = WORK / ("vla-merge-runtime/experiments/tcr-unified-night-20260919/models"
                         "/tcre-r1/block_regmeanpp_manifest.json")
SOLVER = HERE / "materialize_covariance_alpha.py"
RUN = WORK / "vla-merge-runtime/experiments/claude-covariance-20260919"
BANK_SHA = "d61a5f9e56bb0f76d8186a33dc26fe283cf8cfab220e903a460f92e4507f209b"
EXPECTED_ROWS = 1331600
DUAL_MODULES = 414
LOCAL_TEACHER = ["model.action_in_proj", "model.action_out_proj",
                 "model.time_mlp_in", "model.time_mlp_out"]


def sha(path):
    return table3.digest(path)


def build_config(arm, run, smoke_states, soup_sha, ridge_sha,
                 calibration_set='across'):
    alpha, balance = REGISTERED_ARMS[arm]
    config = {
        "schema": "claude_covariance_study_v1",
        "arm": arm,
        "teacher_alpha": alpha,
        "covariance_balance": balance,
        "repeat": 1,
        "row_cap_per_request_module": 16,
        "expected_realized_rows": EXPECTED_ROWS,
        "expert_masses": "covariance_2x2",
        "merged_slots": [],
        "calibration_set": calibration_set,
        "calibration": f"frozen pure-expert caches, draw={calibration_set}",
        "start_point": str(SOUP),
        "start_point_sha256": soup_sha,
        "start_point_role": "initialisation and regularisation anchor (Soup)",
        "ridge_source_manifest": str(RIDGE_MANIFEST),
        "ridge_source_sha256": ridge_sha,
        "ridge_source_role": ("418 numeric ridges from the fresh tcre-r1 build; a ridge "
                              "source only, NOT the initialisation"),
        "dense_expert_bank_sha256": BANK_SHA,
        "dual_propagation_modules": DUAL_MODULES,
        "local_teacher_modules": LOCAL_TEACHER,
        "scope_claim": ("teacher alpha applies to 414 of 418 solved modules; the 4 dense "
                        "interface modules keep the local teacher. Covariance balancing "
                        "applies to all 418, being a property of the regression metric."),
        "attribution": ("covariance-Frobenius normalisation is FeatCal's; this is a "
                        "trace-matched variant, not a new normalisation principle"),
        "smoke_states_per_suite": smoke_states or None,
        "not_a_table3_variant": True,
        "not_a_second_round_contract": ("start point and ridge source are distinct "
                                        "checkpoints in this study"),
    }
    path = run / "configs" / f"{arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def build_command(arm, run, reference, smoke_states, soup_sha, ridge_sha,
                  calibration_set='across'):
    alpha, balance = REGISTERED_ARMS[arm]
    config = build_config(arm, run, smoke_states, soup_sha, ridge_sha, calibration_set)
    command = [str(SOURCE / ".venv/bin/python"), "-u", str(SOLVER),
               f"--ablation-config={config}", f"--dense-expert-bank={table3.BANK}",
               f"--base-model={reference['inputs']['base_model']}",
               f"--prior-model={SOUP}",
               f"--output={run / 'checkpoints' / arm}",
               f"--teacher-alpha={alpha}",
               "--ridge-ratio=.05", "--ridge-scale=feature_energy",
               "--max-correction-ratio=3", "--max-rows-per-sample=16",
               "--device=cuda",
               "--allow-mixed-calibration-policies", "--allow-prior-calibration-mismatch",
               "--expert-loss-normalization=none", "--replay-prefix=merged"]
    if balance:
        command.append("--covariance-balance")
    if smoke_states:
        command += [f"--smoke-states={smoke_states}",
                    f"--max-states-per-calibration-source={smoke_states}"]
    caches = {}
    root, pattern = CALIBRATION_SETS[calibration_set]
    for suite in table3.SUITES:
        folder = root / pattern.format(suite=suite)
        if not (folder / "replay.json").exists():
            raise FileNotFoundError(f"{arm}/{suite}: missing calibration cache {folder}")
        command += [f"--expert={suite}={reference['experts'][suite]}",
                    f"--calibration={suite}={folder / 'replay.safetensors'}",
                    f"--manifest={suite}={folder / 'replay.json'}"]
        caches[suite] = {"path": str(folder),
                         "manifest_sha256": sha(folder / "replay.json"),
                         "tensor_sha256": sha(folder / "replay.safetensors")}
    return {"arm": arm, "command": command, "config": str(config), "caches": caches}


def verify_solve(arm, run, smoke_states, soup_sha):
    """Post-build audit, independent of the solver's own in-run checks."""
    alpha, balance = REGISTERED_ARMS[arm]
    folder = run / "checkpoints" / arm
    manifest = json.loads((folder / "block_regmeanpp_manifest.json").read_text())
    modules = manifest["modules"]
    problems = []
    if len(modules) != 418:
        problems.append(f"scope is {len(modules)} modules, not 418")
    if manifest.get("covariance_study_arm") != arm:
        problems.append(f"manifest arm label is {manifest.get('covariance_study_arm')!r}")
    if manifest.get("teacher_alpha") != alpha:
        problems.append(f"manifest alpha is {manifest.get('teacher_alpha')}")
    if manifest.get("covariance_balance") is not balance:
        problems.append(f"manifest balance is {manifest.get('covariance_balance')}")
    if bool(manifest.get("smoke_run")) != bool(smoke_states):
        problems.append("smoke stamp does not match the run mode")

    dual = {k: v for k, v in modules.items() if v.get("teacher_scope") == "dual_propagation"}
    local = {k: v for k, v in modules.items() if v.get("teacher_scope") == "local_teacher"}
    if len(dual) != DUAL_MODULES:
        problems.append(f"{len(dual)} modules on the interpolated teacher")
    if sorted(local) != sorted(LOCAL_TEACHER):
        problems.append(f"local-teacher modules are {sorted(local)}")
    realised_balance = {v.get("covariance_balance") for v in modules.values()}
    if realised_balance != {balance}:
        problems.append(f"mixed covariance-balance settings: {realised_balance}")

    rows_by_module = {k: v.get("rows_by_expert") or {} for k, v in modules.items()}
    realised = sum(sum(v.values()) for v in rows_by_module.values())
    if not smoke_states:
        time_modules = {"model.time_mlp_in", "model.time_mlp_out"}
        for key, rows in rows_by_module.items():
            want = 50 if key in time_modules else 800
            if len(rows) != 4 or any(r != want for r in rows.values()):
                problems.append(f"{key}: row quota {rows} (expected 4 x {want})")
                break
        if realised != EXPECTED_ROWS:
            problems.append(f"realised rows {realised} != {EXPECTED_ROWS}")

    ridge_reference = json.loads(RIDGE_MANIFEST.read_text())
    frozen = sum(1 for key, value in modules.items()
                 if abs(value["ridge"] - ridge_reference["modules"][key]["ridge"]) < 1e-12)
    if frozen != 418:
        problems.append(f"only {frozen}/418 modules kept the tcre-r1 ridge")

    digest = sha(folder / "model.safetensors")
    if digest != manifest["model_sha256"]:
        problems.append("written checkpoint does not match its manifest")
    if digest == soup_sha:
        problems.append("output is byte-identical to the Soup start point")
    if problems:
        raise ValueError(f"{arm}: " + "; ".join(problems))

    scales = [v["balance_diagnostics"]["trace_scale_s"] for v in dual.values()
              if v.get("balance_diagnostics", {}).get("balance")]
    clipped = sum(1 for v in modules.values() if v.get("trust_scale", 1.0) < 1.0)
    return {"arm": arm, "teacher_alpha": alpha, "covariance_balance": balance,
            "realized_rows": realised, "modules": len(modules),
            "dual_propagation_modules": len(dual), "frozen_ridge_modules": frozen,
            "trust_clipped_modules": clipped,
            "median_trace_scale_s": (round(sorted(scales)[len(scales) // 2], 5)
                                     if scales else None),
            "peak_gpu_memory_allocated_gib": manifest.get("peak_gpu_memory_allocated_gib"),
            "model_sha256": digest, "smoke": bool(smoke_states)}


def worker(gpu, job, run, state, lock, smoke_states, stop, soup_sha):
    arm = job["arm"]
    if stop.stopped:
        with lock:
            state["skipped"].append({"arm": arm, "reason": stop.reason})
        return
    log = run / "logs" / f"{arm}.log"
    env = {**table3.environment(gpu)}
    env["PYTHONPATH"] = ":".join([str(HERE), env.get("PYTHONPATH", "")]).rstrip(":")
    begin = time.monotonic()
    child = Child(job["command"], env, log, cwd=ROOT)
    if not stop.register(child):
        with lock:
            state["skipped"].append({"arm": arm, "reason": stop.reason,
                                     "note": "cancelled during launch; child stopped"})
        return
    with lock:
        state["started"].append({"arm": arm, "gpu": gpu, **child.identity()})
        print(json.dumps({"started": arm, "gpu": gpu, "pid": child.pid,
                          "pgid": child.pgid}), flush=True)
    outcome = classify_exit(child.wait())
    elapsed = time.monotonic() - begin
    if outcome["outcome"] != "success":
        with lock:
            state["errors"].append({"arm": arm, "gpu": gpu, "log": str(log), **outcome})
        stop.trip(f"{arm} exited: {outcome['outcome']}")
        return
    try:
        receipt = verify_solve(arm, run, smoke_states, soup_sha)
    except Exception as error:
        with lock:
            state["errors"].append({"arm": arm, "gpu": gpu, "error": str(error)})
        stop.trip(f"{arm} failed its post-build audit")
        return
    receipt.update(gpu=gpu, minutes=round(elapsed / 60, 1), pid=child.pid)
    with lock:
        state["done"].append(receipt)
        print(json.dumps({"done": len(state["done"]), **receipt}), flush=True)


def main(args):
    os.nice(args.nice)
    run = args.run / "smoke" if args.smoke_states else args.run
    if args.clean_smoke:
        target = args.run / "smoke"
        if target.exists():
            shutil.rmtree(target)
            print(f"removed smoke outputs: {target}", flush=True)
        else:
            print(f"no smoke outputs at {target}", flush=True)
        return
    gpus = validate_gpus(args.gpus, AUTHORISED_GPUS)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in REGISTERED_ARMS]
    if unknown:
        raise ValueError(f"Unregistered arms: {unknown}; registered: {sorted(REGISTERED_ARMS)}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"Duplicate arms requested: {arms}")
    if len(gpus) < len(arms):
        raise ValueError("One worker per GPU: not enough authorised GPUs")
    assignment = dict(zip(arms, gpus, strict=False))
    for arm in arms:
        folder = run / "checkpoints" / arm
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"{arm}: {folder} already exists; audit and claim it")
    run.mkdir(parents=True, exist_ok=True)
    if sha(table3.BANK) != BANK_SHA:
        raise ValueError("Expert bank index changed")
    if not (SOUP / "model.safetensors").exists():
        raise FileNotFoundError(f"Soup start point missing at {SOUP}")
    soup_sha = sha(SOUP / "model.safetensors")
    ridge_sha = sha(RIDGE_MANIFEST)
    reference = json.loads(table3.REFERENCE.read_text())
    jobs = {arm: build_command(arm, run, reference, args.smoke_states, soup_sha, ridge_sha,
                               args.calibration_set) for arm in arms}
    table3.save(run / f"plan-{'-'.join(arms)}.json", {
        "stage": "covariance_study_solve", "arms": arms, "gpus": gpus,
        "hostname": os.uname().nodename, "authorised_gpus": list(AUTHORISED_GPUS),
        "arm_gpu_assignment": assignment,
        "factors": {arm: {"teacher_alpha": REGISTERED_ARMS[arm][0],
                          "covariance_balance": REGISTERED_ARMS[arm][1]} for arm in arms},
        "start_point": {"path": str(SOUP), "model_sha256": soup_sha,
                        "role": "initialisation and regularisation anchor"},
        "ridge_source": {"manifest": str(RIDGE_MANIFEST), "sha256": ridge_sha,
                         "role": "418 numeric ridges only; not the initialisation"},
        "solver": str(SOLVER), "solver_sha256": sha(SOLVER),
        "split_solver_sha256": sha(HERE / "covariance_split_solve.py"),
        "expected_realized_rows": EXPECTED_ROWS,
        "dual_propagation_modules": DUAL_MODULES,
        "local_teacher_modules": LOCAL_TEACHER,
        "jobs": [{k: v for k, v in j.items() if k != "command"} for j in jobs.values()],
        "commands": {arm: jobs[arm]["command"] for arm in arms},
        "smoke": bool(args.smoke_states),
        "smoke_states_per_suite": args.smoke_states or None,
        "safety": "per-child process groups, parent-death guard, stop-on-kill latch",
        "not_a_result": "A solved checkpoint is not an evaluation.",
        "evaluated": False, "goal_achieved": False})
    if args.preflight_only:
        print("PREFLIGHT_PASS", flush=True)
        return
    state = {"done": [], "errors": [], "started": [], "skipped": []}
    lock = threading.Lock()
    stop = BatchStop(report_path=run / "batch-stop-report.json")
    stop.install_signal_handlers()
    threads = [threading.Thread(target=worker,
                                args=(assignment[arm], jobs[arm], run, state, lock,
                                      args.smoke_states, stop, soup_sha))
               for arm in arms]
    begin = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {"status": "complete" if len(state["done"]) == len(arms) and not state["errors"]
                         else "stopped" if stop.stopped else "incomplete",
               "stop_reason": stop.reason, "restarted": False,
               "hostname": os.uname().nodename,
               "arms": sorted(state["done"], key=lambda r: r["arm"]),
               "errors": state["errors"], "skipped": state["skipped"],
               "wall_hours": round((time.monotonic() - begin) / 3600, 2),
               "smoke": bool(args.smoke_states),
               "evaluated": False, "goal_achieved": False}
    table3.save(run / f"summary-{'-'.join(arms)}.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "errors", "wall_hours")}), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--arms", default=",".join(REGISTERED_ARMS))
    parser.add_argument("--gpus", default=",".join(str(g) for g in AUTHORISED_GPUS[:4]))
    parser.add_argument("--calibration-set", default="across",
                        choices=sorted(CALIBRATION_SETS),
                        help="Independent calibration draw; only the replication uses "
                             "a non-default value")
    parser.add_argument("--nice", type=int, default=5)
    parser.add_argument("--smoke-states", type=int, default=0)
    parser.add_argument("--clean-smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    main(parser.parse_args())
