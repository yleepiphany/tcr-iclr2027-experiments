#!/usr/bin/env python3
"""D2 development evaluation of the 2026-09-19 teacher-target arms L0 / T03 / T1.

Three arms x ten offsets (30..39) x four suites x ten tasks x one episode = 1200 episodes,
400 per arm.  Every arm sees the same tasks, the same init states and the same seeds; the
only difference between the checkpoints being scored is the teacher-target alpha.

L0 is evaluated fresh rather than reusing the earlier round-3 numbers.  L0 goes through
the same interpolated-teacher code path as the other two arms (at alpha = 0), so its
checkpoint is not byte-identical to the earlier round-3 build and its 400 episodes are a
*re-evaluation*, registered as such.  Any L0-vs-round3 difference must not be attributed
to the teacher change.

Closed-loop evaluation carries an inherent run-to-run spread: three runs of the same C-M
checkpoint at offset 30 scored 33, 34 and 32.  Differences of one to two tasks per forty
are not distinguishable from re-running the same model.

This is a development evaluation on one build seed.  It is not independent confirmation,
it is not a final proof, and these scores must not be differenced against the FeatCal
70.92% figure, which comes from a different reset bank.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge"
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
sys.path.insert(0, str(ROOT / "scripts"))

import run_pi05_table3_queue as table3  # noqa: E402  read-only reference
from run_featcal_execution_pilot_lane_v2 import environment  # noqa: E402  read-only reference

sys.path.insert(0, str(HERE))
from covariance_split_solve import REGISTERED_ARMS  # noqa: E402

REGISTERED_ALPHAS = {arm: alpha for arm, (alpha, _b) in REGISTERED_ARMS.items()}

RUN = WORK / "vla-merge-runtime/experiments/claude-covariance-20260919"
WRAPPER = HERE / "eval_pi05_expanded_development.py"
ORIGINAL_WRAPPER = ROOT / "scripts/eval_pi05_iteration_development.py"
LIBERO = WORK / ".datasets/LIBERO/20260919/config-standard"
# Registered bases; an arm must declare which one it was built on and must not be
# byte-identical to it.
BASE_SHAS = {"c_e": "3e3a55ce0acd04b3ee1e180eda2a6109b0ef6d0ea4e717f6052a27829adfce95",
             "r01": "4562193825dc9b23e834eb32bfc57242501c7abc3f6c561eeaeee1b5b6f42da8"}
ARMS = tuple(REGISTERED_ARMS)
# Untouched reference checkpoints evaluated as controls. No new solve: the point is to
# separate "better than the start point" from "less degraded by an extra calibration
# round", so the start point itself must be scored under the same runtime.
# Untouched reference checkpoints, evaluated for comparison. No new solve: these are the
# start point itself and the existing FeatCal build. Both are RE-EVALUATIONS under this
# study's runtime (repaired LIBERO tree, current wrapper), because the earlier FeatCal D2
# used a different wrapper and predates the LIBERO repair, so strict reuse is not available.
CONTROLS = {
    "soup_reference": (
        WORK / ("vla-merge-runtime/experiments/iclr2027-table1-20260910/libero/model-soups"
                "/repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model"), None),
    "featcal_reference": (
        WORK / ("vla-merge-runtime/experiments/iclr2027-table1-20260910/preflight"
                "/featcal-formal-full-v1/checkpoint/pretrained_model"),
        "e43de109844431c02e316e57f701d7c06a9a3c8feaa3c41c9f391281f8197efa"),
}
EXPECTED_ROWS = 1331600
DUAL_MODULES = 414
OFFSETS = tuple(range(30, 40))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha(path):
    return table3.digest(path)


def model_for(arm, run):
    if arm in CONTROLS:
        folder, digest = CONTROLS[arm]
        observed = sha(folder / "model.safetensors")
        if digest is not None and observed != digest:
            raise ValueError(f"{arm}: reference checkpoint hash {observed} != registered {digest}")
        return folder, observed
    return _model_for_arm(arm, run)


def _model_for_arm(arm, run):
    """Resolve an arm to its checkpoint, re-checking identity at evaluation time.

    The solver already audited this, but the evaluator re-derives it independently: a
    smoke build or a checkpoint from a different contract must never reach 400 episodes.
    """
    folder = run / "checkpoints" / arm
    manifest = json.loads((folder / "block_regmeanpp_manifest.json").read_text())
    if manifest.get("covariance_study_arm") != arm:
        raise ValueError(f"{arm}: checkpoint is not labelled as this covariance-study arm")
    if manifest.get("covariance_balance") is not REGISTERED_ARMS[arm][1]:
        raise ValueError(f"{arm}: checkpoint balance flag differs from the registered arm")
    if manifest.get("teacher_alpha") != REGISTERED_ALPHAS[arm]:
        raise ValueError(f"{arm}: checkpoint alpha is {manifest.get('teacher_alpha')}")
    if manifest.get("smoke_run"):
        raise ValueError(f"{arm}: refusing to evaluate a smoke build")
    if manifest["realized_row_total"] != EXPECTED_ROWS or len(manifest["modules"]) != 418:
        raise ValueError(f"{arm}: checkpoint budget or scope differs")
    if manifest.get("dual_propagation_modules") != DUAL_MODULES:
        raise ValueError(f"{arm}: interpolated-teacher scope is not {DUAL_MODULES} modules")
    declared = (manifest.get("ablation") or {}).get("start_point_sha256")
    if not declared:
        raise ValueError(f"{arm}: manifest does not declare its start point")
    if manifest["model_sha256"] == declared:
        raise ValueError(f"{arm}: checkpoint is byte-identical to its start point")
    return folder, manifest["model_sha256"]


def build_command(arm, offset, run, model):
    output = run / arm / f"offset-{offset}"
    return [str(SOURCE / ".venv/bin/python"), "-u", str(WRAPPER),
            f"--output_dir={output}", "--env.type=libero",
            "--env.task=" + ",".join(SUITES),
            "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]", "--env.max_parallel_tasks=1",
            "--eval.batch_size=1", "--eval.n_episodes=1",
            f"--seed={391600 + 1000 * (offset - 30)}",
            f"--policy.path={model}", "--policy.device=cuda",
            "--policy.compile_model=false", "--policy.gradient_checkpointing=false",
            "--policy.n_action_steps=10"]


def outcomes_from(path):
    rows = json.loads(Path(path).read_text())["per_task"]
    if len(rows) != 40:
        raise ValueError(f"{path}: expected forty task rows")
    table = {}
    for row in rows:
        values = row["metrics"]["successes"]
        if len(values) != 1 or type(values[0]) is not bool:
            raise ValueError(f"{path}: exactly one episode per task")
        table[f"{row['task_group']}/{row['task_id']}"] = values[0]
    if len(table) != 40:
        raise ValueError(f"{path}: duplicate task rows")
    return table


def verify(arm, offset, run):
    output = run / arm / f"offset-{offset}"
    if json.loads((output / "exit.json").read_text())["return_code"]:
        raise ValueError(f"{arm}/off{offset}: evaluation failed")
    table = outcomes_from(output / "eval_info.json")
    receipts = [json.loads(line) for line in
                (output / "paired-noise.jsonl").read_text().splitlines() if line.strip()]
    if len(receipts) != 40:
        raise ValueError(f"{arm}/off{offset}: expected forty receipts")
    keys = set()
    for receipt in receipts:
        if receipt["offset"] != offset or receipt["actual_init_state_id"] != offset:
            raise ValueError(f"{arm}/off{offset}: receipt offset mismatch")
        if receipt["available_init_states"] < 40:
            raise ValueError(f"{arm}/off{offset}: task has fewer than forty init states")
        key = f"{receipt['suite']}/{receipt['task_id']}"
        if key in keys:
            raise ValueError(f"{arm}/off{offset}: duplicate receipt {key}")
        keys.add(key)
        if table[key] != receipt["successes"][0]:
            raise ValueError(f"{arm}/off{offset}: eval_info and receipt disagree on {key}")
    if keys != set(table):
        raise ValueError(f"{arm}/off{offset}: receipts and eval_info cover different tasks")
    return {"arm": arm, "offset": offset, "successes": sum(table.values()), "episodes": 40,
            "per_suite": {s: sum(v for k, v in table.items() if k.startswith(s + "/")) for s in SUITES},
            "outcomes": table, "eval_info_sha256": sha(output / "eval_info.json"),
            "receipt_sha256": sha(output / "paired-noise.jsonl")}


def arm_agreement(done):
    """Per-offset paired comparison between arms on identical tasks and seeds.

    Reported as agreement/disagreement counts, not as a significance test.  Every arm
    saw the same forty tasks at the same init state with the same seeds, so a paired
    read is the meaningful one; the bootstrap intervals are computed separately in the
    analysis step from the per-episode records kept here.
    """
    by_arm = {}
    for record in done:
        by_arm.setdefault(record["arm"], {})[record["offset"]] = record["outcomes"]
    report = {}
    arms = sorted(by_arm)
    for i, left in enumerate(arms):
        for right in arms[i + 1:]:
            shared = sorted(set(by_arm[left]) & set(by_arm[right]))
            wins = losses = ties = 0
            for offset in shared:
                a, b = by_arm[left][offset], by_arm[right][offset]
                if set(a) != set(b):
                    raise ValueError(f"{left} vs {right} off{offset}: different task sets")
                for key in a:
                    if a[key] and not b[key]:
                        wins += 1
                    elif b[key] and not a[key]:
                        losses += 1
                    else:
                        ties += 1
            report[f"{left}_vs_{right}"] = {
                "offsets_compared": shared, "episodes": wins + losses + ties,
                "{}_only".format(left): wins, "{}_only".format(right): losses,
                "same_outcome": ties,
                "note": "paired counts on identical tasks/seeds; not a significance test"}
    return report


def worker(gpu, jobs, run, state, lock):
    while True:
        with lock:
            if not jobs:
                return
            job = jobs.pop(0)
        output = run / job["arm"] / f"offset-{job['offset']}"
        # Resume support: a job that already has a verified receipt is not re-run, so
        # scaling the worker count mid-experiment costs nothing already paid for.
        existing = output / "verified.json"
        if existing.exists():
            record = json.loads(existing.read_text())
            if record.get("arm") == job["arm"] and record.get("offset") == job["offset"]:
                with lock:
                    state["done"].append(record)
                    print(json.dumps({"resumed": True, "done": len(state["done"]),
                                      "total": state["total"], "arm": record["arm"],
                                      "offset": record["offset"],
                                      "successes": record["successes"]}), flush=True)
                continue
        output.mkdir(parents=True, exist_ok=True)
        env = environment(gpu)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="1",
                   PI05_LIBERO_INIT_STATE_OFFSET=str(job["offset"]),
                   PI05_LIBERO_INIT_STATE_COUNT="1", PI05_TASK_TEXT_MODE="correct",
                   CLAUDE_EVAL_DUTY=str(job["duty"]),
                   ITERATION_PHYSICAL_GPU=str(gpu),
                   LIBERO_CONFIG_PATH=str(LIBERO),
                   ITERATION_NOISE_RECEIPT=str(output / "paired-noise.jsonl"))
        # The repaired 2026-09-19 tree only. The deleted Jenson paths and the "pro"
        # overrides must not leak in from the parent environment; MUJOCO_EGL_DEVICE_ID
        # is dropped so EGL binds to the single device CUDA_VISIBLE_DEVICES exposes.
        for key in ("LIBERO_PRO_REPO", "LIBERO_PRO_ASSET_DIR", "MUJOCO_EGL_DEVICE_ID"):
            env.pop(key, None)
        env["PYTHONPATH"] = ":".join([str(HERE), env.get("PYTHONPATH", "")]).rstrip(":")
        table3.save(output / "launch.json", {"gpu": gpu, "command": job["command"],
                    "model": job["model"], "model_sha256": job["model_sha256"],
                    "offset": job["offset"], "wrapper_sha256": sha(WRAPPER),
                    "duty": job["duty"], "started_unix": time.time()})
        begin = time.monotonic()
        with (output / "worker.log").open("w") as stream:
            code = subprocess.run(job["command"], cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT).returncode
        table3.save(output / "exit.json", {"return_code": code,
                                           "wall_seconds": time.monotonic() - begin})
        try:
            record = verify(job["arm"], job["offset"], run)
        except Exception as error:
            with lock:
                state["errors"].append({"arm": job["arm"], "offset": job["offset"],
                                        "error": str(error)})
            continue
        record.update(gpu=gpu, minutes=round((time.monotonic() - begin) / 60, 1))
        table3.save(output / "verified.json", record)
        with lock:
            state["done"].append(record)
            print(json.dumps({"done": len(state["done"]), "total": state["total"],
                              "arm": record["arm"], "offset": record["offset"],
                              "successes": record["successes"],
                              "minutes": record["minutes"]}), flush=True)


def main(args):
    os.nice(args.nice)
    gpus = [int(x) for x in args.gpus.split(",")]
    offsets = [int(x) for x in args.offsets.split(",")]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.run.mkdir(parents=True, exist_ok=True)
    models = {arm: model_for(arm, args.run) for arm in arms}
    for arm, (path, expected) in models.items():
        if sha(Path(path) / "model.safetensors") != expected:
            raise ValueError(f"{arm}: checkpoint changed since it was registered")
    jobs = [{"arm": arm, "offset": offset, "duty": args.duty,
             "model": str(models[arm][0]), "model_sha256": models[arm][1],
             "command": build_command(arm, offset, args.run, models[arm][0])}
            for arm in arms for offset in offsets]
    table3.save(args.run / f"plan-{'-'.join(arms)}-{offsets[0]}-{offsets[-1]}.json", {
        "stage": "target_study_development", "arms": arms, "offsets": offsets, "gpus": gpus,
        "hostname": os.uname().nodename,
        "teacher_alphas": {a: REGISTERED_ALPHAS.get(a) for a in arms},
        "controls": [a for a in arms if a in CONTROLS],
        "bases": {a: ("(control: untouched start point)" if a in CONTROLS else
                      json.loads((args.run / "checkpoints" / a /
                                  "block_regmeanpp_manifest.json").read_text()
                                 ).get("target_study_base")) for a in arms},
        "libero_config_path": str(LIBERO),
        "l0_is_a_reevaluation": ("L0 runs the same interpolated-teacher path at alpha=0, so "
                                 "it is not byte-identical to the earlier round-3 build; its "
                                 "400 episodes are a re-evaluation, and an L0-vs-round3 "
                                 "difference must not be attributed to the teacher change."),
        "episodes_total": 40 * len(jobs), "episodes_per_job": 40,
        "wrapper": str(WRAPPER), "wrapper_sha256": sha(WRAPPER),
        "original_wrapper": str(ORIGINAL_WRAPPER),
        "original_wrapper_sha256": sha(ORIGINAL_WRAPPER),
        "divergence": "See expanded-development/wrapper-divergence.diff. Offset term in the "
                      "per-task seed, offset/actual-init-state receipt fields, video rendering "
                      "forced off, configurable duty. Offset 30 seeds are numerically identical "
                      "to the original wrapper, verified by CPU tests.",
        "seed_rule": "391600 + 1000*(offset-30) + 100*suite_index + task_id; flow seed +100000; "
                     "generator reset per episode",
        "models": {arm: {"path": str(p), "model_sha256": s} for arm, (p, s) in models.items()},
        "duty": args.duty, "videos": False,
        "offset_30_reused": False,
        "run_to_run_spread": {"arm": "c_m (earlier measurement, other arms)", "offset": 30,
                              "runs": [33, 34, 32],
                              "stable_tasks": 38, "unstable_tasks":
                              ["libero_spatial/1", "libero_spatial/5"],
                              "note": "Measured on one arm with three runs; an estimate of the "
                                      "noise floor, not a full characterisation. Differences of "
                                      "one to two tasks per forty are not distinguishable from "
                                      "re-running the same model."},
        "boundary": "Expanded development on one build seed. Not independent confirmation, not a "
                    "final proof. Offset 30 overlaps the already-explored screen; offsets 31, 32 "
                    "and 33 were already used by the earlier tcr_expanded_development_protocol "
                    "repeats; only 34..39 are new to development. None of 30..39 is untouched "
                    "confirmation.",
        "previously_used_offsets": [30, 31, 32, 33],
        "comparison_boundary": ("These development scores must not be differenced against the "
                                "FeatCal 70.92% figure, which comes from a different reset bank."),
        "goal_achieved": False})
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "jobs": len(jobs),
                          "episodes": 40 * len(jobs)}), flush=True)
        return
    total_jobs = len(jobs)
    state = {"done": [], "errors": [], "total": total_jobs}
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu, jobs, args.run, state, lock))
               for gpu in gpus]
    begin = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done = sorted(state["done"], key=lambda r: (r["arm"], r["offset"]))
    totals = {}
    for arm in arms:
        rows = [r for r in done if r["arm"] == arm]
        totals[arm] = {
            "episodes": 40 * len(rows), "successes": sum(r["successes"] for r in rows),
            "offset_30": next((r["successes"] for r in rows if r["offset"] == 30), None),
            "offsets_31_39": sum(r["successes"] for r in rows if r["offset"] != 30),
            "offsets_31_39_episodes": 40 * len([r for r in rows if r["offset"] != 30])}
    table3.save(args.run / f"summary-{'-'.join(arms)}-{offsets[0]}-{offsets[-1]}.json", {
        "status": "complete" if len(done) == total_jobs and not state["errors"] else "incomplete",
        "totals": totals, "records": [{k: v for k, v in r.items() if k != "outcomes"} for r in done],
        "arm_agreement": arm_agreement(done),
        "errors": state["errors"], "wall_hours": round((time.monotonic() - begin) / 3600, 2),
        "independent_confirmation_used": False, "goal_achieved": False})
    print(json.dumps({"status": "done", "totals": {a: t["successes"] for a, t in totals.items()},
                      "errors": state["errors"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--offsets", default=",".join(str(o) for o in OFFSETS))
    parser.add_argument("--gpus", default="1,2,4,5,6,7")
    parser.add_argument("--duty", type=float, default=0.0)
    parser.add_argument("--nice", type=int, default=5)
    parser.add_argument("--preflight-only", action="store_true")
    main(parser.parse_args())
