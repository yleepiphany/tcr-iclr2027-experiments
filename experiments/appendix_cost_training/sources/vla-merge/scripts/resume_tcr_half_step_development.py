"""Explicit recovery of the interrupted serial half-step run; never overwrite it."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time

import run_tcr_half_step_development as original

DEFAULT = original.WORK / "vla-merge-runtime/experiments/tcr-half-step-recovery-20260918"


def assert_stopped(source):
    if original.read(source / "plan.json")["hostname"] != socket.gethostname():
        raise ValueError("Can only audit source processes on their original host")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(arg.endswith(b"/run_tcr_half_step_development.py") or
               arg.startswith(("--output_dir=" + str(source)).encode()) for arg in args):
            raise RuntimeError(f"Original supervisor or worker still exists: {entry.name}")


def reusable(folder, offset, plan, model):
    if not (folder / "exit.json").exists():
        return None  # Even complete-looking eval_info does not establish successful exit.
    if original.read(folder / "exit.json")["return_code"]:
        return None
    launch = original.read(folder / "launch.json")
    if launch["command"] != original.eval_command(plan, folder, model["path"], offset):
        raise ValueError("Completed source used a different command")
    if launch["model_sha256"] != model["model_sha256"] or launch["offset"] != offset:
        raise ValueError("Completed source provenance differs")
    if launch["wrapper_sha256"] != plan["dependencies"][plan["wrapper"]] or launch["duty"] != plan["duty"]:
        raise ValueError("Completed source wrapper/duty differs")
    receipt = original.verify_job(folder, offset)
    if receipt != original.read(folder / "verified.json"):
        raise ValueError("Completed source receipts changed")
    return receipt


def prepare(args):
    assert_stopped(args.source)
    plan = original.read(args.source / "plan.json")
    original.check_dependencies(plan)
    if plan["alpha"] != .5 or plan["arms"] != list(original.ARMS) or plan["offsets"] != list(range(30, 40)):
        raise ValueError("Not the frozen two-arm half-step experiment")
    models = original.read(args.source / "build-complete.json")["models"]
    for arm in original.ARMS:
        model = models[arm]
        folder = Path(model["path"])
        if not model["reload_bitwise_equal"] or not model["outside_scope_unchanged"]:
            raise ValueError("Missing export validation")
        if original.read(folder / "half_step_manifest.json") != model:
            raise ValueError("Model manifest changed")
        if original.sha(folder / "model.safetensors") != model["model_sha256"]:
            raise ValueError("Exported checkpoint bytes changed")
    jobs = []
    for offset in plan["offsets"]:
        for arm in plan["arms"]:
            source = args.source / "development" / arm / f"offset-{offset}"
            receipt = reusable(source, offset, plan, models[arm])
            jobs.append({"arm": arm, "offset": offset, "source": str(source),
                         "reuse": receipt is not None, "verified": receipt,
                         "reason": "verified exit0 and receipts" if receipt else
                         "No verified successful exit; old artifacts retained, whole 40-episode job rerun"})
    args.run.mkdir(parents=True, exist_ok=False)
    output = {"original_plan": plan, "models": models, "jobs": jobs,
              "source": str(args.source), "source_plan_sha256": original.sha(args.source / "plan.json"),
              "recovery_script_sha256": original.sha(__file__),
              "pid": os.getpid(), "hostname": socket.gethostname(), "created_unix": time.time(),
              "poll_seconds": args.poll_seconds, "new_weights": 0,
              "reused_episodes": 40 * sum(j["reuse"] for j in jobs),
              "scheduled_episodes": 40 * sum(not j["reuse"] for j in jobs),
              "no_kill": True, "goal_achieved": False, "independent_confirmation_used": False}
    original.write(args.run / "recovery-plan.json", output)
    return output


def execute(args, recovery):
    plan, models = recovery["original_plan"], recovery["models"]
    results = []
    for job in recovery["jobs"]:
        arm, offset = job["arm"], job["offset"]
        if job["reuse"]:
            results.append({**job, "folder": job["source"]})
            continue
        while True:
            original.check_dependencies(plan)
            raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                                           "--format=csv,noheader,nounits"], text=True)
            rows = [tuple(map(int, line.split(","))) for line in raw.strip().splitlines()]
            gpu = original.select_gpu(rows, original.protect_repeat02())
            if gpu is not None:
                break
            print("WAIT_GPU", arm, offset, flush=True)
            time.sleep(args.poll_seconds)
        folder = args.run / "development" / arm / f"offset-{offset}"
        folder.mkdir(parents=True, exist_ok=False)
        model = models[arm]
        command = original.eval_command(plan, folder, model["path"], offset)
        env = original.environment(gpu)
        env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="1", PI05_LIBERO_INIT_STATE_OFFSET=str(offset),
                   PI05_LIBERO_INIT_STATE_COUNT="1", PI05_TASK_TEXT_MODE="correct",
                   CLAUDE_EVAL_DUTY=str(plan["duty"]), ITERATION_PHYSICAL_GPU=str(gpu),
                   ITERATION_NOISE_RECEIPT=str(folder / "paired-noise.jsonl"))
        original.write(folder / "launch.json", {
            "command": command, "gpu": gpu, "gpu_snapshot": rows, "offset": offset,
            "model": model["path"], "model_sha256": model["model_sha256"],
            "wrapper_sha256": plan["dependencies"][plan["wrapper"]], "duty": plan["duty"],
            "hostname": socket.gethostname(), "started_unix": time.time()})
        begin = time.monotonic()
        with (folder / "worker.log").open("x") as log:
            worker = subprocess.Popen(command, cwd=original.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            print("START", arm, offset, "gpu", gpu, "worker", worker.pid, flush=True)
            try:
                original.write(folder / "worker.json", {"pid": worker.pid, "hostname": socket.gethostname()})
            finally:
                code = worker.wait()
        original.write(folder / "exit.json", {"return_code": code, "wall_seconds": time.monotonic() - begin})
        verified = original.verify_job(folder, offset)
        original.write(folder / "verified.json", verified)
        results.append({**job, "folder": str(folder), "verified": verified})
        print("COMPLETE", arm, offset, flush=True)
    original.write(args.run / "complete.json", {"jobs": results, "episodes_per_arm": 400,
                   "status": "evaluations_complete_pending_paired_analysis", "ended_unix": time.time(),
                   "goal_achieved": False, "independent_confirmation_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=original.DEFAULT)
    parser.add_argument("--run", type=Path, default=DEFAULT)
    parser.add_argument("--poll-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        raise ValueError("Low-frequency monitoring only")
    os.nice(10)
    with (args.run.parent / ".tcr-half-step-recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepared = prepare(args)
            execute(args, prepared)
        except Exception as error:
            if args.run.is_dir():
                original.write(args.run / "failure.json", {"error": repr(error), "no_retry": True})
            raise
