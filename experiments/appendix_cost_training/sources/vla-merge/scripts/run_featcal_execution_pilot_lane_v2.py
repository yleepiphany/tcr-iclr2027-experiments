#!/usr/bin/env python3
"""Bounded two-GPU pilot orchestration; existing GPU jobs are never stopped."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
SOURCE = WORK / "pi05_lora_finetune_v2_20260826"
PYTHON = SOURCE / ".venv/bin/python"
TABLE1 = WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910"
PILOT = WORK / "vla-merge-runtime/experiments/featcal-execution-pilot-20260916"
RESULT = PILOT / "attempt-v2"
SUITES = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "long": "libero_10"}


def environment(gpu):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_GL": "egl",
            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "TORCHINDUCTOR_COMPILE_THREADS": "1", "PYTHONUNBUFFERED": "1",
            "PALIGEMMA_TOKENIZER_PATH": str(SOURCE / "assets/paligemma-3b-pt-224-tokenizer"),
            "PYTHONPATH": ":".join([str(WORK / "vla-merge-runtime/python-overlay"),
                                       str(SOURCE / "src"), str(ROOT / "scripts"), str(ROOT / "src")])}


def wait_memory(gpu):
    # Shared use authorized by user. Reserve room for the existing workload;
    # our worker separately enforces a 28-GiB allocator ceiling (+ context).
    for _ in range(1440):
        used, total = [float(x) for x in subprocess.check_output([
            "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits"], text=True).strip().split(",")]
        if total - used >= 38 * 1024:
            return
        print(f"WAIT gpu={gpu} free_mib={total-used}", flush=True)
        time.sleep(30)
    raise RuntimeError("No safe GPU memory window within 12 hours")


def run_logged(command, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        print("START", str(log), flush=True)
        subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print("COMPLETE", str(log), flush=True)


def collect_one(name, gpu):
    out = PILOT / "raw-inputs" / name
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite capture attempt: {out}")
    receipt = json.loads((TABLE1 / f"libero/tcr-e/repeat-01/execution-traces-v2/{name}/receipt.json").read_text())
    policy = receipt["dense_checkpoint"]["path"]
    wait_memory(gpu)
    out.mkdir(parents=True)
    env = environment(gpu)
    env.update({"PI05_BLOCK_REGMEANPP_TASK": name,
                "PI05_BLOCK_REGMEANPP_TENSOR_OUTPUT": str(out / "calls.safetensors"),
                "PI05_BLOCK_REGMEANPP_MANIFEST_OUTPUT": str(out / "calls.json"),
                "PI05_BLOCK_REGMEANPP_CALIBRATION_POLICY": policy,
                "PI05_BLOCK_REGMEANPP_MAX_CALLS": "150",
                "PI05_BLOCK_REGMEANPP_MAX_CALLS_PER_PROMPT": "15",
                "PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE": "5",
                "PI05_BLOCK_REGMEANPP_EPISODE_AWARE": "1",
                "PI05_BLOCK_REGMEANPP_FULL_PREFIX": "0",
                "PI05_BLOCK_REGMEANPP_FLOW_INDICES": "0,5,9",
                "PI05_BLOCK_REGMEANPP_REQUEST_MODE": "reservoir",
                "PI05_BLOCK_REGMEANPP_START_SEED": "271001",
                "PI05_LIBERO_INIT_STATE_OFFSET": "0",
                "PI05_LIBERO_INIT_STATE_COUNT": "1",
                "FEATCAL_EXEC_NOISE_SEED": "272001",
                "FEATCAL_EXEC_MEMORY_FRACTION": "0.35"})
    command = [str(PYTHON), str(ROOT / "scripts/collect_pi05_featcal_execution_inputs.py"),
               f"--output_dir={out / 'rollout'}", "--env.type=libero", f"--env.task={SUITES[name]}",
               "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]", "--eval.batch_size=1",
               "--eval.n_episodes=1", "--seed=271001", f"--policy.path={policy}",
               "--policy.device=cuda", "--policy.compile_model=false",
               "--policy.gradient_checkpointing=false", "--policy.n_action_steps=10"]
    with (out / "launch.json").open("x") as f:
        json.dump({"command": command, "host": socket.gethostname(), "gpu": gpu,
                   "source_checkpoint": receipt["dense_checkpoint"],
                   "allocator_fraction": .35, "shared_gpu": True}, f, indent=2)
    run_logged(command, env, out / "collector.log")
    manifest = json.loads((out / "calls.json").read_text())
    if manifest["sample_count"] != 150:
        raise ValueError("Capture incomplete")


def collect_lane(names, gpu):
    for name in names:
        collect_one(name, gpu)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def build_stage(stage, gpu, experts=None):
    wait_memory(gpu)
    command = [str(PYTHON), str(ROOT / "scripts/run_pi05_featcal_execution_pilot_v2.py"), "--stage", stage]
    if experts:
        command += ["--experts", experts]
    run_logged(command, environment(gpu), RESULT / f"build-{stage}-{gpu}.log")


def evaluate_lane(names, gpu):
    complete = json.loads((PILOT / "build-v2/complete.json").read_text())
    if complete["status"] != "complete" or not complete["reload_bitwise_equal"]:
        raise ValueError("Candidate did not pass checkpoint reload")
    candidate = complete["checkpoint"]
    demo = TABLE1 / "preflight/featcal-formal-full-v1/checkpoint/pretrained_model"
    policies = [("execution", candidate["path"], candidate["model_sha256"]),
                ("demo", str(demo), "e43de109844431c02e316e57f701d7c06a9a3c8feaa3c41c9f391281f8197efa")]
    for method, policy, sha in policies:
        if digest(Path(policy) / "model.safetensors") != sha:
            raise ValueError("Evaluation checkpoint hash differs")
        for name in names:
            out = RESULT / "development" / method / name
            if out.exists():
                raise FileExistsError(out)
            wait_memory(gpu)
            env = environment(gpu)
            env.update({"PI05_LIBERO_INIT_STATE_OFFSET": "30", "PI05_LIBERO_INIT_STATE_COUNT": "3",
                        "PI05_LIBERO_SUITE": SUITES[name], "PI05_TASK_TEXT_MODE": "correct",
                        "FEATCAL_EXEC_MEMORY_FRACTION": ".35"})
            command = [str(PYTHON), str(ROOT / "scripts/eval_pi05_capped_development.py"),
                       f"--output_dir={out}", "--env.type=libero", f"--env.task={SUITES[name]}",
                       "--eval.batch_size=1", "--eval.n_episodes=3", "--seed=260101",
                       f"--policy.path={policy}", "--policy.device=cuda",
                       "--policy.compile_model=false", "--policy.gradient_checkpointing=false",
                       "--policy.n_action_steps=10"]
            launch = {"method": method, "suite": name, "command": command, "checkpoint_sha256": sha,
                      "host": socket.gethostname(), "gpu": gpu, "formal_evaluation": False}
            launch_path = RESULT / "development" / f"{method}-{name}-launch.json"
            launch_path.parent.mkdir(parents=True, exist_ok=True)
            with launch_path.open("x") as f:
                json.dump(launch, f, indent=2)
            run_logged(command, env, RESULT / "development" / f"{method}-{name}.log")
            outcomes = json.loads((out / "eval_info.json").read_text())
            if sum(len(t["metrics"]["successes"]) for t in outcomes["per_task"]) != 30:
                raise ValueError("Incomplete development suite")


def continue_after_collection():
    RESULT.mkdir(exist_ok=False)
    # Independent numerical diagnostics must pass before any new solver run.
    for name in ("identity-spatial-strict-fp32-v1", "identity-object-baseline-v1"):
        receipt = json.loads((PILOT / "diagnostics" / name / "summary.json").read_text())
        if receipt.get("status") != "passed":
            raise ValueError("Independent replay-identity diagnosis failed")
    for _ in range(1440):
        if all((PILOT / "raw-inputs" / e / "calls.json").is_file() for e in SUITES):
            break
        print("WAIT_FOR_FOUR_RAW_CAPTURES", flush=True)
        time.sleep(30)
    else:
        raise RuntimeError("Raw captures incomplete after 12 hours; no solve launched")
    build_stage("prepare", 6)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(build_stage, "teachers", 6, "spatial,goal"),
                pool.submit(build_stage, "teachers", 7, "object,long")]
        for job in jobs:
            job.result()
    build_stage("solve", 6)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(evaluate_lane, ["spatial", "goal"], 6),
                pool.submit(evaluate_lane, ["object", "long"], 7)]
        for job in jobs:
            job.result()
    summary = {"formal_result": False, "episodes_per_arm": 120, "arms": {}}
    for method in ["demo", "execution"]:
        successes, total = 0, 0
        suites = {}
        for name in SUITES:
            p = RESULT / "development" / method / name / "eval_info.json"
            data = json.loads(p.read_text())
            outcomes = [v for t in data["per_task"] for v in t["metrics"]["successes"]]
            successes += sum(outcomes)
            total += len(outcomes)
            suites[name] = {"successes": sum(outcomes), "episodes": len(outcomes), "eval_sha256": digest(p)}
        summary["arms"][method] = {"successes": successes, "episodes": total, "success_pct": 100*successes/total, "suites": suites}
    summary["execution_minus_demo_pp"] = summary["arms"]["execution"]["success_pct"] - summary["arms"]["demo"]["success_pct"]
    with (RESULT / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    print("PAIRED_DEVELOPMENT_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["collect", "continue"], required=True)
    args = parser.parse_args()
    if socket.gethostname() != "dsw-824375-57c745db88-n6tv9":
        raise RuntimeError("User authorized GPUs 6/7 of current host only")
    if args.phase == "continue":
        continue_after_collection()
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(collect_lane, ["spatial", "goal"], 6),
                       pool.submit(collect_lane, ["object", "long"], 7)]
            for f in futures:
                f.result()
        print("ALL_FOUR_EXECUTION_INPUT_CAPTURES_COMPLETE", flush=True)
