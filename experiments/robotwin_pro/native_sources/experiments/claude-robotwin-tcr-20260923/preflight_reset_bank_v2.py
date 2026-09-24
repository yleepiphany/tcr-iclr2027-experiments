#!/usr/bin/env python3
"""Isolated, bounded native reset preflight for the 180 RoboTwin A/B identities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
PYTHON = RUNTIME / "envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python"
sys.path[:0] = [str(HERE), str(VLA)]
import capture_contract as contract  # noqa: E402
import preflight_reset_bank as first  # noqa: E402

OLD_PROTOCOL = first.OLD_PROTOCOL
RESET_TIMEOUT_SECONDS = 900
TIMEOUT_CONFIRMATIONS = 2


def child_probe(parent: Path, task: str, seed: int, result_path: Path) -> None:
    first.load_formal(contract.read(parent))
    import torch
    from envs.utils.create_actor import UnStableError
    from lerobot.envs import make_env
    from lerobot.envs.configs import RoboTwinEnvConfig
    from scripts import run_iclr2027_robotwin_native_pi05_smoke_v2 as native
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Reset child requires exactly one visible CUDA device")
    env = None
    try:
        cfg = RoboTwinEnvConfig(task=task, episode_length=None,
                                task_config="demo_clean", action_mode="joint")
        env = make_env(cfg, n_envs=1, use_async_envs=False)[task][0]
        try:
            raw, _ = env.reset(seed=seed)
            native.validate_raw_observation(raw)
            result = {"status": "reset_valid", "observation_sha256": native.observation_digest(raw)}
        except UnStableError as error:
            result = {"status": "simulator_unstable", "error": str(error)}
    finally:
        if env is not None:
            env.close()
        torch.cuda.empty_cache()
    contract.save(result_path, result)


def stop_owned_child(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def probe_once(output: Path, job: dict[str, Any], task: dict[str, Any], seed: int,
               counter: int, trial: int, gpu: int) -> dict[str, Any]:
    result_path = output / "individual" / f"{job['id']}-{task['task_index']}-c{counter}-t{trial}.json"
    log_path = result_path.with_suffix(".log")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--probe-one",
               "--parent", job["parent"]["path"], "--task", task["task"],
               "--seed", str(seed), "--result", str(result_path)]
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=WORK, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=RESET_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            stop_owned_child(process)
            return {"status": "simulator_reset_timeout", "error":
                    f"No reset result within {RESET_TIMEOUT_SECONDS}s", "child_pid": process.pid}
    if code != 0 or not result_path.exists():
        raise RuntimeError(f"Reset child failed technically ({code}): {log_path}")
    result = contract.read(result_path)
    if result.get("status") not in {"reset_valid", "simulator_unstable"}:
        raise ValueError("Reset child emitted an unknown outcome")
    return result


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol = contract.read(OLD_PROTOCOL)
    if protocol.get("schema") != "robotwin_tcr_native_ab_capture_v1" or len(protocol["jobs"]) != 18:
        raise ValueError("Old capture protocol differs")
    gpu, uuid, lease, legacy = first.choose_card("robotwin-180-reset-preflight-v2")
    try:
        output.mkdir(parents=True, exist_ok=False)
        contract.save(output / "started.json", {
            "status": "reset_preflight_running", "pid": os.getpid(),
            "host": socket.gethostname(), "started_at": first.utc(),
            "source_protocol": contract.bind(OLD_PROTOCOL), "gpu": gpu,
            "gpu_uuid": uuid, "min_free_mib": first.MIN_FREE_MIB,
            "reserve_mib": first.RESERVE_MIB, "reset_timeout_seconds": RESET_TIMEOUT_SECONDS,
            "timeout_confirmations": TIMEOUT_CONFIRMATIONS,
            "policy_inference": False, "success_values_read": False,
        })
        try:
            conduct(protocol, output, gpu, uuid)
        except BaseException as error:
            contract.save(output / "failure.json", {"status": "reset_preflight_failed",
                "error_type": type(error).__name__, "error": str(error), "finished_at": first.utc()})
            raise
    finally:
        legacy.release(); lease.release()


def conduct(protocol: dict[str, Any], output: Path, gpu: int, uuid: str) -> None:
    historical = contract.occupied_task_seeds()
    original = {(row["task_index"], row["seed"])
                for job in protocol["jobs"] for row in job["tasks"]}
    if len(original) != 180 or original & historical:
        raise ValueError("Original bank has duplicate/historical reset identity")
    reserved = historical | original
    receipts_path = output / "reset-probes.jsonl"
    accepted: list[dict[str, Any]] = []
    timeout_candidates = 0

    for job in protocol["jobs"]:
        for task in job["tasks"]:
            counter = int(task["derivation_counter"])
            for _ in range(100):
                seed = first.candidate(task, job["repeat"], job["pool"], counter)
                key = (task["task_index"], seed)
                if seed != task["seed"] and key in reserved:
                    counter += 1
                    continue
                for trial in range(1, TIMEOUT_CONFIRMATIONS + 1):
                    result = probe_once(output, job, task, seed, counter, trial, gpu)
                    receipt = {"job": job["id"], "task": task["task"],
                               "task_index": task["task_index"], "old_seed": task["seed"],
                               "seed": seed, "counter": counter, "checked_at": first.utc(),
                               **result}
                    first.append_receipt(receipts_path, receipt)
                    if result["status"] != "simulator_reset_timeout":
                        break
                if result["status"] == "reset_valid":
                    accepted.append(receipt)
                    if seed != task["seed"]:
                        reserved.add(key)
                    if first.gpu_row(gpu)["free_mib"] < first.RESERVE_MIB:
                        raise RuntimeError("GPU free memory fell below 8 GiB")
                    break
                if result["status"] == "simulator_reset_timeout":
                    timeout_candidates += 1
                    reserved.add(key)
                elif result["status"] == "simulator_unstable":
                    reserved.add(key)
                counter += 1
            else:
                raise RuntimeError(f"No valid reset for {job['id']} / {task['task']}")

    amended = first.amend_protocol(protocol, accepted, historical,
                                   output.parent / "capture-v2")
    amended["reset_preflight"] = {
        "status": "all_180_simulator_resets_valid_before_policy_inference",
        "source_protocol": str(OLD_PROTOCOL),
        "source_protocol_sha256": contract.sha256_file(OLD_PROTOCOL),
        "probe_receipts": contract.bind(receipts_path),
        "replacement_count": sum(row["seed"] != row["old_seed"] for row in accepted),
        "timeout_confirmed_candidates": timeout_candidates,
        "timeout_seconds": RESET_TIMEOUT_SECONDS,
        "selection_uses_only_reset_validity": True,
        "policy_inference": False, "success_values_read": False,
    }
    contract.save(output / "candidate-protocol.json", amended)
    contract.save(output / "complete.json", {
        "status": "reset_preflight_complete", "checked": len(accepted),
        "source_protocol_sha256": contract.sha256_file(OLD_PROTOCOL),
        "candidate_protocol": contract.bind(output / "candidate-protocol.json"),
        "probe_receipts": contract.bind(receipts_path),
        "timeout_confirmed_candidates": timeout_candidates,
        "gpu": gpu, "gpu_uuid": uuid, "policy_inference": False,
        "success_values_read": False, "finished_at": first.utc(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--probe-one", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.probe_one:
        if not all((args.parent, args.task, args.seed is not None, args.result)):
            raise ValueError("A reset child requires parent/task/seed/result")
        child_probe(args.parent, args.task, args.seed, args.result)
    else:
        if args.run is None:
            raise ValueError("Reset supervisor requires --run")
        run(args.run.resolve())


if __name__ == "__main__":
    main()
