#!/usr/bin/env python3
"""Isolated 180-reset preflight with exact lazy official-language generation.

Attempt 02 is invalid because its 900-second reset timeout included the
million-string official instruction generator. This attempt starts from the
original frozen capture protocol, never from attempt-02 timeout-selected seeds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_contract as contract  # noqa: E402
import fast_official_instruction as fast  # noqa: E402
import preflight_reset_bank as first  # noqa: E402
import preflight_reset_bank_v2 as v2  # noqa: E402

PYTHON = v2.PYTHON
WORK = v2.WORK
OLD_PROTOCOL = v2.OLD_PROTOCOL
RESET_TIMEOUT_SECONDS = v2.RESET_TIMEOUT_SECONDS
TIMEOUT_CONFIRMATIONS = v2.TIMEOUT_CONFIRMATIONS


def phase(path: Path, value: str) -> None:
    with path.open("w") as stream:
        stream.write(value + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def child_probe(parent: Path, task: str, seed: int, result_path: Path,
                phase_path: Path) -> None:
    first.load_formal(contract.read(parent))
    import torch
    from envs.utils.create_actor import UnStableError
    from lerobot.envs import make_env, robotwin
    from lerobot.envs.configs import RoboTwinEnvConfig
    from scripts import run_iclr2027_robotwin_native_pi05_smoke_v2 as native

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Reset child requires exactly one visible CUDA device")
    if os.environ.get(robotwin.OFFICIAL_INSTRUCTION_MAX_ENV) != "1000000":
        raise ValueError("The historical million-description protocol is not pinned")
    fast.install()
    fast_function = robotwin._generate_robotwin_official_instruction

    def tracked(task_name: str, env: Any) -> str:
        phase(phase_path, "instruction_entered_after_simulator_setup")
        instruction = fast_function(task_name, env)
        phase(phase_path, "instruction_complete")
        return instruction

    robotwin._generate_robotwin_official_instruction = tracked
    env = None
    try:
        cfg = RoboTwinEnvConfig(task=task, episode_length=None,
                                task_config="demo_clean", action_mode="joint")
        env = make_env(cfg, n_envs=1, use_async_envs=False)[task][0]
        try:
            phase(phase_path, "simulator_reset_running")
            raw, _ = env.reset(seed=seed)
            phase(phase_path, "simulator_reset_complete")
            native.validate_raw_observation(raw)
            result = {"status": "reset_valid", "observation_sha256": native.observation_digest(raw)}
        except UnStableError as error:
            result = {"status": "simulator_unstable", "error": str(error)}
    finally:
        if env is not None:
            env.close()
        torch.cuda.empty_cache()
    contract.save(result_path, result)


def probe_once(output: Path, job: dict[str, Any], task: dict[str, Any], seed: int,
               counter: int, trial: int, gpu: int) -> dict[str, Any]:
    result_path = output / "individual" / f"{job['id']}-{task['task_index']}-c{counter}-t{trial}.json"
    log_path = result_path.with_suffix(".log")
    phase_path = result_path.with_suffix(".phase")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "LEROBOT_ROBOTWIN_INSTRUCTION_MAX": "1000000"})
    command = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--probe-one",
               "--parent", job["parent"]["path"], "--task", task["task"],
               "--seed", str(seed), "--result", str(result_path),
               "--phase", str(phase_path)]
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=WORK, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=RESET_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            last_phase = phase_path.read_text().strip() if phase_path.exists() else "unknown"
            v2.stop_owned_child(process)
            if last_phase != "simulator_reset_running":
                raise RuntimeError(f"Technical reset child timeout after phase={last_phase}: {log_path}")
            return {"status": "simulator_reset_timeout", "error":
                    f"Simulator setup did not return within {RESET_TIMEOUT_SECONDS}s",
                    "child_pid": process.pid, "phase": last_phase}
    if code != 0 or not result_path.exists():
        raise RuntimeError(f"Reset child failed technically ({code}): {log_path}")
    result = contract.read(result_path)
    if result.get("status") not in {"reset_valid", "simulator_unstable"}:
        raise ValueError("Reset child emitted an unknown outcome")
    result["phase"] = phase_path.read_text().strip() if phase_path.exists() else "unknown"
    return result


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    protocol = contract.read(OLD_PROTOCOL)
    if protocol.get("schema") != "robotwin_tcr_native_ab_capture_v1" or len(protocol["jobs"]) != 18:
        raise ValueError("Original capture protocol changed")
    gpu, uuid, lease, legacy = first.choose_card("robotwin-180-reset-preflight-v3")
    try:
        output.mkdir(parents=True, exist_ok=False)
        instruction_contract = {
            "schema": "robotwin_exact_lazy_official_instruction_v1",
            "upstream_max_descriptions": 1_000_000,
            "upstream_wrapper": contract.bind(WORK / "pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/envs/robotwin.py"),
            "upstream_generator": contract.bind(WORK / "vla-merge-runtime/environments/robotwin2-0aee-open3d-optional-v2/description/utils/generate_episode_instructions.py"),
            "lazy_implementation": contract.bind(HERE / "fast_official_instruction.py"),
            "behavior_tests": "test_fast_official_instruction.py: 17 passed, matching text and Python/NumPy RNG states",
            "affected_tasks": sorted(fast.TASKS),
            "does_not_change_description_distribution_or_rng_state": True,
        }
        contract.save(output / "instruction-contract.json", instruction_contract)
        contract.save(output / "started.json", {
            "status": "reset_preflight_running", "pid": os.getpid(),
            "host": socket.gethostname(), "started_at": first.utc(),
            "source_protocol": contract.bind(OLD_PROTOCOL), "gpu": gpu,
            "gpu_uuid": uuid, "min_free_mib": first.MIN_FREE_MIB,
            "reserve_mib": first.RESERVE_MIB, "reset_timeout_seconds": RESET_TIMEOUT_SECONDS,
            "timeout_confirmations": TIMEOUT_CONFIRMATIONS,
            "instruction_contract": contract.bind(output / "instruction-contract.json"),
            "policy_inference": False, "success_values_read": False,
        })
        try:
            v2.probe_once = probe_once
            v2.conduct(protocol, output, gpu, uuid)
            candidate = contract.read(output / "candidate-protocol.json")
            candidate["reset_preflight"]["instruction_contract"] = contract.bind(
                output / "instruction-contract.json")
            contract.save(output / "candidate-protocol.json", candidate)
            complete = contract.read(output / "complete.json")
            complete["candidate_protocol"] = contract.bind(output / "candidate-protocol.json")
            complete["instruction_contract"] = contract.bind(output / "instruction-contract.json")
            contract.save(output / "complete.json", complete)
        except BaseException as error:
            contract.save(output / "failure.json", {"status": "reset_preflight_failed",
                "error_type": type(error).__name__, "error": str(error),
                "finished_at": first.utc()})
            raise
    finally:
        legacy.release(); lease.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--probe-one", action="store_true")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--phase", type=Path)
    args = parser.parse_args()
    if args.probe_one:
        if None in (args.parent, args.task, args.seed, args.result, args.phase):
            raise ValueError("A reset child needs parent/task/seed/result/phase")
        child_probe(args.parent, args.task, args.seed, args.result, args.phase)
    elif args.run is not None:
        run(args.run.resolve())
    else:
        parser.error("Choose --run or --probe-one")


if __name__ == "__main__":
    main()
