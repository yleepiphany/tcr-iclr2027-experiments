#!/usr/bin/env python3
"""Check every RoboTwin TCR calibration reset without policy inference.

This is a distinct technical attempt.  It never reads actions, reward or success;
only simulator reset validity can cause a deterministic seed replacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
VLA = HERE.parents[1]
WORK = VLA.parent
RUNTIME = WORK / "vla-merge-runtime"
sys.path[:0] = [str(HERE), str(VLA), str(VLA / "experiments/claude-firstpass-cause-20260920")]
import capture_contract as contract  # noqa: E402
import card_flock  # noqa: E402
from run_capture_queue import gpu_row  # noqa: E402

OLD_PROTOCOL = RUNTIME / "experiments/claude-robotwin-tcr-20260923/capture-v1/protocol.json"
MIN_FREE_MIB = 32768
RESERVE_MIB = 8192


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate(row: dict[str, Any], repeat: int, pool: str, counter: int) -> int:
    if counter == row["derivation_counter"]:
        return int(row["seed"])
    return contract.uint31(contract.SEED_NAMESPACE, "reset", repeat, pool,
                           row["task_index"], counter)


def amend_protocol(protocol: dict[str, Any], checks: list[dict[str, Any]],
                   historical: set[tuple[int, int]], output_root: Path) -> dict[str, Any]:
    """Apply only reset-valid identities and keep every job/task in place."""
    frozen = [(job["id"], row["task_index"]) for job in protocol["jobs"] for row in job["tasks"]]
    keyed = {(row["job"], row["task_index"]): row for row in checks}
    if len(frozen) != len(set(frozen)) or len(keyed) != len(checks) or set(keyed) != set(frozen):
        raise ValueError("Reset checks do not exactly cover the frozen job/task partition")
    amended = json.loads(json.dumps(protocol))
    amended["created_at"] = utc()
    amended["amended_from_source_protocol"] = str(OLD_PROTOCOL)
    amended["reset_selection_uses_only_simulator_validity"] = True
    for job in amended["jobs"]:
        job["output"] = str(output_root / "jobs" / job["id"])
        for row in job["tasks"]:
            checked = keyed[(job["id"], row["task_index"])]
            if checked["task"] != row["task"] or checked["old_seed"] != row["seed"]:
                raise ValueError("Reset check task or old-seed identity differs")
            if checked["status"] != "reset_valid":
                raise ValueError("Reset check did not pass simulator initialization")
            if type(checked["counter"]) is not int or checked["counter"] < row["derivation_counter"]:
                raise ValueError("Reset check derivation counter differs")
            if checked["seed"] != candidate(row, job["repeat"], job["pool"], checked["counter"]):
                raise ValueError("Reset check seed does not follow the frozen namespace")
            row["seed"] = checked["seed"]
            row["derivation_counter"] = checked["counter"]
    amended_keys = {(row["task_index"], row["seed"])
                    for job in amended["jobs"] for row in job["tasks"]}
    if len(amended_keys) != len(frozen) or amended_keys & historical:
        raise ValueError("Amended reset bank has duplicate or formal/historical collision")
    return amended


def choose_card(label: str) -> tuple[int, str, Any, Any]:
    uuids = card_flock.gpu_uuids()
    for gpu in sorted(uuids, key=lambda index: gpu_row(index)["free_mib"], reverse=True):
        row = gpu_row(gpu)
        if row["free_mib"] < MIN_FREE_MIB:
            continue
        lease = card_flock.take_card(gpu, uuids[gpu], label, "robotwin-reset-preflight")
        if lease is None:
            continue
        legacy = card_flock._try_lock(
            RUNTIME / "resource-leases" / socket.gethostname() / f"gpu-{gpu}.lock",
            label, {"job": label, "gpu": gpu, "stage": "robotwin-reset-preflight"},
        )
        if legacy is None:
            lease.release()
            continue
        after = gpu_row(gpu)
        if after["uuid"] != uuids[gpu] or after["free_mib"] < MIN_FREE_MIB:
            legacy.release(); lease.release()
            continue
        return gpu, uuids[gpu], lease, legacy
    raise RuntimeError("No unleased GPU has 32 GiB free for reset preflight")


def append_receipt(path: Path, receipt: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_formal(parent: dict[str, Any]) -> None:
    formal_path = VLA / "scripts/watch_robotwin_experts_formal_v2.py"
    spec = importlib.util.spec_from_file_location("robotwin_reset_preflight_formal", formal_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.slicer.runtime_setup(parent)


def probe(protocol: dict[str, Any], output: Path, gpu: int, uuid: str) -> None:
    # Set CUDA visibility before importing Torch or the simulator.  The parent
    # process holds both host/UUID and legacy host/index leases throughout.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
                       "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    first_parent = contract.read(Path(protocol["jobs"][0]["parent"]["path"]))
    load_formal(first_parent)
    import torch
    from envs.utils.create_actor import UnStableError
    from lerobot.envs import make_env
    from lerobot.envs.configs import RoboTwinEnvConfig
    from scripts import run_iclr2027_robotwin_native_pi05_smoke_v2 as native

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Reset preflight requires exactly one visible CUDA device")

    historical = contract.occupied_task_seeds()
    original = {(row["task_index"], row["seed"])
                for job in protocol["jobs"] for row in job["tasks"]}
    if len(original) != 180 or original & historical:
        raise ValueError("Original reset bank has duplicate or historical collision")
    reserved = historical | original
    checks: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    failure_path = output / "reset-probes.jsonl"

    for job in protocol["jobs"]:
        for task in job["tasks"]:
            counter = int(task["derivation_counter"])
            for _ in range(100):
                seed = candidate(task, job["repeat"], job["pool"], counter)
                key = (task["task_index"], seed)
                if seed != task["seed"] and key in reserved:
                    counter += 1
                    continue
                env_cfg = RoboTwinEnvConfig(task=task["task"], episode_length=None,
                                            task_config="demo_clean", action_mode="joint")
                env = make_env(env_cfg, n_envs=1, use_async_envs=False)[task["task"]][0]
                try:
                    raw, _ = env.reset(seed=seed)
                    native.validate_raw_observation(raw)
                    observation_sha = native.observation_digest(raw)
                except UnStableError as error:
                    append_receipt(failure_path, {"job": job["id"], "task": task["task"],
                        "task_index": task["task_index"], "seed": seed, "counter": counter,
                        "status": "simulator_unstable", "error": str(error), "checked_at": utc()})
                    counter += 1
                    continue
                finally:
                    env.close()
                    torch.cuda.empty_cache()
                row = {"job": job["id"], "task": task["task"],
                       "task_index": task["task_index"], "old_seed": task["seed"],
                       "seed": seed, "counter": counter, "status": "reset_valid",
                       "observation_sha256": observation_sha, "checked_at": utc()}
                append_receipt(failure_path, row)
                checks.append(row)
                if seed != task["seed"]:
                    replacements.append(row)
                    reserved.add(key)
                if gpu_row(gpu)["free_mib"] < RESERVE_MIB:
                    raise RuntimeError("GPU free memory fell below 8 GiB during reset preflight")
                break
            else:
                raise RuntimeError(f"No valid collision-free reset for {job['id']} / {task['task']}")

    if len(checks) != 180 or len({(row["task_index"], row["seed"]) for row in checks}) != 180:
        raise AssertionError("Reset preflight did not cover 180 unique identities")
    amended = amend_protocol(protocol, checks, historical, output.parent / "capture-v2")
    amended["reset_preflight"] = {
        "status": "all_180_simulator_resets_valid_before_policy_inference",
        "source_protocol_sha256": contract.sha256_file(OLD_PROTOCOL),
        "source_protocol": str(OLD_PROTOCOL), "probe_receipts": contract.bind(failure_path),
        "replacement_count": len(replacements), "selection_uses_only_reset_validity": True,
        "policy_inference": False, "success_values_read": False,
    }
    contract.save(output / "candidate-protocol.json", amended)
    contract.save(output / "complete.json", {"status": "reset_preflight_complete",
        "old_protocol_sha256": contract.sha256_file(OLD_PROTOCOL),
        "candidate_protocol": contract.bind(output / "candidate-protocol.json"),
        "probe_receipts": contract.bind(failure_path), "checked": len(checks),
        "replacements": replacements, "gpu": gpu, "gpu_uuid": uuid,
        "policy_inference": False, "success_values_read": False, "finished_at": utc()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = contract.read(OLD_PROTOCOL)
    if protocol.get("schema") != "robotwin_tcr_native_ab_capture_v1" or len(protocol.get("jobs", [])) != 18:
        raise ValueError("Wrong frozen RoboTwin capture protocol")
    gpu, uuid, lease, legacy = choose_card("robotwin-180-reset-preflight")
    try:
        output.mkdir(parents=True, exist_ok=False)
        contract.save(output / "started.json", {"status": "reset_preflight_running",
            "pid": os.getpid(), "host": socket.gethostname(), "started_at": utc(),
            "source_protocol": contract.bind(OLD_PROTOCOL), "gpu": gpu, "gpu_uuid": uuid,
            "min_free_mib": MIN_FREE_MIB, "reserve_mib": RESERVE_MIB,
            "success_values_read": False})
        try:
            probe(protocol, output, gpu, uuid)
        except BaseException as error:
            contract.save(output / "failure.json", {"status": "reset_preflight_failed",
                "error_type": type(error).__name__, "error": str(error), "finished_at": utc()})
            raise
    finally:
        legacy.release(); lease.release()


if __name__ == "__main__":
    main()
