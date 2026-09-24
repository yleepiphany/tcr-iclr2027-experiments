#!/usr/bin/env python3
"""Evaluate OFT checkpoints only on the frozen repair-development stock bank."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "openvla-tcr-20260921"
sys.path[:0] = [str(HERE), str(OLD)]

import dev_bank
from evaluate import HORIZONS, SUITES, rollout, write
from native_oft import NativePolicy, configure_runtime
from vla_merge.libero_procedural_bank import sha256_file, sha256_state


@dataclass(frozen=True)
class DevTask:
    task_key: str
    suite: str
    task_id: int
    task_name: str
    problem_folder: str
    bddl_file: str
    indices: tuple[int, ...]
    states: np.ndarray
    raw_sha256: tuple[str, ...]


@dataclass(frozen=True)
class DevSelection:
    root: Path
    selection_path: Path
    selection_sha256: str
    manifest_sha256: str
    selection_id: str
    eval_seed: int
    tasks: dict[tuple[str, int], DevTask]


def load_development_selection(root: Path, selection_path: Path) -> DevSelection:
    root = root.resolve()
    dev_bank.verify(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selection_path = selection_path.resolve()
    if selection_path.parent != root:
        raise ValueError("Development selection must be inside the frozen bank")
    selection = json.loads(selection_path.read_text())
    manifest_sha = sha256_file(manifest_path)
    if (selection.get("schema") != "oft_tcr_development_selection_v1"
            or selection.get("bank_manifest_sha256") != manifest_sha
            or selection.get("eval_seed") != dev_bank.EVAL_SEED
            or selection.get("policy_outcomes_read") is not False
            or selection.get("success_filter") is not False):
        raise ValueError("Development selection contract differs")
    expected_count = {"development-40": 1, "development-400": 10}.get(selection.get("selection_id"))
    if expected_count is None:
        raise ValueError("Unknown development selection")
    tasks = {}
    for key, row in selection["tasks"].items():
        task = manifest["tasks"][key]
        indices = tuple(row["stock_offsets"])
        hashes = tuple(row["raw_sha256"])
        if len(indices) != expected_count or len(hashes) != expected_count:
            raise ValueError(f"Development episode count differs: {key}")
        source = np.load(root / task["states_file"], allow_pickle=False)
        by_offset = {item["stock_offset"]: i for i, item in enumerate(task["states"])}
        try:
            states = np.ascontiguousarray(source[[by_offset[index] for index in indices]])
        except KeyError as exc:
            raise ValueError(f"Selection references an unfrozen offset: {key}") from exc
        for state, digest in zip(states, hashes, strict=True):
            if sha256_state(state) != digest:
                raise ValueError(f"Selected reset content differs: {key}")
        identity = (task["suite"], int(task["task_id"]))
        tasks[identity] = DevTask(
            task_key=key, suite=identity[0], task_id=identity[1],
            task_name=task["task_name"], problem_folder=task["problem_folder"],
            bddl_file=task["bddl_file"], indices=indices, states=states,
            raw_sha256=hashes,
        )
    if set(tasks) != {(suite, i) for suite in SUITES for i in range(10)}:
        raise ValueError("Development selection does not cover exactly 40 tasks")
    return DevSelection(
        root=root, selection_path=selection_path,
        selection_sha256=sha256_file(selection_path), manifest_sha256=manifest_sha,
        selection_id=selection["selection_id"], eval_seed=selection["eval_seed"], tasks=tasks,
    )


def task_contract(selection: DevSelection, suite: str, task_id: int, task) -> DevTask:
    selected = selection.tasks[(suite, task_id)]
    if (task.name, task.problem_folder, task.bddl_file) != (
            selected.task_name, selected.problem_folder, selected.bddl_file):
        raise ValueError(f"Task metadata mismatch: {suite}/{task_id}")
    return selected


def state_for_episode(selected: DevTask, episode: int) -> tuple[np.ndarray, str]:
    if type(episode) is not int or not 0 <= episode < len(selected.indices):
        raise ValueError("Development episode index is out of range; no wrapping")
    state = selected.states[episode].copy()
    digest = sha256_state(state)
    if digest != selected.raw_sha256[episode]:
        raise ValueError("Actual development reset differs from frozen state")
    return state, digest


def validate_rows(rows: list[dict], selection: DevSelection, suite: str) -> int:
    episodes = len(selection.tasks[(suite, 0)].indices)
    expected = {(task, ep) for task in range(10) for ep in range(episodes)}
    if len(rows) != 10 * episodes or {(r["task_id"], r["episode_index"]) for r in rows} != expected:
        raise ValueError("Missing, duplicate or extra development episodes")
    for row in rows:
        task = selection.tasks[(suite, row["task_id"])]
        ep = row["episode_index"]
        if (row["suite"] != suite or row["state_index"] != task.indices[ep]
                or row["raw_state_sha256"] != task.raw_sha256[ep]
                or row["rollout_seed"] != selection.eval_seed + ep
                or row["valid"] is not True or type(row["success"]) is not bool):
            raise ValueError("Development episode receipt differs from frozen protocol")
    return sum(row["success"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    selection = load_development_selection(args.bank, args.selection)
    configure_runtime()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if suite.n_tasks != 10:
        raise ValueError("Unexpected task count")
    tasks = [task_contract(selection, args.suite, i, suite.get_task(i)) for i in range(10)]
    episodes_per_task = len(tasks[0].indices)
    if any(len(task.indices) != episodes_per_task for task in tasks):
        raise ValueError("Development task episode counts differ")
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {
        "checkpoint": str(args.checkpoint.resolve()), "suite": args.suite,
        "selection_id": selection.selection_id, "eval_seed": selection.eval_seed,
        "selection_sha256": selection.selection_sha256,
        "bank_manifest_sha256": selection.manifest_sha256,
        "entrypoint_sha256": sha256_file(Path(__file__)),
        "native_adapter_sha256": sha256_file(OLD / "native_oft.py"),
        "mode": "preflight" if args.preflight_only else "development_eval",
        "formal_evaluation": False, "used_for_candidate_selection": True,
        "hard_reset": True, "native_chunk": 8, "execute_actions": 8,
        "horizon": HORIZONS[args.suite], "num_steps_wait": 10,
        "episodes": 10 * episodes_per_task,
        "tasks": {task.task_key: {"indices": task.indices, "hashes": task.raw_sha256}
                  for task in tasks},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write(args.output / "contract.json", contract)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "episodes": 10 * episodes_per_task}), flush=True)
        return
    policy = NativePolicy(args.checkpoint, args.suite)
    rows = []
    for task_id, task_bank in enumerate(tasks):
        task = suite.get_task(task_id)
        for episode in range(episodes_per_task):
            state, digest = state_for_episode(task_bank, episode)
            env, _ = policy.native.get_libero_env(task, "openvla", resolution=256)
            try:
                result = rollout(policy, env, task, state, selection.eval_seed + episode,
                                 HORIZONS[args.suite])
            finally:
                env.close()
            row = {"suite": args.suite, "task_id": task_id, "episode_index": episode,
                   "state_index": task_bank.indices[episode], "raw_state_sha256": digest,
                   "rollout_seed": selection.eval_seed + episode, "valid": True, **result}
            with (args.output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            print(json.dumps({"completed_episodes": len(rows)}), flush=True)
    successes = validate_rows(rows, selection, args.suite)
    write(args.output / "summary.json", {
        "complete": True, "episodes": len(rows), "successes": successes,
        "pc_success": 100 * successes / len(rows),
        "episodes_sha256": sha256_file(args.output / "episodes.jsonl"),
        "formal_evaluation": False, "used_for_candidate_selection": True,
    })


if __name__ == "__main__":
    main()
