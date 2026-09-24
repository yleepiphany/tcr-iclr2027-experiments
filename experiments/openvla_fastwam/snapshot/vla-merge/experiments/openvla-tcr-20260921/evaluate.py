#!/usr/bin/env python3
"""OFT evaluation on the immutable main-table reset bank; no stock-state fallback.

Smoke is an exact native-request/replay test, not a success-rate experiment.
Exceptions are fatal and never converted into valid failed episodes.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import numpy as np

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK / "vla-merge/src"))
from vla_merge.libero_procedural_bank import load_selection, sha256_file, sha256_state
from native_oft import NativePolicy, configure_runtime, validate_actions

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
HORIZONS = dict(zip(SUITES, (220, 280, 300, 520)))


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def task_contract(selection, suite, task_id, task):
    selected = selection.tasks[(suite, task_id)]
    if (task.name, task.problem_folder, task.bddl_file) != (
        selected.task_name, selected.problem_folder, selected.bddl_file
    ):
        raise ValueError(f"Task metadata mismatch: {suite}/{task_id}")
    if len(selected.indices) != 10:
        raise ValueError("Expected ten frozen states per task")
    return selected


def state_for_episode(selected, episode):
    if type(episode) is not int or not 0 <= episode < 10:
        raise ValueError("Episode index must be in [0,10); no wrapping")
    state = selected.states[episode].copy()
    digest = sha256_state(state)
    if digest != selected.raw_sha256[episode]:
        raise ValueError("Actual reset input differs from frozen state")
    return state, digest


def rollout(policy, env, task, state, episode_seed, horizon, *, smoke=False, capture_dir=None):
    native = policy.native
    native.set_seed_everywhere(episode_seed)
    env.seed(episode_seed)
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(10):
        obs, _, _, _ = env.step(native.get_libero_dummy_action("openvla"))
    queue, requests = deque(), 0
    for step in range(horizon):
        if not queue:
            observation, _ = native.prepare_observation(obs, policy.resize_size)
            if smoke:
                order = []
                handles = []
                for name, module in policy.linear_modules():
                    def hook(mod, inputs, name=name):
                        order.append({"name": name, "input_shape": list(inputs[0].shape),
                                      "weight_shape": list(mod.weight.shape)})
                    handles.append(module.register_forward_pre_hook(hook))
                try:
                    actions, request = policy.request(observation, task.language, capture=True)
                finally:
                    for handle in handles:
                        handle.remove()
                replay = policy.replay(request)
                error = float(np.max(np.abs(actions - replay)))
                if error != 0.0:
                    raise ValueError(f"Native request replay identity failed: max_abs={error}")
                if capture_dir is not None:
                    policy.torch.save(request, Path(capture_dir) / "request.pt")
                return {"identity_max_abs": error, "action_shape": list(actions.shape),
                        "linear_call_order": order, "requests": 1,
                        "is_success_evaluation": False}
            actions = validate_actions(policy.request(observation, task.language))
            queue.extend(actions)
            requests += 1
        action = native.process_action(queue.popleft().copy(), "openvla")
        obs, _, done, _ = env.step(action.tolist())
        # Match main evaluator's environment success, not any arbitrary termination.
        success = bool(env.check_success())
        if done or success:
            return {"success": success, "action_steps": step + 1, "requests": requests}
    return {"success": False, "action_steps": horizon, "requests": requests}


def validate_rows(rows, selection, suite):
    expected = {(t, e) for t in range(10) for e in range(10)}
    if len(rows) != 100 or {(r["task_id"], r["episode_index"]) for r in rows} != expected:
        raise ValueError("Missing, duplicate or extra episodes")
    for row in rows:
        task = selection.tasks[(suite, row["task_id"])]
        ep = row["episode_index"]
        if (row["suite"] != suite or row["state_index"] != task.indices[ep]
            or row["raw_state_sha256"] != task.raw_sha256[ep]
            or row["rollout_seed"] != selection.eval_seed + ep
            or row["valid"] is not True or type(row["success"]) is not bool):
            raise ValueError("Episode receipt differs from frozen protocol")
    return sum(r["success"] for r in rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()
    selection = load_selection(args.bank, args.selection, verify_source_files=True)
    if selection.repeat_id is None or selection.eval_seed is None:
        raise ValueError("A formal repeat selection is required")
    configure_runtime()
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if suite.n_tasks != 10:
        raise ValueError("Unexpected task count")
    tasks = [task_contract(selection, args.suite, i, suite.get_task(i)) for i in range(10)]
    args.output.mkdir(parents=True, exist_ok=False)
    contract = {"checkpoint": str(args.checkpoint.resolve()), "suite": args.suite,
                "repeat_id": selection.repeat_id, "eval_seed": selection.eval_seed,
                "selection_sha256": selection.selection_sha256,
                "bank_manifest_sha256": selection.bank_manifest_sha256,
                "entrypoint_sha256": sha256_file(Path(__file__)),
                "native_adapter_sha256": sha256_file(Path(__file__).with_name("native_oft.py")),
                "mode": "preflight" if args.preflight_only else "smoke" if args.smoke else "eval",
                "hard_reset": True, "native_chunk": 8, "execute_actions": 8,
                "horizon": HORIZONS[args.suite], "num_steps_wait": 10,
                "tasks": {t.task_key: {"indices": t.indices, "hashes": t.raw_sha256} for t in tasks},
                "created_at": datetime.now(timezone.utc).isoformat()}
    write(args.output / "contract.json", contract)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "episodes": 100}), flush=True)
        return
    policy = NativePolicy(args.checkpoint, args.suite)
    rows = []
    for task_id in range(1 if args.smoke else 10):
        task = suite.get_task(task_id)
        for episode in range(1 if args.smoke else 10):
            state, digest = state_for_episode(tasks[task_id], episode)
            # One fresh simulator per episode: no native task-level soft-reset reuse.
            env, _ = policy.native.get_libero_env(task, "openvla", resolution=256)
            try:
                result = rollout(policy, env, task, state, selection.eval_seed + episode,
                                 HORIZONS[args.suite], smoke=args.smoke, capture_dir=args.output)
            finally:
                env.close()
            row = {"suite": args.suite, "task_id": task_id, "episode_index": episode,
                   "state_index": tasks[task_id].indices[episode], "raw_state_sha256": digest,
                   "rollout_seed": selection.eval_seed + episode, "valid": True, **result}
            if args.smoke:
                write(args.output / "smoke.json", row)
                print(json.dumps({"smoke": "passed", "identity_max_abs": result["identity_max_abs"]}), flush=True)
                return
            with (args.output / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            print(json.dumps({"completed_episodes": len(rows)}), flush=True)
    successes = validate_rows(rows, selection, args.suite)
    write(args.output / "summary.json", {"complete": True, "episodes": 100,
          "successes": successes, "pc_success": successes,
          "episodes_sha256": sha256_file(args.output / "episodes.jsonl")})


if __name__ == "__main__":
    main()
