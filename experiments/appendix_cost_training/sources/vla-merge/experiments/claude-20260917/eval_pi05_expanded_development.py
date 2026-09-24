#!/usr/bin/env python3
"""Parameterised expanded-development evaluator for D2 (init-state offsets 30..39).

An independent copy of `scripts/eval_pi05_iteration_development.py`.  That wrapper
hard-codes offset 30 in its receipt and derives its per-task seed without an offset
term, and it rejects `n_episodes != 1`, so offsets are scheduled one call at a time
here rather than by widening the episode count.

Differences from the original, all recorded in the run plan:

* `task_seed` gains the registered offset term
  ``391600 + 1000*(offset-30) + 100*suite_index + task_id``; at offset 30 this is
  numerically identical to the original, so offset 30 reproduces the existing screen.
* The receipt stores the offset, the *actual* `init_state_id` read back from the
  constructed environment, and how many init states that task really has.
* Video rendering is forced off (`max_episodes_rendered=0`) because 2000 episodes of
  video would not fit the current shared storage.
* The duty-cycle sleep is configurable and defaults to off; it only paces the GPU and
  touches no RNG, weights, environment or ordering.

Everything that can affect an outcome - per-episode generator reset, seeds, init-state
selection rule, action sampler, normalisation, task text, horizon - is unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import time

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
BASE_OFFSET = 30


def task_seed(suite, task_id, offset):
    if not 0 <= task_id < 10:
        raise ValueError("Unexpected task id")
    if suite not in SUITES:
        raise ValueError(f"Unexpected suite {suite}")
    return 391600 + 1000 * (offset - BASE_OFFSET) + 100 * SUITES.index(suite) + task_id


def main():
    import torch
    import eval_with_local_tokenizer  # noqa: F401  install existing runtime compatibility
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from lerobot.scripts import lerobot_eval
    from lerobot.utils.random_utils import set_seed
    from eval_pi05_libero_with_init_offset import main as evaluate
    from run_pi05_generation_path_probe import memory_free, rest_seconds

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    gpu = int(os.environ["ITERATION_PHYSICAL_GPU"])
    offset = int(os.environ["PI05_LIBERO_INIT_STATE_OFFSET"])
    duty = float(os.environ.get("CLAUDE_EVAL_DUTY", "0"))
    output = Path(os.environ["ITERATION_NOISE_RECEIPT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("PI05_LIBERO_INIT_STATE_COUNT", "1")) != 1:
        raise ValueError("This protocol takes exactly one init state per call")

    def actual_init_state(env):
        """Read the init state off the constructed environment for this task.

        Deliberately not a constructor hook: `install_init_state_offset` wraps
        `LiberoEnv.__init__` too, and a hook installed here would observe the value
        before that wrapper adds the offset.
        """
        stack, seen = [env], set()
        while stack:
            obj = stack.pop()
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            states = getattr(obj, "_init_states", None)
            if hasattr(obj, "init_state_id") and states is not None:
                return int(obj.init_state_id), int(len(states))
            for attribute in ("envs", "env", "unwrapped"):
                child = getattr(obj, attribute, None)
                if isinstance(child, (list, tuple)):
                    stack.extend(child)
                elif child is not None and child is not obj:
                    stack.append(child)
        return None, None

    current = {"key": None, "generator": None, "noise_hashes": []}
    native_run_one = lerobot_eval.run_one
    native_noise = PI05Pytorch.sample_noise
    native_sample = PI05Pytorch.sample_actions

    def run_one(task_group, task_id, env, **kwargs):
        if kwargs["n_episodes"] != 1:
            raise ValueError("This paired development protocol is exactly one episode/task")
        seed = task_seed(task_group, task_id, offset)
        set_seed(seed)
        kwargs["start_seed"] = seed
        kwargs["max_episodes_rendered"] = 0
        kwargs["videos_dir"] = None
        current.update(key=f"{task_group}/{task_id}/off{offset}",
                       generator=torch.Generator(device="cuda").manual_seed(seed + 100000),
                       noise_hashes=[])
        # Read before the episode: the environment advances init_state_id after each reset.
        state_id, available = actual_init_state(env)
        result = native_run_one(task_group, task_id, env, **kwargs)
        state_id_after, _ = actual_init_state(env)
        receipt = {"key": current["key"], "suite": task_group, "task_id": task_id,
                   "offset": offset, "environment_seed": seed, "flow_seed": seed + 100000,
                   "init_state_offset": offset,
                   "actual_init_state_id": state_id,
                   "init_state_id_after_episode": state_id_after,
                   "available_init_states": available,
                   "noise_hashes": current["noise_hashes"],
                   "successes": result[2]["successes"]}
        if receipt["actual_init_state_id"] != offset:
            raise ValueError(
                f"Environment used init_state_id={receipt['actual_init_state_id']} "
                f"for the requested offset {offset}")
        if (receipt["available_init_states"] or 0) < 40:
            raise ValueError(
                f"{task_group}/{task_id} has only {receipt['available_init_states']} init states")

        with output.open("a") as stream:
            stream.write(json.dumps(receipt) + "\n")
        return result

    def sample_noise(self, shape, device):
        if current["generator"] is None:
            return native_noise(self, shape, device)
        value = torch.normal(mean=0., std=1., size=shape, dtype=torch.float32,
                             device=device, generator=current["generator"])
        current["noise_hashes"].append(hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest())
        return value

    def sample_actions(self, *args, **kwargs):
        if memory_free(gpu) < 12 * 1024:
            raise RuntimeError("Shared GPU reserve low; preserve incomplete run, no process termination")
        begin = time.monotonic()
        result = native_sample(self, *args, **kwargs)
        if duty > 0:
            torch.cuda.synchronize()
            time.sleep(rest_seconds(time.monotonic() - begin, duty))
        return result

    lerobot_eval.run_one = run_one
    PI05Pytorch.sample_noise = sample_noise
    PI05Pytorch.sample_actions = sample_actions
    evaluate()


if __name__ == "__main__":
    main()
