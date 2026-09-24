#!/usr/bin/env python3
"""Native evaluator with bounded memory/duty and per-task paired noise.

Only a 40-episode development screen, not independent confirmation. All arms
use this same wrapper; no weights, actions, or environment dynamics are edited.
"""
import hashlib
import json
import os
from pathlib import Path
import time


def task_seed(suite, task_id):
    names = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    if not 0 <= task_id < 10:
        raise ValueError("Unexpected task id")
    return 391600 + names.index(suite) * 100 + task_id


def main():
    import torch
    import eval_with_local_tokenizer  # install existing runtime compatibility
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from lerobot.scripts import lerobot_eval
    from lerobot.utils.random_utils import set_seed
    from eval_pi05_libero_with_init_offset import main as evaluate
    from run_pi05_generation_path_probe import memory_free, rest_seconds

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    gpu = int(os.environ["ITERATION_PHYSICAL_GPU"])
    output = Path(os.environ["ITERATION_NOISE_RECEIPT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    current = {"key": None, "generator": None, "noise_hashes": []}
    native_run_one = lerobot_eval.run_one
    native_noise = PI05Pytorch.sample_noise
    native_sample = PI05Pytorch.sample_actions

    def run_one(task_group, task_id, env, **kwargs):
        if kwargs["n_episodes"] != 1:
            raise ValueError("This paired development protocol is exactly one episode/task")
        seed = task_seed(task_group, task_id)
        set_seed(seed)
        kwargs["start_seed"] = seed
        current.update(key=f"{task_group}/{task_id}",
                       generator=torch.Generator(device="cuda").manual_seed(seed + 100000), noise_hashes=[])
        result = native_run_one(task_group, task_id, env, **kwargs)
        receipt = {"key": current["key"], "environment_seed": seed, "flow_seed": seed+100000,
                   "init_state_offset": 30, "noise_hashes": current["noise_hashes"],
                   "successes": result[2]["successes"]}
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
        torch.cuda.synchronize()
        time.sleep(rest_seconds(time.monotonic()-begin, .25))
        return result

    lerobot_eval.run_one = run_one
    PI05Pytorch.sample_noise = sample_noise
    PI05Pytorch.sample_actions = sample_actions
    evaluate()


if __name__ == "__main__":
    main()
