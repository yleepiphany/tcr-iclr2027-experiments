#!/usr/bin/env python3
"""Original formal evaluator with resource accounting only; no RNG replacement."""
import json
import os
from pathlib import Path
import sys
import time


def bounded_sample(native, before, after):
    def wrapped(*args, **kwargs):
        before()
        start = time.monotonic()
        result = native(*args, **kwargs)
        after(time.monotonic() - start)
        return result
    return wrapped


def main():
    started = time.monotonic()
    output = Path(next(x.split("=", 1)[1] for x in sys.argv if x.startswith("--output_dir=")))
    import torch
    import eval_with_local_tokenizer  # existing compatibility, same as formal entry
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
    from eval_pi05_policy_with_procedural_bank import main as evaluate
    from run_pi05_generation_path_probe import memory_free, rest_seconds
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    gpu = int(os.environ["ITERATION_PHYSICAL_GPU"])
    calls, completed = 0, False

    def before():
        if memory_free(gpu) < 12 * 1024:
            raise RuntimeError("GPU reserve low; no process will be terminated")

    def after(elapsed):
        nonlocal calls
        calls += 1
        torch.cuda.synchronize()
        time.sleep(rest_seconds(elapsed, .25))

    PI05Pytorch.sample_actions = bounded_sample(PI05Pytorch.sample_actions, before, after)
    try:
        evaluate()
        completed = True
    finally:
        if output.exists():
            with (output / "bounded_resources.json").open("x") as stream:
                json.dump({"evaluation_completed": completed, "native_calls": calls,
                    "peak_allocator_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                    "wall_seconds_including_imports": time.monotonic() - started,
                    "allocator_fraction_cap": .20, "duty_fraction": .25,
                    "changes_policy_rng": False, "changes_weights": False}, stream, indent=2)


if __name__ == "__main__":
    main()
