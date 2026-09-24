"""Isolated nightly runtime adapter; original scientific entrypoints stay unchanged."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

WORK = Path(__file__).resolve().parents[2]
SCRIPTS = WORK / "vla-merge/scripts"
DATA = WORK / ".datasets/LIBERO/20260919"


def stop_group(signum, frame):
    signal.signal(signum, signal.SIG_DFL)
    if os.getpgrp() == os.getpid():
        os.killpg(os.getpgrp(), signum)
    else:
        raise SystemExit(128 + signum)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=int, required=True)
    p.add_argument("--mode", choices=("formal", "development", "collect", "solve", "uniform"), required=True)
    p.add_argument("--resources", type=Path, required=True)
    args, remaining = p.parse_known_args()
    signal.signal(signal.SIGTERM, stop_group)
    signal.signal(signal.SIGINT, stop_group)
    # Linux parent-death signal closes this entire private worker process group.
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError("Cannot install parent-death guard")
    if os.getppid() != args.parent:
        raise RuntimeError("Supervisor already exited; do not start work")
    sys.path.insert(0, str(SCRIPTS))
    os.environ["LIBERO_CONFIG_PATH"] = str(DATA / "config-standard")
    for key in ("LIBERO_PRO_REPO", "LIBERO_PRO_ASSET_DIR", "MUJOCO_EGL_DEVICE_ID"):
        os.environ.pop(key, None)
    import libero.libero as inner
    inner._assets_path_cache = str(DATA / "runtime/assets")
    import torch
    torch.set_num_threads(2)
    if args.mode != "development":
        torch.set_num_interop_threads(1)
    cap = .70 if args.mode in ("solve", "uniform") else .30
    torch.cuda.set_per_process_memory_fraction(cap, 0)
    from run_pi05_generation_path_probe import memory_free
    gpu = int(os.environ["ITERATION_PHYSICAL_GPU"])
    begin = time.monotonic()
    completed = False
    calls = 0
    target = {
        "formal": SCRIPTS / "eval_pi05_policy_with_procedural_bank.py",
        "development": WORK / "vla-merge/experiments/claude-20260917/eval_pi05_expanded_development.py",
        "collect": SCRIPTS / "collect_pi05_tcr_e_dense_trace_v2.py",
        "solve": SCRIPTS / "materialize_pi05_tcr_e_peft_safe.py",
        "uniform": SCRIPTS / "materialize_tcr_night_uniform_20260919.py",
    }[args.mode]
    try:
        sys.argv = [str(target), *remaining]
        if args.mode in ("solve", "uniform"):
            import importlib
            solver = importlib.import_module(target.stem)
            native = solver.solve_weight_multi
            def guarded(*a, **kw):
                nonlocal calls
                if memory_free(gpu) < 12 * 1024:
                    raise RuntimeError("GPU reserve below 12 GiB")
                result = native(*a, **kw)
                calls += 1
                return result
            solver.solve_weight_multi = guarded
            solver.main()
        elif args.mode == "formal":
            import eval_with_local_tokenizer
            from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch
            native = PI05Pytorch.sample_actions
            def guarded(self, *a, **kw):
                nonlocal calls
                if memory_free(gpu) < 12 * 1024:
                    raise RuntimeError("GPU reserve below 12 GiB")
                calls += 1
                return native(self, *a, **kw)
            PI05Pytorch.sample_actions = guarded
            runpy.run_path(str(target), run_name="__main__")
        else:
            runpy.run_path(str(target), run_name="__main__")
        completed = True
    finally:
        args.resources.parent.mkdir(parents=True, exist_ok=True)
        with args.resources.open("x") as f:
            json.dump({"completed": completed, "mode": args.mode, "calls": calls,
                       "wall_seconds": time.monotonic() - begin,
                       "peak_allocator_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "allocator_fraction": cap, "duty_sleep": 0,
                       "changes_policy_rng": False, "parent_death_guard": True,
                       "runtime_config": os.environ["LIBERO_CONFIG_PATH"]}, f, indent=2)


if __name__ == "__main__":
    main()
