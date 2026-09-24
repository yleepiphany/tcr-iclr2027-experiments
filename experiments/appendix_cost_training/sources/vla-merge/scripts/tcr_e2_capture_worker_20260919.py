"""Isolated collection worker; no model correction or success-rate evaluation."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

from tcr_night_worker_20260919 import stop_group, WORK, SCRIPTS, DATA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--mode", choices=["collect"], required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    signal.signal(signal.SIGTERM, stop_group)
    signal.signal(signal.SIGINT, stop_group)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != args.parent:
        raise RuntimeError("Parent-death guard or parent identity failed")
    import libero.libero as inner
    inner._assets_path_cache = str(DATA / "runtime/assets")
    import torch
    target = SCRIPTS / "collect_tcr_10k_occupancy.py"
    sys.argv = [str(target), *remaining]
    start, complete = time.monotonic(), False
    try:
        # The original collector owns thread settings, noise seed, allocator .20,
        # reserve checks and fixed duty .25. Do not override its scientific path.
        runpy.run_path(str(target), run_name="__main__")
        complete = True
    finally:
        with args.resources.open("x") as stream:
            json.dump(dict(completed=complete, mode="collect", parent_death_guard=True,
                           runtime_config=os.environ["LIBERO_CONFIG_PATH"],
                           wall_seconds=time.monotonic()-start,
                           peak_allocator_gib=torch.cuda.max_memory_allocated()/2**30,
                           allocator_fraction=.20, duty_fraction=.25,
                           generation_noise_seed=272001, collection_only=True), stream, indent=2)


if __name__ == "__main__":
    main()
