#!/usr/bin/env python3
"""Run the existing development evaluator with an explicit allocator ceiling."""
import os
from pathlib import Path
import runpy
import torch

if __name__ == "__main__":
    fraction = float(os.environ.get("FEATCAL_EXEC_MEMORY_FRACTION", ".35"))
    if not 0 < fraction <= .40:
        raise ValueError("Development worker allocator cap must be <= 0.40")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    runpy.run_path(str(Path(__file__).with_name("eval_pi05_libero_with_init_offset.py")), run_name="__main__")
