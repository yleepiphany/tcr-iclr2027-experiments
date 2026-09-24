#!/usr/bin/env python3
"""Run the original RoboTwin solver and record this process's CUDA peak use."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1]
          / "claude-robotwin-tcr-20260923/run_materializer.py")


def main() -> None:
    spec = importlib.util.spec_from_file_location("robotwin_original_materializer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Original RoboTwin materializer is unavailable")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    completed = False
    try:
        original.main()
        completed = True
    finally:
        import torch
        report = {
            "schema": "robotwin_local_cuda_peak_v1",
            "original_materializer": str(SOURCE),
            "solver_completed": completed,
            "cuda_available": torch.cuda.is_available(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        if report["cuda_available"]:
            report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            report["exit_free_mib"] = free_bytes / 1024**2
            report["device_total_mib"] = total_bytes / 1024**2
        target = os.environ.get("ROBOTWIN_TCR_PEAK_PATH")
        if target:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)


if __name__ == "__main__":
    main()
