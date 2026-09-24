#!/usr/bin/env python3
"""Freeze outcome-blind Fast-WAM A/B calibration resets and identities."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
RUNTIME = ROOT / "vla-merge-runtime"
NAMES = ("spatial", "object", "goal", "long")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "vla-merge_table4/DreamZero/run/libero_config")
    sys.path.insert(0, str(RUNTIME / "references/LIBERO-MergeVLA"))
    import h5py
    import numpy as np
    from libero.libero import benchmark

    sources = json.loads((RUNTIME / "experiments/openvla-fastwam-diagnostics-20260923/fastwam-soup/manifest.json").read_text())["sources"]
    jobs = []
    for expert_index, (expert, suite_name) in enumerate(zip(NAMES, SUITES)):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        stats = FAST / f"weights/{expert}/dataset_stats.json"
        source = sources[expert]
        checkpoint = Path(source["path"])
        stat = checkpoint.stat()
        if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
            raise ValueError(f"Checkpoint identity changed: {expert}")
        for task in range(10):
            stock = np.asarray(suite.get_task_init_states(task))
            stock_hashes = {sha(np.ascontiguousarray(row).tobytes()) for row in stock}
            demo = ROOT / ".datasets/LIBERO/20260919/raw-demonstrations" / suite.get_task_demonstration(task)
            demo_stat = demo.stat()
            with h5py.File(demo) as handle:
                for round_name, demo_index in (("A", 0), ("B", 1)):
                    state = np.ascontiguousarray(handle["data"][f"demo_{demo_index}"].attrs["init_state"])
                    state_hash = sha(state.tobytes())
                    if state.shape != stock.shape[1:] or state_hash in stock_hashes:
                        raise ValueError(f"Calibration reset overlaps evaluation: {expert}/{task}/{round_name}")
                    jobs.append({
                        "id": f"{round_name}-{expert}-task{task:02d}", "round": round_name,
                        "expert": expert, "suite": suite_name, "task_id": task,
                        "demo_index": demo_index,
                        "seed": 20260923 + 1000 * expert_index + 10 * task + demo_index,
                        "reset_sha256": state_hash, "reset_shape": list(state.shape),
                        "reset_dtype": state.dtype.str,
                        "stock_eval_states": len(stock),
                        "stock_eval_collision": False,
                        "demo_file": str(demo), "demo_size": demo_stat.st_size,
                        "demo_mtime_ns": demo_stat.st_mtime_ns,
                        "checkpoint": source,
                        "stats_path": str(stats), "stats_sha256": sha(stats.read_bytes()),
                    })
    if len(jobs) != 80 or len({row["reset_sha256"] for row in jobs}) != 80:
        raise ValueError("A/B reset bank is incomplete or contains duplicates")
    source_code = {}
    for name, path in {
        "native_policy": FAST / "source/src/fastwam/models/wan22/fastwam.py",
        "native_mot": FAST / "source/src/fastwam/models/wan22/mot.py",
        "native_evaluator": FAST / "source/experiments/libero/eval_libero_single.py",
    }.items():
        source_code[name] = {"path": str(path), "sha256": sha(path.read_bytes())}
    plan = {"schema": "fastwam_four_expert_ab_calibration_bank_v1",
            "formal_evaluation": False, "outcome_blind": True,
            "selection": "one full closed-loop expert episode per task and round; demo_0=A/demo_1=B; all episodes regardless of success; first/mid/last request; actual native denoise calls 0/5/9",
            "jobs": jobs, "job_count": len(jobs), "source_code": source_code}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    print(json.dumps({"plan": str(output), "job_count": len(jobs),
                      "A": sum(row["round"] == "A" for row in jobs),
                      "B": sum(row["round"] == "B" for row in jobs)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
