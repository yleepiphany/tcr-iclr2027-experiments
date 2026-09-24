#!/usr/bin/env python3
"""Freeze disjoint native LIBERO development and one-repeat formal reset banks.

Development uses stock reset 0, matching the existing expert/Soup diagnostic.
Formal uses stock resets 1..10 (400 episodes). Calibration uses demo resets
that were independently checked against every stock reset. No outcomes are read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = ROOT / "vla-merge_table4/Fast-WAM"
RUNTIME = ROOT / "vla-merge-runtime"
BASE = RUNTIME / "experiments/claude-fastwam-tcr-20260923"
OUTPUT = BASE / "evaluation-bank-v1.json"
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def freeze() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    os.environ["LIBERO_CONFIG_PATH"] = str(FAST.parent / "DreamZero/run/libero_config")
    sys.path.insert(0, str(RUNTIME / "references/LIBERO-MergeVLA"))
    import numpy as np
    from libero.libero import benchmark

    calibration = json.loads((BASE / "calibration-bank-v1.json").read_text())
    if calibration.get("job_count") != 80:
        raise ValueError("Calibration bank is incomplete")
    calibration_hashes = {j["reset_sha256"] for j in calibration["jobs"]}
    if len(calibration_hashes) != 80:
        raise ValueError("Calibration bank repeats a reset")
    source = json.loads((BASE / "capture-queue-v1/plan.json").read_text())
    if source.get("bank_sha256") != digest(BASE / "calibration-bank-v1.json"):
        raise ValueError("Calibration source identity changed")

    jobs = []
    all_hashes = set(calibration_hashes)
    for suite_index, (expert, suite_name) in enumerate(SUITES.items()):
        suite = benchmark.get_benchmark_dict()[suite_name]()
        if suite.n_tasks != 10:
            raise ValueError(f"{suite_name}: expected ten tasks")
        stats = FAST / f"weights/{expert}/dataset_stats.json"
        for task_id in range(10):
            states = np.asarray(suite.get_task_init_states(task_id))
            if states.ndim != 2 or len(states) < 11:
                raise ValueError(f"{suite_name}/{task_id}: fewer than eleven stock resets")
            for stage, indices in (("development", (0,)),
                                   ("formal", tuple(range(1, 11)))):
                hashes = [hashlib.sha256(np.ascontiguousarray(states[i]).tobytes()).hexdigest()
                          for i in indices]
                if len(set(hashes)) != len(hashes) or set(hashes) & all_hashes:
                    raise ValueError(f"{suite_name}/{task_id}/{stage}: reset overlap")
                all_hashes.update(hashes)
                jobs.append({"id": f"{stage}-{expert}-task{task_id:02d}",
                             "stage": stage, "expert": expert,
                             "suite": suite_name, "task_id": task_id,
                             "stock_indices": list(indices), "reset_sha256": hashes,
                             "seed": 371100 + suite_index * 10 + task_id,
                             "stats_path": str(stats), "stats_sha256": digest(stats)})
    assert len(jobs) == 80 and len(all_hashes) == 520
    plan = {"schema": "fastwam_tcr_eval_bank_v1", "calibration_bank_sha256":
            digest(BASE / "calibration-bank-v1.json"), "source_capture_plan_sha256":
            digest(BASE / "capture-queue-v1/plan.json"),
            "development_episodes": 40, "formal_episodes": 400,
            "formal_repeat_count": 1, "formal_stock_indices": list(range(1, 11)),
            "development_stock_indices": [0], "success_blind_selection": True,
            "formal_evaluation_not_started": True, "jobs": jobs}
    OUTPUT.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"path": str(OUTPUT), "sha256": digest(OUTPUT),
            "jobs": len(jobs), "development_episodes": 40,
            "formal_episodes": 400}


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
