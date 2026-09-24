#!/usr/bin/env python3
"""Derive the four-perturbation PRO Average from Soup's terminal audit package."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

WORK = Path(__file__).resolve().parents[3]
RUN = WORK / "vla-merge-runtime/experiments/pi05-pro-soup-codex-20260924/formal-v1"
PLAN_SHA = "6ba9c3450d4f09a15eda940ea6c8398445dad3ef08a0606c58160a2ff426b7a0"
MODEL_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
REPEATS = ("repeat-01", "repeat-02", "repeat-03")
DIMENSIONS = ("swap", "object", "semantic", "task")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def preflight() -> None:
    if sha(RUN / "plan.json") != PLAN_SHA:
        raise ValueError("Frozen Soup formal plan changed")
    plan = read(RUN / "plan.json")
    if (plan.get("model_sha256") != MODEL_SHA or plan.get("jobs_expected") != 480
            or plan.get("episodes_expected") != 4800
            or Counter((j["repeat"], j["dimension"], j["suite"]) for j in plan["jobs"])
            != Counter({(r, d, s): 10 for r in REPEATS for d in DIMENSIONS for s in SUITES})):
        raise ValueError("Soup plan lacks 3 × 4 × 4 × 10 task coverage")


def derive() -> dict:
    preflight()
    source = RUN / "independent-table-data-v1.json"
    package = read(source)
    terminal = RUN / "queue-ended.json"
    ended = read(terminal)
    if (not package.get("accepted") or package.get("jobs") != 480
            or package.get("episodes") != 4800
            or package.get("formal_plan_sha256") != PLAN_SHA
            or package.get("model_sha256") != MODEL_SHA
            or package.get("formal_terminal_sha256") != sha(terminal)
            or ended.get("complete") is not True or ended.get("stopped") is not False):
        raise ValueError("Soup lacks clean terminal independent audit")
    cells = package["by_repeat_suite_dimension"]
    if set(cells) != {f"{r}/{d}/{s}" for r in REPEATS for d in DIMENSIONS for s in SUITES}:
        raise ValueError("Soup audited suite coverage differs")
    repeat_dimensions = {}
    repeat_average = {}
    for repeat in REPEATS:
        values = {}
        for dimension in DIMENSIONS:
            suite_cells = [cells[f"{repeat}/{dimension}/{suite}"] for suite in SUITES]
            if any(cell.get("tasks") != 10 or cell.get("episodes") != 100
                   or type(cell.get("successes")) is not int
                   or not 0 <= cell["successes"] <= 100 for cell in suite_cells):
                raise ValueError("Audited suite cell has invalid denominator")
            percent = sum(cell["successes"] for cell in suite_cells) / 4.0
            if abs(percent - package["dimensions"][dimension]["repeat_success_percent"][REPEATS.index(repeat)]) > 1e-9:
                raise ValueError("Raw suite counts differ from dimension audit")
            values[dimension] = percent
        repeat_dimensions[repeat] = values
        repeat_average[repeat] = mean(values.values())
    averages = [repeat_average[repeat] for repeat in REPEATS]
    return {"schema": "pi05_pro_soup_four_perturbation_average_v1", "accepted": True,
            "model_sha256": MODEL_SHA, "formal_plan_sha256": PLAN_SHA,
            "independent_source_sha256": sha(source),
            "four_dimensions": list(DIMENSIONS), "repeats": list(REPEATS),
            "repeat_dimension_percent": repeat_dimensions,
            "repeat_average_percent": repeat_average,
            "mean_percent": mean(averages), "sample_std_percent": stdev(averages),
            "rule": "Within each repeat, average Pos./Obj./Lang./Task equally; then use the three repeat averages for mean and sample SD."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    args = parser.parse_args()
    if args.action == "preflight":
        preflight()
        print(json.dumps({"valid": True, "jobs": 480, "episodes": 4800}))
    else:
        data = derive()
        output = RUN / "four-perturbation-average-v1.json"
        with output.open("x") as stream:
            stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"path": str(output), "sha256": sha(output), "accepted": True}))


if __name__ == "__main__":
    main()
