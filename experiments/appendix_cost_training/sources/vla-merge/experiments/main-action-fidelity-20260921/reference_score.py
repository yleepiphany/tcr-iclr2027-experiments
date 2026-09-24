#!/usr/bin/env python3
"""Independent raw-tensor recomputation of C146 MSE/cos task macros."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/candidate-actions-attempt-01"
BANK = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02/bank-index.json"
TEACHERS = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/teacher-actions-attempt-06/teachers"
PRIMARY = RUN / "strict-audit-and-metrics.json"
OUTPUT = RUN / "independent-reference-metrics.json"


def main() -> None:
    plan = json.loads((RUN / "plan.json").read_text())
    bank = json.loads(BANK.read_text())
    primary = json.loads(PRIMARY.read_text())
    requests = sorted(bank["requests"], key=lambda row: row["request_id"])
    teachers = {}
    for suite in ("spatial", "object", "goal", "long"):
        manifest = json.loads((TEACHERS / suite / "manifest.json").read_text())
        teachers[suite] = safe_open(manifest["actions_file"], framework="pt")
    results, comparisons = {}, {}
    for model_name in plan["models"]:
        manifest = json.loads((RUN / "actions" / model_name / "manifest.json").read_text())
        candidate = safe_open(manifest["actions_file"], framework="pt")
        grouped = {}
        valid = 0
        for request in requests:
            prediction = candidate.get_tensor(request["request_id"])[0, :10, :6].to(torch.float64)
            teacher = teachers[request["suite_short"]].get_tensor(request["request_id"])[0, :10, :6].to(torch.float64)
            difference = prediction - teacher
            mse = float(torch.sum(difference * difference)) / 60.0
            pn, tn = float(torch.linalg.vector_norm(prediction)), float(torch.linalg.vector_norm(teacher))
            cosine = None
            if pn > 1e-12 and tn > 1e-12:
                cosine = max(-1.0, min(1.0, float(torch.sum(prediction * teacher)) / (pn * tn)))
                valid += 1
            grouped.setdefault((request["suite"], request["task_id"]), []).append((mse, cosine))
        if len(grouped) != 40 or any(len(values) != 10 for values in grouped.values()):
            raise ValueError(f"{model_name}: reference grouping differs")
        task_mse, task_cos = [], []
        for values in grouped.values():
            task_mse.append(sum(value[0] for value in values) / 10.0)
            cosines = [value[1] for value in values if value[1] is not None]
            if len(cosines) != 10:
                raise ValueError(f"{model_name}: incomplete cosine task")
            task_cos.append(sum(cosines) / 10.0)
        result = {"mse": sum(task_mse) / 40.0, "cosine": sum(task_cos) / 40.0,
                  "requests": 400, "tasks": 40, "cosine_valid_requests": valid}
        expected = primary["results"][model_name]["aggregate"]
        differences = {key: abs(result[key] - expected[key]) for key in ("mse", "cosine")}
        if any(not math.isfinite(value) or value > 1e-12 for value in differences.values()):
            raise ValueError(f"{model_name}: independent metric disagreement: {differences}")
        results[model_name] = result
        comparisons[model_name] = differences
    grouped_results = {}
    for label, names in {
        "full": ["tcr_r01", "tcr_r02", "tcr_r03"],
        "demo_observations": ["demo_r01", "demo_r02", "demo_r03"],
    }.items():
        mses = [results[name]["mse"] for name in names]
        cosines = [results[name]["cosine"] for name in names]
        grouped_results[label] = {"builds": names, "mse_mean": statistics.mean(mses),
            "mse_sample_sd": statistics.stdev(mses), "cosine_mean": statistics.mean(cosines),
            "cosine_sample_sd": statistics.stdev(cosines)}
    output = {"schema": "main_fidelity_independent_reference_v1", "accepted": True,
              "implementation": "direct float64 torch formula; does not import metrics.py",
              "max_abs_disagreement": max(value for row in comparisons.values() for value in row.values()),
              "results": results, "comparison_to_primary": comparisons, "build_aggregates": grouped_results}
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"accepted": True, "max_abs_disagreement": output["max_abs_disagreement"],
                      "build_aggregates": grouped_results}, indent=2))


if __name__ == "__main__":
    main()
