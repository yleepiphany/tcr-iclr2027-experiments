#!/usr/bin/env python3
"""Strictly audit the corrected RegMean++ arm and emit the current-paper package."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921"
RUN = ROOT / "regmeanpp-correction-attempt-01"
OLD_RUN = ROOT / "candidate-actions-attempt-01"
BANK = ROOT / "heldout-raw-attempt-02/bank-index.json"
TEACHERS = ROOT / "teacher-actions-attempt-06/teachers"
MODEL_SHA = "49ea499c3ad37329b95baf597b252bc424cef40da312edd943a5102fabfe645a"
OLD_SHA = "38745ba812c2f4ac4327ab40719e0ce8da017873c0a1765608f65dd926cd4ab3"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_module("regmeanpp_correction_metrics", HERE / "metrics.py")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def write(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    plan = json.loads((RUN / "plan.json").read_text())
    terminal = json.loads((RUN / "queue-ended.json").read_text())
    manifest_path = RUN / "actions/regmean_pp/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    bank = json.loads(BANK.read_text())
    identity = plan["models"]["regmean_pp"]
    if terminal.get("complete") is not True or terminal.get("status") != "complete":
        raise ValueError("Correction run is not terminal-success")
    if identity.get("model_sha256") != MODEL_SHA or manifest.get("model_sha256") != MODEL_SHA:
        raise ValueError("Correction is not bound to current paper RegMean++")
    if plan.get("supersedes_model_sha256") != OLD_SHA:
        raise ValueError("Historical checkpoint exclusion is missing")
    if plan.get("bank_sha256") != sha(BANK) or manifest.get("bank_sha256") != sha(BANK):
        raise ValueError("Frozen bank identity differs")
    requests = {row["request_id"]: row for row in bank["requests"]}
    if len(requests) != 400 or manifest.get("requests") != 400:
        raise ValueError("Expected exact 400-request denominator")
    receipts = {row["request_id"]: row for row in manifest["rows"]}
    if set(receipts) != set(requests) or len(receipts) != 400:
        raise ValueError("Request receipt partition differs")
    action_path = Path(manifest["actions_file"])
    if sha(action_path) != manifest.get("actions_sha256"):
        raise ValueError("Candidate action file hash differs")
    teachers, teacher_hashes = {}, {}
    for suite in ("spatial", "object", "goal", "long"):
        teacher_manifest = json.loads((TEACHERS / suite / "manifest.json").read_text())
        teacher_path = Path(teacher_manifest["actions_file"])
        if sha(teacher_path) != teacher_manifest["actions_sha256"]:
            raise ValueError(f"{suite}: teacher action file changed")
        teachers[suite] = safe_open(str(teacher_path), framework="pt")
        teacher_hashes[suite] = {row["request_id"]: row["action_sha256"] for row in teacher_manifest["rows"]}

    rows, direct = [], {}
    with safe_open(str(action_path), framework="pt") as candidates:
        if set(candidates.keys()) != set(requests):
            raise ValueError("Candidate action tensor keys differ")
        for request_id in sorted(requests):
            request, receipt = requests[request_id], receipts[request_id]
            if receipt.get("raw_input_sha256") != request["raw_input_sha256"]:
                raise ValueError(f"{request_id}: raw input identity differs")
            if receipt.get("native_noise_sha256") != request["native_noise_sha256"]:
                raise ValueError(f"{request_id}: noise identity differs")
            candidate = candidates.get_tensor(request_id)
            teacher = teachers[request["suite_short"]].get_tensor(request_id)
            if tensor_hash(candidate) != receipt.get("action_sha256"):
                raise ValueError(f"{request_id}: candidate content differs")
            if tensor_hash(teacher) != teacher_hashes[request["suite_short"]][request_id]:
                raise ValueError(f"{request_id}: teacher content differs")
            row = metrics.continuous_metrics(candidate, teacher, coordinate_scale=plan["coordinate_scale"],
                continuous_dims=tuple(plan["continuous_dims"]), execute_steps=plan["execute_steps"],
                cosine_eps=plan["cosine_eps"])
            row.update({"suite": request["suite"], "task_id": request["task_id"],
                        "episode_id": request["episode_id"], "request_id": request_id})
            rows.append(row)
            p = candidate[0, :10, :6].to(torch.float64)
            t = teacher[0, :10, :6].to(torch.float64)
            mse = float(torch.sum((p - t) ** 2)) / 60.0
            pn, tn = float(torch.linalg.vector_norm(p)), float(torch.linalg.vector_norm(t))
            cosine = max(-1.0, min(1.0, float(torch.sum(p * t)) / (pn * tn))) if pn > 1e-12 and tn > 1e-12 else None
            direct.setdefault((request["suite"], request["task_id"]), []).append((mse, cosine))
    expected_tasks = sorted(direct)
    if len(expected_tasks) != 40 or any(len(values) != 10 for values in direct.values()):
        raise ValueError("Task/request partition differs")
    primary = metrics.task_macro(rows, expected_tasks=expected_tasks, requests_per_task=10)
    task_mse = [sum(v[0] for v in values) / 10 for values in direct.values()]
    task_cos = [sum(v[1] for v in values if v[1] is not None) / 10 for values in direct.values()]
    if len(task_cos) != 40 or any(sum(v[1] is not None for v in values) != 10 for values in direct.values()):
        raise ValueError("Cosine denominator differs")
    reference = {"mse": sum(task_mse) / 40, "cosine": sum(task_cos) / 40,
                 "requests": 400, "tasks": 40, "cosine_valid_requests": 400}
    disagreement = {key: abs(primary[key] - reference[key]) for key in ("mse", "cosine")}
    if any(not math.isfinite(value) or value > 1e-12 for value in disagreement.values()):
        raise ValueError(f"Independent formula disagreement: {disagreement}")
    correction = {"schema": "main_fidelity_regmeanpp_correction_audit_v1", "accepted": True,
        "model_sha256": MODEL_SHA, "superseded_model_sha256": OLD_SHA,
        "plan_sha256": sha(RUN / "plan.json"), "terminal_sha256": sha(RUN / "queue-ended.json"),
        "manifest_sha256": sha(manifest_path), "actions_sha256": sha(action_path),
        "bank_sha256": sha(BANK), "requests": 400, "tasks": 40,
        "primary": primary, "independent_reference": reference,
        "absolute_disagreement": disagreement}
    write(RUN / "strict-audit-and-metrics.json", correction)

    original = json.loads((OLD_RUN / "strict-audit-and-metrics.json").read_text())
    original_reference = json.loads((OLD_RUN / "independent-reference-metrics.json").read_text())
    if original.get("accepted") is not True or original_reference.get("accepted") is not True:
        raise ValueError("Original thirteen-arm package is not accepted")
    results = {name: value for name, value in original["results"].items() if name != "regmean_pp"}
    if len(results) != 13:
        raise ValueError("Expected thirteen retained original models")
    results["regmean_pp"] = {"model_sha256": MODEL_SHA, "group": "main",
                              "aggregate": primary, "rows": rows}
    package = {"schema": "main_fidelity_current_paper_metrics_v2", "accepted": True,
        "models": 14, "requests_per_model": 400,
        "retained_original_models": sorted(name for name in results if name != "regmean_pp"),
        "corrected_model": "regmean_pp", "corrected_model_sha256": MODEL_SHA,
        "excluded_historical_model_sha256": OLD_SHA,
        "old_outputs_preserved": True,
        "original_audit_sha256": sha(OLD_RUN / "strict-audit-and-metrics.json"),
        "correction_audit_sha256": sha(RUN / "strict-audit-and-metrics.json"),
        "results": results}
    write(ROOT / "current-paper-action-metrics-v2.json", package)
    print(json.dumps({"accepted": True, "model_sha256": MODEL_SHA,
                      "mse": primary["mse"], "cosine": primary["cosine"],
                      "max_abs_disagreement": max(disagreement.values()),
                      "current_paper_models": len(results)}, indent=2))


if __name__ == "__main__":
    main()
