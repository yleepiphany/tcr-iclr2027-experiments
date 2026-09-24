#!/usr/bin/env python3
"""Strictly audit C146 candidate actions and compute registered action metrics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
RUN = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/candidate-actions-attempt-01"
BANK = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/heldout-raw-attempt-02/bank-index.json"
TEACHERS = WORK / "vla-merge-runtime/experiments/main-action-fidelity-20260921/teacher-actions-attempt-06/teachers"
OUTPUT = RUN / "strict-audit-and-metrics.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = load_module("c146_metrics", HERE / "metrics.py")


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


def main() -> None:
    plan = json.loads((RUN / "plan.json").read_text())
    terminal = json.loads((RUN / "queue-ended.json").read_text())
    bank = json.loads(BANK.read_text())
    if terminal.get("complete") is not True or terminal.get("status") != "complete":
        raise ValueError("Candidate queue is not complete")
    if plan.get("bank_sha256") != sha(BANK) or bank.get("request_count") != 400:
        raise ValueError("Candidate plan is not bound to the frozen bank")
    requests = {row["request_id"]: row for row in bank["requests"]}
    if len(requests) != 400:
        raise ValueError("Request IDs are not unique")
    expected_tasks = sorted({(row["suite"], row["task_id"]) for row in requests.values()})
    if len(expected_tasks) != 40:
        raise ValueError("Expected forty tasks")
    teacher_handles, teacher_hashes = {}, {}
    for suite in ("spatial", "object", "goal", "long"):
        manifest_path = TEACHERS / suite / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        action_path = Path(manifest["actions_file"])
        if sha(action_path) != manifest["actions_sha256"]:
            raise ValueError(f"{suite}: teacher action file changed")
        teacher_handles[suite] = safe_open(str(action_path), framework="pt")
        teacher_hashes[suite] = {row["request_id"]: row["action_sha256"] for row in manifest["rows"]}
    summaries, manifest_hashes = {}, {}
    scale = plan["coordinate_scale"]
    for model_name, identity in plan["models"].items():
        manifest_path = RUN / "actions" / model_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        required = {"schema": "main_fidelity_candidate_action_output_v1", "model_name": model_name,
                    "model_path": identity["path"], "model_sha256": identity["model_sha256"],
                    "smoke": False, "requests": 400, "bank_sha256": sha(BANK),
                    "native_steps": 10, "determinism_first_request": True}
        for key, value in required.items():
            if manifest.get(key) != value:
                raise ValueError(f"{model_name}: {key} differs")
        action_path = Path(manifest["actions_file"])
        if sha(action_path) != manifest.get("actions_sha256"):
            raise ValueError(f"{model_name}: action file changed")
        receipts = {row["request_id"]: row for row in manifest["rows"]}
        if set(receipts) != set(requests) or len(receipts) != 400:
            raise ValueError(f"{model_name}: request receipts differ")
        rows = []
        with safe_open(str(action_path), framework="pt") as candidates:
            if set(candidates.keys()) != set(requests):
                raise ValueError(f"{model_name}: action keys differ")
            for request_id in sorted(requests):
                request, receipt = requests[request_id], receipts[request_id]
                if receipt.get("raw_input_sha256") != request["raw_input_sha256"] or receipt.get("native_noise_sha256") != request["native_noise_sha256"]:
                    raise ValueError(f"{model_name}/{request_id}: input identity differs")
                prediction = candidates.get_tensor(request_id)
                if tensor_hash(prediction) != receipt.get("action_sha256"):
                    raise ValueError(f"{model_name}/{request_id}: candidate action hash differs")
                teacher = teacher_handles[request["suite_short"]].get_tensor(request_id)
                if tensor_hash(teacher) != teacher_hashes[request["suite_short"]][request_id]:
                    raise ValueError(f"{request_id}: teacher action hash differs")
                row = metrics.continuous_metrics(prediction, teacher, coordinate_scale=scale,
                    continuous_dims=tuple(plan["continuous_dims"]), execute_steps=plan["execute_steps"],
                    cosine_eps=plan["cosine_eps"])
                row.update({"suite": request["suite"], "task_id": request["task_id"],
                            "episode_id": request["episode_id"], "request_id": request_id})
                rows.append(row)
        aggregate = metrics.task_macro(rows, expected_tasks=expected_tasks, requests_per_task=10)
        summaries[model_name] = {"model_sha256": identity["model_sha256"], "group": identity["group"],
                                 "aggregate": aggregate, "rows": rows}
        manifest_hashes[model_name] = sha(manifest_path)
    output = {"schema": "main_fidelity_strict_audit_and_metrics_v1", "accepted": True,
              "models": len(summaries), "requests_per_model": 400,
              "candidate_plan_sha256": sha(RUN / "plan.json"),
              "candidate_terminal_sha256": sha(RUN / "queue-ended.json"),
              "bank_sha256": sha(BANK), "teacher_manifests": sorted(teacher_hashes),
              "candidate_manifest_sha256": manifest_hashes,
              "metric_contract": {"continuous_dims": plan["continuous_dims"],
                  "execute_steps": plan["execute_steps"], "coordinate_scale": scale,
                  "cosine_eps": plan["cosine_eps"], "aggregation": "task macro, 40 tasks x 10 requests"},
              "results": summaries}
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"accepted": True, "models": len(summaries),
        "results": {name: value["aggregate"] | {"per_task": "omitted"} for name, value in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
