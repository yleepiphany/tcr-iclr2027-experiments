#!/usr/bin/env python3
"""Exact native-bf16 proprio fc2-input mismatch over frozen A/B requests.

No GPU, rollout, model export, success filtering, or change to the R2 runner.
The first row is independently checked against full-7B native replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("Set CUDA_VISIBLE_DEVICES='' for this CPU-only diagnostic")

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
OLD = WORK / "vla-merge-runtime/experiments/openvla-tcr-20260921"
REPAIR = WORK / "vla-merge-runtime/experiments/claude-openvla-tcr-repair-20260922"
CAPTURE = OLD / "expert-ab-capture-attempt-01"
LEDGER = OLD / "local-expert-dynamic-01/identities.json"
ACCEPTANCE = WORK / "coordination/2026-09-21/openvla-ab-capture-acceptance.json"
FINAL = REPAIR / "candidate-build-attempt-02/jobs/R2/build/final-manifest.json"
ANCHOR = WORK / "vla-merge-runtime/experiments/openvla-path-diagnosis-20260923/spatial-proprio_projector.json"
OUT = WORK / "vla-merge-runtime/experiments/openvla-path-diagnosis-20260923/proprio-all-A-B.json"
SUITES = {"spatial": "libero_spatial", "object": "libero_object",
          "goal": "libero_goal", "long": "libero_10"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projector(path):
    files = list(Path(path).glob("proprio_projector--*_checkpoint.pt"))
    if len(files) != 1:
        raise ValueError("Exactly one proprio checkpoint required")
    state = torch.load(str(files[0]), map_location="cpu", weights_only=True, mmap=True)
    state = {key.removeprefix("module."): value.to(torch.bfloat16)
             for key, value in state.items()}
    if set(state) != {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}:
        raise ValueError("Native proprio state scope differs")
    if list(state["fc1.weight"].shape) != [4096, 8]:
        raise ValueError("Native proprio input width differs")
    return state, {"path": str(files[0]), "sha256": sha(files[0])}


def metrics(left, right):
    a, b = left.float().double().flatten(), right.float().double().flatten()
    mse = float((a - b).square().mean() / max(float(b.square().mean()), 1e-12))
    cosine = float(torch.dot(a, b) / max(float(a.norm() * b.norm()), 1e-12))
    if not math.isfinite(mse) or not math.isfinite(cosine):
        raise ValueError("Nonfinite projector path metric")
    return {"relative_mse": mse, "cosine": cosine}


def summarize(rows):
    values = sorted(row["fc2_input_solve_vs_live"]["relative_mse"] for row in rows)
    cos = sorted(row["fc2_input_solve_vs_live"]["cosine"] for row in rows)
    if not values:
        raise ValueError("Empty projector path group")
    return {"count": len(rows), "relative_mse_mean": sum(values) / len(values),
            "relative_mse_median": (values[(len(values)-1)//2] + values[len(values)//2]) / 2,
            "relative_mse_min": values[0], "relative_mse_max": values[-1],
            "cosine_mean": sum(cos) / len(cos), "cosine_min": cos[0],
            "cosine_max": cos[-1]}


def main():
    if torch.cuda.is_available():
        raise RuntimeError("CUDA unexpectedly available")
    torch.set_num_threads(2)
    if json.loads(ACCEPTANCE.read_text()).get("accepted") is not True:
        raise ValueError("A/B native capture was not independently accepted")
    final = json.loads(FINAL.read_text())
    if final.get("complete") is not True or final.get("candidate") != "R2":
        raise ValueError("Frozen R2 checkpoint differs")
    ledger = json.loads(LEDGER.read_text())
    if ledger.get("complete") is not True or set(ledger["experts"]) != set(SUITES):
        raise ValueError("Four expert identities required")
    merged, merged_file = projector(final["checkpoint"])
    rows, sources = [], {"R2": merged_file}
    request_files = {}
    with torch.no_grad():
        for expert, suite in SUITES.items():
            source, sources[expert] = projector(ledger["experts"][expert]["local_path"])
            for pool in ("A", "B"):
                folder = CAPTURE / "jobs" / f"{pool}-{expert}" / "capture"
                files = sorted(folder.glob("task-*.pt"))
                if len(files) != 10:
                    raise ValueError("Missing frozen A/B task payload")
                for task, path in enumerate(files):
                    payload = torch.load(str(path), map_location="cpu", weights_only=False)
                    if (payload.get("pool"), payload.get("suite"), payload.get("task_id")) != (pool, suite, task):
                        raise ValueError("Frozen A/B request identity differs")
                    if len(payload.get("records", [])) != 5:
                        raise ValueError("Every task requires five actual requests")
                    request_files[str(path)] = sha(path)
                    for ordinal, record in enumerate(payload["records"]):
                        if record.get("identity_max_abs") != 0.0:
                            raise ValueError("Native replay discrepancy")
                        proprio = record["inputs"].get("proprio")
                        if proprio is None or list(proprio.shape) != [8]:
                            raise ValueError("Native proprio input differs")
                        # modeling_prismatic.py uses torch.Tensor(np_array), then
                        # casts to the projected patch dtype (bf16 in this policy).
                        x = torch.as_tensor(proprio, dtype=torch.float32).to(torch.bfloat16).reshape(1, 8)
                        live = F.gelu(F.linear(x, merged["fc1.weight"], merged["fc1.bias"]))
                        solve = F.gelu(F.linear(x, source["fc1.weight"], source["fc1.bias"]))
                        live_out = F.linear(live, merged["fc2.weight"], merged["fc2.bias"])
                        solve_out = F.linear(solve, merged["fc2.weight"], merged["fc2.bias"])
                        rows.append({"pool": pool, "expert": expert, "suite": suite,
                                     "task_id": task, "ordinal": ordinal,
                                     "request_index": record["request_index"],
                                     "fc2_input_solve_vs_live": metrics(solve, live),
                                     "fc2_output_solve_vs_live": metrics(solve_out, live_out)})
    if len(rows) != 400 or len(request_files) != 80:
        raise ValueError("Frozen 4 expert × 2 pool × 50 request coverage differs")
    anchor = json.loads(ANCHOR.read_text())["linear_paths"]["proprio_projector.fc2"]["solve_vs_live"][0]
    first = next(row for row in rows if row["expert"] == "spatial" and row["pool"] == "A"
                 and row["task_id"] == 0 and row["ordinal"] == 0)
    if (abs(first["fc2_input_solve_vs_live"]["relative_mse"] - anchor["relative_mse"]) > 1e-5
            or abs(first["fc2_input_solve_vs_live"]["cosine"] - anchor["cosine"]) > 1e-5):
        raise ValueError("Lightweight projector replay differs from full-7B native hook")
    groups = {f"{expert}/{pool}": summarize([row for row in rows
              if row["expert"] == expert and row["pool"] == pool])
              for expert in SUITES for pool in ("A", "B")}
    result = {"cpu_only": True, "success_evaluation": False,
              "candidate": "R2", "requests": len(rows), "payload_files": len(request_files),
              "source_checkpoint_files": sources,
              "source_manifest_sha256": sha(FINAL), "expert_ledger_sha256": sha(LEDGER),
              "accepted_capture_sha256": sha(ACCEPTANCE),
              "full_native_anchor_sha256": sha(ANCHOR),
              "full_native_anchor_match": True,
              "request_files_sha256": request_files,
              "groups": groups, "overall": summarize(rows), "rows": rows,
              "interpretation_limit": "A/B calibration requests, not independent held-out success evidence"}
    with OUT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({"output": str(OUT), "requests": len(rows),
                      "overall": result["overall"]}), flush=True)


if __name__ == "__main__":
    main()
