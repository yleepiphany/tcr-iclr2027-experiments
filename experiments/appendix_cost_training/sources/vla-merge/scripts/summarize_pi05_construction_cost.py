#!/usr/bin/env python3
"""Harvest construction cost from merge manifests without re-running any merge.

Only quantities that a completed run actually recorded are emitted. Realized token rows are
summed from the per-module ``<expert>_rows`` fields of ``block_regmeanpp_manifest.json``; merge
wall-clock and peak accelerator memory are reported only when the run was wrapped in
``run_with_resource_accounting`` and therefore left a ``resource_accounting.json``. Missing
quantities are emitted as null and must stay ``\\tbd{}`` in the paper rather than being inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "vla-merge-runtime/experiments/pi05-spatial-object-lora-fusion-20260830-v1"
CANDIDATES = RUNTIME / "candidates"

# Paper row label -> candidate directory under candidates/.
METHODS: dict[str, str] = {
    "Dense Mean": "multi_mean_m4_spatial_object_goal_long_dense_equivalent",
    "ASWUDI": "pi05_m4_aswudi_official_20260907_v1",
    "Dense RegMean (traj.)": "ctb_prefix_control_full_uniform_20260906_v1_expert",
    "Block RegMean++ (traj.)": "ctb_prefix_control_full_uniform_20260906_v1_merged",
    "TCR-E": "multi_block_regmeanpp_full418_m4_expertcal_k10e1_r5_f3_rows16_priornorm_reconstructed",
}


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def harvest(candidate: Path) -> dict:
    """Read one candidate checkpoint's recorded construction cost."""
    model = candidate / "pretrained_model"
    manifest_path = model / "block_regmeanpp_manifest.json"
    record: dict = {
        "candidate": candidate.name,
        "model_sha256": digest(model / "model.safetensors"),
        "manifest_present": manifest_path.is_file(),
        "token_rows_total": None,
        "token_rows_by_expert": None,
        "module_count": None,
        "max_rows_per_sample": None,
        "gradient_or_backward": None,
        "merge_wall_seconds": None,
        "peak_gpu_memory_mib": None,
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["module_count"] = manifest.get("module_count")
        record["max_rows_per_sample"] = manifest.get("max_rows_per_sample")
        record["gradient_or_backward"] = manifest.get("gradient_or_backward")
        by_expert: dict[str, int] = {}
        for entry in (manifest.get("modules") or {}).values():
            for key, value in entry.items():
                if key.endswith("_rows") and isinstance(value, (int, float)):
                    by_expert[key[: -len("_rows")]] = by_expert.get(key[: -len("_rows")], 0) + int(value)
        if by_expert:
            record["token_rows_by_expert"] = dict(sorted(by_expert.items()))
            record["token_rows_total"] = sum(by_expert.values())

    # Merge time and peak memory exist only for runs wrapped in resource accounting.
    for accounting in sorted(RUNTIME.glob("mechanism/*/resource_accounting.json")):
        payload = json.loads(accounting.read_text(encoding="utf-8"))
        if payload.get("return_code") != 0:
            continue
        if candidate.name in json.dumps(payload.get("command", [])):
            record["merge_wall_seconds"] = payload.get("wall_seconds")
            record["peak_gpu_memory_mib"] = (payload.get("gpu") or {}).get("peak_memory_used_mib")
            break
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="write a JSON receipt here")
    args = parser.parse_args()

    rows = []
    for label, name in METHODS.items():
        candidate = CANDIDATES / name
        if not candidate.exists():
            rows.append({"method": label, "missing_candidate": str(candidate)})
            continue
        rows.append({"method": label, **harvest(candidate)})

    receipt = {"schema_version": 1, "runtime": str(RUNTIME), "methods": rows}
    text = json.dumps(receipt, indent=2, sort_keys=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
