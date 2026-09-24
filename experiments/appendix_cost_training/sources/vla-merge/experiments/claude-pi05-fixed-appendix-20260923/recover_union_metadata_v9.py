#!/usr/bin/env python3
"""Accept the existing A∪B model with an explicit manifest compatibility audit.

The original build, exit receipt and failed queue remain untouched. This script
copies the original manifest into a new recovery directory, verifies the model
and every module's fixed ridge, and derives the omitted top-level ridge reference
from the embedded frozen config. The checkpoint and its manifest remain untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import run_union_v8 as build
import appendix_contract as appendix
import appendix_union_contract as union

ROOT = build.ROOT
ORIGINAL = ROOT / "union-build-attempt-02"
RECOVERY = ROOT / "union-build-artifact-recovery-attempt-03"
EXPECTED_ERROR = "artifact gate: ValueError: one-pass union model identity, coverage or hash differs"
ORIGINAL_SOLVER_SHA = "4243db468e0dd8c7259b9112bf23b37fd9db40b3e94b94b45b3496b625815079"
CORRECTED_SOLVER = build.HERE / "materialize_union_v9.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path: Path, value: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    if RECOVERY.exists():
        raise FileExistsError(f"Recovery is one-shot: {RECOVERY}")
    old_plan = json.loads((ORIGINAL / "plan.json").read_text())
    old_exit = json.loads((ORIGINAL / "exits" / f"{union.ARM}.json").read_text())
    old_end = json.loads((ORIGINAL / "queue-ended.json").read_text())
    assert old_plan["schema"] == "pi05_fixed_appendix_union_build_queue_v1"
    assert old_plan["solver_sha256"] == ORIGINAL_SOLVER_SHA
    assert [job["id"] for job in old_plan["jobs"]] == [union.ARM]
    assert old_plan["source_audit_sha256"] == sha(build.SOURCE_AUDIT)
    assert old_plan["row_budget_sha256"] == sha(build.ROW_BUDGET)
    assert old_plan["dense_bank_sha256"] == sha(appendix.DENSE_BANK)
    assert old_plan["contract_sha256"] == sha(build.HERE / "appendix_union_contract.py")
    for bound in old_plan["trace_bindings"].values():
        assert bound["manifest_sha256"] == sha(Path(bound["manifest"]))
        assert bound["replay_sha256"] == sha(Path(bound["replay"]))
    assert old_exit["returncode"] == 0 and not old_exit["accepted"]
    assert old_exit["error"] == EXPECTED_ERROR
    assert old_end["complete"] is False and old_end["failed"][union.ARM] == EXPECTED_ERROR
    job = old_plan["jobs"][0]
    output = Path(job["output"])
    manifest_path = output / "block_regmeanpp_manifest.json"
    model_path = output / "model.safetensors"
    config_path = Path(job["config"])
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    ridge_path = Path(config["ridge_source_manifest"])
    ridge = json.loads(ridge_path.read_text())
    assert config["schema"] == union.SCHEMA and config["arm"] == union.ARM
    assert ridge_path == appendix.RIDGE and sha(ridge_path) == config["ridge_source_sha256"]
    assert manifest.get("fixed_ridge_reference") is None
    assert manifest["ablation"]["ridge_source_manifest"] == str(ridge_path)
    assert manifest["ablation_config_sha256"] == sha(config_path)
    assert manifest["model_sha256"] == sha(model_path)
    modules = manifest["modules"]
    assert set(modules) == set(ridge["modules"]) and len(modules) == 418
    assert all(modules[key]["ridge"] == ridge["modules"][key]["ridge"] for key in modules)
    rows = union.validate_rows(modules, list(appendix.SUITES))
    assert manifest["fixed_appendix_arm"] == union.ARM
    assert manifest["fixed_appendix_pass"] == 1
    assert manifest["realized_row_total"] == rows == union.ROW_TOTAL
    assert manifest["modified_tensor_count"] == 422
    assert manifest["calibration_source_counts"] == {name: 2 for name in appendix.SUITES}
    union.validate_config(config)
    old_manifest_sha = sha(manifest_path)
    assert sha(build.SOLVER) == ORIGINAL_SOLVER_SHA
    corrected_text = CORRECTED_SOLVER.read_text()
    assert "in (three_level.SCHEMA, subset.SCHEMA, appendix.SCHEMA, union.SCHEMA)" in corrected_text
    RECOVERY.mkdir(parents=True, exist_ok=False)
    original_copy = RECOVERY / "block_regmeanpp_manifest.original_attempt02.json"
    shutil.copy2(manifest_path, original_copy)
    assert sha(original_copy) == old_manifest_sha
    result = {"id": union.ARM, "model_sha256": manifest["model_sha256"],
              "manifest_sha256": old_manifest_sha, "modules": len(modules), "rows": rows}
    assert sha(manifest_path) == old_manifest_sha  # source artifact was not modified
    plan = {
        **old_plan,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recovery_of": str(ORIGINAL),
        "recovery_reason": "union schema omitted from fixed_ridge_reference serialization only",
        "original_plan_sha256": sha(ORIGINAL / "plan.json"),
        "original_manifest_sha256": old_manifest_sha,
        "compatibility_reference": str(ridge_path),
        "compatibility_check": "embedded config and all 418 output ridge values equal frozen reference",
        "original_artifacts_unchanged": True,
        "model_recomputed": False,
        "original_solver_sha256": ORIGINAL_SOLVER_SHA,
        "corrected_solver_sha256": sha(CORRECTED_SOLVER),
    }
    save(RECOVERY / "plan.json", plan)
    save(RECOVERY / "state.json", {
        "accepted": {union.ARM: result}, "failed": {}, "skipped": {},
        "active": {}, "pending": [], "stopped": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    save(RECOVERY / "queue-ended.json", {
        "complete": True, "accepted": {union.ARM: result}, "failed": {},
        "skipped": {}, "pending": [], "stopped": False,
        "reused_model_without_recompute": True,
        "original_manifest_sha256": old_manifest_sha,
        "manifest_sha256": old_manifest_sha,
        "ridge_modules_exact": len(modules),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"recovery": str(RECOVERY), "model_sha256": result["model_sha256"],
                      "manifest_sha256": old_manifest_sha,
                      "ridge_modules_exact": len(modules)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
