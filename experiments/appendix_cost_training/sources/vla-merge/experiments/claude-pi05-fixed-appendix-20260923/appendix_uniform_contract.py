"""Only first-pass expert mass changes: relative error → uniform."""
from __future__ import annotations

import json
import math
from pathlib import Path

import appendix_contract as base

SCHEMA = "pi05_fixed_appendix_uniform_first_v4"
ROOT = base.RUNTIME / "claude-pi05-fixed-appendix-20260923"
ARMS = {
    "uniform_A": {"pass_index": 1, "pool": "A_new", "parent": "Soup"},
    "uniform_AB": {"pass_index": 2, "pool": "B_new", "parent": "uniform_A"},
}


def parent_path(arm_name: str) -> Path:
    return base.SOUP if arm_name == "uniform_A" else ROOT / "checkpoints-v4/uniform_A"


def validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA or config.get("arm") not in ARMS:
        raise ValueError("unregistered uniform-first-pass control")
    arm = ARMS[config["arm"]]
    index = arm["pass_index"]
    expected = {
        "variant": "full", "pass_index": index,
        "pool": arm["pool"], "parent_arm": arm["parent"],
        "row_cap_per_request_module": base.CAP[index],
        "expected_realized_rows": base.ROWS[index],
        "mass_rule": "uniform_quarter",
        "expert_loss_normalization": "none", "replay_prefix": "merged",
        "module_count": 418, "dense_expert_bank_sha256": base.sha(base.DENSE_BANK),
        "historical_table1_full": False, "evaluation_repeat": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"uniform control {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if start != parent_path(config["arm"]).resolve() or \
            base.sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("uniform control parent model differs")
    if index == 2 and Path(config.get("parent_manifest", "")).resolve() != \
            (start / "block_regmeanpp_manifest.json").resolve():
        raise ValueError("uniform pass-B parent manifest differs")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != base.RIDGE.resolve() or base.sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("uniform control fixed ridge source differs")
    return arm


def validate_trace(manifest: dict, dense_path: Path, suite: str, index: int) -> None:
    base.validate_trace(manifest, dense_path, suite, index)


def validate_rows(metrics: dict, names: list[str], arm_name: str) -> int:
    if arm_name not in ARMS:
        raise ValueError("unregistered uniform control")
    total = base.validate_rows(metrics, names, ARMS[arm_name]["pass_index"])
    fixed = json.loads(base.RIDGE.read_text())["modules"]
    if set(metrics) != set(fixed) or any(
        not math.isfinite(row.get("ridge", math.nan)) or row["ridge"] != fixed[name]["ridge"]
        for name, row in metrics.items()
    ):
        raise ValueError("uniform control fixed ridge map differs")
    return total
