"""Fixed A→C and A→B→{B,C} appendix cache contrasts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import appendix_contract as base
import matched_main_contract as matched

SCHEMA = "pi05_fixed_appendix_cache_followon_v3"
ROOT = base.RUNTIME / "claude-pi05-fixed-appendix-20260923"
ARMS = {
    "AC": {"parent": "A", "pool": "C_new", "pass_index": 2, "tier": "base"},
    "ABB": {"parent": "AB", "pool": "B_new", "pass_index": 3, "tier": "base"},
    "ABC": {"parent": "AB", "pool": "C_new", "pass_index": 3, "tier": "base"},
}


def parent_path(arm_name: str) -> Path:
    return ROOT / "checkpoints-v8" / ARMS[arm_name]["parent"]


def validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA or config.get("arm") not in ARMS:
        raise ValueError("unregistered cache follow-on")
    arm = ARMS[config["arm"]]
    expected = {
        "variant": "full", "pass_index": arm["pass_index"],
        "pool": arm["pool"], "row_tier": "base", "row_cap_per_request_module": 16,
        "expected_realized_rows": base.ROWS[2],
        "mass_rule": "uniform_quarter",
        "expert_loss_normalization": "none", "replay_prefix": "merged",
        "module_count": 418,
        "dense_expert_bank_sha256": base.sha(base.DENSE_BANK),
        "historical_table1_full": False, "evaluation_repeat": 1,
        "parent_arm": arm["parent"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"cache follow-on {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if start != parent_path(config["arm"]).resolve() or \
            base.sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("cache follow-on parent model differs")
    if Path(config.get("parent_manifest", "")).resolve() != \
            (start / "block_regmeanpp_manifest.json").resolve():
        raise ValueError("cache follow-on parent manifest differs")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != base.RIDGE.resolve() or base.sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("cache follow-on numerical ridge source differs")
    return arm


def validate_trace(manifest: dict, dense_path: Path, suite: str, pool: str) -> None:
    repeat = {"B_new": 2, "C_new": 3}.get(pool)
    if repeat is None:
        raise ValueError("cache follow-on pool differs")
    matched.validate_trace(manifest, dense_path, suite, repeat, 1)


def validate_rows(metrics: dict, names: list[str], arm_name: str) -> int:
    if arm_name not in ARMS:
        raise ValueError("unregistered cache follow-on")
    total = base.validate_rows(metrics, names, 2)
    fixed = json.loads(base.RIDGE.read_text())["modules"]
    if set(metrics) != set(fixed) or any(
        not math.isfinite(row.get("ridge", math.nan)) or row["ridge"] != fixed[name]["ridge"]
        for name, row in metrics.items()
    ):
        raise ValueError("cache follow-on fixed ridge map differs")
    return total
