"""Fixed A-parent controls: repeat A and nested B-row budgets.

These three models are separate appendix constructions.  The A parent, frozen
numerical ridge map, source requests, expert masses, and evaluation repeat are
identical; only the registered cache or row subset changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import appendix_contract as base
from appendix_row_budget import EXPECTED_TOTALS, TIERS, TIME_MODULES, realized_budget

SCHEMA = "pi05_fixed_appendix_followon_v2"
ROOT = base.RUNTIME / "claude-pi05-fixed-appendix-20260923"
A = ROOT / "checkpoints-v1/A"
ARMS = {
    "AA": {"pool": "A_new", "tier": "base", "rows": EXPECTED_TOTALS["base"]},
    "AB_low": {"pool": "B_new", "tier": "low", "rows": EXPECTED_TOTALS["low"]},
    "AB_high": {"pool": "B_new", "tier": "high", "rows": EXPECTED_TOTALS["high"]},
}


def validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA or config.get("arm") not in ARMS:
        raise ValueError("unregistered appendix follow-on")
    arm = ARMS[config["arm"]]
    expected = {
        "variant": "full", "pass_index": 2,
        "pool": arm["pool"], "row_tier": arm["tier"],
        "row_cap_per_request_module": TIERS[arm["tier"]],
        "expected_realized_rows": arm["rows"],
        "mass_rule": "uniform_quarter",
        "expert_loss_normalization": "none", "replay_prefix": "merged",
        "module_count": 418,
        "dense_expert_bank_sha256": base.sha(base.DENSE_BANK),
        "historical_table1_full": False, "evaluation_repeat": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"appendix follow-on {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if start != A.resolve() or base.sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("follow-on start differs from accepted A")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != base.RIDGE.resolve() or base.sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("follow-on fixed numerical ridge map differs")
    if Path(config.get("parent_A_manifest", "")).resolve() != (A / "block_regmeanpp_manifest.json").resolve():
        raise ValueError("follow-on A parent manifest differs")
    return arm


def validate_trace(manifest: dict, dense_path: Path, suite: str, pool: str) -> None:
    pass_index = 1 if pool == "A_new" else 2 if pool == "B_new" else None
    if pass_index is None:
        raise ValueError("follow-on pool differs")
    base.validate_trace(manifest, dense_path, suite, pass_index)


def validate_rows(metrics: dict, names: list[str], arm_name: str) -> int:
    if arm_name not in ARMS or len(metrics) != 418 or set(names) != set(base.SUITES):
        raise ValueError("follow-on module/expert scope differs")
    arm = ARMS[arm_name]
    per_regular = 50 * TIERS[arm["tier"]]
    total = 0
    frozen_ridges = json.loads(base.RIDGE.read_text())["modules"]
    if set(metrics) != set(frozen_ridges):
        raise ValueError("follow-on module names differ from fixed ridge reference")
    for name, row in metrics.items():
        per_expert = 50 if name in TIME_MODULES else per_regular
        if row.get("rows_by_expert") != {expert: per_expert for expert in names}:
            raise ValueError(f"follow-on rows differ for {name}")
        if not math.isfinite(row.get("ridge", math.nan)) or row["ridge"] != frozen_ridges[name]["ridge"]:
            raise ValueError(f"follow-on numerical ridge differs for {name}")
        total += per_expert * len(names)
    if total != arm["rows"] or total != realized_budget(set(metrics), arm["tier"]):
        raise ValueError("follow-on realized total differs")
    return total
