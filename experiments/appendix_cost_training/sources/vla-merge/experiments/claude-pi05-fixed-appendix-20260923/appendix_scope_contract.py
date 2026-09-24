"""Fixed action-only and conditioning+action appendix scope controls."""
from __future__ import annotations

import json
import math
from pathlib import Path

import appendix_contract as base

SCHEMA = "pi05_fixed_appendix_scope_v5"
ROOT = base.RUNTIME / "claude-pi05-fixed-appendix-20260923"
ARMS = {
    "action_A": {"scope": "action_only", "pass_index": 1, "parent": "Soup", "pool": "A_new"},
    "action_AB": {"scope": "action_only", "pass_index": 2, "parent": "action_A", "pool": "B_new"},
    "conditioning_A": {"scope": "conditioning_action", "pass_index": 1, "parent": "Soup", "pool": "A_new"},
    "conditioning_AB": {"scope": "conditioning_action", "pass_index": 2, "parent": "conditioning_A", "pool": "B_new"},
}
SCOPE_MODULES = {"action_only": 130, "conditioning_action": 256}
ROWS = {
    ("action_only", 1): 769_200,
    ("action_only", 2): 410_000,
    ("conditioning_action", 1): 1_525_200,
    ("conditioning_action", 2): 813_200,
}


def parent_path(arm_name: str) -> Path:
    arm = ARMS[arm_name]
    return base.SOUP if arm["pass_index"] == 1 else ROOT / "checkpoints-v5" / arm["parent"]


def validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA or config.get("arm") not in ARMS:
        raise ValueError("unregistered appendix scope control")
    arm = ARMS[config["arm"]]
    index = arm["pass_index"]
    expected = {
        "variant": arm["scope"], "scope": arm["scope"],
        "pass_index": index, "pool": arm["pool"], "parent_arm": arm["parent"],
        "row_cap_per_request_module": base.CAP[index],
        "expected_realized_rows": ROWS[(arm["scope"], index)],
        "mass_rule": "relative_prior_error" if index == 1 else "uniform_quarter",
        "expert_loss_normalization": base.NORMALIZATION[index],
        "replay_prefix": "merged", "module_count": 418,
        "calibrated_module_count": SCOPE_MODULES[arm["scope"]],
        "dense_expert_bank_sha256": base.sha(base.DENSE_BANK),
        "historical_table1_full": False, "evaluation_repeat": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"scope control {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if start != parent_path(config["arm"]).resolve() or \
            base.sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("scope control parent model differs")
    if index == 2 and Path(config.get("parent_manifest", "")).resolve() != \
            (start / "block_regmeanpp_manifest.json").resolve():
        raise ValueError("scope control pass-B parent manifest differs")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != base.RIDGE.resolve() or base.sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("scope control fixed ridge source differs")
    return arm


def validate_trace(manifest: dict, dense_path: Path, suite: str, index: int) -> None:
    base.validate_trace(manifest, dense_path, suite, index)


def validate_rows(metrics: dict, names: list[str], arm_name: str) -> int:
    if arm_name not in ARMS or len(metrics) != 418 or set(names) != set(base.SUITES):
        raise ValueError("scope control full module/expert accounting differs")
    arm = ARMS[arm_name]
    scope, index = arm["scope"], arm["pass_index"]
    fixed = json.loads(base.RIDGE.read_text())["modules"]
    if set(metrics) != set(fixed):
        raise ValueError("scope control module names differ")
    solved = {name: row for name, row in metrics.items() if row.get("kind") != "frozen_prior"}
    frozen = set(metrics) - set(solved)
    vision = {name for name in metrics if ".vision_tower." in name}
    language = {name for name in metrics
                if ".paligemma." in name and ".vision_tower." not in name}
    expected_frozen = vision | language if scope == "action_only" else vision
    if (len(solved) != SCOPE_MODULES[scope]
            or len(vision) != 162 or len(language) != 126
            or frozen != expected_frozen):
        raise ValueError("scope control calibrated/frozen module set differs")
    total = 0
    for name, row in solved.items():
        per_expert = (150 if index == 1 else 50) if name in base.TIME_MODULES \
            else (1500 if index == 1 else 800)
        if row.get("rows_by_expert") != {expert: per_expert for expert in names}:
            raise ValueError(f"scope control rows differ for {name}")
        if not math.isfinite(row.get("ridge", math.nan)) or row["ridge"] != fixed[name]["ridge"]:
            raise ValueError(f"scope control fixed ridge differs for {name}")
        total += per_expert * len(names)
    if total != ROWS[(scope, index)]:
        raise ValueError("scope control realized total differs")
    return total
