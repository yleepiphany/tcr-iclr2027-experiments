"""Scope controls freeze exactly the registered perception/conditioning modules."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts")]
import appendix_contract as base  # noqa: E402
import appendix_scope_contract as scope  # noqa: E402


def _metrics(arm_name: str) -> dict:
    arm = scope.ARMS[arm_name]
    source = json.loads(base.RIDGE.read_text())["modules"]
    index = arm["pass_index"]
    frozen = {name for name in source if ".vision_tower." in name}
    if arm["scope"] == "action_only":
        frozen.update(name for name in source if ".paligemma." in name)
    return {
        name: ({"kind": "frozen_prior"} if name in frozen else {
            "kind": "multi_dense_augmented",
            "rows_by_expert": {
                expert: ((150 if index == 1 else 50) if name in base.TIME_MODULES
                         else (1500 if index == 1 else 800))
                for expert in base.SUITES},
            "ridge": row["ridge"],
        }) for name, row in source.items()
    }


def test_all_four_scope_builds_have_exact_calibrated_modules_rows_and_ridges() -> None:
    for arm_name, arm in scope.ARMS.items():
        metrics = _metrics(arm_name)
        assert scope.validate_rows(metrics, list(base.SUITES), arm_name) == \
            scope.ROWS[(arm["scope"], arm["pass_index"])]
        frozen = next(name for name, row in metrics.items() if row["kind"] == "frozen_prior")
        metrics[frozen] = {"kind": "multi_dense_augmented", "rows_by_expert": {}, "ridge": 0.05}
        with pytest.raises(ValueError, match="module set"):
            scope.validate_rows(metrics, list(base.SUITES), arm_name)
