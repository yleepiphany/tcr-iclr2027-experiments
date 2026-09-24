"""A-parent follow-ons preserve fixed ridges and realize their exact row totals."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts")]
import appendix_contract as base  # noqa: E402
import appendix_followon_contract as followon  # noqa: E402
from appendix_row_budget import TIERS  # noqa: E402


def test_all_three_followon_totals_and_fixed_ridges() -> None:
    source = json.loads(base.RIDGE.read_text())["modules"]
    names = list(base.SUITES)
    for arm_name, arm in followon.ARMS.items():
        regular = 50 * TIERS[arm["tier"]]
        metrics = {module: {
            "rows_by_expert": {name: 50 if module in base.TIME_MODULES else regular
                               for name in names},
            "ridge": reference["ridge"],
        } for module, reference in source.items()}
        assert followon.validate_rows(metrics, names, arm_name) == arm["rows"]
        metric = next(module for module in metrics if module not in base.TIME_MODULES)
        metrics[metric]["ridge"] *= 1.01
        with pytest.raises(ValueError, match="ridge differs"):
            followon.validate_rows(metrics, names, arm_name)


def test_followon_pools_are_the_actual_A_and_B_sources() -> None:
    bank = json.loads(base.DENSE_BANK.read_text())
    dense = {row["name"]: Path(row["dense_checkpoint"]["path"])
             for row in bank["experts"]}
    for arm in followon.ARMS.values():
        for suite in base.SUITES:
            manifest = json.loads((base.INPUTS / arm["pool"] / "inputs" / suite /
                                   "across/replay.json").read_text())
            followon.validate_trace(manifest, dense[suite], suite, arm["pool"])
