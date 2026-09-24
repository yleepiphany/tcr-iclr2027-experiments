"""Actual B/C cache identity and fixed module row/ridge tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts"),
                str(WORK / "vla-merge/experiments/claude-three-level-main-20260921")]
import appendix_cache_followon_contract as cache  # noqa: E402
import appendix_contract as base  # noqa: E402


def test_B_and_C_actual_traces_have_distinct_registered_pool_identity() -> None:
    bank = json.loads(base.DENSE_BANK.read_text())
    dense = {row["name"]: Path(row["dense_checkpoint"]["path"])
             for row in bank["experts"]}
    for arm in cache.ARMS.values():
        for suite in base.SUITES:
            manifest = json.loads((base.INPUTS / arm["pool"] / "inputs" / suite /
                                   "across/replay.json").read_text())
            cache.validate_trace(manifest, dense[suite], suite, arm["pool"])
            wrong = "B_new" if arm["pool"] == "C_new" else "C_new"
            with pytest.raises(ValueError, match="trace provenance"):
                cache.validate_trace(manifest, dense[suite], suite, wrong)


def test_three_cache_followons_share_exact_rows_and_ridge_values() -> None:
    source = json.loads(base.RIDGE.read_text())["modules"]
    names = list(base.SUITES)
    rows = {module: {"rows_by_expert": {
        expert: 50 if module in base.TIME_MODULES else 800 for expert in names},
        "ridge": reference["ridge"]}
        for module, reference in source.items()}
    for arm in cache.ARMS:
        assert cache.validate_rows(rows, names, arm) == 1_331_600
    first = next(iter(rows))
    rows[first]["ridge"] *= 1.001
    with pytest.raises(ValueError, match="fixed ridge"):
        cache.validate_rows(rows, names, "AC")
