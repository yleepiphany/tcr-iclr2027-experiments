"""Uniform first pass retains both frozen row budgets and cache identities."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts")]
import appendix_contract as base  # noqa: E402
import appendix_uniform_contract as uniform  # noqa: E402


def test_uniform_A_and_B_source_and_rows_are_fixed() -> None:
    bank = json.loads(base.DENSE_BANK.read_text())
    dense = {row["name"]: Path(row["dense_checkpoint"]["path"])
             for row in bank["experts"]}
    source = json.loads(base.RIDGE.read_text())["modules"]
    assert uniform.validate_rows(source, list(base.SUITES), "uniform_A") == 4_441_200
    B = {module: {"rows_by_expert": {
        name: 50 if module in base.TIME_MODULES else 800 for name in base.SUITES},
        "ridge": reference["ridge"]}
        for module, reference in source.items()}
    assert uniform.validate_rows(B, list(base.SUITES), "uniform_AB") == 1_331_600
    for index, pool in base.POOLS.items():
        for suite in base.SUITES:
            manifest = json.loads((base.INPUTS / pool / "inputs" / suite /
                                   "across/replay.json").read_text())
            uniform.validate_trace(manifest, dense[suite], suite, index)
            with pytest.raises(ValueError, match="source identity"):
                uniform.validate_trace(manifest, dense[suite], suite, 3 - index)
