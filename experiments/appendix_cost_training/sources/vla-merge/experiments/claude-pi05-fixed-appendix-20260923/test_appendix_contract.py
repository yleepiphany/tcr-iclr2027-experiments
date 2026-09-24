"""Actual-source gates for the shared fixed appendix A→B reference."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts")]

import appendix_contract as contract  # noqa: E402


def test_actual_A_and_B_pool_manifests_bind_to_distinct_sources() -> None:
    bank = json.loads(contract.DENSE_BANK.read_text())
    dense = {row["name"]: Path(row["dense_checkpoint"]["path"])
             for row in bank["experts"]}
    for pass_index, pool in contract.POOLS.items():
        for suite in contract.SUITES:
            path = contract.INPUTS / pool / "inputs" / suite / "across/replay.json"
            manifest = json.loads(path.read_text())
            contract.validate_trace(manifest, dense[suite], suite, pass_index)
            with pytest.raises(ValueError, match="source identity"):
                contract.validate_trace(manifest, dense[suite], suite, 3 - pass_index)


def test_real_A_rows_and_exact_B_scope_budget() -> None:
    source = json.loads(contract.RIDGE.read_text())["modules"]
    names = list(contract.SUITES)
    assert contract.validate_rows(source, names, 1) == 4_441_200
    B = {key: {"rows_by_expert": {name: 50 if key in contract.TIME_MODULES else 800
                                    for name in names}, "ridge": value["ridge"]}
         for key, value in source.items()}
    assert contract.validate_rows(B, names, 2) == 1_331_600
    first = next(key for key in B if key not in contract.TIME_MODULES)
    B[first]["rows_by_expert"][names[0]] -= 1
    with pytest.raises(ValueError, match="realized rows"):
        contract.validate_rows(B, names, 2)
