"""Behavioral tests of budget nesting and the frozen full-module totals."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts")]

from appendix_row_budget import (  # noqa: E402
    EXPECTED_SINGLE_UNION, EXPECTED_TOTALS, nested_row_indices,
    realized_budget, single_union_budget,
)
from pi05_table3_contract import row_indices  # noqa: E402


def _identities(nrows: int, tier: str, slot: int, cameras: int) -> set[tuple[int, int, int]]:
    return {(flow, camera, row)
            for flow in (0, 5, 9)
            for camera in range(cameras)
            for row in nested_row_indices(nrows, tier, flow, slot,
                                          cameras=cameras, camera=camera)}


def test_nested_rows_preserve_the_existing_base_for_all_requests_and_views() -> None:
    for nrows, cameras in ((50, 1), (968, 1), (256, 3)):
        for slot in range(5):
            low = _identities(nrows, "low", slot, cameras)
            base = _identities(nrows, "base", slot, cameras)
            high = _identities(nrows, "high", slot, cameras)
            assert len(low) == 8 and len(base) == 16 and len(high) == 24
            assert low < base < high
            for flow in (0, 5, 9):
                for camera in range(cameras):
                    assert nested_row_indices(nrows, "base", flow, slot,
                                              cameras=cameras, camera=camera) == row_indices(
                                                  nrows, 16, flow, slot,
                                                  cameras=cameras, camera=camera)


def test_scalar_time_projection_cannot_be_scaled() -> None:
    for slot in range(5):
        assert _identities(1, "low", slot, 1) == _identities(1, "base", slot, 1)
        assert _identities(1, "high", slot, 1) == _identities(1, "base", slot, 1)


def test_budget_totals_use_actual_418_module_manifest() -> None:
    manifest = (WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910/"
                "libero/tcr-e/repeat-01/merge/attempt-01-peft-safe-v1/"
                "block_regmeanpp_manifest.json")
    names = set(json.loads(manifest.read_text())["modules"])
    assert {tier: realized_budget(names, tier) for tier in EXPECTED_TOTALS} == EXPECTED_TOTALS
    assert single_union_budget(names) == EXPECTED_SINGLE_UNION
