#!/usr/bin/env python3
"""Recompute the fixed appendix row counts from the audited 418-module scope."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path[:0] = [str(HERE), str(WORK / "vla-merge/scripts")]
from appendix_row_budget import (  # noqa: E402
    EXPECTED_SINGLE_UNION, EXPECTED_TOTALS, realized_budget,
    single_union_budget,
)

SCOPE = (WORK / "vla-merge-runtime/experiments/iclr2027-table1-20260910/"
         "libero/tcr-e/repeat-01/merge/attempt-01-peft-safe-v1/"
         "block_regmeanpp_manifest.json")
INPUT_AUDIT = WORK / "coordination/2026-09-23/appendix-abc-pools-content-audit.json"
OUTPUT = WORK / "coordination/2026-09-23/appendix-row-budget-contract.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    scope, audit = json.loads(SCOPE.read_text()), json.loads(INPUT_AUDIT.read_text())
    if audit.get("accepted") is not True or audit.get("jobs") != 12 or audit.get("samples") != 1800:
        raise ValueError("A/B/C source content audit is not complete")
    names = set(scope["modules"])
    totals = {tier: realized_budget(names, tier) for tier in EXPECTED_TOTALS}
    union = single_union_budget(names)
    if totals != EXPECTED_TOTALS or union != EXPECTED_SINGLE_UNION:
        raise ValueError("appendix row totals differ from the frozen contract")
    result = {
        "schema": "pi05_fixed_appendix_row_budget_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_content_audit": str(INPUT_AUDIT),
        "input_content_audit_sha256": sha(INPUT_AUDIT),
        "module_scope_manifest": str(SCOPE),
        "module_scope_manifest_sha256": sha(SCOPE),
        "module_count": len(names),
        "vision_module_count": sum(".vision_tower." in name for name in names),
        "time_module_count": 2,
        "experts": 4,
        "requests_per_expert_pool": 50,
        "flow_indices": [0, 5, 9],
        "A_first_pass_rows": 4_441_200,
        "B_second_pass_rows": totals,
        "B_realized_ratio_to_base": {tier: total / totals["base"]
                                     for tier, total in totals.items()},
        "single_A_union_B_rows": union,
        "row_identity_rule": "low subset of frozen cap16, high superset of frozen cap16; time modules unchanged",
        "scope_limit": "time projections have one unique row/request; realized B total differs by 200 rows from exact 0.5x or 1.5x",
        "status": "cpu_budget_contract_only_no_model_or_evaluation",
    }
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"accepted": True, "B_rows": totals,
                      "single_union_rows": union, "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
