"""Two-pass early-request control with all other Full settings fixed."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import appendix_contract as base

SCHEMA = "pi05_fixed_appendix_early_requests_v1"
ROOT = base.RUNTIME / "claude-pi05-fixed-appendix-20260923"
CAPTURE = ROOT / "early-capture-attempt-04"
INPUTS = CAPTURE
DENSE_BANK = base.DENSE_BANK
SOUP = base.SOUP
RIDGE = base.RIDGE
SUITES = base.SUITES
sha = base.sha
POOLS = {1: "A_early", 2: "B_early"}
CAP = base.CAP
ROWS = base.ROWS
NORMALIZATION = base.NORMALIZATION
ARMS = {1: "early_A", 2: "early_AB"}


def parent_path(index: int) -> Path:
    return base.SOUP if index == 1 else ROOT / "checkpoints-v14/early_A"


def validate_config(config: dict) -> int:
    index = config.get("pass_index")
    if config.get("schema") != SCHEMA or index not in POOLS:
        raise ValueError("unregistered early-request pass")
    expected = {
        "arm": ARMS[index], "variant": "full", "pool": POOLS[index],
        "row_cap_per_request_module": CAP[index],
        "expected_realized_rows": ROWS[index],
        "mass_rule": "relative_prior_error" if index == 1 else "uniform_quarter",
        "expert_loss_normalization": NORMALIZATION[index],
        "replay_prefix": "merged", "module_count": 418,
        "dense_expert_bank_sha256": base.sha(base.DENSE_BANK),
        "historical_table1_full": False, "evaluation_repeat": 1,
        "request_selection": "early", "native_requests_per_task": 5,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"early-request {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if start != parent_path(index).resolve() or \
            base.sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("early-request pass start differs")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != base.RIDGE.resolve() or base.sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("early-request fixed ridge source differs")
    if index == 2 and Path(config.get("parent_A_manifest", "")).resolve() != \
            (start / "block_regmeanpp_manifest.json").resolve():
        raise ValueError("early-request B must name its accepted early-A parent")
    return index


def validate_trace(manifest: dict, dense_path: Path, suite: str, index: int) -> None:
    if (index not in POOLS or suite not in base.SUITES or
            manifest.get("task") != suite or manifest.get("repeat_id") != POOLS[index] or
            Path(manifest.get("calibration_policy", "")).resolve() != dense_path.resolve() or
            manifest.get("source_kind") != "expert_execution" or
            manifest.get("table3_capture_version") != 1 or
            manifest.get("table3_request_selection") != "early"):
        raise ValueError("early-request trace identity differs")
    samples = manifest.get("samples") or []
    if manifest.get("sample_count") != 150 or len(samples) != 150:
        raise ValueError("early-request trace must contain 150 flow states")
    grouped: dict[tuple[int, int], list[int]] = {}
    for sample in samples:
        if sample.get("vision_count") != 3:
            raise ValueError("early-request trace lacks three-camera prefix")
        grouped.setdefault((sample.get("prompt_signature"),
                            sample.get("selected_request_slot")), []).append(
                                sample.get("flow_index"))
    if len(grouped) != 50 or any(sorted(values) != [0, 5, 9] for values in grouped.values()) or \
            set(Counter(key[0] for key in grouped).values()) != {5}:
        raise ValueError("early-request task/request/flow allocation differs")
    from audit_early_requests import check_early
    result = check_early(manifest)
    if result["accepted"] is not True:
        raise ValueError(f"early-request native selection differs: {result['issues']}")


def validate_rows(metrics: dict, names: list[str], index: int) -> int:
    total = base.validate_rows(metrics, names, index)
    fixed = json.loads(base.RIDGE.read_text())["modules"]
    if set(metrics) != set(fixed) or any(
            not math.isfinite(row.get("ridge", math.nan)) or
            row["ridge"] != fixed[name]["ridge"] for name, row in metrics.items()):
        raise ValueError("early-request numeric ridge map differs")
    return total
