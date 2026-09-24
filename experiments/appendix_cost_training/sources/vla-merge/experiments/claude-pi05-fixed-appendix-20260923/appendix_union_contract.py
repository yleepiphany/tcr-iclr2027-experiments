"""Frozen one-pass A∪B row and source identity contract.

A contributes its original first-pass per-flow cap-10 rows. B contributes its
original per-request cap-16 rows. Both are fitted together from Soup in one
merged-prefix, relative-error-weighted solve; no intermediate A checkpoint is
used. The effective total is 4,441,200 + 1,331,600 rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import appendix_contract as base
from appendix_row_budget import TIME_MODULES

SCHEMA = "pi05_fixed_appendix_single_a_union_b_v1"
ARM = "A_union_B"
POOLS = ("A_new", "B_new")
ROW_TOTAL = base.ROWS[1] + base.ROWS[2]
SAMPLES_PER_SOURCE = 150


def validate_config(config: dict) -> None:
    expected = {
        "schema": SCHEMA, "arm": ARM, "variant": "full", "pass_index": 1,
        "pools": list(POOLS), "row_caps": {"A_new": 10, "B_new": 16},
        "expected_realized_rows": ROW_TOTAL, "mass_rule": "relative_prior_error",
        "expert_loss_normalization": "prior", "replay_prefix": "merged",
        "module_count": 418, "dense_expert_bank_sha256": base.sha(base.DENSE_BANK),
        "historical_table1_full": False, "evaluation_repeat": 1,
        "single_pass": True, "parent_arm": "Soup",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"one-pass A∪B {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if start != base.SOUP.resolve() or base.sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("one-pass A∪B must start from frozen Soup")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != base.RIDGE.resolve() or base.sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("one-pass A∪B fixed ridge source differs")


def validate_sources(calibrations: dict, manifests: dict) -> None:
    if set(calibrations) != set(base.SUITES) or set(manifests) != set(base.SUITES):
        raise ValueError("one-pass A∪B expert scope differs")
    for suite in base.SUITES:
        wanted = [base.INPUTS / pool / "inputs" / suite / "across" for pool in POOLS]
        if [Path(p).resolve() for p in calibrations[suite]] != [
                (root / "replay.safetensors").resolve() for root in wanted]:
            raise ValueError(f"one-pass A∪B tensor sources differ: {suite}")
        if [Path(p).resolve() for p in manifests[suite]] != [
                (root / "replay.json").resolve() for root in wanted]:
            raise ValueError(f"one-pass A∪B manifest sources differ: {suite}")


def validate_trace(manifest: dict, dense_path: Path, suite: str) -> None:
    pool = manifest.get("repeat_id")
    if pool not in POOLS:
        raise ValueError("one-pass A∪B has an unfamiliar source pool")
    base.validate_trace(manifest, dense_path, suite, 1 if pool == "A_new" else 2)


def source_for_state(index: int, *, is_vision: bool) -> tuple[str, int, int]:
    if type(index) is not int or index < 0:
        raise ValueError("negative or noninteger flow-state index")
    cameras = 3 if is_vision else 1
    total = SAMPLES_PER_SOURCE * cameras
    if index >= 2 * total:
        raise ValueError("one-pass A∪B flow-state index exceeds two sources")
    return (POOLS[index // total], (index % total) // cameras,
            index % cameras if is_vision else 0)


def select_rows(value, index: int, module_name: str, samples: list[dict], *,
                sample_rows, row_indices):
    """Apply A's per-flow quota or B's per-request quota to one native input."""
    is_vision = ".vision_tower." in module_name
    pool, _local_state, camera = source_for_state(index, is_vision=is_vision)
    if len(samples) != 2 * SAMPLES_PER_SOURCE:
        raise ValueError("one-pass A∪B requires 150 states from each source")
    if pool == "A_new":
        return sample_rows(value, 10)
    state_index = index // 3 if is_vision else index
    sample = samples[state_index]
    if sample.get("flow_index") not in (0, 5, 9) or \
            type(sample.get("selected_request_slot")) is not int:
        raise ValueError("B source request/flow identity differs")
    flat = value.reshape(-1, value.shape[-1]).float()
    ids = row_indices(len(flat), 16,
                      sample["flow_index"], sample["selected_request_slot"],
                      cameras=3 if is_vision else 1, camera=camera)
    return flat[ids]


def validate_rows(metrics: dict, names: list[str]) -> int:
    if len(metrics) != 418 or set(names) != set(base.SUITES):
        raise ValueError("one-pass A∪B module/expert scope differs")
    fixed = json.loads(base.RIDGE.read_text())["modules"]
    if set(metrics) != set(fixed):
        raise ValueError("one-pass A∪B ridge module scope differs")
    total = 0
    for module, row in metrics.items():
        first = 150 if module in TIME_MODULES else 4500 if ".vision_tower." in module else 1500
        second = 50 if module in TIME_MODULES else 800
        expected = first + second
        if row.get("rows_by_expert") != {name: expected for name in names}:
            raise ValueError(f"one-pass A∪B row budget differs: {module}")
        if not math.isfinite(row.get("ridge", math.nan)) or row["ridge"] != fixed[module]["ridge"]:
            raise ValueError(f"one-pass A∪B numeric ridge differs: {module}")
        total += expected * len(names)
    if total != ROW_TOTAL:
        raise ValueError("one-pass A∪B total row budget differs")
    return total
