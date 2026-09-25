"""Frozen source and row contract for the shared A_new→B_new appendix reference.

This reference is a new single-build, single-repeat experiment.  It never
inherits the historical Full success rate; only the historical matching Full
pass-A numerical ridge values are reused as a fixed solver hyperparameter.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from appendix_row_budget import EXPECTED_TOTALS, TIME_MODULES, realized_budget

WORK = Path(__file__).resolve().parents[3]
RUNTIME = WORK / "vla-merge-runtime/experiments"
SCHEMA = "pi05_fixed_appendix_ab_reference_v1"
SUITES = ("spatial", "object", "goal", "long")
POOLS = {1: "A_new", 2: "B_new"}
CAP = {1: 10, 2: 16}
ROWS = {1: 4_441_200, 2: EXPECTED_TOTALS["base"]}
NORMALIZATION = {1: "prior", 2: "none"}
SOUP = (RUNTIME / "iclr2027-table1-20260910/libero/model-soups/"
        "repeat-shared/merge/attempt-02-peft-safe-v2/pretrained_model")
RIDGE = (RUNTIME / "iclr2027-table1-20260910/libero/tcr-e/repeat-01/"
         "merge/attempt-01-peft-safe-v1/block_regmeanpp_manifest.json")
DENSE_BANK = RUNTIME / "iclr2027-table1-20260910/expert-dense-bank-peft-v2.json"
INPUTS = RUNTIME / "claude-cache-sequence-budget-20260921/pools-attempt-01"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> int:
    if config.get("schema") != SCHEMA:
        raise ValueError("wrong appendix reference schema")
    pass_index = config.get("pass_index")
    if pass_index not in POOLS:
        raise ValueError("unregistered appendix reference pass")
    expected = {
        "arm": "A" if pass_index == 1 else "AB",
        "variant": "full",
        "pool": POOLS[pass_index],
        "row_cap_per_request_module": CAP[pass_index],
        "expected_realized_rows": ROWS[pass_index],
        "mass_rule": "relative_prior_error" if pass_index == 1 else "uniform_quarter",
        "expert_loss_normalization": NORMALIZATION[pass_index],
        "replay_prefix": "merged",
        "module_count": 418,
        "dense_expert_bank_sha256": sha(DENSE_BANK),
        "historical_table1_full": False,
        "evaluation_repeat": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"appendix reference {key} differs")
    start = Path(config.get("start_point", "")).resolve()
    if pass_index == 1 and start != SOUP.resolve():
        raise ValueError("pass A must start from the frozen four-expert Soup")
    model = start / "model.safetensors"
    if not model.is_file() or sha(model) != config.get("start_point_sha256"):
        raise ValueError("appendix pass start model hash differs")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != RIDGE.resolve() or sha(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("fixed historical pass-A numerical ridge source differs")
    modules = json.loads(ridge.read_text()).get("modules") or {}
    if len(modules) != 418 or not all(math.isfinite(row.get("ridge", math.nan)) and row["ridge"] > 0
                                      for row in modules.values()):
        raise ValueError("ridge source does not contain 418 finite positive values")
    if pass_index == 2 and Path(config.get("parent_A_manifest", "")).resolve() != \
            (start / "block_regmeanpp_manifest.json").resolve():
        raise ValueError("pass B must name its own accepted A parent manifest")
    return pass_index


def validate_trace(manifest: dict, dense_path: Path, suite: str,
                   pass_index: int) -> None:
    if (pass_index not in POOLS or suite not in SUITES
            or manifest.get("task") != suite
            or manifest.get("repeat_id") != POOLS[pass_index]
            or Path(manifest.get("calibration_policy", "")).resolve() != dense_path.resolve()
            or manifest.get("source_kind") != "expert_execution"
            or manifest.get("table3_capture_version") != 1):
        raise ValueError("appendix trace source identity differs")
    samples = manifest.get("samples") or []
    if manifest.get("sample_count") != 150 or len(samples) != 150:
        raise ValueError("appendix trace must contain 150 flow states")
    grouped: dict[tuple[int, int], list[int]] = {}
    for sample in samples:
        if sample.get("vision_count") != 3:
            raise ValueError("appendix trace lacks the three-camera prefix")
        grouped.setdefault((sample.get("prompt_signature"),
                            sample.get("selected_request_slot")), []).append(
                                sample.get("flow_index"))
    if len(grouped) != 50 or any(sorted(values) != [0, 5, 9]
                                 for values in grouped.values()):
        raise ValueError("appendix request/flow allocation differs")
    if set(Counter(key[0] for key in grouped).values()) != {5}:
        raise ValueError("appendix task/request allocation differs")


def validate_rows(metrics: dict, names: list[str], pass_index: int) -> int:
    if pass_index not in POOLS or len(metrics) != 418 or set(names) != set(SUITES):
        raise ValueError("appendix module/expert scope differs")
    total = 0
    for module, row in metrics.items():
        if pass_index == 1:
            per_expert = (150 if module in TIME_MODULES else
                          4500 if ".vision_tower." in module else 1500)
        else:
            per_expert = 50 if module in TIME_MODULES else 800
        if row.get("rows_by_expert") != {name: per_expert for name in names}:
            raise ValueError(f"appendix realized rows differ for {module}")
        if not math.isfinite(row.get("ridge", math.nan)) or row["ridge"] <= 0:
            raise ValueError(f"appendix ridge invalid for {module}")
        total += per_expert * len(names)
    if total != ROWS[pass_index]:
        raise ValueError("appendix realized total differs")
    if pass_index == 2 and total != realized_budget(set(metrics), "base"):
        raise ValueError("appendix B budget accounting differs")
    return total
