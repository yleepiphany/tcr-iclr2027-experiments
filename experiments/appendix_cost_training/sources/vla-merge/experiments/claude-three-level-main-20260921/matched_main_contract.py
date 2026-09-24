"""Fresh matched Full/last-call/expert-prefix main-ablation contract.

All three arms use the same new expert-execution pools and the largest common
per-request budget supported by the last-call arm.  These models are an
independent matched diagnostic batch; they do not replace the historical Full
headline score.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


WORK = Path(__file__).resolve().parents[3]
RUNTIME = WORK / "vla-merge-runtime/experiments"
SCHEMA = "three_level_matched_refresh_v1"
REPEATS = (1, 2, 3)
VARIANTS = ("full", "last", "expert_prefix")
SUITES = ("spatial", "object", "goal", "long")
AUTHORISED_GPUS = tuple(range(8))
RESERVE_MIB = 12 * 1024
FORBIDDEN_EVAL_ENV = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                      "PI05_LIBERO_INIT_STATE_OFFSET", "PI05_LIBERO_INIT_STATE_COUNT")
POOL_NAMES = {
    1: {1: "A_new", 2: "D_new"},
    2: {1: "B_new", 2: "E_new"},
    3: {1: "C_new", 2: "F_new"},
}
EXISTING_POOLS = RUNTIME / "claude-cache-sequence-budget-20260921/pools-attempt-01"
NEW_POOLS = RUNTIME / "claude-three-level-main-20260921/matched-refresh-pools-attempt-01"
SOUP = (RUNTIME / "iclr2027-table1-20260910/libero/model-soups/repeat-shared"
        / "merge/attempt-02-peft-safe-v2/pretrained_model")
SOUP_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
BANK = RUNTIME / "iclr2027-table1-20260910/expert-dense-bank-peft-v2.json"
BANK_SHA = "d61a5f9e56bb0f76d8186a33dc26fe283cf8cfab220e903a460f92e4507f209b"
PASS = {
    1: {"row_cap": 16, "expected_rows": 1_331_600,
        "mass_rule": "relative_prior_error", "normalization": "prior"},
    2: {"row_cap": 16, "expected_rows": 1_331_600,
        "mass_rule": "uniform_quarter", "normalization": "none"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pool_root(repeat: int, pass_index: int) -> Path:
    name = POOL_NAMES[repeat][pass_index]
    return (EXISTING_POOLS if name in {"A_new", "B_new", "C_new"} else NEW_POOLS) / name / "inputs"


def validate_config(config: dict) -> tuple[int, int, str]:
    if config.get("schema") != SCHEMA:
        raise ValueError("wrong matched-ablation schema")
    repeat, pass_index, variant = config.get("repeat"), config.get("pass_index"), config.get("variant")
    if repeat not in REPEATS or pass_index not in PASS or variant not in VARIANTS:
        raise ValueError("invalid matched-ablation identity")
    recipe = PASS[pass_index]
    expected = {
        "arm": variant,
        "row_cap_per_request_module": recipe["row_cap"],
        "expected_realized_rows": recipe["expected_rows"],
        "mass_rule": recipe["mass_rule"],
        "expert_loss_normalization": recipe["normalization"],
        "replay_prefix": "expert" if variant == "expert_prefix" else "merged",
        "module_count": 418,
        "dense_expert_bank_sha256": BANK_SHA,
        "input_pool": POOL_NAMES[repeat][pass_index],
        "matched_budget_role": "fresh_common_budget_diagnostic",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} differs from matched contract")
    start = Path(config.get("start_point", "")).resolve()
    if pass_index == 1 and start != SOUP.resolve():
        raise ValueError("pass A must start from Soup")
    if not (start / "model.safetensors").is_file() or sha256(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("start checkpoint identity differs")
    full_manifest = config.get("full_manifest")
    if variant == "full":
        if full_manifest is not None:
            raise ValueError("matched Full cannot consume its own future manifest")
    elif not full_manifest:
        raise ValueError("matched variant requires matching Full numerical manifest")
    return repeat, pass_index, variant


def validate_trace(manifest: dict, dense_path: Path, suite: str, repeat: int, pass_index: int) -> None:
    if manifest.get("task") != suite or Path(manifest.get("calibration_policy", "")).resolve() != Path(dense_path).resolve():
        raise ValueError("wrong expert trace")
    samples = manifest.get("samples") or []
    if manifest.get("sample_count") != 150 or len(samples) != 150:
        raise ValueError("matched trace must contain 150 states")
    grouped = {}
    for sample in samples:
        if sample.get("vision_count") != 3:
            raise ValueError("matched trace lacks three-camera prefix")
        grouped.setdefault((sample.get("prompt_signature"), sample.get("selected_request_slot")), []).append(
            sample.get("flow_index"))
    if len(grouped) != 50 or any(sorted(value) != [0, 5, 9] for value in grouped.values()):
        raise ValueError("request/flow allocation differs")
    if set(Counter(key[0] for key in grouped).values()) != {5}:
        raise ValueError("task/request allocation differs")
    if (manifest.get("table3_capture_version") != 1 or manifest.get("source_kind") != "expert_execution"
            or manifest.get("repeat_id") != POOL_NAMES[repeat][pass_index]):
        raise ValueError("trace provenance differs from the registered matched pool")


def validate_rows(metrics: dict, names: list[str]) -> int:
    if len(metrics) != 418:
        raise ValueError("matched build must solve all 418 modules")
    total = 0
    for key, value in metrics.items():
        rows = 50 if key in ("model.time_mlp_in", "model.time_mlp_out") else 800
        if value.get("rows_by_expert") != {name: rows for name in names}:
            raise ValueError(f"row mismatch for {key}")
        if not math.isfinite(value.get("ridge", float("nan"))) or value["ridge"] <= 0:
            raise ValueError(f"invalid ridge for {key}")
        total += rows * len(names)
    if total != 1_331_600:
        raise ValueError("matched total row budget differs")
    return total
