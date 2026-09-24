"""Frozen contract for the feasible branch of the three-level main ablation.

The paper's Full row is the completed two-pass model.  The demo replacement is made
in both passes.  Pass A is matched to each Full repeat and Pass B uses the same shared
calibration acquisition as Full.  The original pass-A replay tensors were deleted, so
the other two replacements cannot be reconstructed from a different acquisition.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
RUNTIME = WORK / "vla-merge-runtime/experiments"

SCHEMA = "claude_three_level_demo_twopass_v1"
REPEATS = (1, 2, 3)
SUITES = ("spatial", "object", "goal", "long")
AUTHORISED_GPUS = (1, 2, 4, 5, 6, 7)
RESERVE_MIB = 12 * 1024
FORBIDDEN_EVAL_ENV = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                      "PI05_LIBERO_INIT_STATE_OFFSET",
                      "PI05_LIBERO_INIT_STATE_COUNT")
START_SEEDS = {1: 271001, 2: 271002, 3: 271003}
FLOW_SEEDS = {1: 272001, 2: 272002, 3: 272003}
INIT_STATE_OFFSETS = {1: 0, 2: 1, 3: 2}
DEMO_EPISODE_RANK = {1: 0, 2: 1, 3: 2}

SOUP = (RUNTIME / "iclr2027-table1-20260910/libero/model-soups/repeat-shared"
        / "merge/attempt-02-peft-safe-v2/pretrained_model")
SOUP_SHA = "a92aacc43146dc41663c0057f2999bd90252fb3fb316ef963f165be67e1be01a"
BANK = RUNTIME / "iclr2027-table1-20260910/expert-dense-bank-peft-v2.json"
BANK_SHA = "d61a5f9e56bb0f76d8186a33dc26fe283cf8cfab220e903a460f92e4507f209b"

FULL_A_MANIFEST = {
    repeat: RUNTIME / ("iclr2027-table1-20260910/libero/tcr-e/"
                       f"repeat-{repeat:02d}/merge/attempt-01-peft-safe-v1/"
                       "block_regmeanpp_manifest.json")
    for repeat in REPEATS
}
FULL_B_CACHE = RUNTIME / "table3-ablation-20260916/inputs"
FULL_MODELS = {
    1: RUNTIME / "claude-tcr-new-methods-20260917/arms/checkpoints/r01",
    2: RUNTIME / "claude-tcr-new-methods-20260917/arms/checkpoints/r02",
    3: RUNTIME / "claude-tcr-new-methods-20260917/arms/checkpoints/c_e",
}
FULL_MODEL_SHA = {
    1: "4562193825dc9b23e834eb32bfc57242501c7abc3f6c561eeaeee1b5b6f42da8",
    2: "cc3185acd526f9fd70a0640a9932991e21b0a3f32e0ff994a4fb3fc5c250602c",
    3: "3e3a55ce0acd04b3ee1e180eda2a6109b0ef6d0ea4e717f6052a27829adfce95",
}

PASS = {
    1: {"row_cap": 10, "expected_rows": 4_441_200,
        "mass_rule": "relative_prior_error", "normalization": "prior"},
    2: {"row_cap": 16, "expected_rows": 1_331_600,
        "mass_rule": "uniform_quarter", "normalization": "none"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def owner_of(repeat: int, stage: str = "demo_build") -> str:
    if repeat not in REPEATS:
        raise ValueError(f"unregistered repeat {repeat}")
    return "claude-remote"


def validate_config(config: dict) -> tuple[int, int]:
    if config.get("schema") != SCHEMA or config.get("arm") != "demo":
        raise ValueError("not the frozen demo two-pass main-ablation config")
    repeat, pass_index = config.get("repeat"), config.get("pass_index")
    if repeat not in REPEATS or pass_index not in PASS:
        raise ValueError("invalid repeat/pass")
    recipe = PASS[pass_index]
    for key, want in (
        ("row_cap_per_request_module", recipe["row_cap"]),
        ("expected_realized_rows", recipe["expected_rows"]),
        ("mass_rule", recipe["mass_rule"]),
        ("expert_loss_normalization", recipe["normalization"]),
        ("replay_prefix", "merged"),
        ("module_count", 418),
        ("dense_expert_bank_sha256", BANK_SHA),
    ):
        if config.get(key) != want:
            raise ValueError(f"{key}={config.get(key)!r}, expected {want!r}")
    ridge = Path(config.get("ridge_source_manifest", "")).resolve()
    if ridge != FULL_A_MANIFEST[repeat].resolve():
        raise ValueError("numeric ridges must come from the matching Full pass-A manifest")
    if sha256(ridge) != config.get("ridge_source_sha256"):
        raise ValueError("ridge-source manifest hash drift")
    if pass_index == 1:
        if Path(config.get("start_point", "")).resolve() != SOUP.resolve():
            raise ValueError("pass A must start from the frozen Soup")
        if config.get("start_point_sha256") != SOUP_SHA:
            raise ValueError("Soup identity drift")
    else:
        if not config.get("start_point") or not config.get("start_point_sha256"):
            raise ValueError("pass B must name its own completed pass-A checkpoint")
    return repeat, pass_index


def validate_demo_trace(manifest: dict, dense_path: Path, suite: str,
                        repeat: int, pass_index: int) -> None:
    if manifest.get("task") != suite:
        raise ValueError(f"demo trace task is {manifest.get('task')!r}, expected {suite!r}")
    if Path(manifest.get("calibration_policy", "")).resolve() != Path(dense_path).resolve():
        raise ValueError("demo trace was not generated by the registered dense expert")
    samples = manifest.get("samples") or []
    if manifest.get("sample_count") != 150 or len(samples) != 150:
        raise ValueError("demo trace must contain 10 tasks x 5 slots x 3 flows")
    grouped = {}
    for sample in samples:
        if sample.get("vision_count") != 3:
            raise ValueError("demo trace is missing a three-camera prefix")
        grouped.setdefault((sample.get("prompt_signature"),
                            sample.get("selected_request_slot")), []).append(
                                sample.get("flow_index"))
    if len(grouped) != 50 or any(sorted(v) != [0, 5, 9] for v in grouped.values()):
        raise ValueError("demo trace request/flow structure differs")
    if set(Counter(key[0] for key in grouped).values()) != {5}:
        raise ValueError("demo trace does not contain five requests per task")
    if manifest.get("table3_capture_version") != 1:
        raise ValueError("demo trace predates the matched-input cache contract")
    if manifest.get("demonstration_actions_used") is not False:
        raise ValueError("demonstration actions must never be regression targets")
    if pass_index == 1:
        if manifest.get("source_kind") != "demo_observation_expert_generation":
            raise ValueError("pass-A demo trace lacks native expert-generation provenance")
        if manifest.get("repeat") != repeat:
            raise ValueError("pass-A demo trace belongs to another repeat")
        pairing = manifest.get("paired_noise_verified") or {}
        if pairing != {"flow0_pairs_checked": 50, "flow0_equal": 50,
                       "max_abs_diff": 0.0}:
            raise ValueError("pass-A demo trace does not prove tensor-level noise pairing")
        if manifest.get("demo_episode_rank") != DEMO_EPISODE_RANK[repeat]:
            raise ValueError("pass-A demo trace uses the wrong demonstration rank")
    else:
        if manifest.get("source_kind") != "demonstration_observations":
            raise ValueError("pass-B demo trace is not the frozen shared replacement")
        expected = (FULL_B_CACHE / suite / "across").resolve()
        if Path(manifest.get("paired_noise_template", "")).resolve() != expected:
            raise ValueError("pass-B demo trace is not paired to Full's shared B cache")


def validate_rows(metrics: dict, names: list[str], pass_index: int) -> int:
    if len(metrics) != 418:
        raise ValueError(f"solver produced {len(metrics)} modules, expected 418")
    total = 0
    for module, entry in metrics.items():
        if pass_index == 1:
            if module in ("model.time_mlp_in", "model.time_mlp_out"):
                rows = 150
            elif ".vision_tower." in module:
                rows = 4500
            else:
                rows = 1500
        else:
            rows = 50 if module in ("model.time_mlp_in", "model.time_mlp_out") else 800
        expected = {name: rows for name in names}
        if entry.get("rows_by_expert") != expected:
            raise ValueError(f"row mismatch for {module}: {entry.get('rows_by_expert')}")
        if not math.isfinite(entry.get("ridge", float("nan"))) or entry["ridge"] <= 0:
            raise ValueError(f"invalid ridge for {module}")
        total += rows * len(names)
    if total != PASS[pass_index]["expected_rows"]:
        raise ValueError(f"row total {total} differs from frozen pass-{pass_index} budget")
    return total
