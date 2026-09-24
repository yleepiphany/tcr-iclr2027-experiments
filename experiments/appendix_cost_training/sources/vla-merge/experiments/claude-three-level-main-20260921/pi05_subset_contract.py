"""Opt-in fixed-per-expert two-pass TCR contract for all 2/3/4 subsets.

This is a fresh A_new -> B_new study. The historical Table-1 A replay tensors
are absent, so these builds must never be labeled historical Full repeats.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from pi05_tcr_e_dense_contract import NAMES, validate_dense_bank as validate_full_bank

SCHEMA = "pi05_fixed_per_expert_subset_tcr_v1"
POOL_NAMES = {1: "A_new", 2: "B_new"}
ROWS_PER_EXPERT = 332_900
CAP = 16
SUITES = tuple(NAMES)


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def validate_names(names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(names)
    if len(names) not in (2, 3, 4) or len(set(names)) != len(names):
        raise ValueError("Subset requires two, three or four distinct experts")
    if names != tuple(name for name in SUITES if name in names):
        raise ValueError("Subset must preserve frozen expert order")
    return names


def validate_config(config: dict) -> tuple[tuple[str, ...], int]:
    if config.get("schema") != SCHEMA or config.get("variant") != "full":
        raise ValueError("Not the frozen subset Full TCR config")
    names = validate_names(config.get("experts", []))
    pass_index = config.get("pass_index")
    if pass_index not in (1, 2):
        raise ValueError("Invalid subset pass")
    expected = {
        "row_cap_per_request_module": CAP,
        "expected_realized_rows": ROWS_PER_EXPERT * len(names),
        "mass_rule": "relative_prior_error" if pass_index == 1 else "uniform",
        "expert_loss_normalization": "prior" if pass_index == 1 else "none",
        "replay_prefix": "merged", "module_count": 418,
        "pool": POOL_NAMES[pass_index],
        "historical_table1_full": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Subset contract differs at {key}")
    start = Path(config.get("start_point", ""))
    if not (start / "model.safetensors").is_file() or \
            sha(start / "model.safetensors") != config.get("start_point_sha256"):
        raise ValueError("Subset prior checkpoint identity differs")
    ridge = config.get("ridge_source_manifest")
    if pass_index == 1 and ridge is not None:
        raise ValueError("Pass A must solve its own numerical ridges")
    if pass_index == 2:
        if not ridge or Path(ridge) != start / "block_regmeanpp_manifest.json":
            raise ValueError("Pass B must reuse this subset's own A manifest")
        if sha(Path(ridge)) != config.get("ridge_source_sha256"):
            raise ValueError("Subset A ridge manifest changed")
        manifest = json.loads(Path(ridge).read_text())
        if (manifest.get("subset_experts") != list(names)
                or manifest.get("subset_pass") != 1
                or len(manifest.get("modules", {})) != 418):
            raise ValueError("Pass B ridge manifest is not this subset's A")
        ridges = [m.get("ridge") for m in manifest["modules"].values()]
        if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                   for x in ridges):
            raise ValueError("Pass A has invalid numeric ridges")
    return names, pass_index


def validate_subset_bank(path: Path, experts: dict[str, Path], *,
                         verify_weights: bool = True) -> dict:
    names = validate_names(list(experts))
    full = validate_full_bank(path, None, verify_weights=False)
    rows = json.loads(Path(path).read_text())["experts"]
    by_name = {row["name"]: row for row in rows}
    selected = {}
    for name in names:
        row = by_name[name]
        if Path(experts[name]).resolve() != Path(row["source_adapter"]["path"]).resolve():
            raise ValueError(f"{name}: adapter differs from four-expert bank")
        model = Path(full[name]["path"]) / "model.safetensors"
        if verify_weights and sha(model) != full[name]["model_sha256"]:
            raise ValueError(f"{name}: dense checkpoint digest differs")
        selected[name] = full[name]
    return selected


def validate_trace(manifest: dict, dense_path: Path, name: str,
                   pass_index: int) -> None:
    if (manifest.get("task") != name or
            Path(manifest.get("calibration_policy", "")).resolve() != Path(dense_path).resolve()):
        raise ValueError("Subset trace belongs to another expert")
    samples = manifest.get("samples") or []
    if manifest.get("sample_count") != 150 or len(samples) != 150:
        raise ValueError("Subset trace must have 150 states")
    grouped = {}
    for sample in samples:
        if sample.get("vision_count") != 3:
            raise ValueError("Subset trace lacks all three cameras")
        grouped.setdefault((sample.get("prompt_signature"),
                            sample.get("selected_request_slot")), []).append(
                                sample.get("flow_index"))
    if len(grouped) != 50 or any(sorted(v) != [0, 5, 9] for v in grouped.values()):
        raise ValueError("Subset trace request/flow allocation differs")
    if set(Counter(key[0] for key in grouped).values()) != {5}:
        raise ValueError("Subset trace does not have five requests per task")
    if (manifest.get("table3_capture_version") != 1
            or manifest.get("source_kind") != "expert_execution"
            or manifest.get("repeat_id") != POOL_NAMES[pass_index]):
        raise ValueError("Subset trace pool provenance differs")


def validate_rows(metrics: dict, names: tuple[str, ...]) -> int:
    if len(metrics) != 418:
        raise ValueError("Subset solve must cover 418 modules")
    total = 0
    for module, entry in metrics.items():
        rows = 50 if module in ("model.time_mlp_in", "model.time_mlp_out") else 800
        if entry.get("rows_by_expert") != {name: rows for name in names}:
            raise ValueError(f"{module}: subset realized rows differ")
        ridge = entry.get("ridge")
        if not isinstance(ridge, (float, int)) or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError(f"{module}: invalid ridge")
        total += rows * len(names)
    if total != ROWS_PER_EXPERT * len(names):
        raise ValueError("Subset total row budget differs")
    return total
