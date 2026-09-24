"""Strict M=3 recipe/identity contract for RoboTwin two-pass TCR."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


NAMES = ("coordination", "receptacle", "precision")
COMMON = {
    "ridge_ratio": 0.05,
    "ridge_scale": "feature_energy",
    "max_correction_ratio": 3.0,
    "expert_loss_normalization_power": 1.0,
    "expert_aggregation": "mean",
    "objective_grouping": "expert",
    "replay_prefix": "merged",
    "action_block_solver": "independent_linear",
    "require_objective_improvement": False,
    "freeze_prefix_from_prior": False,
    "freeze_action_interface_from_prior": False,
    "max_states_per_calibration_source": 0,
    "expert_weight_plan": None,
}


def digest(path: Path | str) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pass() -> str:
    value = os.environ.get("ROBOTWIN_TCR_PASS")
    if value not in {"A", "B"}:
        raise ValueError("ROBOTWIN_TCR_PASS must be A or B")
    return value


def validate_recipe(args: Any) -> None:
    which = _pass()
    expected = {**COMMON,
                "max_rows_per_sample": 10 if which == "A" else 16,
                "expert_loss_normalization": "prior" if which == "A" else "none"}
    for key, value in expected.items():
        if getattr(args, key) != value:
            raise ValueError(f"RoboTwin pass {which} recipe drift: {key} must be {value!r}")
    if args.prior_model is None:
        raise ValueError(f"RoboTwin pass {which} requires an immutable dense prior")


def validate_dense_bank(path: Path, experts=None, *, verify_weights=False):
    bank = json.loads(Path(path).read_text())
    if (bank.get("kind") != "robotwin_three_expert_peft_safe_dense_bank"
            or bank.get("status") != "passed_parameter_exact"):
        raise ValueError("Not the accepted RoboTwin dense expert bank")
    if bank.get("deployment_policy") != "dense_only_no_unmerged_adapter_fallback":
        raise ValueError("Dense-only expert semantics required")
    rows = bank.get("experts") or []
    if [row.get("name") for row in rows] != list(NAMES):
        raise ValueError("RoboTwin expert order or membership differs")
    result = {}
    for row in rows:
        name = row["name"]
        source = row["source_adapter"]
        dense = row["dense_checkpoint"]
        if experts is not None and Path(experts[name]).resolve() != Path(source["path"]).resolve():
            raise ValueError(f"{name}: adapter provenance differs")
        if digest(Path(source["path"]) / "adapter_model.safetensors") != source["model_sha256"]:
            raise ValueError(f"{name}: adapter hash differs")
        manifest = dense["manifest"]
        if digest(manifest["path"]) != manifest["sha256"]:
            raise ValueError(f"{name}: dense manifest hash differs")
        model = Path(dense["path"]) / "model.safetensors"
        if not model.is_file() or (verify_weights and digest(model) != dense["model_sha256"]):
            raise ValueError(f"{name}: dense model identity differs")
        result[name] = {"path": dense["path"], "model_sha256": dense["model_sha256"],
                        "source_adapter": source["path"]}
    return result


def validate_trace(manifest: dict[str, Any], dense: Any, name: str) -> None:
    # `dense` includes source_adapter when supplied by this contract.  The
    # capture correctly ran the native PEFT expert, while solve targets use its
    # independently exact dense equivalent.
    if isinstance(dense, dict):
        expected_policy = dense.get("source_adapter")
    else:
        dense_manifest = json.loads(
            (Path(dense) / "dense_equivalent_manifest.json").read_text()
        )
        expected_policy = dense_manifest.get("inputs", {}).get("adapter")
    if manifest.get("task") != name or manifest.get("group") != name:
        raise ValueError("Trace expert identity differs")
    if expected_policy is None or Path(manifest["calibration_policy"]).resolve() != Path(expected_policy).resolve():
        raise ValueError("Trace does not identify the frozen source expert")
    required = {
        "sample_count": 150, "prompt_count": 10, "requests_per_episode": 5,
        "episode_aware": True, "flow_indices": [0, 5, 9], "request_mode": "initial",
        "method": "pi05_full_vision_language_action_block_regmeanpp_replay_calibration",
        "source_kind": "physical_simulator_expert_execution", "success_filtering": False,
        "demonstration_actions_used": False, "closed_loop_expert_execution": True,
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"Incomplete or mismatched RoboTwin trace: {key}")
    samples = manifest.get("samples") or []
    if len(samples) != 150 or set((manifest.get("prompt_sample_counts") or {}).values()) != {15}:
        raise ValueError("RoboTwin trace must have 15 rows per task")
    grouped = {}
    identities = set()
    for sample in samples:
        key = (sample["prompt_signature"], sample["selected_request_slot"])
        grouped.setdefault(key, []).append(sample["flow_index"])
        identities.add((sample["task"], sample["task_index"], sample["seed"]))
        if sample.get("vision_count") != 3:
            raise ValueError("RoboTwin trace lacks its three-camera prefix")
    if len(grouped) != 50 or any(sorted(value) != [0, 5, 9] for value in grouped.values()):
        raise ValueError("RoboTwin request/flow groups differ")
    if set(Counter(key[0] for key in grouped).values()) != {5} or len(identities) != 10:
        raise ValueError("RoboTwin task/request allocation differs")


def validate_realized_rows(metrics: dict[str, Any]) -> int:
    which = _pass()
    cap = 10 if which == "A" else 16
    counts = Counter()
    total = 0
    for module, entry in metrics.items():
        if module in ("model.time_mlp_in", "model.time_mlp_out"):
            expected = 150
        elif ".vision_tower." in module:
            expected = 150 * 3 * cap
        else:
            expected = 150 * cap
        expected_by_expert = {name: expected for name in NAMES}
        if entry.get("rows_by_expert") != expected_by_expert:
            raise ValueError(f"Per-module row contract differs: {module}")
        counts[expected] += 1
        total += len(NAMES) * expected
    if counts != Counter({150: 2, 150 * cap: 254, 450 * cap: 162}):
        raise ValueError(f"RoboTwin full-scope module classes differ: {dict(counts)}")
    return total


def validate_second_round_config(config: dict[str, Any]) -> None:
    required = {
        "schema": "claude_second_round_v1", "arm": "robotwin_b",
        "row_cap_per_request_module": 16,
        "expert_masses": "uniform_three", "merged_slots": [],
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"RoboTwin pass-B config differs at {key}")
    if type(config.get("repeat")) is not int or config["repeat"] not in (1, 2, 3):
        raise ValueError("RoboTwin pass-B requires one of the three frozen repeats")
    for key in ("start_point", "start_point_manifest", "start_point_sha256"):
        if not config.get(key):
            raise ValueError(f"RoboTwin pass-B config requires {key}")


def validate_second_round_trace(manifest: dict[str, Any], dense: Any, start_point: Any,
                                name: str, arm: str, merged_slots=()) -> dict[str, int]:
    if arm != "robotwin_b" or tuple(merged_slots):
        raise ValueError("RoboTwin pass B must use independent expert-execution pool B")
    validate_trace(manifest, dense, name)
    if manifest.get("pool") != "B":
        raise ValueError("RoboTwin pass B trace is not pool B")
    return {"expert_slots": 150, "merged_slots": 0}
