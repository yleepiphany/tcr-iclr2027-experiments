"""Fail-closed planning, shard identity, and resume logic for PI0.5 FeatCal caches."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import Tensor

from .featcal_forward_order import FeatCalForwardStep


class FeatCalCachePlanError(ValueError):
    """Raised when a formal-cache plan, shard, or resume invariant differs."""


EXPERTS = ("spatial", "object", "goal", "long")
ROLES = ("teacher", "student")
TASKS_PER_EXPERT = 10
EPISODES_PER_TASK = 1
REQUESTS_PER_EPISODE = 5
FLOW_INDICES = (0, 5, 9)
CALLS_PER_TASK = EPISODES_PER_TASK * REQUESTS_PER_EPISODE * len(FLOW_INDICES)
CALLS_PER_EXPERT = TASKS_PER_EXPERT * CALLS_PER_TASK
ROW_CAP = 10
EXPECTED_MODULE_COUNTS = {150: 2, 1500: 254, 4500: 162}
EXPECTED_ROWS_PER_EXPERT = 1_110_300
EXPECTED_TOTAL_ROWS = 4_441_200
ROW_KEY_COLUMNS = (
    "call_ordinal_within_task",
    "module_occurrence",
    "source_row_index",
    "selection_rank",
)
CALL_STATE_HASH_FIELDS = (
    "processed_images_and_masks_sha256",
    "language_tokens_and_mask_sha256",
    "action_chunk_sha256",
    "padded_normalized_action_sha256",
    "noise_sha256",
    "x_t_sha256",
    "timestep_sha256",
    "source_expert_sha256",
)


@dataclass(frozen=True)
class FeatCalModuleQuota:
    module_path: str
    step: int
    layer_id: str
    input_width: int
    module_class: str
    occurrences_per_call: int
    rows_per_occurrence: int
    rows_per_call: int
    rows_per_task_shard: int
    rows_per_expert: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "step": self.step,
            "layer_id": self.layer_id,
            "input_width": self.input_width,
            "module_class": self.module_class,
            "occurrences_per_call": self.occurrences_per_call,
            "rows_per_occurrence": self.rows_per_occurrence,
            "rows_per_call": self.rows_per_call,
            "rows_per_task_shard": self.rows_per_task_shard,
            "rows_per_expert": self.rows_per_expert,
        }


@dataclass(frozen=True)
class StreamingBatchCapture:
    trace: tuple[str, ...]
    inputs_by_module: dict[str, Tensor]
    row_keys_by_module: dict[str, Tensor]
    selected_bytes: int


def expected_full_forward_trace(
    forward_plan: Sequence[FeatCalForwardStep],
) -> tuple[str, ...]:
    vision = [step for step in forward_plan if step.stage == "vision_prefix"]
    remainder = [step for step in forward_plan if step.stage != "vision_prefix"]
    if not vision or any(step.repetitions_per_forward != 3 for step in vision):
        raise FeatCalCachePlanError("Full-forward vision plan differs")
    if any(step.repetitions_per_forward != 1 for step in remainder):
        raise FeatCalCachePlanError("Full-forward non-vision repetitions differ")
    trace = []
    for _camera in range(3):
        for step in vision:
            trace.extend(step.target_module_paths)
    for step in remainder:
        trace.extend(step.target_module_paths)
    return tuple(trace)


class MultiLayerStreamingRowCollector:
    """Hook every target once, retaining only deterministic capped rows on CPU."""

    def __init__(
        self,
        root: torch.nn.Module,
        forward_plan: Sequence[FeatCalForwardStep],
        quotas: Sequence[FeatCalModuleQuota],
        *,
        call_slot_ids: Sequence[str],
        call_ordinals_within_task: Sequence[int],
        row_mask_provider: Any,
        selection_seed: int,
        max_selected_bytes: int = 2 * 1024**3,
    ) -> None:
        if root.training:
            raise FeatCalCachePlanError("Streaming FeatCal collection requires eval mode")
        if (
            not call_slot_ids
            or len(call_slot_ids) != len(call_ordinals_within_task)
            or len(set(call_ordinals_within_task)) != len(call_ordinals_within_task)
        ):
            raise FeatCalCachePlanError("Streaming call-slot batch identity differs")
        if not callable(row_mask_provider) or max_selected_bytes < 1:
            raise FeatCalCachePlanError("Streaming mask provider/cap is invalid")
        modules = dict(root.named_modules())
        quota_by_module = {quota.module_path: quota for quota in quotas}
        expected_modules = {path for step in forward_plan for path in step.target_module_paths}
        if set(quota_by_module) != expected_modules:
            raise FeatCalCachePlanError("Streaming quota and forward target scopes differ")
        selected_modules = {}
        for path in expected_modules:
            module = modules.get(path)
            if not isinstance(module, torch.nn.Linear):
                raise FeatCalCachePlanError(f"Streaming target is not Linear: {path}")
            selected_modules[path] = module
        self.root = root
        self.forward_plan = tuple(forward_plan)
        self.expected_trace = expected_full_forward_trace(forward_plan)
        self.quotas = quota_by_module
        self.modules = selected_modules
        self.call_slot_ids = tuple(call_slot_ids)
        self.call_ordinals = tuple(int(value) for value in call_ordinals_within_task)
        self.row_mask_provider = row_mask_provider
        self.selection_seed = int(selection_seed)
        self.max_selected_bytes = int(max_selected_bytes)

    def capture(self, forward: Any) -> StreamingBatchCapture:
        if self.root.training:
            raise FeatCalCachePlanError("Streaming collector root entered training mode")
        trace = []
        occurrences: dict[str, int] = Counter()
        input_parts: dict[str, list[Tensor]] = {path: [] for path in self.modules}
        row_key_parts: dict[str, list[Tensor]] = {path: [] for path in self.modules}
        selected_bytes = 0
        handles = []

        def make_hook(path: str):
            def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                nonlocal selected_bytes
                trace_index = len(trace)
                expected = (
                    self.expected_trace[trace_index]
                    if trace_index < len(self.expected_trace)
                    else None
                )
                if expected != path:
                    raise FeatCalCachePlanError(
                        f"Streaming full-forward order differs at {trace_index}: {path} != {expected}"
                    )
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise FeatCalCachePlanError(f"Streaming target input is missing: {path}")
                value = inputs[0].detach()
                if (
                    value.ndim < 2
                    or value.shape[0] != len(self.call_slot_ids)
                    or value.shape[-1] != module.in_features
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise FeatCalCachePlanError(f"Streaming target input contract differs: {path}")
                occurrence = occurrences[path]
                quota = self.quotas[path]
                if occurrence >= quota.occurrences_per_call:
                    raise FeatCalCachePlanError(f"Streaming target occurrence overflow: {path}")
                mask = torch.as_tensor(
                    self.row_mask_provider(path, occurrence, value),
                    dtype=torch.bool,
                    device="cpu",
                )
                if tuple(mask.shape) != tuple(value.shape[:-1]):
                    raise FeatCalCachePlanError(f"Streaming row mask shape differs: {path}")
                for batch_index, slot_id in enumerate(self.call_slot_ids):
                    flat_value = value[batch_index].reshape(-1, value.shape[-1])
                    selected = deterministic_row_selection(
                        mask[batch_index].reshape(-1),
                        cap=quota.rows_per_occurrence,
                        identity=(
                            f"seed={self.selection_seed}/{slot_id}/{path}/occurrence={occurrence}"
                        ),
                    )
                    chosen = (
                        flat_value[selected.to(flat_value.device)]
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                    keys = torch.tensor(
                        [
                            [
                                self.call_ordinals[batch_index],
                                occurrence,
                                int(source_row),
                                rank,
                            ]
                            for rank, source_row in enumerate(selected.tolist())
                        ],
                        dtype=torch.int64,
                    )
                    selected_bytes += (
                        chosen.numel() * chosen.element_size()
                        + keys.numel() * keys.element_size()
                    )
                    if selected_bytes > self.max_selected_bytes:
                        raise MemoryError("Streaming selected rows exceed max_selected_bytes")
                    input_parts[path].append(chosen)
                    row_key_parts[path].append(keys)
                occurrences[path] += 1
                trace.append(path)

            return hook

        for path, module in self.modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(path)))
        try:
            with torch.inference_mode():
                forward()
        finally:
            for handle in handles:
                handle.remove()
        if tuple(trace) != self.expected_trace:
            raise FeatCalCachePlanError("Streaming full-forward trace is incomplete")
        for path, quota in self.quotas.items():
            if occurrences[path] != quota.occurrences_per_call:
                raise FeatCalCachePlanError(f"Streaming target call count differs: {path}")
        return StreamingBatchCapture(
            trace=tuple(trace),
            inputs_by_module={path: torch.cat(parts) for path, parts in input_parts.items()},
            row_keys_by_module={path: torch.cat(parts) for path, parts in row_key_parts.items()},
            selected_bytes=selected_bytes,
        )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + value.view(torch.uint8).numpy().tobytes()).hexdigest()


def build_call_slots() -> list[dict[str, Any]]:
    slots = []
    global_ordinal = 0
    for expert in EXPERTS:
        expert_ordinal = 0
        for task in range(TASKS_PER_EXPERT):
            task_ordinal = 0
            for request in range(REQUESTS_PER_EPISODE):
                for flow_index in FLOW_INDICES:
                    slot_id = (
                        f"expert={expert}/task={task:02d}/episode=00/"
                        f"request={request:02d}/flow={flow_index:02d}"
                    )
                    slots.append(
                        {
                            "slot_id": slot_id,
                            "global_ordinal": global_ordinal,
                            "expert": expert,
                            "expert_ordinal": expert_ordinal,
                            "task": task,
                            "task_ordinal": task_ordinal,
                            "episode_slot": 0,
                            "request": request,
                            "flow_index": flow_index,
                            "concrete_state_status": "pending_call_state_manifest",
                        }
                    )
                    global_ordinal += 1
                    expert_ordinal += 1
                    task_ordinal += 1
    if len(slots) != len(EXPERTS) * CALLS_PER_EXPERT:
        raise AssertionError("Internal FeatCal call-slot cardinality differs")
    return slots


def deterministic_row_selection(
    eligible_mask: Tensor,
    *,
    cap: int,
    identity: str,
) -> Tensor:
    """Select exactly ``cap`` eligible rows by stable SHA256 order."""

    mask = torch.as_tensor(eligible_mask, dtype=torch.bool).reshape(-1).cpu()
    if cap < 1 or not identity:
        raise FeatCalCachePlanError("Row selection cap/identity is invalid")
    eligible = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    if len(eligible) < cap:
        raise FeatCalCachePlanError(
            f"Insufficient eligible rows for exact cap: {len(eligible)} < {cap}"
        )
    ranked = sorted(
        eligible,
        key=lambda row: hashlib.sha256(f"{identity}:row={row}".encode("utf-8")).digest(),
    )
    return torch.tensor(ranked[:cap], dtype=torch.int64)


def validate_call_state_manifest(
    manifest: Mapping[str, Any],
    call_slots: Sequence[Mapping[str, Any]],
    *,
    expert_model_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "pi05_featcal_call_state_bank"
        or manifest.get("status") != "complete"
        or manifest.get("labels_used") is not False
        or manifest.get("final_evaluation_disjoint") is not True
        or manifest.get("forward_interface") != "full_joint_explicit_xt_and_timestep_label_free"
    ):
        raise FeatCalCachePlanError("Call-state manifest header differs")
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != len(call_slots):
        raise FeatCalCachePlanError("Call-state manifest cardinality differs")
    if set(expert_model_sha256) != set(EXPERTS):
        raise FeatCalCachePlanError("Call-state expert model identities differ")

    def is_sha256(value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    for ordinal, (slot, state) in enumerate(zip(call_slots, states, strict=True)):
        if not isinstance(state, dict):
            raise FeatCalCachePlanError("Call-state row is not an object")
        for field in ("slot_id", "expert", "task", "episode_slot", "request", "flow_index"):
            if state.get(field) != slot.get(field):
                raise FeatCalCachePlanError(
                    f"Call-state structural identity differs at ordinal {ordinal}: {field}"
                )
        if not state.get("episode_identity") or state.get("suite_task_id") is None:
            raise FeatCalCachePlanError("Call-state concrete episode/task identity is missing")
        if not isinstance(state.get("noise_seed"), int) or not isinstance(
            state.get("timestep"), float
        ):
            raise FeatCalCachePlanError("Call-state noise/time identity is missing")
        if any(not is_sha256(state.get(field)) for field in CALL_STATE_HASH_FIELDS):
            raise FeatCalCachePlanError("Call-state tensor/source hash is invalid")
        if state["source_expert_sha256"] != expert_model_sha256[state["expert"]]:
            raise FeatCalCachePlanError("Call-state source expert hash differs")
    counts = Counter(state["expert"] for state in states)
    if counts != Counter({expert: CALLS_PER_EXPERT for expert in EXPERTS}):
        raise FeatCalCachePlanError("Call-state per-expert quotas differ")
    return {
        "status": "call_state_manifest_valid",
        "state_count": len(states),
        "states_per_expert": dict(sorted(counts.items())),
        "manifest_sha256": canonical_json_sha256(dict(manifest)),
    }


def _weight_shapes(expert_bank: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    logical = expert_bank.get("adaptation_domain", {}).get("logical_tensors")
    if not isinstance(logical, list) or len(logical) != 422:
        raise FeatCalCachePlanError("Expected 422 logical adapted tensors")
    shapes = {}
    for row in logical:
        if not isinstance(row, dict):
            raise FeatCalCachePlanError("Malformed logical tensor row")
        key = row.get("base_key")
        shape = row.get("base_shape")
        if (
            isinstance(key, str)
            and key.endswith(".weight")
            and isinstance(shape, list)
            and len(shape) == 2
        ):
            shapes[key[: -len(".weight")]] = (int(shape[0]), int(shape[1]))
    if len(shapes) != 418:
        raise FeatCalCachePlanError(f"Expected 418 adapted Linear weights, got {len(shapes)}")
    return shapes


def _quota_class(module_path: str) -> tuple[str, int, int]:
    if ".vision_tower.vision_model.encoder.layers." in module_path:
        return "vision_three_physical_camera_calls", 3, ROW_CAP
    if module_path.endswith((".time_mlp_in", ".time_mlp_out")):
        return "time_condition_one_row", 1, 1
    return "language_action_or_interface", 1, ROW_CAP


def derive_module_quotas(
    expert_bank: Mapping[str, Any],
    forward_plan: Sequence[FeatCalForwardStep],
) -> list[FeatCalModuleQuota]:
    shapes = _weight_shapes(expert_bank)
    plan_targets = {
        path: (step.step, step.layer_id)
        for step in forward_plan
        for path in step.target_module_paths
    }
    if set(plan_targets) != set(shapes):
        missing = sorted(set(shapes) - set(plan_targets))
        extra = sorted(set(plan_targets) - set(shapes))
        raise FeatCalCachePlanError(
            f"Forward plan and expert bank differ: missing={missing[:5]}, extra={extra[:5]}"
        )
    quotas = []
    for module_path in sorted(shapes):
        module_class, occurrences, rows_per_occurrence = _quota_class(module_path)
        rows_per_call = occurrences * rows_per_occurrence
        rows_per_task = CALLS_PER_TASK * rows_per_call
        rows_per_expert = TASKS_PER_EXPERT * rows_per_task
        step, layer_id = plan_targets[module_path]
        quotas.append(
            FeatCalModuleQuota(
                module_path=module_path,
                step=step,
                layer_id=layer_id,
                input_width=shapes[module_path][1],
                module_class=module_class,
                occurrences_per_call=occurrences,
                rows_per_occurrence=rows_per_occurrence,
                rows_per_call=rows_per_call,
                rows_per_task_shard=rows_per_task,
                rows_per_expert=rows_per_expert,
            )
        )
    counts = Counter(quota.rows_per_expert for quota in quotas)
    if dict(counts) != EXPECTED_MODULE_COUNTS:
        raise FeatCalCachePlanError(f"Per-module quota classes differ: {dict(counts)}")
    rows_per_expert = sum(quota.rows_per_expert for quota in quotas)
    if rows_per_expert != EXPECTED_ROWS_PER_EXPERT:
        raise FeatCalCachePlanError("Per-expert row total differs")
    if rows_per_expert * len(EXPERTS) != EXPECTED_TOTAL_ROWS:
        raise FeatCalCachePlanError("Grand row total differs")
    return quotas


def _task_slot_ids(call_slots: Sequence[Mapping[str, Any]], expert: str, task: int) -> list[str]:
    result = [
        str(slot["slot_id"])
        for slot in call_slots
        if slot.get("expert") == expert and slot.get("task") == task
    ]
    if len(result) != CALLS_PER_TASK:
        raise FeatCalCachePlanError(f"Call slots differ for {expert}/task={task}")
    return result


def build_shard_index(
    quotas: Sequence[FeatCalModuleQuota],
    forward_plan: Sequence[FeatCalForwardStep],
    call_slots: Sequence[Mapping[str, Any]],
    *,
    contract_sha256: str,
    expert_model_sha256: Mapping[str, str],
    initial_student_sha256: str,
) -> dict[str, Any]:
    if set(expert_model_sha256) != set(EXPERTS) or not initial_student_sha256:
        raise FeatCalCachePlanError("Shard index model identities differ")
    by_module = {quota.module_path: quota for quota in quotas}
    shards = []
    for role in ROLES:
        for expert in EXPERTS:
            for task in range(TASKS_PER_EXPERT):
                slot_ids = _task_slot_ids(call_slots, expert, task)
                for step in forward_plan:
                    targets = []
                    factor_bytes = 0
                    row_key_bytes = 0
                    for target_index, module_path in enumerate(step.target_module_paths):
                        quota = by_module[module_path]
                        rows = quota.rows_per_task_shard
                        factor_bytes += rows * quota.input_width * 4
                        row_key_bytes += rows * len(ROW_KEY_COLUMNS) * 8
                        targets.append(
                            {
                                "target_index": target_index,
                                "module_path": module_path,
                                "input_width": quota.input_width,
                                "module_class": quota.module_class,
                                "occurrences_per_call": quota.occurrences_per_call,
                                "rows_per_occurrence": quota.rows_per_occurrence,
                                "rows_per_call": quota.rows_per_call,
                                "expected_rows": rows,
                                "input_tensor": f"target_{target_index:03d}.input",
                                "row_keys_tensor": f"target_{target_index:03d}.row_keys",
                            }
                        )
                    stem = f"shards/{role}/{expert}/task_{task:02d}/step_{step.step:02d}"
                    shards.append(
                        {
                            "shard_id": f"{role}/{expert}/task_{task:02d}/step_{step.step:02d}",
                            "role": role,
                            "expert": expert,
                            "task": task,
                            "step": step.step,
                            "layer_id": step.layer_id,
                            "call_slot_ids": slot_ids,
                            "data_file": f"{stem}.safetensors",
                            "manifest_file": f"{stem}.json",
                            "source_checkpoint_sha256": (
                                expert_model_sha256[expert]
                                if role == "teacher"
                                else (
                                    initial_student_sha256
                                    if step.step == 0
                                    else "previous_step_solve_output"
                                )
                            ),
                            "targets": targets,
                            "factor_bytes_float32": factor_bytes,
                            "row_key_bytes_int64": row_key_bytes,
                        }
                    )
    expected_count = len(ROLES) * len(EXPERTS) * TASKS_PER_EXPERT * len(forward_plan)
    if len(shards) != expected_count or len({row["shard_id"] for row in shards}) != expected_count:
        raise AssertionError("Internal FeatCal shard index cardinality differs")
    totals = {
        role: {
            "shards": sum(row["role"] == role for row in shards),
            "factor_bytes_float32": sum(
                row["factor_bytes_float32"] for row in shards if row["role"] == role
            ),
            "row_key_bytes_int64": sum(
                row["row_key_bytes_int64"] for row in shards if row["role"] == role
            ),
        }
        for role in ROLES
    }
    return {
        "schema_version": 1,
        "kind": "pi05_featcal_row_factor_shard_index",
        "contract_sha256": contract_sha256,
        "shard_count": len(shards),
        "roles": list(ROLES),
        "totals": totals,
        "shards": shards,
    }


def validate_row_keys(row_keys: Tensor, target: Mapping[str, Any]) -> None:
    if row_keys.dtype != torch.int64 or tuple(row_keys.shape) != (
        int(target["expected_rows"]),
        len(ROW_KEY_COLUMNS),
    ):
        raise FeatCalCachePlanError("Shard row-key dtype/shape differs")
    rows_per_occurrence = int(target["rows_per_occurrence"])
    occurrences = int(target["occurrences_per_call"])
    expected = Counter(
        (call, occurrence, rank)
        for call in range(CALLS_PER_TASK)
        for occurrence in range(occurrences)
        for rank in range(rows_per_occurrence)
    )
    observed = Counter(
        (int(row[0]), int(row[1]), int(row[3])) for row in row_keys.cpu().tolist()
    )
    if observed != expected:
        raise FeatCalCachePlanError("Shard row-key call/occurrence/rank quota differs")
    if torch.any(row_keys[:, 2] < 0):
        raise FeatCalCachePlanError("Shard source row index is negative")


def validate_completed_shard(
    cache_root: Path,
    spec: Mapping[str, Any],
    *,
    plan_sha256: str,
    call_state_manifest_sha256: str,
    full_tensor_check: bool = True,
) -> dict[str, Any]:
    data_path = cache_root / str(spec["data_file"])
    manifest_path = cache_root / str(spec["manifest_file"])
    if not data_path.is_file() or not manifest_path.is_file():
        raise FeatCalCachePlanError(f"Shard pair is incomplete: {spec['shard_id']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise FeatCalCachePlanError("Shard manifest is not an object")
    identity_fields = ("shard_id", "role", "expert", "task", "step", "layer_id")
    if any(manifest.get(field) != spec.get(field) for field in identity_fields):
        raise FeatCalCachePlanError(f"Shard identity differs: {spec['shard_id']}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "pi05_featcal_row_factor_shard"
        or manifest.get("status") != "complete"
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("call_state_manifest_sha256") != call_state_manifest_sha256
        or manifest.get("call_slot_ids") != spec.get("call_slot_ids")
        or manifest.get("data_file") != spec.get("data_file")
        or manifest.get("data_sha256") != sha256_file(data_path)
    ):
        raise FeatCalCachePlanError(f"Shard manifest contract differs: {spec['shard_id']}")
    if not isinstance(manifest.get("source_checkpoint_sha256"), str):
        raise FeatCalCachePlanError("Shard source checkpoint hash is missing")
    expected_source = spec.get("source_checkpoint_sha256")
    if expected_source == "previous_step_solve_output":
        if manifest.get("source_checkpoint_sha256") != manifest.get(
            "student_prefix_input_sha256"
        ):
            raise FeatCalCachePlanError("Student shard source/prefix hashes differ")
    elif manifest.get("source_checkpoint_sha256") != expected_source:
        raise FeatCalCachePlanError("Shard source checkpoint identity differs")
    if spec["role"] == "student" and not isinstance(
        manifest.get("student_prefix_input_sha256"), str
    ):
        raise FeatCalCachePlanError("Student shard prefix input hash is missing")

    expected_keys = {
        key
        for target in spec["targets"]
        for key in (target["input_tensor"], target["row_keys_tensor"])
    }
    manifest_targets = manifest.get("targets")
    if not isinstance(manifest_targets, list) or len(manifest_targets) != len(spec["targets"]):
        raise FeatCalCachePlanError("Shard target manifest count differs")
    manifest_by_module = {row.get("module_path"): row for row in manifest_targets}
    with safe_open(data_path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected_keys:
            raise FeatCalCachePlanError("Shard safetensors keys differ")
        for target in spec["targets"]:
            input_key = target["input_tensor"]
            row_key = target["row_keys_tensor"]
            input_slice = handle.get_slice(input_key)
            row_slice = handle.get_slice(row_key)
            if input_slice.get_dtype() != "F32" or tuple(input_slice.get_shape()) != (
                int(target["expected_rows"]),
                int(target["input_width"]),
            ):
                raise FeatCalCachePlanError("Shard input factor header differs")
            if row_slice.get_dtype() != "I64" or tuple(row_slice.get_shape()) != (
                int(target["expected_rows"]),
                len(ROW_KEY_COLUMNS),
            ):
                raise FeatCalCachePlanError("Shard row-key header differs")
            row_manifest = manifest_by_module.get(target["module_path"])
            if not isinstance(row_manifest, dict):
                raise FeatCalCachePlanError("Shard target manifest is missing")
            if full_tensor_check:
                inputs = handle.get_tensor(input_key)
                row_keys = handle.get_tensor(row_key)
                if not torch.isfinite(inputs).all():
                    raise FeatCalCachePlanError("Shard input factor is non-finite")
                validate_row_keys(row_keys, target)
                if (
                    row_manifest.get("input_sha256") != tensor_sha256(inputs)
                    or row_manifest.get("row_keys_sha256") != tensor_sha256(row_keys)
                ):
                    raise FeatCalCachePlanError("Shard target tensor hash differs")
    return {
        "shard_id": spec["shard_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "data_sha256": sha256_file(data_path),
        "full_tensor_check": full_tensor_check,
        "student_prefix_input_sha256": manifest.get("student_prefix_input_sha256"),
        "previous_step_solve_receipt": manifest.get("previous_step_solve_receipt"),
        "previous_step_solve_receipt_sha256": manifest.get(
            "previous_step_solve_receipt_sha256"
        ),
    }


def write_completed_shard_atomic(
    cache_root: Path,
    spec: Mapping[str, Any],
    tensors: Mapping[str, Tensor],
    *,
    plan_sha256: str,
    call_state_manifest_sha256: str,
    source_checkpoint_sha256: str,
    student_prefix_input_sha256: str | None = None,
    previous_step_solve_receipt: str | None = None,
    previous_step_solve_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    data_path = cache_root / str(spec["data_file"])
    manifest_path = cache_root / str(spec["manifest_file"])
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing existing shard: {spec['shard_id']}")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    expected_keys = {
        key
        for target in spec["targets"]
        for key in (target["input_tensor"], target["row_keys_tensor"])
    }
    if set(tensors) != expected_keys:
        raise FeatCalCachePlanError("Shard write tensor keys differ")
    target_rows = []
    normalized = {}
    for target in spec["targets"]:
        inputs = tensors[target["input_tensor"]].detach().cpu().to(torch.float32).contiguous()
        row_keys = tensors[target["row_keys_tensor"]].detach().cpu().to(torch.int64).contiguous()
        if tuple(inputs.shape) != (
            int(target["expected_rows"]),
            int(target["input_width"]),
        ) or not torch.isfinite(inputs).all():
            raise FeatCalCachePlanError("Shard write input factor differs")
        validate_row_keys(row_keys, target)
        normalized[target["input_tensor"]] = inputs
        normalized[target["row_keys_tensor"]] = row_keys
        target_rows.append(
            {
                **dict(target),
                "input_sha256": tensor_sha256(inputs),
                "row_keys_sha256": tensor_sha256(row_keys),
            }
        )
    if spec["role"] == "student" and not student_prefix_input_sha256:
        raise FeatCalCachePlanError("Student shard write requires prefix input hash")
    if spec["role"] == "student" and int(spec["step"]) > 0 and not (
        previous_step_solve_receipt and previous_step_solve_receipt_sha256
    ):
        raise FeatCalCachePlanError("Student shard write requires previous solve receipt")

    with tempfile.TemporaryDirectory(prefix=".featcal-shard-", dir=data_path.parent) as tmp:
        temporary = Path(tmp)
        temp_data = temporary / data_path.name
        temp_manifest = temporary / manifest_path.name
        save_file(normalized, temp_data, metadata={"format": "pt"})
        manifest = {
            "schema_version": 1,
            "kind": "pi05_featcal_row_factor_shard",
            "status": "complete",
            **{field: spec[field] for field in ("shard_id", "role", "expert", "task", "step", "layer_id")},
            "plan_sha256": plan_sha256,
            "call_state_manifest_sha256": call_state_manifest_sha256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "student_prefix_input_sha256": student_prefix_input_sha256,
            "previous_step_solve_receipt": previous_step_solve_receipt,
            "previous_step_solve_receipt_sha256": previous_step_solve_receipt_sha256,
            "call_slot_ids": spec["call_slot_ids"],
            "row_key_columns": list(ROW_KEY_COLUMNS),
            "selection": "sha256_ordered_without_replacement_per_call_and_occurrence",
            "storage_dtype": "float32_lossless_for_bfloat16_and_float32_hook_inputs",
            "data_file": spec["data_file"],
            "data_sha256": sha256_file(temp_data),
            "targets": target_rows,
        }
        temp_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for temporary_file in (temp_data, temp_manifest):
            with temporary_file.open("rb") as stream:
                os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses an existing destination;
        # a crash between the two links leaves a detectable half-written pair.
        os.link(temp_data, data_path)
        os.link(temp_manifest, manifest_path)
        directory_fd = os.open(data_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return validate_completed_shard(
        cache_root,
        spec,
        plan_sha256=plan_sha256,
        call_state_manifest_sha256=call_state_manifest_sha256,
        full_tensor_check=True,
    )


def scan_resume_state(
    cache_root: Path,
    shard_index: Mapping[str, Any],
    *,
    plan_sha256: str,
    call_state_manifest_sha256: str,
    initial_student_sha256: str | None = None,
    full_tensor_check: bool = True,
) -> dict[str, Any]:
    completed = []
    pending = []
    for spec in shard_index.get("shards", []):
        data_exists = (cache_root / spec["data_file"]).exists()
        manifest_exists = (cache_root / spec["manifest_file"]).exists()
        if data_exists != manifest_exists:
            raise FeatCalCachePlanError(f"Half-written shard found: {spec['shard_id']}")
        if not data_exists:
            pending.append(spec["shard_id"])
            continue
        validation = validate_completed_shard(
            cache_root,
            spec,
            plan_sha256=plan_sha256,
            call_state_manifest_sha256=call_state_manifest_sha256,
            full_tensor_check=full_tensor_check,
        )
        if spec["role"] == "student":
            prefix_sha = validation["student_prefix_input_sha256"]
            if int(spec["step"]) == 0:
                if not initial_student_sha256 or prefix_sha != initial_student_sha256:
                    raise FeatCalCachePlanError("Student step-0 prefix is not the frozen initializer")
            else:
                receipt_rel = validation["previous_step_solve_receipt"]
                receipt_sha = validation["previous_step_solve_receipt_sha256"]
                if not isinstance(receipt_rel, str) or not isinstance(receipt_sha, str):
                    raise FeatCalCachePlanError("Student previous-step solve receipt is missing")
                solve_path = cache_root / receipt_rel
                if not solve_path.is_file() or sha256_file(solve_path) != receipt_sha:
                    raise FeatCalCachePlanError("Student previous-step solve receipt identity differs")
                solve = json.loads(solve_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(solve, dict)
                    or solve.get("status") != "complete"
                    or solve.get("step") != int(spec["step"]) - 1
                    or solve.get("output_student_prefix_sha256") != prefix_sha
                ):
                    raise FeatCalCachePlanError("Student prefix hash chain differs")
        completed.append(spec["shard_id"])

    teacher_total = sum(spec["role"] == "teacher" for spec in shard_index["shards"])
    teacher_complete = sum(value.startswith("teacher/") for value in completed)
    student_counts = Counter(
        int(value.rsplit("step_", 1)[1])
        for value in completed
        if value.startswith("student/")
    )
    touched_student_steps = set(student_counts)
    if teacher_complete < teacher_total and touched_student_steps:
        raise FeatCalCachePlanError("Student shards exist before the teacher cache is complete")
    fully_completed_student_steps = []
    if touched_student_steps:
        largest = max(touched_student_steps)
        for step in range(largest + 1):
            count = student_counts.get(step, 0)
            if step < largest and count != len(EXPERTS) * TASKS_PER_EXPERT:
                raise FeatCalCachePlanError("Student resume has later shards after an incomplete step")
            if step == largest and count not in range(1, len(EXPERTS) * TASKS_PER_EXPERT + 1):
                raise FeatCalCachePlanError("Student resume step cardinality differs")
            if count == len(EXPERTS) * TASKS_PER_EXPERT:
                fully_completed_student_steps.append(step)
    next_phase = "teacher" if teacher_complete < teacher_total else "student"
    if not pending:
        next_phase = "complete"
    return {
        "status": "resume_scan_passed",
        "completed_shards": len(completed),
        "pending_shards": len(pending),
        "teacher_completed": teacher_complete,
        "teacher_total": teacher_total,
        "fully_completed_student_steps": fully_completed_student_steps,
        "student_shards_by_step": dict(sorted(student_counts.items())),
        "next_phase": next_phase,
        "next_pending_shard_ids": pending[:20],
        "full_tensor_check": full_tensor_check,
    }


__all__ = [
    "CALLS_PER_EXPERT",
    "CALLS_PER_TASK",
    "EXPECTED_MODULE_COUNTS",
    "EXPECTED_ROWS_PER_EXPERT",
    "EXPECTED_TOTAL_ROWS",
    "EXPERTS",
    "FLOW_INDICES",
    "FeatCalCachePlanError",
    "FeatCalModuleQuota",
    "MultiLayerStreamingRowCollector",
    "MultiLayerStreamingRowCollector",
    "REQUESTS_PER_EPISODE",
    "ROLES",
    "ROW_KEY_COLUMNS",
    "TASKS_PER_EXPERT",
    "StreamingBatchCapture",
    "build_call_slots",
    "build_shard_index",
    "canonical_json_sha256",
    "derive_module_quotas",
    "deterministic_row_selection",
    "expected_full_forward_trace",
    "expected_full_forward_trace",
    "scan_resume_state",
    "sha256_file",
    "tensor_sha256",
    "validate_completed_shard",
    "validate_call_state_manifest",
    "validate_row_keys",
    "write_completed_shard_atomic",
]
