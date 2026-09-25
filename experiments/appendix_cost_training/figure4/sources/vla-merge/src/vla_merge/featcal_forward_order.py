"""Fail-closed forward-order contracts and paired-input moments for PI0.5 FeatCal.

The real PI0.5 graph is not a single transformer chain.  This module freezes an
adapted-linear-only calibration schedule and provides a hook collector that is
safe to use one atomic step at a time.  A caller must rerun the full student and
matching expert forwards after every calibrated step so downstream student
features come from the deployed, prefix-calibrated state.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor, nn


class FeatCalForwardOrderError(ValueError):
    """Raised when topology, call order, or paired-feature invariants differ."""


VISION_LAYERS = 27
JOINT_TOWER_LAYERS = 18
VISION_CALLS_PER_FORWARD = 3

VISION_ROOT = (
    "model.paligemma_with_expert.paligemma.model.vision_tower."
    "vision_model.encoder.layers"
)
LANGUAGE_ROOT = "model.paligemma_with_expert.paligemma.model.language_model.layers"
EXPERT_ROOT = "model.paligemma_with_expert.gemma_expert.model.layers"

VISION_SUFFIX_ORDER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "mlp.fc1",
    "mlp.fc2",
)
TOWER_QKV_ORDER = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
)
TOWER_OUTPUT_MLP_ORDER = (
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class FeatCalForwardStep:
    step: int
    layer_id: str
    stage: str
    target_module_paths: tuple[str, ...]
    repetitions_per_forward: int = 1

    def __post_init__(self) -> None:
        if self.step < 0 or not self.layer_id or not self.stage:
            raise FeatCalForwardOrderError("FeatCal step identity is invalid")
        if not self.target_module_paths or len(set(self.target_module_paths)) != len(
            self.target_module_paths
        ):
            raise FeatCalForwardOrderError("FeatCal step targets are empty or duplicated")
        if self.repetitions_per_forward < 1:
            raise FeatCalForwardOrderError("FeatCal repetitions must be positive")

    @property
    def expected_trace(self) -> tuple[str, ...]:
        return self.target_module_paths * self.repetitions_per_forward

    @property
    def weight_keys(self) -> tuple[str, ...]:
        return tuple(f"{path}.weight" for path in self.target_module_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "layer_id": self.layer_id,
            "stage": self.stage,
            "target_module_paths": list(self.target_module_paths),
            "target_weight_keys": list(self.weight_keys),
            "repetitions_per_forward": self.repetitions_per_forward,
            "within_step_update": "simultaneous_after_all_inputs_captured",
            "next_step_student_source": "full_model_rerun_from_updated_prefix",
        }


def _vision_targets(layer: int) -> tuple[str, ...]:
    root = f"{VISION_ROOT}.{layer}"
    return tuple(f"{root}.{suffix}" for suffix in VISION_SUFFIX_ORDER)


def _tower_targets(root: str, layer: int, suffixes: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{root}.{layer}.{suffix}" for suffix in suffixes)


def build_pi05_adapted_linear_forward_plan(
    adapted_weight_keys: Sequence[str] | set[str],
) -> tuple[FeatCalForwardStep, ...]:
    """Build and validate the 47-step PI0.5 adapted-linear FeatCal schedule."""

    steps: list[FeatCalForwardStep] = []
    for layer in range(VISION_LAYERS):
        steps.append(
            FeatCalForwardStep(
                step=len(steps),
                layer_id=f"vision_encoder.layer.{layer:02d}",
                stage="vision_prefix",
                target_module_paths=_vision_targets(layer),
                repetitions_per_forward=VISION_CALLS_PER_FORWARD,
            )
        )

    steps.append(
        FeatCalForwardStep(
            step=len(steps),
            layer_id="action_time_embedding",
            stage="action_time_embedding",
            target_module_paths=(
                "model.action_in_proj",
                "model.time_mlp_in",
                "model.time_mlp_out",
            ),
        )
    )

    for layer in range(JOINT_TOWER_LAYERS):
        # compute_layer_complete evaluates both QKV sets before joint attention,
        # then the language output/MLP path before the action-expert path.
        targets = (
            *_tower_targets(LANGUAGE_ROOT, layer, TOWER_QKV_ORDER),
            *_tower_targets(EXPERT_ROOT, layer, TOWER_QKV_ORDER),
            *_tower_targets(LANGUAGE_ROOT, layer, TOWER_OUTPUT_MLP_ORDER),
            *_tower_targets(EXPERT_ROOT, layer, TOWER_OUTPUT_MLP_ORDER),
        )
        steps.append(
            FeatCalForwardStep(
                step=len(steps),
                layer_id=f"joint_language_action.layer.{layer:02d}",
                stage="joint_language_action_towers",
                target_module_paths=targets,
            )
        )

    steps.append(
        FeatCalForwardStep(
            step=len(steps),
            layer_id="action_output_projection",
            stage="action_output",
            target_module_paths=("model.action_out_proj",),
        )
    )

    plan = tuple(steps)
    expected_keys = {key for step in plan for key in step.weight_keys}
    observed_keys = set(adapted_weight_keys)
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    if missing or extra:
        raise FeatCalForwardOrderError(
            "PI0.5 adapted-linear scope differs: "
            f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
        )
    if [step.step for step in plan] != list(range(len(plan))):
        raise AssertionError("Internal FeatCal plan steps are not contiguous")
    return plan


def expected_pi05_adapted_weight_keys() -> set[str]:
    """Return the exact 418-weight Table-1 FeatCal scope."""

    provisional: list[FeatCalForwardStep] = []
    for layer in range(VISION_LAYERS):
        provisional.append(
            FeatCalForwardStep(
                step=len(provisional),
                layer_id=f"vision_encoder.layer.{layer:02d}",
                stage="vision_prefix",
                target_module_paths=_vision_targets(layer),
                repetitions_per_forward=VISION_CALLS_PER_FORWARD,
            )
        )
    provisional.append(
        FeatCalForwardStep(
            step=len(provisional),
            layer_id="action_time_embedding",
            stage="action_time_embedding",
            target_module_paths=(
                "model.action_in_proj",
                "model.time_mlp_in",
                "model.time_mlp_out",
            ),
        )
    )
    for layer in range(JOINT_TOWER_LAYERS):
        provisional.append(
            FeatCalForwardStep(
                step=len(provisional),
                layer_id=f"joint_language_action.layer.{layer:02d}",
                stage="joint_language_action_towers",
                target_module_paths=(
                    *_tower_targets(LANGUAGE_ROOT, layer, TOWER_QKV_ORDER),
                    *_tower_targets(EXPERT_ROOT, layer, TOWER_QKV_ORDER),
                    *_tower_targets(LANGUAGE_ROOT, layer, TOWER_OUTPUT_MLP_ORDER),
                    *_tower_targets(EXPERT_ROOT, layer, TOWER_OUTPUT_MLP_ORDER),
                ),
            )
        )
    provisional.append(
        FeatCalForwardStep(
            step=len(provisional),
            layer_id="action_output_projection",
            stage="action_output",
            target_module_paths=("model.action_out_proj",),
        )
    )
    return {key for step in provisional for key in step.weight_keys}


def forward_plan_sha256(plan: Sequence[FeatCalForwardStep]) -> str:
    payload = json.dumps(
        [step.to_dict() for step in plan],
        indent=2,
        sort_keys=True,
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapturedLinearInput:
    module_path: str
    occurrence: int
    value: Tensor

    @property
    def identity(self) -> tuple[str, int]:
        return self.module_path, self.occurrence


@dataclass(frozen=True)
class ForwardInputCapture:
    step: FeatCalForwardStep
    forward_identity: str
    calls: tuple[CapturedLinearInput, ...]

    @property
    def trace(self) -> tuple[str, ...]:
        return tuple(call.module_path for call in self.calls)


@dataclass(frozen=True)
class FeatCalRowFactors:
    student: Tensor
    target: Tensor
    row_weights: Tensor
    forward_identity: str

    def __post_init__(self) -> None:
        if self.student.ndim != 2 or self.student.shape != self.target.shape:
            raise FeatCalForwardOrderError("FeatCal row factors must be equal-shape matrices")
        if self.row_weights.ndim != 1 or self.row_weights.shape[0] != self.student.shape[0]:
            raise FeatCalForwardOrderError("FeatCal row-factor weights differ")
        if not self.forward_identity:
            raise FeatCalForwardOrderError("FeatCal row-factor identity is empty")
        if not all(
            torch.isfinite(value).all()
            for value in (self.student, self.target, self.row_weights)
        ):
            raise FeatCalForwardOrderError("FeatCal row factors contain non-finite values")
        if torch.any(self.row_weights <= 0):
            raise FeatCalForwardOrderError("FeatCal retained row weights must be positive")


class ForwardOrderInputCollector:
    """Capture one atomic step and reject every call-order or shape drift."""

    def __init__(
        self,
        root: nn.Module,
        step: FeatCalForwardStep,
        *,
        max_capture_bytes: int = 2 * 1024**3,
    ) -> None:
        if root.training:
            raise FeatCalForwardOrderError("FeatCal collection requires eval mode")
        if max_capture_bytes <= 0:
            raise FeatCalForwardOrderError("max_capture_bytes must be positive")
        modules = dict(root.named_modules())
        selected: dict[str, nn.Linear] = {}
        for path in step.target_module_paths:
            module = modules.get(path)
            if not isinstance(module, nn.Linear):
                raise FeatCalForwardOrderError(f"FeatCal target is not a Linear: {path}")
            selected[path] = module
        if len({id(module) for module in selected.values()}) != len(selected):
            raise FeatCalForwardOrderError("FeatCal targets alias the same Linear module")
        self.root = root
        self.step = step
        self.modules = selected
        self.max_capture_bytes = int(max_capture_bytes)

    def capture(
        self,
        forward: Callable[[], Any],
        *,
        forward_identity: str,
    ) -> ForwardInputCapture:
        if self.root.training:
            raise FeatCalForwardOrderError("FeatCal root entered training mode")
        if not forward_identity:
            raise FeatCalForwardOrderError("forward_identity must be non-empty")
        expected_trace = self.step.expected_trace
        calls: list[CapturedLinearInput] = []
        occurrences: dict[str, int] = defaultdict(int)
        handles = []
        captured_bytes = 0

        def make_hook(path: str):
            def hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
                nonlocal captured_bytes
                call_index = len(calls)
                if call_index >= len(expected_trace) or expected_trace[call_index] != path:
                    expected = expected_trace[call_index] if call_index < len(expected_trace) else None
                    raise FeatCalForwardOrderError(
                        f"Forward call order differs at {call_index}: observed={path}, expected={expected}"
                    )
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise FeatCalForwardOrderError(f"Linear input is not a positional tensor: {path}")
                value = inputs[0].detach()
                if not value.is_floating_point() or not torch.isfinite(value).all():
                    raise FeatCalForwardOrderError(f"Linear input is non-finite/non-floating: {path}")
                if value.ndim < 2 or value.shape[-1] != module.in_features:
                    raise FeatCalForwardOrderError(
                        f"Linear input width differs for {path}: {tuple(value.shape)}"
                    )
                captured_bytes += value.numel() * value.element_size()
                if captured_bytes > self.max_capture_bytes:
                    raise FeatCalForwardOrderError("FeatCal capture exceeds max_capture_bytes")
                occurrence = occurrences[path]
                occurrences[path] += 1
                calls.append(
                    CapturedLinearInput(
                        module_path=path,
                        occurrence=occurrence,
                        value=value.to(device="cpu", copy=True).contiguous(),
                    )
                )

            return hook

        for path, module in self.modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(path)))
        try:
            with torch.inference_mode():
                forward()
        finally:
            for handle in handles:
                handle.remove()
        observed_trace = tuple(call.module_path for call in calls)
        if observed_trace != expected_trace:
            raise FeatCalForwardOrderError(
                f"Forward trace incomplete: observed={observed_trace}, expected={expected_trace}"
            )
        return ForwardInputCapture(
            step=self.step,
            forward_identity=forward_identity,
            calls=tuple(calls),
        )


def paired_featcal_moments(
    student: ForwardInputCapture,
    expert: ForwardInputCapture,
    *,
    row_weights: Mapping[tuple[str, int], Tensor],
    teacher_interp_alpha: float,
) -> tuple[dict[str, dict[str, Tensor | int | float]], dict[str, Any]]:
    """Compute exact weighted sufficient statistics from paired hook captures."""

    alpha = float(teacher_interp_alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise FeatCalForwardOrderError("teacher_interp_alpha must lie in [0, 1]")
    if student.step != expert.step or student.forward_identity != expert.forward_identity:
        raise FeatCalForwardOrderError("Student/expert captures are not the same forward identity")
    student_by_id = {call.identity: call for call in student.calls}
    expert_by_id = {call.identity: call for call in expert.calls}
    expected_ids = set(student_by_id)
    if expected_ids != set(expert_by_id) or expected_ids != set(row_weights):
        raise FeatCalForwardOrderError("Paired calls or explicit row-weight identities differ")

    accumulators: dict[str, dict[str, Tensor | int | float]] = {}
    for identity in (call.identity for call in student.calls):
        xs = student_by_id[identity].value.to(dtype=torch.float64)
        xe = expert_by_id[identity].value.to(dtype=torch.float64)
        if xs.shape != xe.shape:
            raise FeatCalForwardOrderError(f"Student/expert input shapes differ: {identity}")
        weights = torch.as_tensor(row_weights[identity], dtype=torch.float64)
        if tuple(weights.shape) != tuple(xs.shape[:-1]):
            raise FeatCalForwardOrderError(f"Explicit row-weight shape differs: {identity}")
        if not torch.isfinite(weights).all() or torch.any(weights < 0):
            raise FeatCalForwardOrderError(f"Explicit row weights are invalid: {identity}")
        flat_weights = weights.reshape(-1)
        flat_xs = xs.reshape(-1, xs.shape[-1])
        flat_xe = xe.reshape(-1, xe.shape[-1])
        flat_xt = alpha * flat_xe + (1.0 - alpha) * flat_xs
        weight_sum = flat_weights.sum()
        weighted = flat_weights.unsqueeze(-1)
        update: dict[str, Tensor | int | float] = {
            "sum_xs": (flat_xs * weighted).sum(dim=0),
            "sum_xt": (flat_xt * weighted).sum(dim=0),
            "sum_xsxs": flat_xs.transpose(0, 1) @ (flat_xs * weighted),
            "sum_xsxt": flat_xs.transpose(0, 1) @ (flat_xt * weighted),
            "sum_w": float(weight_sum),
            "rows": int(torch.count_nonzero(flat_weights).item()),
            "calls": 1,
        }
        path = identity[0]
        if path not in accumulators:
            accumulators[path] = update
        else:
            accumulator = accumulators[path]
            for key in ("sum_xs", "sum_xt", "sum_xsxs", "sum_xsxt"):
                accumulator[key] = accumulator[key] + update[key]
            for key in ("sum_w", "rows", "calls"):
                accumulator[key] = accumulator[key] + update[key]

    statistics: dict[str, dict[str, Tensor | int | float]] = {}
    receipt_targets: dict[str, Any] = {}
    for path, accumulator in accumulators.items():
        sum_w = float(accumulator["sum_w"])
        if sum_w <= 0:
            raise FeatCalForwardOrderError(f"Aggregate row weights sum to zero: {path}")
        statistics[path] = {
            "cov_ss": accumulator["sum_xsxs"] / sum_w,
            "cross_st": accumulator["sum_xsxt"] / sum_w,
            "mean_student": accumulator["sum_xs"] / sum_w,
            "mean_teacher": accumulator["sum_xt"] / sum_w,
            "rows": int(accumulator["rows"]),
            "weight_sum": sum_w,
        }
        receipt_targets[path] = {
            "calls": int(accumulator["calls"]),
            "positive_weight_rows": int(accumulator["rows"]),
            "weight_sum": sum_w,
            "input_width": int(statistics[path]["mean_student"].shape[0]),
        }
    receipt = {
        "step": student.step.step,
        "layer_id": student.step.layer_id,
        "forward_identity": student.forward_identity,
        "same_sample_pairing": True,
        "explicit_row_weights": True,
        "teacher_interp_alpha": alpha,
        "targets": receipt_targets,
    }
    return statistics, receipt


def _tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def paired_featcal_factor_probe_receipt(
    student: ForwardInputCapture,
    expert: ForwardInputCapture,
    *,
    row_weights: Mapping[tuple[str, int], Tensor],
    teacher_interp_alpha: float,
    probe_seed: int,
    probe_columns: int = 4,
    dense_width_limit: int = 1024,
) -> dict[str, Any]:
    """Verify exact moment factors without materializing unsafe large D-by-D matrices.

    For every target, the exact moments are represented by ``X_s``, ``X_t`` and
    explicit row weights. Deterministic right probes exercise ``G @ P`` and
    ``C @ P`` directly from those factors. For widths up to
    ``dense_width_limit`` the function additionally materializes the full
    moments and checks their probe products against the factor path.
    """

    alpha = float(teacher_interp_alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise FeatCalForwardOrderError("teacher_interp_alpha must lie in [0, 1]")
    if probe_columns < 1 or dense_width_limit < 1:
        raise FeatCalForwardOrderError("Probe and dense-width settings must be positive")
    if student.step != expert.step or student.forward_identity != expert.forward_identity:
        raise FeatCalForwardOrderError("Student/expert captures are not the same forward identity")
    student_by_id = {call.identity: call for call in student.calls}
    expert_by_id = {call.identity: call for call in expert.calls}
    expected_ids = set(student_by_id)
    if expected_ids != set(expert_by_id) or expected_ids != set(row_weights):
        raise FeatCalForwardOrderError("Paired calls or explicit row-weight identities differ")

    targets: dict[str, Any] = {}
    total_factor_bytes = 0
    for path in student.step.target_module_paths:
        identities = [call.identity for call in student.calls if call.module_path == path]
        student_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        weight_rows: list[Tensor] = []
        for identity in identities:
            xs = student_by_id[identity].value.to(dtype=torch.float64)
            xe = expert_by_id[identity].value.to(dtype=torch.float64)
            if xs.shape != xe.shape:
                raise FeatCalForwardOrderError(f"Student/expert input shapes differ: {identity}")
            weights = torch.as_tensor(row_weights[identity], dtype=torch.float64)
            if tuple(weights.shape) != tuple(xs.shape[:-1]):
                raise FeatCalForwardOrderError(f"Explicit row-weight shape differs: {identity}")
            if not torch.isfinite(weights).all() or torch.any(weights < 0):
                raise FeatCalForwardOrderError(f"Explicit row weights are invalid: {identity}")
            flat_xs = xs.reshape(-1, xs.shape[-1])
            flat_xe = xe.reshape(-1, xe.shape[-1])
            student_rows.append(flat_xs)
            target_rows.append(alpha * flat_xe + (1.0 - alpha) * flat_xs)
            weight_rows.append(weights.reshape(-1))

        all_xs = torch.cat(student_rows, dim=0)
        all_xt = torch.cat(target_rows, dim=0)
        all_weights = torch.cat(weight_rows, dim=0)
        weight_sum = all_weights.sum()
        if weight_sum <= 0:
            raise FeatCalForwardOrderError(f"Aggregate row weights sum to zero: {path}")
        width = int(all_xs.shape[1])
        seed_material = f"{probe_seed}:{path}".encode("utf-8")
        target_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(target_seed)
        probe = torch.randn(width, probe_columns, generator=generator, dtype=torch.float64)
        weighted = all_weights.unsqueeze(-1)
        cov_probe = all_xs.transpose(0, 1) @ (weighted * (all_xs @ probe)) / weight_sum
        cross_probe = all_xs.transpose(0, 1) @ (weighted * (all_xt @ probe)) / weight_sum
        if not torch.isfinite(cov_probe).all() or not torch.isfinite(cross_probe).all():
            raise FeatCalForwardOrderError(f"Factor moment probe is non-finite: {path}")

        dense_materialized = width <= dense_width_limit
        cov_probe_error = None
        cross_probe_error = None
        if dense_materialized:
            covariance = all_xs.transpose(0, 1) @ (weighted * all_xs) / weight_sum
            cross = all_xs.transpose(0, 1) @ (weighted * all_xt) / weight_sum
            cov_probe_error = float((covariance @ probe - cov_probe).abs().max())
            cross_probe_error = float((cross @ probe - cross_probe).abs().max())
            if cov_probe_error > 1e-9 or cross_probe_error > 1e-9:
                raise FeatCalForwardOrderError(f"Dense/factor moment probe parity differs: {path}")

        factor_bytes = (
            all_xs.numel() * all_xs.element_size()
            + all_xt.numel() * all_xt.element_size()
            + all_weights.numel() * all_weights.element_size()
        )
        total_factor_bytes += factor_bytes
        mean_student = (all_xs * weighted).sum(dim=0) / weight_sum
        mean_teacher = (all_xt * weighted).sum(dim=0) / weight_sum
        targets[path] = {
            "calls": len(identities),
            "input_width": width,
            "physical_rows": int(all_weights.numel()),
            "positive_weight_rows": int(torch.count_nonzero(all_weights).item()),
            "weight_sum": float(weight_sum),
            "factor_bytes_float64": factor_bytes,
            "dense_cov_cross_bytes_float64": 2 * width * width * 8,
            "dense_materialized": dense_materialized,
            "cov_probe_sha256": _tensor_sha256(cov_probe),
            "cross_probe_sha256": _tensor_sha256(cross_probe),
            "mean_student_sha256": _tensor_sha256(mean_student),
            "mean_teacher_sha256": _tensor_sha256(mean_teacher),
            "cov_probe_max_abs_error_vs_dense": cov_probe_error,
            "cross_probe_max_abs_error_vs_dense": cross_probe_error,
        }
    return {
        "step": student.step.step,
        "layer_id": student.step.layer_id,
        "forward_identity": student.forward_identity,
        "same_sample_pairing": True,
        "explicit_row_weights": True,
        "teacher_interp_alpha": alpha,
        "probe_seed": int(probe_seed),
        "probe_columns": int(probe_columns),
        "dense_width_limit": int(dense_width_limit),
        "moment_representation": "exact_row_factors_with_deterministic_right_probes",
        "total_factor_bytes_float64": total_factor_bytes,
        "targets": targets,
    }


def paired_featcal_row_factors(
    student: ForwardInputCapture,
    expert: ForwardInputCapture,
    *,
    row_weights: Mapping[tuple[str, int], Tensor],
    teacher_interp_alpha: float,
) -> dict[str, FeatCalRowFactors]:
    """Return exact positive-weight row factors for every target in one atomic step."""

    alpha = float(teacher_interp_alpha)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise FeatCalForwardOrderError("teacher_interp_alpha must lie in [0, 1]")
    if student.step != expert.step or student.forward_identity != expert.forward_identity:
        raise FeatCalForwardOrderError("Student/expert captures are not the same forward identity")
    student_by_id = {call.identity: call for call in student.calls}
    expert_by_id = {call.identity: call for call in expert.calls}
    expected_ids = set(student_by_id)
    if expected_ids != set(expert_by_id) or expected_ids != set(row_weights):
        raise FeatCalForwardOrderError("Paired calls or explicit row-weight identities differ")

    result: dict[str, FeatCalRowFactors] = {}
    for path in student.step.target_module_paths:
        student_rows: list[Tensor] = []
        target_rows: list[Tensor] = []
        retained_weights: list[Tensor] = []
        for call in student.calls:
            if call.module_path != path:
                continue
            identity = call.identity
            xs = call.value
            xe = expert_by_id[identity].value
            if xs.shape != xe.shape:
                raise FeatCalForwardOrderError(f"Student/expert input shapes differ: {identity}")
            weights = torch.as_tensor(row_weights[identity], dtype=torch.float64).reshape(-1)
            flat_xs = xs.reshape(-1, xs.shape[-1])
            flat_xe = xe.reshape(-1, xe.shape[-1])
            if weights.shape[0] != flat_xs.shape[0]:
                raise FeatCalForwardOrderError(f"Explicit row-weight shape differs: {identity}")
            if not torch.isfinite(weights).all() or torch.any(weights < 0):
                raise FeatCalForwardOrderError(f"Explicit row weights are invalid: {identity}")
            keep = weights > 0
            if torch.any(keep):
                kept_xs = flat_xs[keep].detach().cpu().to(dtype=torch.float64).contiguous()
                kept_xe = flat_xe[keep].detach().cpu().to(dtype=torch.float64).contiguous()
                student_rows.append(kept_xs)
                target_rows.append(alpha * kept_xe + (1.0 - alpha) * kept_xs)
                retained_weights.append(weights[keep].cpu().contiguous())
        if not student_rows:
            raise FeatCalForwardOrderError(f"Target has no positive-weight rows: {path}")
        result[path] = FeatCalRowFactors(
            student=torch.cat(student_rows, dim=0),
            target=torch.cat(target_rows, dim=0),
            row_weights=torch.cat(retained_weights, dim=0),
            forward_identity=student.forward_identity,
        )
    return result


def featcal_rowspace_linear_weight(
    expert_weights: Sequence[Tensor],
    task_factors: Sequence[FeatCalRowFactors],
    *,
    soup_weight: Tensor,
    base_weight: Tensor,
    ridge_lambda: float,
    anchor_blend_rho: float,
    covariance_eps: float,
    solve_dtype: torch.dtype = torch.float64,
    solve_device: torch.device | str = "cpu",
    output_chunk_size: int = 128,
    max_workspace_bytes: int = 2 * 1024**3,
) -> tuple[Tensor, dict[str, Any]]:
    """Solve FeatCal's ridge system exactly in row space via Woodbury.

    If ``Z`` stacks the task-normalized, weighted student row factors and
    ``Q`` stacks the corresponding target rows multiplied by each expert
    weight, the primal system is

    ``(mu I + Z.T Z) X = Z.T Q + lambda W_anchor.T``.

    Woodbury gives

    ``X = (B - Z.T (mu I + Z Z.T)^-1 Z B) / mu``.

    Thus the largest square matrix is ``R x R`` for total retained rows R;
    no ``D x D`` Gram is constructed. Output columns are solved in chunks.
    """

    if solve_dtype not in (torch.float32, torch.float64):
        raise FeatCalForwardOrderError("solve_dtype must be float32 or float64")
    if len(expert_weights) != len(task_factors) or not expert_weights:
        raise FeatCalForwardOrderError("Expert weights and task factors must align")
    if output_chunk_size < 1 or max_workspace_bytes < 1:
        raise FeatCalForwardOrderError("Chunk size and workspace cap must be positive")
    ridge = float(ridge_lambda)
    rho = float(anchor_blend_rho)
    eps = float(covariance_eps)
    if not all(math.isfinite(value) for value in (ridge, rho, eps)):
        raise FeatCalForwardOrderError("FeatCal solve settings must be finite")
    if ridge < 0 or eps <= 0:
        raise FeatCalForwardOrderError("FeatCal row-space solve requires lambda>=0 and eps>0")
    mu = ridge + eps
    output_shape = tuple(soup_weight.shape)
    if soup_weight.ndim != 2 or base_weight.shape != soup_weight.shape:
        raise FeatCalForwardOrderError("Soup/base weights must be equal-shape matrices")
    output_width, input_width = output_shape
    for index, (weight, factors) in enumerate(zip(expert_weights, task_factors, strict=True)):
        if weight.shape != soup_weight.shape or factors.student.shape[1] != input_width:
            raise FeatCalForwardOrderError(f"Task {index} weight/factor dimensions differ")
        if factors.target.shape != factors.student.shape:
            raise FeatCalForwardOrderError(f"Task {index} target factors differ")

    device = torch.device(solve_device)
    element_bytes = torch.empty((), dtype=solve_dtype).element_size()
    total_rows = sum(factors.student.shape[0] for factors in task_factors)
    maximum_task_rows = max(factors.student.shape[0] for factors in task_factors)
    chunk = min(int(output_chunk_size), output_width)
    persistent_elements = (
        2 * total_rows * input_width
        + 2 * total_rows * output_width
        + 2 * total_rows * total_rows
        + output_width * input_width
    )
    temporary_elements = max(
        3 * maximum_task_rows * input_width
        + maximum_task_rows * maximum_task_rows
        + maximum_task_rows * output_width,
        2 * input_width * chunk + 3 * total_rows * chunk,
    )
    estimated_workspace = element_bytes * (persistent_elements + temporary_elements)
    if estimated_workspace > max_workspace_bytes:
        raise MemoryError(
            f"FeatCal row-space estimated workspace exceeds cap: "
            f"{estimated_workspace} > {max_workspace_bytes}"
        )

    z_parts: list[Tensor] = []
    q_parts: list[Tensor] = []
    task_info = []
    for task_index, (expert_weight, factors) in enumerate(
        zip(expert_weights, task_factors, strict=True)
    ):
        xs = factors.student.to(device=device, dtype=solve_dtype)
        xt = factors.target.to(device=device, dtype=solve_dtype)
        weights = factors.row_weights.to(device=device, dtype=solve_dtype)
        weight_sum = weights.sum()
        sqrt_weights = weights.sqrt().unsqueeze(-1)
        weighted_xs = sqrt_weights * xs
        row_gram = weighted_xs @ weighted_xs.transpose(0, 1)
        covariance_norm = row_gram.norm(p="fro") / weight_sum
        covariance_norm = covariance_norm.clamp_min(eps)
        factor_scale = torch.rsqrt(weight_sum * covariance_norm)
        z_part = factor_scale * weighted_xs
        target_part = factor_scale * sqrt_weights * xt
        q_part = torch.empty(
            xs.shape[0],
            output_width,
            device=device,
            dtype=solve_dtype,
        )
        for start in range(0, output_width, chunk):
            stop = min(start + chunk, output_width)
            weight_chunk = expert_weight[start:stop].to(device=device, dtype=solve_dtype)
            q_part[:, start:stop] = target_part @ weight_chunk.transpose(0, 1)
        z_parts.append(z_part)
        q_parts.append(q_part)
        task_info.append(
            {
                "task": task_index,
                "rows": int(xs.shape[0]),
                "weight_sum": float(weight_sum),
                "covariance_frobenius_norm": float(covariance_norm),
                "normalization_factor": float(factor_scale),
                "forward_identity": factors.forward_identity,
            }
        )
        del xs, xt, weights, sqrt_weights, weighted_xs, row_gram, target_part

    z = torch.cat(z_parts, dim=0)
    q = torch.cat(q_parts, dim=0)
    kernel = z @ z.transpose(0, 1)
    kernel.diagonal().add_(mu)
    cholesky, info = torch.linalg.cholesky_ex(kernel)
    if torch.any(info != 0):
        raise FeatCalForwardOrderError(f"Row-space Cholesky failed: {info.tolist()}")

    output = torch.empty(
        output_width,
        input_width,
        dtype=soup_weight.dtype,
        device="cpu",
    )
    max_abs_residual = 0.0
    rhs_norm_squared = 0.0
    residual_norm_squared = 0.0
    for start in range(0, output_width, chunk):
        stop = min(start + chunk, output_width)
        anchor_chunk = (
            rho * soup_weight[start:stop].to(device=device, dtype=solve_dtype)
            + (1.0 - rho) * base_weight[start:stop].to(device=device, dtype=solve_dtype)
        )
        rhs = z.transpose(0, 1) @ q[:, start:stop] + ridge * anchor_chunk.transpose(0, 1)
        projected_rhs = z @ rhs
        correction = torch.cholesky_solve(projected_rhs, cholesky)
        solved = (rhs - z.transpose(0, 1) @ correction) / mu
        residual = mu * solved + z.transpose(0, 1) @ (z @ solved) - rhs
        max_abs_residual = max(max_abs_residual, float(residual.abs().max()))
        rhs_norm_squared += float(rhs.square().sum())
        residual_norm_squared += float(residual.square().sum())
        output[start:stop].copy_(solved.transpose(0, 1).to(device="cpu", dtype=output.dtype))
    if not torch.isfinite(output).all():
        raise FeatCalForwardOrderError("Row-space FeatCal solution is non-finite")
    relative_residual = math.sqrt(residual_norm_squared) / max(
        math.sqrt(rhs_norm_squared), torch.finfo(solve_dtype).eps
    )
    metadata = {
        "strategy": "exact_featcal_woodbury_rowspace",
        "input_width": input_width,
        "output_width": output_width,
        "task_count": len(task_factors),
        "total_rows": total_rows,
        "rowspace_kernel_shape": [total_rows, total_rows],
        "dense_input_gram_allocated": False,
        "dense_cov_cross_bytes_avoided": 2 * input_width * input_width * element_bytes,
        "ridge_lambda": ridge,
        "covariance_eps": eps,
        "mu": mu,
        "anchor_blend_rho": rho,
        "solve_dtype": str(solve_dtype),
        "solve_device": str(device),
        "output_chunk_size": chunk,
        "estimated_workspace_bytes": estimated_workspace,
        "workspace_cap_bytes": max_workspace_bytes,
        "max_abs_primal_residual": max_abs_residual,
        "relative_primal_residual": relative_residual,
        "tasks": task_info,
    }
    return output, metadata


__all__ = [
    "CapturedLinearInput",
    "FeatCalForwardOrderError",
    "FeatCalForwardStep",
    "ForwardInputCapture",
    "ForwardOrderInputCollector",
    "build_pi05_adapted_linear_forward_plan",
    "expected_pi05_adapted_weight_keys",
    "forward_plan_sha256",
    "paired_featcal_moments",
    "paired_featcal_factor_probe_receipt",
    "FeatCalRowFactors",
    "paired_featcal_row_factors",
    "featcal_rowspace_linear_weight",
]
