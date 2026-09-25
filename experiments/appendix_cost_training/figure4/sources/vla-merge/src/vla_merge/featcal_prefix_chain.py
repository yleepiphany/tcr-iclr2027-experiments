"""Fail-closed helpers for a non-final PI0.5 FeatCal prefix-chain gate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .featcal_cache_plan import (
    FeatCalCachePlanError,
    FeatCalModuleQuota,
    sha256_file,
    validate_completed_shard,
)
from .featcal_forward_order import FeatCalForwardStep


class FeatCalPrefixChainError(ValueError):
    """Raised when an engineering prefix-chain invariant differs."""


@dataclass(frozen=True)
class SelectedStepCapture:
    trace: tuple[str, ...]
    inputs_by_module: dict[str, Tensor]
    row_keys_by_module: dict[str, Tensor]
    selected_bytes: int


def expected_single_step_trace(step: FeatCalForwardStep) -> tuple[str, ...]:
    if step.repetitions_per_forward not in (1, 3):
        raise FeatCalPrefixChainError("Atomic step repetition count differs")
    return tuple(step.target_module_paths) * step.repetitions_per_forward


class SingleStepSelectedRowCollector:
    """Capture one atomic step at teacher-frozen source row indices."""

    def __init__(
        self,
        root: torch.nn.Module,
        step: FeatCalForwardStep,
        quotas: Sequence[FeatCalModuleQuota],
        *,
        teacher_row_keys: Mapping[str, Tensor],
        call_slot_ids: Sequence[str],
        call_ordinals_within_task: Sequence[int],
        row_mask_provider: Callable[[str, int, Tensor], Tensor],
        max_selected_bytes: int = 512 * 1024**2,
    ) -> None:
        if root.training:
            raise FeatCalPrefixChainError("Prefix-chain collection requires eval mode")
        if (
            not call_slot_ids
            or len(call_slot_ids) != len(call_ordinals_within_task)
            or len(set(call_ordinals_within_task)) != len(call_ordinals_within_task)
        ):
            raise FeatCalPrefixChainError("Prefix-chain call batch identity differs")
        quota_by_module = {quota.module_path: quota for quota in quotas}
        targets = set(step.target_module_paths)
        if set(quota_by_module) != targets or set(teacher_row_keys) != targets:
            raise FeatCalPrefixChainError("Step quota/teacher row-key target scopes differ")
        modules = dict(root.named_modules())
        selected_modules = {}
        normalized_keys = {}
        for path in step.target_module_paths:
            module = modules.get(path)
            if not isinstance(module, torch.nn.Linear):
                raise FeatCalPrefixChainError(f"Step target is not Linear: {path}")
            keys = torch.as_tensor(teacher_row_keys[path], dtype=torch.int64).cpu().contiguous()
            quota = quota_by_module[path]
            if tuple(keys.shape) != (quota.rows_per_task_shard, 4):
                raise FeatCalPrefixChainError(f"Teacher row-key shape differs: {path}")
            selected_modules[path] = module
            normalized_keys[path] = keys
        if not callable(row_mask_provider) or max_selected_bytes < 1:
            raise FeatCalPrefixChainError("Step mask provider/cap is invalid")
        self.root = root
        self.step = step
        self.expected_trace = expected_single_step_trace(step)
        self.quotas = quota_by_module
        self.modules = selected_modules
        self.teacher_row_keys = normalized_keys
        self.call_slot_ids = tuple(call_slot_ids)
        self.call_ordinals = tuple(int(value) for value in call_ordinals_within_task)
        self.row_mask_provider = row_mask_provider
        self.max_selected_bytes = int(max_selected_bytes)

    def capture(self, forward: Callable[[], Any]) -> SelectedStepCapture:
        if self.root.training:
            raise FeatCalPrefixChainError("Prefix-chain root entered training mode")
        trace: list[str] = []
        occurrences: dict[str, int] = Counter()
        inputs: dict[str, list[Tensor]] = {path: [] for path in self.modules}
        row_keys: dict[str, list[Tensor]] = {path: [] for path in self.modules}
        handles = []
        selected_bytes = 0

        def make_hook(path: str):
            def hook(module: torch.nn.Module, values: tuple[Any, ...]) -> None:
                nonlocal selected_bytes
                trace_index = len(trace)
                expected = (
                    self.expected_trace[trace_index]
                    if trace_index < len(self.expected_trace)
                    else None
                )
                if path != expected:
                    raise FeatCalPrefixChainError(
                        f"Atomic step forward order differs at {trace_index}: {path} != {expected}"
                    )
                if not values or not isinstance(values[0], Tensor):
                    raise FeatCalPrefixChainError(f"Step target input is missing: {path}")
                value = values[0].detach()
                if (
                    value.ndim < 2
                    or value.shape[0] != len(self.call_slot_ids)
                    or value.shape[-1] != module.in_features
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise FeatCalPrefixChainError(f"Step target input contract differs: {path}")
                occurrence = occurrences[path]
                quota = self.quotas[path]
                if occurrence >= quota.occurrences_per_call:
                    raise FeatCalPrefixChainError(f"Step target occurrence overflow: {path}")
                eligible = torch.as_tensor(
                    self.row_mask_provider(path, occurrence, value),
                    dtype=torch.bool,
                    device="cpu",
                )
                if tuple(eligible.shape) != tuple(value.shape[:-1]):
                    raise FeatCalPrefixChainError(f"Step row mask shape differs: {path}")
                all_keys = self.teacher_row_keys[path]
                for batch_index, call_ordinal in enumerate(self.call_ordinals):
                    keep = (all_keys[:, 0] == call_ordinal) & (
                        all_keys[:, 1] == occurrence
                    )
                    chosen_keys = all_keys[keep]
                    if chosen_keys.shape[0] != quota.rows_per_occurrence:
                        raise FeatCalPrefixChainError(
                            f"Teacher row-key quota differs: {path}/call={call_ordinal}"
                        )
                    order = torch.argsort(chosen_keys[:, 3], stable=True)
                    chosen_keys = chosen_keys[order]
                    expected_ranks = torch.arange(
                        quota.rows_per_occurrence, dtype=torch.int64
                    )
                    if (
                        not torch.equal(chosen_keys[:, 3], expected_ranks)
                        or len(torch.unique(chosen_keys[:, 2]))
                        != quota.rows_per_occurrence
                    ):
                        raise FeatCalPrefixChainError(
                            f"Teacher row-key rank/source uniqueness differs: {path}"
                        )
                    source_rows = chosen_keys[:, 2]
                    flat_value = value[batch_index].reshape(-1, value.shape[-1])
                    if (
                        torch.any(source_rows < 0)
                        or torch.any(source_rows >= flat_value.shape[0])
                        or not torch.all(eligible[batch_index].reshape(-1)[source_rows])
                    ):
                        raise FeatCalPrefixChainError(
                            f"Teacher-selected source row is ineligible: {path}"
                        )
                    chosen = (
                        flat_value[source_rows.to(flat_value.device)]
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                    copied_keys = chosen_keys.clone().contiguous()
                    selected_bytes += (
                        chosen.numel() * chosen.element_size()
                        + copied_keys.numel() * copied_keys.element_size()
                    )
                    if selected_bytes > self.max_selected_bytes:
                        raise MemoryError("Step selected rows exceed memory cap")
                    inputs[path].append(chosen)
                    row_keys[path].append(copied_keys)
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
            raise FeatCalPrefixChainError("Atomic step forward trace is incomplete")
        if any(
            occurrences[path] != quota.occurrences_per_call
            for path, quota in self.quotas.items()
        ):
            raise FeatCalPrefixChainError("Atomic step target occurrence count differs")
        return SelectedStepCapture(
            trace=tuple(trace),
            inputs_by_module={path: torch.cat(parts) for path, parts in inputs.items()},
            row_keys_by_module={path: torch.cat(parts) for path, parts in row_keys.items()},
            selected_bytes=selected_bytes,
        )


def prefix_chain_sha256(
    input_prefix_sha256: str,
    step: int,
    solved_tensor_sha256: Mapping[str, str],
) -> str:
    if len(input_prefix_sha256) != 64 or step < 0 or not solved_tensor_sha256:
        raise FeatCalPrefixChainError("Prefix-chain transition identity is invalid")
    payload = {
        "input_prefix_sha256": input_prefix_sha256,
        "step": int(step),
        "solved_tensor_sha256": dict(sorted(solved_tensor_sha256.items())),
    }
    return hashlib.sha256(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def scan_prefix_chain_resume(
    cache_root: Path,
    student_specs: Sequence[Mapping[str, Any]],
    *,
    plan_sha256: str,
    call_state_manifest_sha256: str,
    initial_prefix_sha256: str,
) -> dict[str, Any]:
    """Validate a strictly sequential student-shard/solve-receipt chain."""

    ordered = sorted(student_specs, key=lambda row: int(row["step"]))
    if [int(row["step"]) for row in ordered] != list(range(len(ordered))):
        raise FeatCalPrefixChainError("Student prefix-chain steps are not contiguous")
    current_prefix = initial_prefix_sha256
    previous_solve: dict[str, str] | None = None
    completed = 0
    pending_solve = None
    gap_seen = False
    for spec in ordered:
        step = int(spec["step"])
        data_path = cache_root / str(spec["data_file"])
        manifest_path = cache_root / str(spec["manifest_file"])
        solve_path = cache_root / f"solves/step_{step:02d}.json"
        pair_exists = data_path.exists() and manifest_path.exists()
        if data_path.exists() != manifest_path.exists():
            raise FeatCalPrefixChainError(f"Half-written student shard: step {step}")
        if solve_path.exists() and not pair_exists:
            raise FeatCalPrefixChainError(f"Solve receipt exists without student shard: step {step}")
        if not pair_exists:
            gap_seen = True
            continue
        if gap_seen:
            raise FeatCalPrefixChainError("Later student shard exists after a missing step")
        shard = validate_completed_shard(
            cache_root,
            spec,
            plan_sha256=plan_sha256,
            call_state_manifest_sha256=call_state_manifest_sha256,
            full_tensor_check=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("student_prefix_input_sha256") != current_prefix:
            raise FeatCalPrefixChainError(f"Student prefix input chain differs: step {step}")
        expected_previous_path = previous_solve["path"] if previous_solve else None
        expected_previous_sha = previous_solve["sha256"] if previous_solve else None
        if (
            shard["previous_step_solve_receipt"] != expected_previous_path
            or shard["previous_step_solve_receipt_sha256"] != expected_previous_sha
        ):
            raise FeatCalPrefixChainError(
                f"Student shard previous-solve reference differs: step {step}"
            )
        if not solve_path.exists():
            pending_solve = step
            gap_seen = True
            continue
        solve = json.loads(solve_path.read_text(encoding="utf-8"))
        if (
            not isinstance(solve, dict)
            or solve.get("status") != "engineering_prefix_step_complete"
            or solve.get("step") != step
            or solve.get("input_prefix_sha256") != current_prefix
            or solve.get("student_shard_manifest_sha256") != shard["manifest_sha256"]
            or solve.get("student_shard_data_sha256") != shard["data_sha256"]
            or solve.get("checkpoint_created") is not False
        ):
            raise FeatCalPrefixChainError(f"Student solve receipt chain differs: step {step}")
        output_prefix = solve.get("output_prefix_sha256")
        solved_hashes = solve.get("solved_tensor_sha256")
        if (
            not isinstance(output_prefix, str)
            or not isinstance(solved_hashes, dict)
            or output_prefix
            != prefix_chain_sha256(current_prefix, step, solved_hashes)
        ):
            raise FeatCalPrefixChainError(f"Student output prefix chain differs: step {step}")
        current_prefix = output_prefix
        previous_solve = {
            "path": str(solve_path.relative_to(cache_root)),
            "sha256": sha256_file(solve_path),
        }
        completed += 1
    return {
        "status": "prefix_chain_resume_scan_passed",
        "completed_steps": completed,
        "total_steps": len(ordered),
        "pending_solve_step": pending_solve,
        "next_step": None if completed == len(ordered) else completed,
        "current_prefix_sha256": current_prefix,
        "complete": completed == len(ordered),
    }


def rollback_safe_step_load(
    modules: Mapping[str, torch.nn.Linear],
    replacements: Mapping[str, Tensor],
) -> dict[str, Any]:
    """Load one whole step or roll that step back before propagating an error."""

    if set(modules) != set(replacements) or not modules:
        raise FeatCalPrefixChainError("Atomic step load scopes differ")
    originals = {
        path: module.weight.detach().cpu().clone().contiguous()
        for path, module in modules.items()
    }
    try:
        with torch.no_grad():
            for path, module in modules.items():
                value = replacements[path]
                if value.shape != module.weight.shape or not torch.isfinite(value).all():
                    raise FeatCalPrefixChainError(f"Step replacement differs: {path}")
                module.weight.copy_(value.to(module.weight.device, module.weight.dtype))
        loaded = {
            path: module.weight.detach().cpu().clone().contiguous()
            for path, module in modules.items()
        }
        if any(not torch.equal(loaded[path], replacements[path].to(loaded[path].dtype)) for path in modules):
            raise FeatCalPrefixChainError("Atomic step loaded values differ")
    except BaseException:
        with torch.no_grad():
            for path, module in modules.items():
                module.weight.copy_(originals[path].to(module.weight.device, module.weight.dtype))
        raise
    return {"originals": originals, "loaded": loaded}


def _memmap_random_normal(
    path: Path,
    shape: tuple[int, int],
    *,
    seed: int,
    scale: float,
    row_block: int = 128,
) -> np.memmap:
    if path.exists():
        raise FileExistsError(path)
    value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    generator = np.random.default_rng(seed)
    for start in range(0, shape[0], row_block):
        stop = min(start + row_block, shape[0])
        value[start:stop] = generator.standard_normal(
            (stop - start, shape[1]), dtype=np.float32
        ) * np.float32(scale)
    value.flush()
    return value


def synthetic_out_of_core_rowspace_benchmark(
    workspace: Path,
    *,
    rows: int = 6000,
    input_width: int = 16_384,
    output_width: int = 2_048,
    feature_tile: int = 128,
    output_chunk: int = 64,
    mu: float = 0.05000001,
    seed: int = 2026091205,
    device: str = "cuda",
    max_workspace_bytes: int = 2 * 1024**3,
    return_output_for_testing: bool = False,
) -> dict[str, Any]:
    """Exercise the exact Woodbury kernel without allocating a D-by-D Gram."""

    if (
        min(rows, input_width, output_width, feature_tile, output_chunk) < 1
        or not math.isfinite(mu)
        or mu <= 0
    ):
        raise FeatCalPrefixChainError("Synthetic row-space dimensions/mu differ")
    if return_output_for_testing and output_width * input_width > 1_000_000:
        raise FeatCalPrefixChainError("Testing output capture is limited to small problems")
    element_bytes = torch.empty((), dtype=torch.float64).element_size()
    chunk = min(output_chunk, output_width)
    estimated_peak_bytes = element_bytes * (
        3 * rows * rows
        + rows * feature_tile
        + rows * chunk * 4
        + input_width * chunk * 3
    )
    if estimated_peak_bytes > max_workspace_bytes:
        raise MemoryError(
            "Synthetic row-space estimated workspace exceeds cap: "
            f"{estimated_peak_bytes} > {max_workspace_bytes}"
        )
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    z_path = workspace / "z.npy"
    q_path = workspace / "q.npy"
    started = time.monotonic()
    z = _memmap_random_normal(
        z_path,
        (rows, input_width),
        seed=seed,
        scale=1.0 / math.sqrt(input_width),
    )
    q = _memmap_random_normal(
        q_path,
        (rows, output_width),
        seed=seed + 1,
        scale=1.0,
    )
    data_generation_seconds = time.monotonic() - started
    device_value = torch.device(device)
    if device_value.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_value)

    def z_tile(start: int, stop: int) -> Tensor:
        array = np.array(z[:, start:stop], copy=True)
        return torch.from_numpy(array).to(device=device_value, dtype=torch.float64)

    kernel_started = time.monotonic()
    kernel = torch.zeros(
        (rows, rows), device=device_value, dtype=torch.float64
    )
    for feature_start in range(0, input_width, feature_tile):
        feature_stop = min(feature_start + feature_tile, input_width)
        tile = z_tile(feature_start, feature_stop)
        kernel.addmm_(tile, tile.transpose(0, 1))
        del tile
    kernel.diagonal().add_(float(mu))
    cholesky, info = torch.linalg.cholesky_ex(kernel)
    if torch.any(info != 0):
        raise FeatCalPrefixChainError(
            f"Synthetic row-space Cholesky failed: {info.detach().cpu().tolist()}"
        )
    del kernel, info
    if device_value.type == "cuda":
        torch.cuda.synchronize(device_value)
    kernel_seconds = time.monotonic() - kernel_started

    solve_started = time.monotonic()
    testing_output_chunks: list[Tensor] = []
    output_digest = hashlib.sha256()
    output_digest.update(
        json.dumps(
            {"dtype": "float32", "shape": [output_width, input_width]},
            sort_keys=True,
        ).encode("utf-8")
        + b"\0"
    )
    rhs_norm_squared = 0.0
    residual_norm_squared = 0.0
    max_abs_residual = 0.0
    solved_columns = 0
    for output_start in range(0, output_width, chunk):
        output_stop = min(output_start + chunk, output_width)
        width = output_stop - output_start
        q_chunk = torch.from_numpy(
            np.array(q[:, output_start:output_stop], copy=True)
        ).to(device=device_value, dtype=torch.float64)
        rhs = torch.empty(
            (input_width, width), device=device_value, dtype=torch.float64
        )
        for feature_start in range(0, input_width, feature_tile):
            feature_stop = min(feature_start + feature_tile, input_width)
            tile = z_tile(feature_start, feature_stop)
            rhs[feature_start:feature_stop].copy_(
                tile.transpose(0, 1) @ q_chunk
            )
            del tile
        projected_rhs = torch.zeros(
            (rows, width), device=device_value, dtype=torch.float64
        )
        for feature_start in range(0, input_width, feature_tile):
            feature_stop = min(feature_start + feature_tile, input_width)
            tile = z_tile(feature_start, feature_stop)
            projected_rhs.add_(tile @ rhs[feature_start:feature_stop])
            del tile
        correction = torch.cholesky_solve(projected_rhs, cholesky)
        solution = torch.empty_like(rhs)
        for feature_start in range(0, input_width, feature_tile):
            feature_stop = min(feature_start + feature_tile, input_width)
            tile = z_tile(feature_start, feature_stop)
            solution[feature_start:feature_stop].copy_(
                (
                    rhs[feature_start:feature_stop]
                    - tile.transpose(0, 1) @ correction
                )
                / float(mu)
            )
            del tile

        z_solution = torch.zeros(
            (rows, width), device=device_value, dtype=torch.float64
        )
        for feature_start in range(0, input_width, feature_tile):
            feature_stop = min(feature_start + feature_tile, input_width)
            tile = z_tile(feature_start, feature_stop)
            z_solution.add_(tile @ solution[feature_start:feature_stop])
            del tile
        residual = float(mu) * solution - rhs
        for feature_start in range(0, input_width, feature_tile):
            feature_stop = min(feature_start + feature_tile, input_width)
            tile = z_tile(feature_start, feature_stop)
            residual[feature_start:feature_stop].add_(
                tile.transpose(0, 1) @ z_solution
            )
            del tile
        if not torch.isfinite(solution).all() or not torch.isfinite(residual).all():
            raise FeatCalPrefixChainError("Synthetic row-space output/residual is non-finite")
        rhs_norm_squared += float(rhs.square().sum())
        residual_norm_squared += float(residual.square().sum())
        max_abs_residual = max(max_abs_residual, float(residual.abs().max()))
        output_chunk_cpu = solution.transpose(0, 1).float().cpu().contiguous()
        output_digest.update(output_chunk_cpu.numpy().tobytes())
        if return_output_for_testing:
            testing_output_chunks.append(output_chunk_cpu.clone())
        solved_columns += width
        del (
            q_chunk,
            rhs,
            projected_rhs,
            correction,
            solution,
            z_solution,
            residual,
            output_chunk_cpu,
        )
    if device_value.type == "cuda":
        torch.cuda.synchronize(device_value)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device_value))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device_value))
    else:
        peak_allocated_bytes = None
        peak_reserved_bytes = None
    solve_seconds = time.monotonic() - solve_started
    relative_residual = math.sqrt(residual_norm_squared) / max(
        math.sqrt(rhs_norm_squared), torch.finfo(torch.float64).eps
    )
    if solved_columns != output_width or relative_residual > 1e-8:
        raise FeatCalPrefixChainError(
            f"Synthetic row-space solve residual/cardinality differs: {relative_residual}"
        )
    if peak_allocated_bytes is not None and (
        peak_allocated_bytes >= max_workspace_bytes
        or peak_reserved_bytes is None
        or peak_reserved_bytes >= max_workspace_bytes
    ):
        raise MemoryError("Synthetic CUDA workspace reached the 2 GiB cap")
    result = {
        "status": "synthetic_r6000_d16384_out_of_core_rowspace_passed",
        "rows": rows,
        "input_width": input_width,
        "output_width": output_width,
        "feature_tile": feature_tile,
        "output_chunk": chunk,
        "seed": seed,
        "mu": float(mu),
        "solve_dtype": "float64",
        "device": str(device_value),
        "dense_input_gram_allocated": False,
        "dense_input_gram_bytes_forbidden": input_width * input_width * element_bytes,
        "rowspace_kernel_shape": [rows, rows],
        "estimated_peak_bytes": estimated_peak_bytes,
        "max_workspace_bytes": max_workspace_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "relative_primal_residual": relative_residual,
        "max_abs_primal_residual": max_abs_residual,
        "output_sha256": output_digest.hexdigest(),
        "data_generation_seconds": data_generation_seconds,
        "kernel_and_cholesky_seconds": kernel_seconds,
        "chunked_solve_and_residual_seconds": solve_seconds,
        "wall_seconds": time.monotonic() - started,
        "temporary_factor_bytes": z_path.stat().st_size + q_path.stat().st_size,
    }
    if return_output_for_testing:
        result["_output_tensor"] = torch.cat(testing_output_chunks, dim=0)
    del cholesky, z, q
    if device_value.type == "cuda":
        torch.cuda.empty_cache()
    return result


__all__ = [
    "FeatCalPrefixChainError",
    "SelectedStepCapture",
    "SingleStepSelectedRowCollector",
    "expected_single_step_trace",
    "prefix_chain_sha256",
    "rollback_safe_step_load",
    "scan_prefix_chain_resume",
    "synthetic_out_of_core_rowspace_benchmark",
]
