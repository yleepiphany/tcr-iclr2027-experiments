"""Replay a recorded Fast-WAM action-only call through the native modules.

This is a parity gate for real traces, not a substitute for closed-loop capture.
It reuses the observed visual-prefill inputs and observed action latents/times;
it never constructs a denoising trajectory or substitutes demonstration actions.
"""
from __future__ import annotations

from typing import Any

import torch


def _to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("Native replay prediction shape/dtype differs")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Native replay contains a nonfinite prediction")
    return float((actual.float() - expected.float()).abs().max().item())


@torch.no_grad()
def replay_native_trace(model: Any, payload: dict[str, Any], *,
                        device: torch.device | str, atol: float = 0.0,
                        rtol: float = 0.0) -> dict[str, Any]:
    """Fail closed unless native prefill and every selected prediction agree."""
    if payload.get("schema") != "fastwam_native_action_trace_v1" or \
            payload.get("mode") != "action_only_with_visual_prefill":
        raise ValueError("Not a Fast-WAM native action-only trace")
    if atol < 0 or rtol < 0:
        raise ValueError("Replay tolerances must be nonnegative")
    indices = payload.get("selected_call_indices")
    calls = payload.get("calls")
    steps = payload.get("expected_denoising_calls")
    if type(steps) is not int or steps < 1 or not isinstance(indices, list) or \
            not indices or indices != sorted(set(indices)) or indices[-1] >= steps or \
            indices[0] < 0 or not isinstance(calls, list) or len(calls) != len(indices) or \
            [row.get("call_index") for row in calls] != indices:
        raise ValueError("Native trace action-call coverage differs")
    prefill = payload.get("prefill_inputs")
    if not isinstance(prefill, dict) or set(prefill) != {
            "video_tokens", "video_freqs", "video_t_mod", "video_context_payload",
            "video_attention_mask"}:
        raise ValueError("Native visual-prefill input contract differs")
    cache = model.mot.prefill_video_cache(**_to_device(prefill, device))
    expected_shapes = payload.get("video_cache_shapes")
    actual_shapes = [{key: tuple(layer[key].shape) for key in ("k", "v")}
                     for layer in cache]
    if actual_shapes != expected_shapes:
        raise ValueError("Native visual cache shapes differ")
    errors = []
    for index, row in zip(indices, calls):
        args = row.get("inputs")
        if not isinstance(args, dict) or set(args) != {
                "latents_action", "timestep_action", "context", "context_mask",
                "attention_mask", "video_seq_len"}:
            raise ValueError(f"Native action-call contract differs at {index}")
        expected = row.get("prediction")
        if not torch.is_tensor(expected):
            raise ValueError(f"Missing native prediction at {index}")
        actual = model._predict_action_noise_with_cache(
            **_to_device(args, device), video_kv_cache=cache)
        expected = expected.to(device=device)
        errors.append(_max_error(actual, expected))
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            raise ValueError(f"Native prediction parity failed at call {index}")
    return {"accepted": True, "selected_call_indices": indices,
            "max_absolute_error": max(errors), "per_call_max_absolute_error": errors,
            "atol": atol, "rtol": rtol}
