"""Capture Fast-WAM's actual action-only prefill and denoising call inputs.

This is an instrumentation primitive, not a merger. It patches only the given
model instance during one native ``infer_action`` call, leaves computations
untouched, and restores both methods even when inference raises. The captured
prefill inputs can rebuild the video K/V cache; sampled action states remain
paired with their real denoising times. No future-video branch is invoked here.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch


def snapshot(value: Any) -> Any:
    """Copy the selected native call inputs to CPU without retaining GPU views."""
    if torch.is_tensor(value):
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Native trace contains a nonfinite tensor")
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        copied = [snapshot(item) for item in value]
        return type(value)(copied)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise TypeError(f"Unsupported native trace value: {type(value).__name__}")


class NativeActionTrace(AbstractContextManager):
    """Record one video prefill and selected calls from the native action loop.

    ``selected_steps`` are zero-based denoising call positions. This recorder
    never synthesizes an intermediate action state or changes the model's RNG.
    """

    def __init__(self, model: Any, *, expected_steps: int = 10,
                 selected_steps: tuple[int, ...] = (0, 5, 9)) -> None:
        if (type(expected_steps) is not int or expected_steps < 1
                or not selected_steps or tuple(sorted(set(selected_steps))) != selected_steps
                or any(type(index) is not int or index < 0 or index >= expected_steps
                       for index in selected_steps)):
            raise ValueError("Invalid frozen native action-call selection")
        self.model = model
        self.expected_steps = expected_steps
        self.selected_steps = selected_steps
        self.prefill_inputs: dict[str, Any] | None = None
        self.cache_shapes: list[dict[str, tuple[int, ...]]] | None = None
        self.calls: list[dict[str, Any]] = []
        self.observed_steps = 0
        self._cache_ref: Any = None
        self._original_prefill = None
        self._original_denoise = None
        self._prior_prefill_override: Any = None
        self._prior_denoise_override: Any = None
        self._had_prefill_override = False
        self._had_denoise_override = False

    def __enter__(self) -> "NativeActionTrace":
        if self._original_prefill is not None or getattr(self.model, "_fastwam_trace_active", False):
            raise RuntimeError("Native trace cannot be nested or entered twice")
        self._original_prefill = self.model.mot.prefill_video_cache
        self._original_denoise = self.model._predict_action_noise_with_cache
        self._had_prefill_override = "prefill_video_cache" in self.model.mot.__dict__
        self._had_denoise_override = "_predict_action_noise_with_cache" in self.model.__dict__
        self._prior_prefill_override = self.model.mot.__dict__.get("prefill_video_cache")
        self._prior_denoise_override = self.model.__dict__.get("_predict_action_noise_with_cache")

        def prefill(*args: Any, **kwargs: Any) -> Any:
            if args or self.prefill_inputs is not None or self.observed_steps:
                raise ValueError("Expected one keyword-only visual prefill before denoising")
            result = self._original_prefill(**kwargs)
            if not isinstance(result, list) or not result or any(
                    not isinstance(layer, dict) or set(layer) != {"k", "v"}
                    or not all(torch.is_tensor(layer[key]) for key in ("k", "v"))
                    for layer in result):
                raise ValueError("Native video cache layout differs")
            self.prefill_inputs = snapshot(kwargs)
            self.cache_shapes = [{key: tuple(layer[key].shape) for key in ("k", "v")}
                                 for layer in result]
            # Keep the original object alive: comparing integer ids alone can
            # accept a replacement after Python recycles the old object's id.
            self._cache_ref = result
            return result

        def denoise(*args: Any, **kwargs: Any) -> Any:
            if args or self.prefill_inputs is None or self.observed_steps >= self.expected_steps:
                raise ValueError("Native action denoising call count/order differs")
            if kwargs.get("video_kv_cache") is not self._cache_ref:
                raise ValueError("Action denoising did not reuse the visual prefill cache")
            index = self.observed_steps
            self.observed_steps += 1
            if index in self.selected_steps:
                required = {"latents_action", "timestep_action", "context", "context_mask",
                            "attention_mask", "video_seq_len", "video_kv_cache"}
                if set(kwargs) != required:
                    raise ValueError("Native action-call input contract differs")
                row = {"call_index": index,
                       "inputs": snapshot({key: value for key, value in kwargs.items()
                                           if key != "video_kv_cache"})}
            else:
                row = None
            output = self._original_denoise(**kwargs)
            if row is not None:
                row["prediction"] = snapshot(output)
                self.calls.append(row)
            return output

        self.model._fastwam_trace_active = True
        self.model.mot.prefill_video_cache = prefill
        self.model._predict_action_noise_with_cache = denoise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._had_prefill_override:
            self.model.mot.prefill_video_cache = self._prior_prefill_override
        else:
            del self.model.mot.prefill_video_cache
        if self._had_denoise_override:
            self.model._predict_action_noise_with_cache = self._prior_denoise_override
        else:
            del self.model._predict_action_noise_with_cache
        del self.model._fastwam_trace_active
        self._cache_ref = None
        if exc_type is None and (self.prefill_inputs is None
                                 or self.observed_steps != self.expected_steps
                                 or [row["call_index"] for row in self.calls]
                                 != list(self.selected_steps)):
            raise ValueError("Native action-only trace is incomplete")
        return False

    def payload(self, *, native_action: Any) -> dict[str, Any]:
        if (self.prefill_inputs is None or self.cache_shapes is None
                or self.observed_steps != self.expected_steps
                or len(self.calls) != len(self.selected_steps)):
            raise ValueError("Cannot export an incomplete native trace")
        return {"schema": "fastwam_native_action_trace_v1",
                "mode": "action_only_with_visual_prefill",
                "expected_denoising_calls": self.expected_steps,
                "selected_call_indices": list(self.selected_steps),
                "prefill_inputs": self.prefill_inputs,
                "video_cache_shapes": self.cache_shapes,
                "calls": self.calls, "native_action": snapshot(native_action)}
