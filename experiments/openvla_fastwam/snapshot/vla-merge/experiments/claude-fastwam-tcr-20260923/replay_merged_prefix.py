"""Replay expert action states after the *current merged* Fast-WAM prefix.

The native request rebuilds text/proprio, VAE latents, video pre-DiT and all
video K/V layers under the current checkpoint. The native prefill is stopped
before its own scheduler loop. Observed expert x_t/time pairs then enter the
normal action-only predictor with this current visual cache. This preserves
the native branch and does not create new flow states or future video.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch


class _AfterPrefill(Exception):
    pass


@contextmanager
def _override(instance: Any, name: str, replacement: Any):
    had = name in instance.__dict__
    previous = instance.__dict__.get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if had:
            setattr(instance, name, previous)
        else:
            delattr(instance, name)


def _device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _device(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_device(item, device) for item in value)
    return value


@torch.no_grad()
def merged_native_prefill(model: Any, request: dict[str, Any], *, device: torch.device | str) -> dict:
    """Return native prefix outputs, aborting only after real video prefill."""
    if getattr(model, "_fastwam_merged_replay_active", False):
        raise RuntimeError("Merged-prefix replay cannot be nested")
    if (request.get("num_inference_steps") != 10 or request.get("action_horizon") != 32
            or not torch.is_tensor(request.get("input_image"))
            or not torch.is_tensor(request.get("proprio"))):
        raise ValueError("Captured native request identity differs")
    original_context = model._append_proprio_to_context
    original_mask = model._build_mot_attention_mask
    original_prefill = model.mot.prefill_video_cache
    captured: dict[str, Any] = {}

    def context(*args: Any, **kwargs: Any) -> Any:
        if "context" in captured:
            raise ValueError("Native context was constructed more than once")
        result = original_context(*args, **kwargs)
        captured["context"], captured["context_mask"] = result
        return result

    def mask(*args: Any, **kwargs: Any) -> Any:
        if "attention_mask" in captured:
            raise ValueError("Native attention mask was constructed more than once")
        result = original_mask(*args, **kwargs)
        captured["attention_mask"] = result
        return result

    def prefill(*args: Any, **kwargs: Any) -> Any:
        if args or "video_kv_cache" in captured:
            raise ValueError("Native visual prefill was not unique and keyword-only")
        cache = original_prefill(**kwargs)
        if not isinstance(cache, list) or len(cache) != model.mot.num_layers:
            raise ValueError("Native K/V cache has wrong block coverage")
        captured["video_kv_cache"] = cache
        captured["video_seq_len"] = int(kwargs["video_tokens"].shape[1])
        raise _AfterPrefill

    model._fastwam_merged_replay_active = True
    try:
        with _override(model, "_append_proprio_to_context", context), \
                _override(model, "_build_mot_attention_mask", mask), \
                _override(model.mot, "prefill_video_cache", prefill):
            try:
                model.infer_action(**_device(request, device))
            except _AfterPrefill:
                pass
            else:
                raise ValueError("Native inference skipped visual prefill")
    finally:
        del model._fastwam_merged_replay_active
    if set(captured) != {"context", "context_mask", "attention_mask",
                         "video_kv_cache", "video_seq_len"}:
        raise ValueError("Native merged prefix is incomplete")
    if captured["attention_mask"].shape[0] != captured["video_seq_len"] + 32:
        raise ValueError("Native video/action attention layout changed")
    return captured


@torch.no_grad()
def replay_selected_expert_states(model: Any, request: dict[str, Any], trace: dict,
                                  *, device: torch.device | str) -> tuple[list[torch.Tensor], dict]:
    """Run recorded expert states at 0/5/9 under the current model prefix."""
    if trace.get("schema") != "fastwam_native_action_trace_v1" or \
            trace.get("selected_call_indices") != [0, 5, 9] or \
            [row.get("call_index") for row in trace.get("calls", [])] != [0, 5, 9]:
        raise ValueError("Expected the frozen native expert flow states")
    prefix = merged_native_prefill(model, request, device=device)
    predictions = []
    for row in trace["calls"]:
        state = row["inputs"]
        if set(state) != {"latents_action", "timestep_action", "context",
                          "context_mask", "attention_mask", "video_seq_len"}:
            raise ValueError("Native expert flow-state input differs")
        latents = _device(state["latents_action"], device)
        timestep = _device(state["timestep_action"], device)
        if tuple(latents.shape) != (1, 32, 7) or tuple(timestep.shape) != (1,):
            raise ValueError("Native expert flow-state shape differs")
        prediction = model._predict_action_noise_with_cache(
            latents_action=latents, timestep_action=timestep,
            context=prefix["context"], context_mask=prefix["context_mask"],
            video_kv_cache=prefix["video_kv_cache"],
            attention_mask=prefix["attention_mask"],
            video_seq_len=prefix["video_seq_len"])
        if not torch.is_tensor(prediction) or tuple(prediction.shape) != (1, 32, 7) or \
                not torch.isfinite(prediction).all():
            raise ValueError("Current-prefix native action prediction is invalid")
        predictions.append(prediction)
    return predictions, prefix
