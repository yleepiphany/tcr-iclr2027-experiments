"""Bind native Fast-WAM action traces to requests actually used by LIBERO.

The native evaluator remains responsible for reset, processing, action
denormalization, and environment steps. This wrapper records its own request
inputs and returned executable chunks without changing any of those choices.
The caller must supply a calibration-only reset bank and checkpoint identity.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from native_trace import NativeActionTrace, snapshot


def observation_sha256(obs: dict[str, Any]) -> str:
    if not isinstance(obs, dict) or not obs:
        raise ValueError("A nonempty native observation dict is required")
    digest = hashlib.sha256()
    for key in sorted(obs):
        value = np.asarray(obs[key])
        if value.dtype.hasobject:
            raise TypeError(f"Object-valued observation field: {key}")
        value = np.ascontiguousarray(value)
        header = json.dumps({"key": key, "shape": list(value.shape),
                             "dtype": value.dtype.str}, sort_keys=True).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.tobytes())
    return digest.hexdigest()


def observation_snapshot(obs: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Keep the raw closed-loop observation for independent receipt auditing."""
    copied = {}
    for key, value in obs.items():
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype.hasobject:
            raise TypeError(f"Object-valued observation field: {key}")
        copied[key] = torch.from_numpy(array.copy())
    return copied


def request_quantiles(count: int) -> list[int]:
    """Position-only first/middle/last rule, independent of episode success."""
    if type(count) is not int or count < 1:
        raise ValueError("A completed episode must have at least one request")
    return sorted({0, (count - 1) // 2, count - 1})


class ClosedLoopEpisodeCapture(AbstractContextManager):
    """Instrument one synchronous native `run_single_episode` call."""

    def __init__(self, native: Any, model: Any, output: Path,
                 identity: dict[str, Any], *, expected_steps: int = 10) -> None:
        self.native = native
        self.model = model
        self.output = Path(output)
        self.identity = dict(identity)
        self.expected_steps = expected_steps
        self.request_count = 0
        self._pending: dict[str, Any] | None = None
        self._in_chunk = False
        self._original_chunk = None
        self._original_infer = None
        self._had_infer_override = False
        self._prior_infer_override = None

    def __enter__(self) -> "ClosedLoopEpisodeCapture":
        if self._original_chunk is not None or getattr(self.model, "_fastwam_episode_capture", False):
            raise RuntimeError("Closed-loop capture cannot be nested")
        self.output.mkdir(parents=True, exist_ok=False)
        self._original_chunk = self.native._predict_action_chunk
        self._original_infer = self.model.infer_action
        self._had_infer_override = "infer_action" in self.model.__dict__
        self._prior_infer_override = self.model.__dict__.get("infer_action")

        def infer_action(**kwargs: Any) -> Any:
            if not self._in_chunk or self._pending is not None:
                raise ValueError("Native inference was not paired one-to-one with an action chunk")
            with NativeActionTrace(self.model, expected_steps=self.expected_steps) as trace:
                prediction = self._original_infer(**kwargs)
            if not isinstance(prediction, dict) or not torch.is_tensor(prediction.get("action")):
                raise ValueError("Native action output contract differs")
            self._pending = {"native_request": snapshot(kwargs),
                             "trace": trace.payload(native_action=prediction["action"])}
            return prediction

        def action_chunk(*args: Any, **kwargs: Any) -> Any:
            if args or self._in_chunk or self._pending is not None:
                raise ValueError("Expected one keyword-only native action chunk")
            obs = kwargs.get("obs")
            digest = observation_sha256(obs)
            self._in_chunk = True
            try:
                result = self._original_chunk(**kwargs)
            finally:
                self._in_chunk = False
            if self._pending is None or not isinstance(result, tuple) or len(result) != 3:
                raise ValueError("Native chunk did not produce exactly one action inference")
            action = np.ascontiguousarray(result[0])
            if action.ndim != 2 or not np.issubdtype(action.dtype, np.number) or \
                    not np.isfinite(action).all():
                raise ValueError("Native executable action chunk is invalid")
            row = {**self._pending, "request_index": self.request_count,
                   "raw_observation_sha256": digest,
                   "raw_observation": observation_snapshot(obs),
                   "executed_action_chunk": torch.from_numpy(action.copy()),
                   "executed_action_sha256": hashlib.sha256(action.tobytes()).hexdigest()}
            target = self.output / f"request-{self.request_count:06d}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(row, temporary)
            os.replace(temporary, target)
            self.request_count += 1
            self._pending = None
            return result

        self.model._fastwam_episode_capture = True
        self.model.infer_action = infer_action
        self.native._predict_action_chunk = action_chunk
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.native._predict_action_chunk = self._original_chunk
        if self._had_infer_override:
            self.model.infer_action = self._prior_infer_override
        else:
            del self.model.infer_action
        del self.model._fastwam_episode_capture
        if exc_type is None and (self._pending is not None or self.request_count < 1):
            raise ValueError("Closed-loop episode capture ended without complete requests")
        return False

    def complete(self, *, initial_state: np.ndarray, success: bool) -> dict[str, Any]:
        """Seal an episode after the original evaluator returned successfully."""
        if type(success) is not bool or self.request_count < 1 or self._pending is not None:
            raise ValueError("Cannot seal an incomplete native episode")
        initial = np.ascontiguousarray(initial_state)
        if initial.dtype.hasobject:
            raise TypeError("Object-valued initial state")
        payload = {"schema": "fastwam_closed_loop_episode_v1",
                   "complete": True, "identity": self.identity,
                   "initial_state_shape": list(initial.shape),
                   "initial_state_dtype": initial.dtype.str,
                   "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                   "request_count": self.request_count,
                   "selected_request_indices": request_quantiles(self.request_count),
                   "selection_rule": "first_mid_last_floor_position_only",
                   "success": success}
        with (self.output / "episode.json").open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload
