#!/usr/bin/env python3
"""Collect replayable PI0.5 action-block inputs during a LIBERO rollout.

The resulting calibration file contains the action/timestep front-end inputs,
the cached language-prefix K/V tensors, and the exact suffix attention inputs
seen by Action Expert block 0.  It is deliberately independent of simulator
state after collection so a strict sequential block merge can replay one fixed
set of examples without restarting LIBERO for every block.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import types
from typing import Any
import zlib

from safetensors.torch import save_file
import torch

import eval_with_local_tokenizer  # noqa: F401  (installs local runtime overrides)


LANGUAGE_TOKENS_KEY = "observation.language.tokens"
LANGUAGE_ATTENTION_MASK_KEY = "observation.language.attention_mask"


def _required_path(name: str, *, output: bool = False) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    path = Path(value).expanduser()
    path = path.absolute() if output else path.resolve()
    if not output and not path.exists():
        raise FileNotFoundError(path)
    return path


def _resolve_suffix(modules: dict[str, torch.nn.Module], suffix: str) -> tuple[str, torch.nn.Module]:
    matches = [(name, module) for name, module in modules.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one module ending in {suffix!r}, found {[name for name, _ in matches]}")
    return matches[0]


class BlockCalibrationCollector:
    """Keep a deterministic reservoir of complete denoising-step inputs."""

    def __init__(self, policy: torch.nn.Module) -> None:
        self.policy = policy
        self.task = os.environ.get("PI05_BLOCK_REGMEANPP_TASK", "").strip().lower()
        if not self.task:
            raise ValueError("PI05_BLOCK_REGMEANPP_TASK must be non-empty")
        self.tensor_output = _required_path("PI05_BLOCK_REGMEANPP_TENSOR_OUTPUT", output=True)
        self.manifest_output = _required_path("PI05_BLOCK_REGMEANPP_MANIFEST_OUTPUT", output=True)
        self.calibration_policy = _required_path("PI05_BLOCK_REGMEANPP_CALIBRATION_POLICY")
        self.max_calls = int(os.environ.get("PI05_BLOCK_REGMEANPP_MAX_CALLS", "8"))
        self.max_calls_per_prompt = int(
            os.environ.get("PI05_BLOCK_REGMEANPP_MAX_CALLS_PER_PROMPT", "0")
        )
        self.requests_per_episode = int(
            os.environ.get("PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE", "1")
        )
        self.episode_aware = os.environ.get("PI05_BLOCK_REGMEANPP_EPISODE_AWARE", "0") == "1"
        self.init_state_offset = int(os.environ.get("PI05_LIBERO_INIT_STATE_OFFSET", "0"))
        self.calibration_seed = int(os.environ.get("PI05_BLOCK_REGMEANPP_START_SEED", "1000"))
        self.full_prefix = os.environ.get("PI05_BLOCK_REGMEANPP_FULL_PREFIX", "0") == "1"
        flow_indices_raw = os.environ.get("PI05_BLOCK_REGMEANPP_FLOW_INDICES", "").strip()
        self.flow_indices = tuple(
            int(item.strip()) for item in flow_indices_raw.split(",") if item.strip()
        )
        self.request_mode = os.environ.get(
            "PI05_BLOCK_REGMEANPP_REQUEST_MODE", "call_reservoir"
        ).strip().lower()
        if self.max_calls <= 0:
            raise ValueError("PI05_BLOCK_REGMEANPP_MAX_CALLS must be positive")
        if self.max_calls_per_prompt < 0:
            raise ValueError("PI05_BLOCK_REGMEANPP_MAX_CALLS_PER_PROMPT must be non-negative")
        if self.requests_per_episode <= 0:
            raise ValueError("PI05_BLOCK_REGMEANPP_REQUESTS_PER_EPISODE must be positive")
        if self.episode_aware and self.max_calls_per_prompt <= 0:
            raise ValueError("Episode-aware sampling requires MAX_CALLS_PER_PROMPT > 0")
        if self.flow_indices:
            if not self.episode_aware:
                raise ValueError("Structured flow sampling requires episode-aware collection")
            if len(set(self.flow_indices)) != len(self.flow_indices) or any(
                index < 0 for index in self.flow_indices
            ):
                raise ValueError(f"Invalid flow indices: {self.flow_indices}")
            if tuple(sorted(self.flow_indices)) != self.flow_indices:
                raise ValueError(f"Flow indices must be sorted: {self.flow_indices}")
            if self.request_mode not in {"initial", "reservoir", "last"}:
                raise ValueError(
                    "Structured flow sampling request mode must be initial, reservoir, or last"
                )
        elif self.request_mode != "call_reservoir":
            raise ValueError(
                "PI05_BLOCK_REGMEANPP_REQUEST_MODE requires PI05_BLOCK_REGMEANPP_FLOW_INDICES"
            )

        modules = dict(policy.named_modules())
        self.action_name, self.action_module = _resolve_suffix(modules, "action_in_proj")
        self.time_name, self.time_module = _resolve_suffix(modules, "time_mlp_in")
        self.layer_name, self.layer0 = _resolve_suffix(
            modules, "paligemma_with_expert.gemma_expert.model.layers.0"
        )
        self.vision_layer_name: str | None = None
        self.vision_layer0: torch.nn.Module | None = None
        self.language_layer_name: str | None = None
        self.language_layer0: torch.nn.Module | None = None
        if self.full_prefix:
            self.vision_layer_name, self.vision_layer0 = _resolve_suffix(
                modules,
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0",
            )
            self.language_layer_name, self.language_layer0 = _resolve_suffix(
                modules, "paligemma_with_expert.paligemma.model.language_model.layers.0"
            )
        self.latest_action_input: torch.Tensor | None = None
        self.latest_time_input: torch.Tensor | None = None
        self.pending_vision_calls: list[dict[str, torch.Tensor]] = []
        self.latest_prefix: dict[str, torch.Tensor] | None = None
        self.latest_prompt_signature: int | None = None
        self.prompt_texts: dict[int, str] = {}
        self.prompt_example_token_ids: dict[int, list[int]] = {}
        self.records: list[dict[str, torch.Tensor]] = []
        self.records_by_prompt: dict[int, list[dict[str, torch.Tensor]]] = {}
        self.seen_calls_by_prompt: dict[int, int] = {}
        self.episode_slots_by_prompt: dict[int, dict[int, int]] = {}
        self.structured_episode_slots_by_prompt: dict[
            int, dict[int, dict[int, list[int]]]
        ] = {}
        self.episode_ordinals_by_prompt: dict[int, dict[int, int]] = {}
        self.seen_calls_by_prompt_episode: dict[tuple[int, int], int] = {}
        self.record_metadata_by_prompt: dict[int, list[dict[str, Any]]] = {}
        self.current_episode_serial = -1
        self.current_initial_observation_hash: str | None = None
        self.current_request_index = -1
        self.current_flow_index = -1
        self.current_request_selected = False
        self.current_request_slots: list[int] = []
        self.current_captured_flow_indices: set[int] = set()
        self.seen_calls = 0
        self.handles: list[Any] = []
        self.original_predict_action_chunk: Any | None = None
        self.original_policy_reset: Any | None = None

    @staticmethod
    def _cpu(value: torch.Tensor) -> torch.Tensor:
        # Preserve the evaluator's dtype.  In particular, x_t and the additive
        # attention mask are float32; downcasting them would make strict replay
        # differ from the policy that generated the calibration trajectory.
        return value.detach().cpu().contiguous()

    def _input_hook(self, kind: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"Missing tensor input for {kind}")
            value = self._cpu(inputs[0])
            if kind == "action":
                self.latest_action_input = value
            else:
                self.latest_time_input = value

        return hook

    def capture_task_texts(self, tasks: Any) -> None:
        """Identify the task before PI0.5 appends observation-dependent state tokens."""
        if isinstance(tasks, str):
            task_list = [tasks]
        elif isinstance(tasks, list | tuple) and all(isinstance(task, str) for task in tasks):
            task_list = list(tasks)
        else:
            raise TypeError(f"Expected task text or a sequence of task texts, got {type(tasks)}")
        if len(task_list) != 1:
            raise ValueError(
                "Task-stratified calibration currently requires eval.batch_size=1; "
                f"received {len(task_list)} task descriptions"
            )
        task_text = task_list[0].strip().replace("_", " ").replace("\n", " ")
        prompt_signature = zlib.crc32(task_text.encode("utf-8"))
        previous = self.prompt_texts.setdefault(prompt_signature, task_text)
        if previous != task_text:
            raise RuntimeError(f"CRC32 collision between task prompts: {prompt_signature}")
        self.latest_prompt_signature = prompt_signature

    def _capture_prompt_tokens_from_policy_batch(self, batch: dict[str, Any]) -> None:
        """Keep one full tokenized example per task for calibration provenance."""
        if self.latest_prompt_signature is None:
            raise RuntimeError("Policy inference started before the raw task text was captured")
        tokens = batch.get(LANGUAGE_TOKENS_KEY)
        attention_mask = batch.get(LANGUAGE_ATTENTION_MASK_KEY)
        if not torch.is_tensor(tokens) or not torch.is_tensor(attention_mask):
            raise RuntimeError(
                "Policy batch is missing tokenized task fields needed for task-stratified sampling"
            )
        if tokens.ndim != 2 or attention_mask.shape != tokens.shape:
            raise ValueError(
                f"Expected task tokens/mask with the same [batch, sequence] shape, got "
                f"{tuple(tokens.shape)} and {tuple(attention_mask.shape)}"
            )
        if tokens.shape[0] != 1:
            raise ValueError(
                "Task-stratified calibration currently requires eval.batch_size=1; "
                f"received batch size {tokens.shape[0]}"
            )

        valid_token_ids = (
            tokens[0][attention_mask[0].to(dtype=torch.bool)]
            .detach()
            .to(device="cpu", dtype=torch.int64)
            .contiguous()
        )
        token_id_list = valid_token_ids.tolist()
        self.prompt_example_token_ids.setdefault(self.latest_prompt_signature, token_id_list)

        if self.episode_aware:
            if self.current_episode_serial < 0:
                raise RuntimeError("Policy inference started before policy.reset()")
            if self.current_initial_observation_hash is None:
                self.current_initial_observation_hash = self._observation_hash(batch)
            ordinals = self.episode_ordinals_by_prompt.setdefault(self.latest_prompt_signature, {})
            if self.current_episode_serial not in ordinals:
                ordinals[self.current_episode_serial] = len(ordinals)

    @staticmethod
    def _observation_hash(batch: dict[str, Any]) -> str:
        """Hash the first processed visual/proprioceptive observation of an episode."""
        digest = hashlib.sha256()
        tensor_count = 0
        for key, value in sorted(batch.items()):
            if not torch.is_tensor(value):
                continue
            if not (key.startswith("observation.images.") or key == "observation.state"):
                continue
            tensor = value.detach().to(device="cpu").contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
            tensor_count += 1
        if tensor_count == 0:
            raise RuntimeError("Could not find image/state tensors for initial-observation hashing")
        return digest.hexdigest()

    def _start_episode(self) -> None:
        self.current_episode_serial += 1
        self.current_initial_observation_hash = None
        self.latest_prompt_signature = None
        self.current_request_index = -1
        self.current_flow_index = -1
        self.current_request_selected = False
        self.current_request_slots = []
        self.current_captured_flow_indices = set()

    def _begin_structured_request(self) -> None:
        if not self.flow_indices:
            return
        if self.latest_prompt_signature is None:
            raise RuntimeError("Structured request started without a prompt signature")
        if self.current_episode_serial < 0:
            raise RuntimeError("Structured request started before policy.reset()")
        self.current_request_index += 1
        self.current_flow_index = 0
        self.current_captured_flow_indices = set()

        signature = self.latest_prompt_signature
        records = self.records_by_prompt.setdefault(signature, [])
        slots_by_episode = self.structured_episode_slots_by_prompt.setdefault(signature, {})
        if self.current_episode_serial not in slots_by_episode:
            if len(slots_by_episode) >= self.max_calls_per_prompt:
                self.current_request_selected = False
                self.current_request_slots = []
                return
            slots_by_episode[self.current_episode_serial] = {}

        request_index = self.current_request_index
        reservoir_index: int | None
        if self.request_mode == "initial":
            reservoir_index = (
                request_index if request_index < self.requests_per_episode else None
            )
        elif self.request_mode == "last":
            reservoir_index = request_index % self.requests_per_episode
        elif request_index < self.requests_per_episode:
            reservoir_index = request_index
        else:
            # Deterministic Algorithm-R reservoir sampling.  After seeing n+1
            # requests, each request has equal probability R/(n+1) of being
            # represented, without requiring simulator state to be replayed.
            salt = (
                zlib.crc32(self.task.encode("utf-8"))
                ^ signature
                ^ self.current_episode_serial
            )
            draw = ((request_index + 1) * 1103515245 + salt) & 0x7FFFFFFF
            candidate = draw % (request_index + 1)
            reservoir_index = (
                candidate if candidate < self.requests_per_episode else None
            )

        self.current_request_selected = reservoir_index is not None
        if reservoir_index is None:
            self.current_request_slots = []
            return

        episode_slots = slots_by_episode[self.current_episode_serial]
        metadata = self.record_metadata_by_prompt.setdefault(signature, [])
        ordinal = self.episode_ordinals_by_prompt[signature][self.current_episode_serial]
        if reservoir_index not in episode_slots:
            slots = []
            for flow_index in self.flow_indices:
                slots.append(len(records))
                records.append({})
                metadata.append({})
            episode_slots[reservoir_index] = slots
        self.current_request_slots = episode_slots[reservoir_index]
        for slot, flow_index in zip(self.current_request_slots, self.flow_indices):
            metadata[slot] = {
                "episode_serial": self.current_episode_serial,
                "task_episode_index": ordinal,
                "init_state_id": self.init_state_offset + ordinal,
                "simulator_seed": self.calibration_seed + ordinal,
                "initial_observation_sha256": self.current_initial_observation_hash,
                "request_index": request_index,
                "request_reservoir_index": reservoir_index,
                "flow_index": flow_index,
            }

    def _end_structured_request(self) -> None:
        if not self.flow_indices or not self.current_request_selected:
            return
        missing = set(self.flow_indices) - self.current_captured_flow_indices
        if missing:
            raise RuntimeError(
                f"Request {self.current_request_index} did not reach flow indices {sorted(missing)}; "
                f"observed {self.current_flow_index} Action Expert calls"
            )

    def _reservoir_slot(self, prompt_signature: int | None) -> tuple[int | None, list[dict[str, torch.Tensor]]]:
        self.seen_calls += 1
        if self.max_calls_per_prompt > 0:
            if prompt_signature is None:
                raise RuntimeError("Missing prompt signature for stratified reservoir")
            records = self.records_by_prompt.setdefault(prompt_signature, [])
            seen = self.seen_calls_by_prompt.get(prompt_signature, 0) + 1
            self.seen_calls_by_prompt[prompt_signature] = seen
            if self.episode_aware:
                if self.current_episode_serial < 0:
                    raise RuntimeError("Missing episode serial for episode-aware reservoir")
                episode_key = (prompt_signature, self.current_episode_serial)
                episode_seen = self.seen_calls_by_prompt_episode.get(episode_key, 0) + 1
                self.seen_calls_by_prompt_episode[episode_key] = episode_seen
                slots = self.episode_slots_by_prompt.setdefault(prompt_signature, {})
                if self.current_episode_serial not in slots:
                    if len(slots) >= self.max_calls_per_prompt:
                        return None, records
                    slot = len(records)
                    slots[self.current_episode_serial] = slot
                    ordinal = self.episode_ordinals_by_prompt[prompt_signature][
                        self.current_episode_serial
                    ]
                    self.record_metadata_by_prompt.setdefault(prompt_signature, []).append(
                        {
                            "episode_serial": self.current_episode_serial,
                            "task_episode_index": ordinal,
                            "init_state_id": self.init_state_offset + ordinal,
                            "simulator_seed": self.calibration_seed + ordinal,
                            "initial_observation_sha256": self.current_initial_observation_hash,
                        }
                    )
                    return slot, records
                slot = slots[self.current_episode_serial]
                salt = (
                    zlib.crc32(self.task.encode("utf-8"))
                    ^ prompt_signature
                    ^ self.current_episode_serial
                )
                selected = ((episode_seen * 1103515245 + salt) & 0x7FFFFFFF) % episode_seen == 0
                return (slot if selected else None), records
            if len(records) < self.max_calls_per_prompt:
                return len(records), records
            salt = zlib.crc32(self.task.encode("utf-8")) ^ prompt_signature
            slot = ((seen * 1103515245 + salt) & 0x7FFFFFFF) % seen
            return (slot if slot < self.max_calls_per_prompt else None), records

        records = self.records
        if len(self.records) < self.max_calls:
            return len(self.records), records
        salt = zlib.crc32(self.task.encode("utf-8"))
        slot = ((self.seen_calls * 1103515245 + salt) & 0x7FFFFFFF) % self.seen_calls
        return (slot if slot < self.max_calls else None), records

    def _vision_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Vision block 0 did not receive hidden_states")
        record = {"hidden": self._cpu(args[0])}
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) > 1:
            attention_mask = args[1]
        if torch.is_tensor(attention_mask):
            record["attention_mask"] = self._cpu(attention_mask)
        self.pending_vision_calls.append(record)

    def _language_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Language block 0 did not receive hidden_states")
        if not self.pending_vision_calls:
            raise RuntimeError("Language prefix started without any preceding vision calls")
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        if not torch.is_tensor(attention_mask) or not torch.is_tensor(position_ids):
            raise RuntimeError("Language block 0 is missing attention_mask/position_ids")
        language_hidden = self._cpu(args[0])
        prefix: dict[str, torch.Tensor] = {
            "language_hidden_mean": language_hidden,
            "language_attention_mask": self._cpu(attention_mask),
            "language_position_ids": self._cpu(position_ids),
            "vision_count": torch.tensor(len(self.pending_vision_calls), dtype=torch.int64),
        }
        for index, vision in enumerate(self.pending_vision_calls):
            for name, tensor in vision.items():
                prefix[f"vision_{index:02d}.{name}"] = tensor
        self.latest_prefix = prefix
        self.pending_vision_calls = []

    def _layer_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Action Expert block 0 did not receive hidden_states")
        if self.latest_action_input is None or self.latest_time_input is None:
            raise RuntimeError("Front-end inputs were not observed before Action Expert block 0")
        required = ("attention_mask", "position_ids", "past_key_values", "adarms_cond")
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise RuntimeError(f"Action Expert block 0 is missing replay arguments: {missing}")

        structured_flow_index: int | None = None
        if self.flow_indices:
            if self.latest_prompt_signature is None:
                raise RuntimeError("Structured flow hook is missing a prompt signature")
            structured_flow_index = self.current_flow_index
            self.current_flow_index += 1
            self.seen_calls += 1
            self.seen_calls_by_prompt[self.latest_prompt_signature] = (
                self.seen_calls_by_prompt.get(self.latest_prompt_signature, 0) + 1
            )
            if (
                not self.current_request_selected
                or structured_flow_index not in self.flow_indices
            ):
                return
            flow_slot = self.flow_indices.index(structured_flow_index)
            slot = self.current_request_slots[flow_slot]
            destination = self.records_by_prompt[self.latest_prompt_signature]
            self.current_captured_flow_indices.add(structured_flow_index)
        else:
            slot, destination = self._reservoir_slot(self.latest_prompt_signature)
            if slot is None:
                return
        cache = list(kwargs["past_key_values"])
        if len(cache) != 18:
            raise ValueError(f"Expected 18 cached prefix layers, found {len(cache)}")
        record = {
            "action_input": self.latest_action_input.clone(),
            "time_input": self.latest_time_input.clone(),
            "hidden_mean": self._cpu(args[0]),
            "cond_mean": self._cpu(kwargs["adarms_cond"]),
            "attention_mask": self._cpu(kwargs["attention_mask"]),
            "position_ids": self._cpu(kwargs["position_ids"]),
        }
        if self.full_prefix:
            if self.latest_prefix is None:
                raise RuntimeError("Action suffix started without a captured vision/language prefix")
            record.update({name: tensor.clone() for name, tensor in self.latest_prefix.items()})
        for layer_idx, (keys, values, _sliding_window) in enumerate(cache):
            record[f"prefix_key_{layer_idx:02d}"] = self._cpu(keys)
            record[f"prefix_value_{layer_idx:02d}"] = self._cpu(values)
        if slot == len(destination):
            destination.append(record)
        else:
            destination[slot] = record
        if structured_flow_index is not None:
            metadata = self.record_metadata_by_prompt[self.latest_prompt_signature][slot]
            metadata["request_index"] = self.current_request_index
            metadata["flow_index"] = structured_flow_index

    def install(self) -> None:
        if self.max_calls_per_prompt > 0:
            self.original_predict_action_chunk = self.policy.predict_action_chunk
            if self.episode_aware:
                self.original_policy_reset = self.policy.reset

                def reset_with_episode_tracking(
                    _policy: torch.nn.Module,
                    *args: Any,
                    **kwargs: Any,
                ) -> Any:
                    self._start_episode()
                    return self.original_policy_reset(*args, **kwargs)

                self.policy.reset = types.MethodType(  # type: ignore[method-assign]
                    reset_with_episode_tracking, self.policy
                )

            def predict_action_chunk_with_prompt(
                _policy: torch.nn.Module,
                batch: dict[str, Any],
                *args: Any,
                **kwargs: Any,
            ) -> torch.Tensor:
                self._capture_prompt_tokens_from_policy_batch(batch)
                self._begin_structured_request()
                try:
                    return self.original_predict_action_chunk(batch, *args, **kwargs)
                finally:
                    self._end_structured_request()

            self.policy.predict_action_chunk = types.MethodType(  # type: ignore[method-assign]
                predict_action_chunk_with_prompt, self.policy
            )
        self.handles.append(self.action_module.register_forward_pre_hook(self._input_hook("action")))
        self.handles.append(self.time_module.register_forward_pre_hook(self._input_hook("time")))
        if self.full_prefix:
            assert self.vision_layer0 is not None and self.language_layer0 is not None
            self.handles.append(
                self.vision_layer0.register_forward_pre_hook(self._vision_hook, with_kwargs=True)
            )
            self.handles.append(
                self.language_layer0.register_forward_pre_hook(self._language_hook, with_kwargs=True)
            )
        self.handles.append(self.layer0.register_forward_pre_hook(self._layer_hook, with_kwargs=True))
        print(
            f"Installed strict block calibration collector: task={self.task} "
            f"max_calls={self.max_calls} max_calls_per_prompt={self.max_calls_per_prompt} "
            f"requests_per_episode={self.requests_per_episode} "
            f"episode_aware={self.episode_aware} full_prefix={self.full_prefix} "
            f"request_mode={self.request_mode} flow_indices={self.flow_indices} "
            f"layer={self.layer_name}",
            flush=True,
        )

    def save(self) -> None:
        if self.original_predict_action_chunk is not None:
            self.policy.predict_action_chunk = self.original_predict_action_chunk
        if self.original_policy_reset is not None:
            self.policy.reset = self.original_policy_reset
        for handle in self.handles:
            handle.remove()
        prompt_signatures: list[int | None]
        if self.max_calls_per_prompt > 0:
            ordered = sorted(self.records_by_prompt.items())
            records = [record for _signature, group in ordered for record in group]
            prompt_signatures = [
                signature for signature, group in ordered for _record in group
            ]
            record_metadata = [
                metadata
                for signature, _group in ordered
                for metadata in self.record_metadata_by_prompt.get(signature, [{}] * len(_group))
            ]
        else:
            records = self.records
            prompt_signatures = [None] * len(records)
            record_metadata = [{} for _record in records]
        if not records:
            raise RuntimeError("No Action Expert calibration records were collected")
        self.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        flattened = {
            f"sample_{sample_idx:03d}.{name}": tensor
            for sample_idx, record in enumerate(records)
            for name, tensor in record.items()
        }
        save_file(flattened, self.tensor_output, metadata={"format": "pt"})
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "method": (
                "pi05_full_vision_language_action_block_regmeanpp_replay_calibration"
                if self.full_prefix
                else "pi05_action_block_regmeanpp_replay_calibration"
            ),
            "task": self.task,
            "sample_count": len(records),
            "seen_denoise_calls": self.seen_calls,
            "sampling": (
                "structured_request_flow"
                if self.flow_indices
                else (
                    "prompt_stratified_reservoir"
                    if self.max_calls_per_prompt > 0
                    else "global_reservoir"
                )
            ),
            "max_calls_per_prompt": self.max_calls_per_prompt,
            "requests_per_episode": self.requests_per_episode,
            "episode_aware": self.episode_aware,
            "request_mode": self.request_mode,
            "flow_indices": list(self.flow_indices),
            "samples_per_episode": (
                self.requests_per_episode * len(self.flow_indices)
                if self.flow_indices
                else 1
            ),
            "init_state_offset": self.init_state_offset,
            "start_seed": self.calibration_seed,
            "prompt_count": len(self.records_by_prompt),
            "prompt_sample_counts": {
                str(signature): len(group)
                for signature, group in sorted(self.records_by_prompt.items())
            },
            "prompt_seen_call_counts": {
                str(signature): count
                for signature, count in sorted(self.seen_calls_by_prompt.items())
            },
            "prompt_texts": {
                str(signature): prompt_text
                for signature, prompt_text in sorted(self.prompt_texts.items())
            },
            "prompt_example_token_ids": {
                str(signature): token_ids
                for signature, token_ids in sorted(self.prompt_example_token_ids.items())
            },
            "action_module": self.action_name,
            "time_module": self.time_name,
            "first_action_block": self.layer_name,
            "cached_prefix_layer_count": 18,
            "calibration_policy": str(self.calibration_policy),
            "tensor_file": str(self.tensor_output),
            "samples": [
                {
                    "index": index,
                    "prompt_signature": prompt_signatures[index],
                    **record_metadata[index],
                    "action_shape": list(record["action_input"].shape),
                    "prefix_length": int(record["prefix_key_00"].shape[-2]),
                    "suffix_length": int(record["hidden_mean"].shape[-2]),
                    "vision_count": int(record.get("vision_count", torch.tensor(0))),
                }
                for index, record in enumerate(records)
            ],
        }
        self.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"Saved {len(records)} replay records across {len(self.records_by_prompt)} prompts "
            f"after {self.seen_calls} denoise calls "
            f"to {self.tensor_output}",
            flush=True,
        )


def main() -> None:
    from eval_pi05_libero_with_init_offset import install_init_state_offset
    from lerobot.scripts import lerobot_eval
    from lerobot.lerobot_types import TransitionKey
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

    original_make_policy = lerobot_eval.make_policy
    original_prepare_state_call = Pi05PrepareStateTokenizerProcessorStep.__call__
    collector: BlockCalibrationCollector | None = None

    install_init_state_offset()

    def make_policy_with_collector(*args: Any, **kwargs: Any) -> torch.nn.Module:
        nonlocal collector
        policy = original_make_policy(*args, **kwargs)
        collector = BlockCalibrationCollector(policy)
        collector.install()
        return policy

    def prepare_state_with_task_capture(
        step: Pi05PrepareStateTokenizerProcessorStep,
        transition: Any,
    ) -> Any:
        if collector is not None:
            complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            collector.capture_task_texts(complementary_data.get(step.task_key))
        return original_prepare_state_call(step, transition)

    lerobot_eval.make_policy = make_policy_with_collector
    Pi05PrepareStateTokenizerProcessorStep.__call__ = prepare_state_with_task_capture
    try:
        lerobot_eval.main()
    finally:
        Pi05PrepareStateTokenizerProcessorStep.__call__ = original_prepare_state_call
        if collector is not None:
            collector.save()


if __name__ == "__main__":
    main()
