#!/usr/bin/env python3
"""Capture raw native call inputs for a separate FeatCal execution-source pilot.

Reuse the existing episode/request reservoir, but store processed images, tokens,
actual denoising latents and times, not model-specific hidden/prefix features.
No trajectories are selected by success. Never overwrites an existing capture.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "pi05_lora_finetune_v2_20260826"
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "lerobot/src"), str(ROOT / "src")]

import torch
from safetensors.torch import save_file
import collect_pi05_block_regmeanpp_calibration as native


def cpu(value):
    return value.detach().cpu().clone().contiguous()


class RawExecutionCollector(native.BlockCalibrationCollector):
    def install(self):
        super().install()
        model = self.policy.model
        self.original_sample = model.sample_actions
        self.original_denoise = model.denoise_step
        self.original_noise = model.sample_noise
        self.noise_seed = int(os.environ.get("FEATCAL_EXEC_NOISE_SEED", "272001"))
        generator = torch.Generator(device="cuda").manual_seed(self.noise_seed)
        model.sample_noise = lambda shape, device: torch.randn(
            shape, device=device, dtype=torch.float32, generator=generator
        )
        signature = inspect.signature(self.original_sample)

        def sample(*args, **kwargs):
            values = signature.bind(*args, **kwargs)
            values.apply_defaults()
            v = values.arguments
            self.request_inputs = {"tokens": cpu(v["tokens"]), "masks": cpu(v["masks"])}
            for i, (im, mask) in enumerate(zip(v["images"], v["img_masks"], strict=True)):
                self.request_inputs[f"image_{i}"] = cpu(im)
                self.request_inputs[f"image_mask_{i}"] = cpu(mask)
            if len(v["images"]) != 3:
                raise ValueError("Expected exactly three physical camera calls")
            for name in ("states", "state_masks"):
                if v.get(name) is not None:
                    self.request_inputs[name] = cpu(v[name])
            return self.original_sample(*args, **kwargs)

        denoise_signature = inspect.signature(self.original_denoise)

        def denoise(*args, **kwargs):
            values = denoise_signature.bind(*args, **kwargs).arguments
            self.current_native_time = cpu(values["timestep"])
            self.raw_selected_record = None
            result = self.original_denoise(*args, **kwargs)
            if self.raw_selected_record is not None:
                self.raw_selected_record["native_velocity"] = cpu(result)
            return result

        model.sample_actions = sample
        model.denoise_step = denoise

    def _layer_hook(self, module, args, kwargs):
        flow = self.current_flow_index
        super()._layer_hook(module, args, kwargs)
        if not self.current_request_selected or flow not in self.flow_indices:
            return
        slot = self.current_request_slots[self.flow_indices.index(flow)]
        record = {key: value.clone() for key, value in self.request_inputs.items()}
        record["x_t"] = self.latest_action_input.clone()
        record["time"] = self.current_native_time.clone()
        self.records_by_prompt[self.latest_prompt_signature][slot] = record
        self.raw_selected_record = record

    def save(self):
        if self.original_predict_action_chunk is not None:
            self.policy.predict_action_chunk = self.original_predict_action_chunk
        if self.original_policy_reset is not None:
            self.policy.reset = self.original_policy_reset
        for handle in self.handles:
            handle.remove()
        self.policy.model.sample_actions = self.original_sample
        self.policy.model.denoise_step = self.original_denoise
        self.policy.model.sample_noise = self.original_noise
        if self.tensor_output.exists() or self.manifest_output.exists():
            raise FileExistsError("Execution capture exists; use a new attempt")
        groups = sorted(self.records_by_prompt.items())
        if len(groups) != 10 or any(len(records) != 15 for _, records in groups):
            raise ValueError("Incomplete capture: need ten tasks and 15 states per task")
        tensors, samples = {}, []
        for task_slot, (signature, records) in enumerate(groups):
            metadata = self.record_metadata_by_prompt[signature]
            for ordinal, (record, meta) in enumerate(zip(records, metadata, strict=True)):
                if not record or "native_velocity" not in record:
                    raise ValueError("Missing native call record")
                if meta["flow_index"] != (0, 5, 9)[ordinal % 3]:
                    raise ValueError("Flow order differs from frozen reservoir")
                index = len(samples)
                samples.append({**meta, "index": index, "task_slot": task_slot,
                                "task_ordinal": ordinal, "prompt_signature": signature,
                                "prompt": self.prompt_texts[signature]})
                for key, tensor in record.items():
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise ValueError(f"Nonfinite raw input: {key}")
                    tensors[f"sample_{index:03d}.{key}"] = tensor.contiguous()
        self.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, self.tensor_output, metadata={"format": "pt"})
        digest = hashlib.sha256()
        with self.tensor_output.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        manifest = {"schema_version": 1, "method": "featcal_native_execution_input_capture",
                    "expert": self.task, "source_policy": str(self.calibration_policy),
                    "sample_count": len(samples), "samples": samples,
                    "task_slot_order": "sorted prompt signatures, not LIBERO numeric task IDs",
                    "noise_seed": self.noise_seed, "calibration_seed": self.calibration_seed,
                    "init_state_offset": self.init_state_offset,
                    "flow_indices": list(self.flow_indices), "request_mode": self.request_mode,
                    "demonstration_actions_used": False, "success_filtering": False,
                    "tensor_file": str(self.tensor_output), "tensor_sha256": digest.hexdigest()}
        with self.manifest_output.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        print(f"RAW_EXECUTION_CAPTURE_COMPLETE {self.task} states={len(samples)}", flush=True)


if __name__ == "__main__":
    fraction = float(os.environ.get("FEATCAL_EXEC_MEMORY_FRACTION", "0.35"))
    if not 0 < fraction <= 0.40:
        raise ValueError("Pilot allocator fraction must be in (0, 0.40]")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    native.BlockCalibrationCollector = RawExecutionCollector
    native.main()
