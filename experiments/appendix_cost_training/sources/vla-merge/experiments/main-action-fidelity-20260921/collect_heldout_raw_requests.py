#!/usr/bin/env python3
"""Capture complete raw native inputs for the frozen C146 held-out requests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT.parent / "pi05_lora_finetune_v2_20260826"
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "lerobot/src"),
                str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from safetensors.torch import save_file
import collect_pi05_block_regmeanpp_calibration as native
from collect_pi05_featcal_execution_inputs import RawExecutionCollector


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def select_quantiles(metadata):
    requests = sorted({int(row["request_index"]) for row in metadata})
    if len(requests) < 5:
        raise ValueError("Fewer than five native requests")
    if len(requests) >= 128:
        raise ValueError("Episode reached request cap")
    selected = [requests[(len(requests) - 1) * k // 4] for k in range(5)]
    if len(set(selected)) != 5:
        raise ValueError("Request quantiles are not distinct")
    slots = []
    for request in selected:
        matches = [i for i, row in enumerate(metadata)
                   if row["request_index"] == request and row["flow_index"] == 0]
        if len(matches) != 1:
            raise ValueError("Selected request lacks exactly one flow-zero record")
        slots.append(matches[0])
    return slots, selected


class HeldoutRawCollector(RawExecutionCollector):
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
            raise FileExistsError("Held-out raw capture exists")

        groups = sorted(self.records_by_prompt.items())
        if len(groups) != 10:
            raise ValueError("Need exactly ten task prompts")
        tensors, samples, provenance = {}, [], {}
        for task_slot, (signature, records) in enumerate(groups):
            metadata = self.record_metadata_by_prompt[signature]
            slots, selected = select_quantiles(metadata)
            provenance[str(signature)] = selected
            for request_slot, index_in_group in enumerate(slots):
                record, meta = records[index_in_group], metadata[index_in_group]
                if not record or "native_velocity" not in record:
                    raise ValueError("Missing complete native input/teacher velocity")
                index = len(samples)
                samples.append({**meta, "index": index, "task_slot": task_slot,
                                "request_slot": request_slot,
                                "prompt_signature": signature,
                                "prompt": self.prompt_texts[signature]})
                for key, tensor in record.items():
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise ValueError(f"Nonfinite raw input: {key}")
                    tensors[f"sample_{index:03d}.{key}"] = tensor.contiguous()
        if len(samples) != 50:
            raise ValueError(f"Expected 50 selected requests, got {len(samples)}")
        self.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, self.tensor_output, metadata={"format": "pt"})
        digest = hashlib.sha256()
        with self.tensor_output.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        manifest = {
            "schema_version": 1,
            "method": "main_fidelity_native_execution_input_capture",
            "source_kind": "heldout_expert_execution",
            "heldout_only": True,
            "forbidden_as_calibration": True,
            "expert": self.task,
            "episode_id": required("PI05_MAIN_FIDELITY_EPISODE_ID"),
            "source_policy": str(self.calibration_policy),
            "source_policy_sha256": required("PI05_MAIN_FIDELITY_POLICY_SHA256"),
            "dense_expert_bank_sha256": required("PI05_MAIN_FIDELITY_DENSE_BANK_SHA256"),
            "sample_count": len(samples), "samples": samples,
            "task_slot_order": "sorted prompt signatures; prompt text maps to LIBERO task id",
            "selected_requests": provenance,
            "request_selection": "five trajectory-time quantiles 0,.25,.5,.75,1",
            "noise_seed": self.noise_seed,
            "capture_seed": self.calibration_seed,
            "init_state_offset": self.init_state_offset,
            "flow_indices": list(self.flow_indices),
            "request_mode": self.request_mode,
            "demonstration_actions_used": False,
            "success_filtering": False,
            "failures_retained": True,
            "tensor_file": str(self.tensor_output),
            "tensor_sha256": digest.hexdigest(),
        }
        with self.manifest_output.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
        print(f"MAIN_FIDELITY_RAW_CAPTURE_COMPLETE {self.task} requests=50", flush=True)


if __name__ == "__main__":
    fraction = float(os.environ.get("FEATCAL_EXEC_MEMORY_FRACTION", "0.35"))
    if not 0 < fraction <= 0.40:
        raise ValueError("Allocator fraction must be in (0, .40]")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    native.BlockCalibrationCollector = HeldoutRawCollector
    native.main()
