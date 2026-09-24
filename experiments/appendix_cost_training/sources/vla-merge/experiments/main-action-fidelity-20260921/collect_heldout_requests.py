#!/usr/bin/env python3
"""Capture five fixed request quantiles from one held-out expert episode/task.

This is a thin adapter around the audited native PI0.5 replay collector.  It
keeps only flow index zero (the initial native noise) because the artifact is a
common action-fidelity input bank, never a regression/calibration cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def quantile_requests(metadata):
    requests = sorted({int(row["request_index"]) for row in metadata})
    if len(requests) < 5:
        raise ValueError("Fewer than five requests; do not duplicate requests")
    if len(requests) >= 128:
        raise ValueError("Episode reached capture cap; cannot certify full trajectory")
    selected = [requests[(len(requests) - 1) * k // 4] for k in range(5)]
    if len(set(selected)) != 5:
        raise ValueError("Request quantiles are not distinct")
    slots = []
    for request in selected:
        matches = [i for i, row in enumerate(metadata)
                   if row["request_index"] == request and row["flow_index"] == 0]
        if len(matches) != 1:
            raise ValueError(f"Request {request} lacks one unique initial-noise record")
        slots.append(matches[0])
    return slots, selected


def main() -> None:
    work = Path(__file__).resolve().parents[3]
    data = work / ".datasets/LIBERO/20260919"
    os.environ["LIBERO_CONFIG_PATH"] = str(data / "config-standard")
    import libero.libero as inner
    inner._assets_path_cache = str(data / "runtime/assets")
    import torch

    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("PI05_MAIN_FIDELITY_MEMORY_FRACTION", "0.30")), 0
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import collect_pi05_block_regmeanpp_calibration as native
    from run_pi05_generation_path_probe import memory_free

    episode_id = required("PI05_MAIN_FIDELITY_EPISODE_ID")
    expert_name = required("PI05_MAIN_FIDELITY_EXPERT_NAME")
    policy_sha = required("PI05_MAIN_FIDELITY_POLICY_SHA256")
    bank_sha = required("PI05_MAIN_FIDELITY_DENSE_BANK_SHA256")
    start_seed = int(required("PI05_MAIN_FIDELITY_START_SEED"))
    flow_seed = int(required("PI05_MAIN_FIDELITY_FLOW_SEED"))
    init_offset = int(required("PI05_MAIN_FIDELITY_INIT_STATE_OFFSET"))

    class HeldoutCollector(native.BlockCalibrationCollector):
        def install(self):
            super().install()
            generator = torch.Generator(device="cuda").manual_seed(flow_seed)
            self.noise_hashes = []

            def noise(shape, device):
                if memory_free(int(required("ITERATION_PHYSICAL_GPU"))) < 12 * 1024:
                    raise RuntimeError("Held-out capture GPU reserve below 12 GiB")
                value = torch.randn(shape, device=device, dtype=torch.float32,
                                    generator=generator)
                self.noise_hashes.append(hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                return value

            self.policy.model.sample_noise = noise

        def save(self):
            records, metadata = self.records_by_prompt, self.record_metadata_by_prompt
            if len(records) != 10:
                raise ValueError("Incomplete ten-task held-out capture")
            root = self.tensor_output.parent
            chosen, chosen_meta, provenance = {}, {}, {}
            for prompt, group in records.items():
                slots, requests = quantile_requests(metadata[prompt])
                chosen[prompt] = [group[i] for i in slots]
                chosen_meta[prompt] = [
                    {**metadata[prompt][i], "selected_request_slot": slot}
                    for slot, i in enumerate(slots)
                ]
                provenance[str(prompt)] = requests
            self.records_by_prompt, self.record_metadata_by_prompt = chosen, chosen_meta
            self.tensor_output = root / "heldout" / "requests.safetensors"
            self.manifest_output = root / "heldout" / "requests.json"
            self.requests_per_episode = 5
            super().save()
            manifest = json.loads(self.manifest_output.read_text())
            manifest.update(
                main_fidelity_capture_version=1,
                source_kind="heldout_expert_execution",
                heldout_only=True,
                forbidden_as_calibration=True,
                request_selection="five trajectory-time quantiles 0,.25,.5,.75,1",
                selected_requests=provenance,
                native_noise_sha256=self.noise_hashes,
                generation_noise_seed=flow_seed,
                episode_id=episode_id,
                expert_name=expert_name,
                capture_start_seed=start_seed,
                capture_init_state_offset=init_offset,
                calibration_policy_sha256=policy_sha,
                dense_expert_bank_sha256=bank_sha,
                failures_retained=True,
            )
            self.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            self.records_by_prompt, self.record_metadata_by_prompt = records, metadata

    native.BlockCalibrationCollector = HeldoutCollector
    native.main()


if __name__ == "__main__":
    main()
