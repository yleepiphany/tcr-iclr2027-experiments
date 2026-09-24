#!/usr/bin/env python3
"""Parameterised adapter for the frozen Table-3 execution collector.

This file intentionally leaves ``scripts/collect_pi05_table3_execution.py`` unchanged.
It reproduces only that wrapper's small amount of Table-3 selection logic while taking
the repeat/start/noise identities from the environment.  The native collector remains
the implementation of the hooks and replay tensors; this adapter only makes the
registered repeat identity explicit.

The adapter writes the ``early`` view only.  It retains the complete request sequence
up to the registered cap, then selects the first five requests from the true observed request
indices.  It never filters by rollout success.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    work = Path(__file__).resolve().parents[3]
    data = work / ".datasets/LIBERO/20260919"
    os.environ["LIBERO_CONFIG_PATH"] = str(data / "config-standard")
    import libero.libero as inner
    inner._assets_path_cache = str(data / "runtime/assets")
    import torch

    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("PI05_MAINLINE_MEMORY_FRACTION", "0.30")), 0
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import collect_pi05_block_regmeanpp_calibration as native
    from pi05_table3_contract import request_selection
    from run_pi05_generation_path_probe import memory_free

    repeat_id = _required("PI05_MAINLINE_REPEAT_ID")
    expert_name = _required("PI05_MAINLINE_EXPERT_NAME")
    policy_sha = _required("PI05_MAINLINE_POLICY_SHA256")
    bank_sha = _required("PI05_MAINLINE_DENSE_BANK_SHA256")
    start_seed = int(_required("PI05_MAINLINE_START_SEED"))
    flow_seed = int(_required("PI05_MAINLINE_FLOW_SEED"))
    init_offset = int(_required("PI05_MAINLINE_INIT_STATE_OFFSET"))

    class MainlineCollector(native.BlockCalibrationCollector):
        def install(self):
            super().install()
            generator = torch.Generator(device="cuda").manual_seed(flow_seed)
            self.noise_hashes = []

            def noise(shape, device):
                if memory_free(int(_required("ITERATION_PHYSICAL_GPU"))) < 12 * 1024:
                    raise RuntimeError("Capture GPU reserve below12GiB; stop this batch")
                value = torch.randn(
                    shape, device=device, dtype=torch.float32, generator=generator
                )
                self.noise_hashes.append(
                    hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                )
                return value

            self.policy.model.sample_noise = noise

        def save(self):
            records = self.records_by_prompt
            metadata = self.record_metadata_by_prompt
            if len(records) != 10:
                raise ValueError("Incomplete ten-task execution capture")

            root = self.tensor_output.parent
            chosen, chosen_meta, provenance = {}, {}, {}
            for prompt, group in records.items():
                slots, requests = request_selection(metadata[prompt], "early")
                if len({s["request_index"] for s in metadata[prompt]}) >= 128:
                    raise ValueError(
                        "Episode reached the registered capture cap; do not truncate"
                    )
                chosen[prompt] = [group[i] for i in slots]
                chosen_meta[prompt] = [
                    {**metadata[prompt][i], "selected_request_slot": j // 3}
                    for j, i in enumerate(slots)
                ]
                provenance[str(prompt)] = requests

            self.records_by_prompt = chosen
            self.record_metadata_by_prompt = chosen_meta
            self.tensor_output = root / "early" / "replay.safetensors"
            self.manifest_output = root / "early" / "replay.json"
            self.requests_per_episode = 5
            super().save()

            manifest = json.loads(self.manifest_output.read_text(encoding="utf-8"))
            manifest.update(
                table3_capture_version=1,
                source_kind="expert_execution",
                table3_request_selection="early",
                selected_requests=provenance,
                native_noise_sha256=self.noise_hashes,
                generation_noise_seed=flow_seed,
                selection_rule=(
                    "first five observed native request indices; no success filtering"
                ),
                single_repeat=True,
                mainline_capture_version=1,
                repeat_id=repeat_id,
                expert_name=expert_name,
                calibration_start_seed=start_seed,
                calibration_init_state_offset=init_offset,
                calibration_policy_sha256=policy_sha,
                dense_expert_bank_sha256=bank_sha,
                capture_policy_role="10k dense single-task expert execution",
                failures_retained=True,
            )
            self.manifest_output.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.records_by_prompt = records
            self.record_metadata_by_prompt = metadata

    native.BlockCalibrationCollector = MainlineCollector
    native.main()


if __name__ == "__main__":
    main()
