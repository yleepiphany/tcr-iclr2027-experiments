#!/usr/bin/env python3
"""Collect complete merged-policy episodes; retain five native requests/task.

This records INPUTS, not expert labels or a trained/refined checkpoint.
Failures are retained. No policy or environment actions are changed.
"""
import hashlib
import json
import os
import time


def main():
    import torch
    import collect_pi05_block_regmeanpp_calibration as native
    from pi05_table3_contract import request_selection
    from run_pi05_generation_path_probe import memory_free, rest_seconds
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = int(os.environ["ITERATION_PHYSICAL_GPU"])
    base = native.BlockCalibrationCollector

    class OccupancyCollector(base):
        def install(self):
            super().install()
            generator = torch.Generator(device="cuda").manual_seed(272001)
            self.noise_hashes = []
            self.calls = 0
            self.started = time.monotonic()

            def noise(shape, device):
                value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                self.noise_hashes.append(hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest())
                return value

            self.policy.model.sample_noise = noise
            sample = self.policy.model.sample_actions

            def bounded_sample(*args, **kwargs):
                if memory_free(gpu) < 12 * 1024:
                    raise RuntimeError("GPU reserve low; no process will be terminated")
                started = time.monotonic()
                value = sample(*args, **kwargs)
                self.calls += 1
                torch.cuda.synchronize()
                time.sleep(rest_seconds(time.monotonic() - started, .25))
                return value

            self.policy.model.sample_actions = bounded_sample

        def save(self):
            records, metadata = self.records_by_prompt, self.record_metadata_by_prompt
            if len(records) != 10:
                raise ValueError("Incomplete collection; do not fill absent tasks")
            chosen, meta, provenance = {}, {}, {}
            for prompt, group in records.items():
                slots, requests = request_selection(metadata[prompt], "across")
                if len({s["request_index"] for s in metadata[prompt]}) >= 128:
                    raise ValueError("Capture reached cap; do not silently truncate")
                chosen[prompt] = [group[i] for i in slots]
                meta[prompt] = [{**metadata[prompt][i], "selected_request_slot": j // 3}
                                for j, i in enumerate(slots)]
                provenance[str(prompt)] = requests
            self.records_by_prompt, self.record_metadata_by_prompt = chosen, meta
            self.requests_per_episode = 5
            super().save()
            manifest = json.loads(self.manifest_output.read_text())
            manifest.update(occupancy_capture_version=1, source_kind="merged_policy_execution",
                selected_requests=provenance, generation_noise_seed=272001,
                native_noise_sha256=self.noise_hashes, calibration_policy_sha256=os.environ["TCR_OCCUPANCY_MODEL_SHA256"],
                source_expert_bank_sha256=os.environ["TCR_OCCUPANCY_BANK_SHA256"],
                selection_rule="Five quantiles across the complete execution; no success filtering",
                expert_targets_collected=False, refined_checkpoint_produced=False,
                native_calls=self.calls, wall_seconds=time.monotonic() - self.started,
                peak_allocator_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                allocator_fraction_cap=.20, duty_fraction=.25)
            self.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n")

    native.BlockCalibrationCollector = OccupancyCollector
    native.main()


if __name__ == "__main__":
    main()
