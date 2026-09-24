#!/usr/bin/env python3
"""Dense trace collector with a per-process PyTorch allocator ceiling."""
import os
import hashlib
import json
import torch

if __name__ == "__main__":
    fraction = float(os.environ.get("TCR_E_CUDA_MEMORY_FRACTION", "0.30"))
    if not 0 < fraction <= 0.50:
        raise ValueError("TCR-E collector allocator fraction must be in (0, 0.50]")
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"TCR-E collector CUDA allocator fraction={fraction}; CUDA context is additional", flush=True)
    import collect_pi05_block_regmeanpp_calibration as native
    base = native.BlockCalibrationCollector

    class SeededDenseCollector(base):
        def install(self):
            super().install()
            self.generation_noise_seed = int(os.environ["TCR_E_GENERATION_NOISE_SEED"])
            self.noise_hashes = []
            generator = torch.Generator(device="cuda").manual_seed(self.generation_noise_seed)
            def fixed_noise(shape, device):
                noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                self.noise_hashes.append(hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest())
                return noise
            self.policy.model.sample_noise = fixed_noise

        def save(self):
            super().save()
            receipt = json.loads(self.manifest_output.read_text())
            receipt.update(generation_noise_seed=self.generation_noise_seed,
                           generation_noise_schedule="single explicit CUDA generator, request execution order",
                           native_noise_sha256=self.noise_hashes)
            self.manifest_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    native.BlockCalibrationCollector = SeededDenseCollector
    native.main()
