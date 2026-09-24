"""Collect one complete episode/task, save paired early and across-execution traces."""
import hashlib
import json
import os
from pathlib import Path


def main():
    import torch
    torch.cuda.set_per_process_memory_fraction(.40, 0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import collect_pi05_block_regmeanpp_calibration as native
    from pi05_table3_contract import request_selection
    base = native.BlockCalibrationCollector

    class Table3Collector(base):
        def install(self):
            super().install()
            generator = torch.Generator(device='cuda').manual_seed(272001)
            self.noise_hashes = []
            def noise(shape, device):
                value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                self.noise_hashes.append(hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest())
                return value
            self.policy.model.sample_noise = noise

        def save(self):
            records = self.records_by_prompt
            metadata = self.record_metadata_by_prompt
            if len(records) != 10:
                raise ValueError('Incomplete ten-task execution capture')
            root = self.tensor_output.parent
            for mode in ('across', 'early'):
                chosen, meta, provenance = {}, {}, {}
                for prompt, group in records.items():
                    slots, requests = request_selection(metadata[prompt], mode)
                    if len({s['request_index'] for s in metadata[prompt]}) >= 128:
                        raise ValueError('Episode reached capture cap; do not silently truncate')
                    chosen[prompt] = [group[i] for i in slots]
                    meta[prompt] = [{**metadata[prompt][i], 'selected_request_slot': j//3}
                                    for j,i in enumerate(slots)]
                    provenance[str(prompt)] = requests
                self.records_by_prompt, self.record_metadata_by_prompt = chosen, meta
                self.tensor_output = root / mode / 'replay.safetensors'
                self.manifest_output = root / mode / 'replay.json'
                self.requests_per_episode = 5
                super().save()
                manifest = json.loads(self.manifest_output.read_text())
                manifest.update(table3_capture_version=1, source_kind='expert_execution',
                                table3_request_selection=mode, selected_requests=provenance,
                                native_noise_sha256=self.noise_hashes, generation_noise_seed=272001,
                                selection_rule='early first five / across quantiles 0, .25, .5, .75, 1; no success filtering',
                                single_repeat=True)
                self.manifest_output.write_text(json.dumps(manifest, indent=2)+'\n')
            self.records_by_prompt, self.record_metadata_by_prompt = records, metadata

    native.BlockCalibrationCollector = Table3Collector
    native.main()


if __name__ == '__main__':
    main()
