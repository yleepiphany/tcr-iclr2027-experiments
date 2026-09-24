#!/usr/bin/env python3
"""Read-only CPU audit of expert versus merged OpenVLA auxiliary weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

WORK = Path(__file__).resolve().parents[3]
REPAIR = WORK / 'vla-merge-runtime/experiments/claude-openvla-tcr-repair-20260922'
TABLE4 = WORK / 'vla-merge_table4/OpenVLA-OFT/weights'
SOURCES = {
    'old_R2': REPAIR / 'candidate-build-attempt-02/jobs/R2/build/pass-B/export/checkpoint',
    'ordered_R2': REPAIR / 'ordered-aux-build-attempt-01/jobs/R2_aux_ordered/build/pass-B/export/checkpoint',
    **{name: next((TABLE4 / name).glob('*')) for name in ('spatial', 'object', 'goal', 'long')},
}
TARGETS = {
    'proprio_projector': 'module.fc2.weight',
    'action_head': 'module.model.fc1.weight',
}
EXPERTS = ('spatial', 'object', 'goal', 'long')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load(component: str, key: str) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    values, sources = {}, {}
    for name, root in SOURCES.items():
        files = list(root.glob(component + '--*_checkpoint.pt'))
        if len(files) != 1:
            raise ValueError(f'{name} has no unique {component} component')
        path = files[0]
        state = torch.load(path, map_location='cpu', weights_only=True)
        if key not in state:
            raise ValueError(f'{name} lacks {key}')
        value = state[key].detach().float().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f'{name}/{key} has nonfinite parameters')
        values[name] = value
        sources[name] = {'path': str(path.resolve()), 'sha256': digest(path)}
    if len({tuple(x.shape) for x in values.values()}) != 1:
        raise ValueError(f'{component}/{key} tensor shapes differ')
    return values, sources


def audit() -> dict:
    result = {'schema': 'openvla_aux_parameter_sanity_readonly_v1',
              'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Existing released experts and old/corrected R2 pass-B auxiliary tensors',
              'policy_evaluation': False, 'success_values_read': False,
              'targets': {}}
    for component, key in TARGETS.items():
        values, sources = load(component, key)
        expert_norms = {name: float(torch.linalg.vector_norm(values[name]).item()) for name in EXPERTS}
        low = torch.stack([values[name] for name in EXPERTS]).amin(dim=0)
        high = torch.stack([values[name] for name in EXPERTS]).amax(dim=0)
        merged = {}
        for name in ('old_R2', 'ordered_R2'):
            value = values[name]
            norm = float(torch.linalg.vector_norm(value).item())
            merged[name] = {'norm': norm, 'norm_ratio_to_largest_expert': norm / max(expert_norms.values()),
                            'outside_expert_elementwise_envelope_fraction': float(((value < low) | (value > high)).float().mean().item()),
                            'max_abs': float(value.abs().max().item()), 'finite': True}
        result['targets'][component] = {'key': key, 'shape': list(next(iter(values.values())).shape),
            'expert_norms': expert_norms, 'merged': merged, 'sources': sources}
        del values, low, high
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    for component, row in value['targets'].items():
        print(component, row['merged'])


if __name__ == '__main__':
    main()
