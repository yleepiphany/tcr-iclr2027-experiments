"""Read-only CPU diagnosis of frozen Fast-WAM affine interface incompatibility.

No model edits, GPU, environment rollout, or hyperparameter selection. The
unregularized fit is a local-objective lower-bound diagnostic, not a new model.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'vla-merge-runtime/experiments/claude-fastwam-tcr-20260923'
RUN = ROOT / 'vla-merge-runtime/experiments/fastwam-local-diagnosis-20260924'
sys.path.insert(0, str(ROOT / 'vla-merge/experiments/openvla-tcr-20260921'))
from linear_calibration import row_indices

torch.set_num_threads(2)
plan = json.loads((RUN / 'attempt-01/plan.json').read_text())
names = ('spatial', 'object', 'goal', 'long')
paths = {s: plan['models']['expert-' + s]['path'] for s in names}
paths.update(soup=plan['models']['soup']['path'])
paths.update({p: str(RUN / f'corrected-attempt-01/tcr-build-v5/pass-{p}/checkpoint.pt')
              for p in ('A', 'B')})
keys = {'encoder': ('mot', 'mixtures.action.action_encoder'),
        'head': ('mot', 'mixtures.action.head'),
        'proprio': ('proprio_encoder', ''),
        'action_q0': ('mot', 'mixtures.action.blocks.0.self_attn.q')}
weights = {}
for name, path in paths.items():
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    weights[name] = {}
    for label, (part, prefix) in keys.items():
        prefix = prefix + '.' if prefix else ''
        state = payload[part]
        w = state[prefix + 'weight'].float().clone()
        b = state.get(prefix + 'bias')
        weights[name][label] = (w, b.float().clone() if b is not None else None)
    del payload

geometry = {}
for label in keys:
    ws = [weights[s][label][0] for s in names]
    cos = {a + ':' + b: float(torch.nn.functional.cosine_similarity(
        weights[a][label][0].flatten(), weights[b][label][0].flatten(), dim=0))
        for a, b in itertools.combinations(names, 2)}
    norm = sum(float(w.norm()) for w in ws) / 4
    geometry[label] = {'pairwise_cosine': cos,
        'mean_weight_norm': norm,
        'soup_norm_ratio': float(weights['soup'][label][0].norm()) / norm,
        'A_norm_ratio': float(weights['A'][label][0].norm()) / norm,
        'B_norm_ratio': float(weights['B'][label][0].norm()) / norm}

accepted = json.loads((BASE / 'calibration-acceptance-v1.json').read_text())
datasets = {pool: {kind: {s: [] for s in names}
    for kind in ('encoder_sampled', 'encoder_all', 'encoder_call0', 'proprio')}
    for pool in ('A', 'B')}
noise_hashes = {s: set() for s in names}
seed_values = set()
request_count = 0
module = 'mot.mixtures.action.action_encoder'
for job in accepted['jobs_detail']:
    pool, expert = job['round'], job['expert']
    for index in job['selected_request_indices']:
        path = BASE / 'calibration-v1' / job['id'] / f'request-{index:06d}.pt'
        row = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
        request_count += 1
        seed_values.add(str(row['native_request'].get('seed')))
        proprio = row['native_request']['proprio'].reshape(-1, 8).double()
        datasets[pool]['proprio'][expert].append(proprio)
        rid = f'{pool}/{expert}/task-{job["task_id"]:02d}/request-{index:06d}'
        for ordinal, call in enumerate(row['trace']['calls']):
            x = call['inputs']['latents_action'].reshape(-1, 7).double()
            datasets[pool]['encoder_all'][expert].append(x)
            if ordinal == 0:
                datasets[pool]['encoder_call0'][expert].append(x)
                noise_hashes[expert].add(hashlib.sha256(x.numpy().tobytes()).hexdigest())
            indices = row_indices(len(x), 2, seed=2026092301 if pool == 'A' else 2026092302,
                request_id=rid, module_name=module, call_index=ordinal)
            datasets[pool]['encoder_sampled'][expert].append(x[indices])
        del row

def affine(label, name):
    w, b = weights[name][label]
    return torch.cat((w.double(), b.double()[:, None]), dim=1)

stats = {}
for pool, kinds in datasets.items():
    stats[pool] = {}
    for kind, by_expert in kinds.items():
        label = 'proprio' if kind == 'proprio' else 'encoder'
        xs = [torch.cat(by_expert[s]) for s in names]
        xs = [torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), dim=1) for x in xs]
        if len(set(map(len, xs))) != 1:
            raise ValueError('This diagnostic requires balanced expert rows')
        ys = [x @ affine(label, s).T for s, x in zip(names, xs)]
        x = torch.cat(xs)
        y = torch.cat(ys)
        optimal = torch.linalg.lstsq(x, y, driver='gelsd').solution
        energy = float(y.square().mean())
        losses = {arm: float((x @ affine(label, arm).T - y).square().mean())
                  for arm in ('soup', 'A', 'B')}
        best = float((x @ optimal - y).square().mean())
        stats[pool][kind] = {'rows_per_expert': len(xs[0]), 'target_energy': energy,
            'mse': losses, 'unregularized_min_mse': best,
            'unregularized_min_nmse': best / energy,
            'B_nmse': losses['B'] / energy,
            'rank': int(torch.linalg.matrix_rank(x))}
        if kind == 'encoder_call0':
            # Same x through all four frozen encoders: exact conflict independent
            # of sampling differences among expert trajectories.
            common_x = xs[0]
            targets = torch.stack([common_x @ affine(label, s).T for s in names])
            target_mean = targets.mean(0)
            stats[pool][kind]['common_input_variance_fraction'] = float(
                (targets - target_mean).square().mean() / targets.square().mean())

report = {'request_count': request_count, 'geometry': geometry,
    'seed_values': sorted(seed_values),
    'unique_call0_noise_per_expert': {s: len(h) for s, h in noise_hashes.items()},
    'unique_call0_noise_total': len(set.union(*noise_hashes.values())),
    'local_affine_fits': stats,
    'limitations': ['Local calibration fit, not end-to-end causal proof or success metric',
                    'No historical step-zero checkpoints tested',
                    'No saved weights modified; no GPU or rollouts']}
stored = json.loads((RUN / 'corrected-attempt-01/tcr-build-v5/pass-B/block-32.json').read_text())
recorded_mse = stored['modules'][module]['dense_regmean_loss']
observed_mse = stats['B']['encoder_sampled']['mse']['B']
if abs(recorded_mse - observed_mse) > 1e-5:
    raise ValueError('Independent sampled encoder objective does not match build report')
report['sampled_B_objective_check'] = {'recorded_mse': recorded_mse,
    'independent_exported_weight_mse': observed_mse,
    'absolute_difference': abs(recorded_mse - observed_mse), 'accepted': True}
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path)
args = parser.parse_args()
serialized = json.dumps(report, indent=2) + '\n'
if args.output:
    with args.output.open('x') as stream:
        stream.write(serialized)
    print(json.dumps({'output': str(args.output),
        'sha256': hashlib.sha256(serialized.encode()).hexdigest(),
        'sampled_B': stats['B']['encoder_sampled'],
        'all_B': stats['B']['encoder_all']}))
else:
    print(serialized)
