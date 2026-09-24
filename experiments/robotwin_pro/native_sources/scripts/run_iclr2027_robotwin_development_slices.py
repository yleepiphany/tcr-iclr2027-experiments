#!/usr/bin/env python3
"""Partition the frozen 200-episode development panel without changing its runner."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('checkpoint_development_base', ROOT / 'scripts/run_iclr2027_robotwin_checkpoint_development.py')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def keys(m):
    result = [(t['task_index'], s) for t in m['tasks'] for s in t['seeds']]
    if len(m['tasks']) != 10 or any(len(t['seeds']) != 20 for t in m['tasks']) or len(result) != 200 or len(set(result)) != 200:
        raise ValueError('Parent must contain exactly 10 tasks x 20 unique seeds')
    return result


def validate_parent(m):
    base.validate(m)
    expected = dict(episodes_per_task=20, full_horizon=True, task_config='demo_clean',
                    action_mode='joint', invalid_seed_policy='fail attempt; never replace seed')
    if m['simulator'] != expected:
        raise ValueError('Frozen simulator protocol differs')
    keys(m)


def partition(m, index, count):
    if not 1 <= count <= 20 or not 0 <= index < count:
        raise ValueError('Invalid slice index/count')
    return keys(m)[index::count]


def selected_manifest(m, selected):
    selected = set(map(tuple, selected))
    result = copy.deepcopy(m)
    result['tasks'] = [dict(t, seeds=[s for s in t['seeds'] if (t['task_index'], s) in selected]) for t in m['tasks']]
    result['tasks'] = [t for t in result['tasks'] if t['seeds']]
    # A partial panel can never establish a checkpoint plateau.
    result['plateau_eligible'] = False
    return result


def build(parent, output, count):
    m = base.read(parent)
    validate_parent(m)
    if not 1 <= count <= 20:
        raise ValueError('Slice count must be 1..20')
    output.mkdir(parents=True, exist_ok=False)
    for index in range(count):
        base.write(output / f'slice-{index:02d}.json', dict(
            schema_version=1, purpose='development_checkpoint_slice', formal_result=False,
            parent=base.bind(parent), runner=base.bind(__file__), slice_index=index,
            slice_count=count, task_seed_keys=partition(m, index, count)))
    print(json.dumps(dict(status='slice_manifests_ready_no_gpu', slices=count, episodes=200, output=str(output))))


def validate_slice(path):
    s = base.read(path)
    if s['schema_version'] != 1 or s['purpose'] != 'development_checkpoint_slice' or s['formal_result'] is not False:
        raise ValueError('Invalid slice scope')
    for b in (s['parent'], s['runner']):
        if base.sha(b['path']) != b['sha256']:
            raise ValueError('Bound parent/runner SHA changed')
    if Path(s['runner']['path']).resolve() != Path(__file__).resolve():
        raise ValueError('Different slice runner')
    m = base.read(s['parent']['path'])
    validate_parent(m)
    expected = partition(m, s['slice_index'], s['slice_count'])
    if list(map(tuple, s['task_seed_keys'])) != expected:
        raise ValueError('Slice keys differ from deterministic partition')
    return s, m


def runtime_setup(m):
    os.environ.update(m['runtime_environment'])
    sys.path.insert(0, str(Path(m['lerobot']['root']) / 'src'))
    sys.path.insert(0, m['robotwin']['root'])
    os.chdir(m['robotwin']['root'])


def preflight(path):
    s, m = validate_slice(path)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Preflight requires CUDA_VISIBLE_DEVICES empty')
    runtime_setup(m)
    import torch
    import yaml
    from scripts import run_iclr2027_robotwin_native_pi05_smoke_v2 as native
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa
    from lerobot.envs.configs import RoboTwinEnvConfig
    cfg = PreTrainedConfig.from_pretrained(Path(m['checkpoint']) / 'pretrained_model')
    native.apply_runtime_visual_override(cfg)
    horizons = yaml.safe_load((Path(m['robotwin']['root']) / 'task_config/_eval_step_limit.yml').read_text())
    for task in m['tasks']:
        assert int(horizons[task['task']]) > 0
        env = RoboTwinEnvConfig(task=task['task'], episode_length=None, task_config='demo_clean', action_mode='joint')
        assert env.episode_length is None
    if torch.cuda.is_initialized():
        raise ValueError('Preflight unexpectedly initialized CUDA')
    print(json.dumps(dict(status='runtime_preflight_passed_no_cuda', slice_index=s['slice_index'], episodes=len(s['task_seed_keys']), parent_sha256=s['parent']['sha256'])))


def run(path, output, gpu):
    s, m = validate_slice(path)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('CUDA_VISIBLE_DEVICES must match explicit physical --gpu')
    from scripts.run_iclr2027_robotwin_native_pi05_smoke_v2 import exclusive_gpu_lock, check_gpu_admission
    with exclusive_gpu_lock(base.RUNTIME / f'resource-leases/gpu-{gpu}.lock'):
        admission = check_gpu_admission(gpu)
        output.mkdir(parents=True, exist_ok=False)
        base.write(output / 'started.json', dict(slice_manifest=base.bind(path), parent=s['parent'], admission=admission, formal_result=False))
        try:
            runtime_setup(m)
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError('Exactly one CUDA device required')
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            result = base.simulator_audit(selected_manifest(m, s['task_seed_keys']), output, False)
            if result['episodes'] != len(s['task_seed_keys']):
                raise ValueError('Incomplete slice result')
            row_files = [output / f'{t["task"]}-{seed}.json' for t in m['tasks'] for seed in t['seeds'] if [t['task_index'], seed] in s['task_seed_keys']]
            base.write(output / 'complete.json', dict(status='development_slice_complete',
                formal_result=False, plateau_eligible=False, slice_manifest=base.bind(path), parent=s['parent'],
                runner=s['runner'], slice_index=s['slice_index'], slice_count=s['slice_count'],
                rows=[base.bind(p) for p in row_files], result=result, seconds=time.monotonic()-begin,
                peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2))
        except BaseException as error:
            base.write(output / 'failure.json', dict(error_type=type(error).__name__, error=str(error)))
            raise


def summarize_rows(m, rows, horizons):
    expected = set(keys(m))
    seen = set()
    by_task = {t['task_index']: t for t in m['tasks']}
    successes = {i: 0 for i in by_task}
    for row in rows:
        k = (row['task_index'], row['seed'])
        if k not in expected or k in seen:
            raise ValueError('Duplicate or foreign task-seed key')
        seen.add(k)
        task = by_task[k[0]]
        if row['task'] != task['task'] or type(row['success']) is not bool:
            raise ValueError('Task identity/success type mismatch')
        horizon = int(horizons[task['task']])
        if row['horizon'] != horizon or type(row['steps']) is not int or not 0 < row['steps'] <= horizon:
            raise ValueError('Invalid native horizon or step count')
        if not math.isfinite(row['seconds']) or row['seconds'] < 0:
            raise ValueError('Invalid episode duration')
        digest = row['initial_observation_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Missing reset observation digest')
        successes[k[0]] += int(row['success'])
    if seen != expected:
        raise ValueError(f'Incomplete panel: {len(seen)}/200')
    means = {by_task[i]['task']: successes[i]/20 for i in by_task}
    return dict(status='development_rollout_complete', formal_result=False, episodes=200,
                task_episode_counts={t['task']:20 for t in m['tasks']}, task_success=means,
                macro_success=sum(means.values())/10, plateau_eligible=m['plateau_eligible'])


def aggregate(parent, receipts, output):
    m = base.read(parent)
    validate_parent(m)
    parent_binding = base.bind(parent)
    all_rows, indices, count, refs = [], set(), None, []
    for path in receipts:
        receipt = base.read(path)
        if (receipt['status'] != 'development_slice_complete' or receipt['formal_result'] is not False
                or receipt['plateau_eligible'] is not False or receipt['parent'] != parent_binding):
            raise ValueError('Receipt scope/parent mismatch')
        binding = receipt['slice_manifest']
        if base.sha(binding['path']) != binding['sha256']:
            raise ValueError('Slice manifest SHA mismatch')
        s, _ = validate_slice(binding['path'])
        if receipt['runner'] != s['runner'] or receipt['slice_index'] != s['slice_index'] or receipt['slice_count'] != s['slice_count']:
            raise ValueError('Receipt identity mismatch')
        if count is None:
            count = s['slice_count']
        if count != s['slice_count'] or s['slice_index'] in indices:
            raise ValueError('Mixed partition or duplicate slice')
        indices.add(s['slice_index'])
        rows = []
        for b in receipt['rows']:
            if Path(b['path']).resolve().parent != path.resolve().parent or base.sha(b['path']) != b['sha256']:
                raise ValueError('Episode row path/SHA mismatch')
            rows.append(base.read(b['path']))
        actual = [(r['task_index'], r['seed']) for r in rows]
        if len(actual) != len(set(actual)) or set(actual) != set(map(tuple, s['task_seed_keys'])):
            raise ValueError('Slice row coverage mismatch')
        if receipt['result']['episodes'] != len(rows):
            raise ValueError('Receipt episode count mismatch')
        all_rows.extend(rows)
        refs.append(base.bind(path))
    if count is None or indices != set(range(count)):
        raise ValueError('Missing slice receipt')
    import yaml
    horizon_path = Path(m['robotwin']['root']) / 'task_config/_eval_step_limit.yml'
    result = summarize_rows(m, all_rows, yaml.safe_load(horizon_path.read_text()))
    base.write(output, dict(**result, parent=parent_binding, runner=base.bind(__file__), slices=refs,
        horizon_config=base.bind(horizon_path), checkpoint=m['checkpoint'], group=m['group'], step=m['step'],
        note='Rollout component only; heldout loss and all-five-expert checkpoint criteria remain required.'))
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build','validate','preflight','run','aggregate'])
    p.add_argument('--parent', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--output', type=Path)
    p.add_argument('--gpu', type=int)
    p.add_argument('--receipts', type=Path, nargs='+')
    a = p.parse_args()
    if a.action == 'build':
        return build(a.parent.resolve(), a.output.resolve(), a.count)
    if a.action == 'aggregate':
        return aggregate(a.parent.resolve(), a.receipts, a.output.resolve())
    if a.action == 'preflight':
        return preflight(a.manifest.resolve())
    if a.action == 'validate':
        s, _ = validate_slice(a.manifest.resolve())
        print(json.dumps(dict(status='slice_validated_no_gpu', episodes=len(s['task_seed_keys']))))
        return
    if a.gpu is None or a.output is None:
        p.error('run requires --gpu and --output')
    run(a.manifest.resolve(), a.output.resolve(), a.gpu)


if __name__ == '__main__':
    main()
