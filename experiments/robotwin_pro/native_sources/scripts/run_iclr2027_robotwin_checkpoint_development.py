#!/usr/bin/env python3
"""Fixed development-only RoboTwin checkpoint audit. Build/validate never use CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNTIME = Path('/mnt/workspace/Wilson/parameter-fusion/vla-merge-runtime')
TABLE5 = RUNTIME / 'experiments/iclr2027-table5-20260910'
SPLIT = TABLE5 / 'dataset-splits/robotwin-clean50-content-hash-40-5-5-v1.json'
SPLIT_SHA = '7e3823f16902541e5acc5510fedde263eebea455c53b990c83c9b3028f1c5868'
NATIVE = TABLE5 / 'native-smoke-plans/robotwin-native-joint-rgb-smoke-v2.json'
STEPS = (5000, 7500, 10000, 15000)
GROUPS = ('coordination', 'receptacle', 'precision', 'device', 'dynamic')
WORLD4_GROUPS = ('precision', 'device')
WORLD4_PLAN = TABLE5 / 'expert-plans/robotwin-world4-remaining-experts-v1'
WORLD4_TRAINING = TABLE5 / 'expert-training/robotwin-world4-remaining-experts-v1'
WORLD4_MANIFEST = TABLE5 / 'manifest-revision16.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, sort_keys=True, indent=2)
        f.write('\n')


def seeds(task_index):
    namespace = 'iclr2027-table5-robotwin-development-rollout-v1'
    return [int(hashlib.sha256(f'{namespace}\0{task_index}\0{r}'.encode()).hexdigest()[:8], 16) % 2**31 for r in range(20)]


def task_panel(split, sampling):
    tasks = {int(t['task_index']): t for t in split['tasks']}
    rows = []
    for i in sampling['task_indices']:
        t = tasks[int(i)]
        dev = sorted(int(e['converted_episode']) for e in t['splits']['development'])
        train = set(int(e['converted_episode']) for e in t['splits']['train'])
        calibration = set(int(e['converted_episode']) for e in t['splits']['calibration'])
        if len(dev) != 5 or len(train) != 40 or len(calibration) != 5 or set(dev) & (train | calibration) or train & calibration:
            raise ValueError('Development split is not disjoint 40/5/5')
        if any(e // 550 != int(i) for e in dev):
            raise ValueError('Source task identity mismatch')
        rows.append(dict(task=t['task'], task_index=int(i), episode_ids=dev, seeds=seeds(int(i))))
    if len(rows) != 10 or len({r['task_index'] for r in rows}) != 10:
        raise ValueError('Expected 10 unique own-group tasks')
    dev = sorted(e for r in rows for e in r['episode_ids'])
    if dev != sorted(sampling['development_episode_ids']):
        raise ValueError('Sampling plan development IDs differ')
    if set(dev) & set(sampling['train_episode_ids'] + sampling['calibration_episode_ids']):
        raise ValueError('Heldout leakage in sampling plan')
    return rows


def bind(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))


def source_paths(group, step):
    """Resolve the frozen training source without silently mixing world3/world4."""
    if group in WORLD4_GROUPS:
        plan_root = WORLD4_PLAN / group
        summary_path = plan_root / 'summary.json'
        summary = read(summary_path)
        if (summary.get('schema_version') != 1
                or summary.get('status') != 'robotwin_world4_full_expert_plan_ready'
                or summary.get('group_id') != group
                or summary.get('world_size') != 4
                or summary.get('save_freq') != 2500):
            raise ValueError('World4 plan identity/contract differs')
        for label in ('train_config', 'sampling_plan', 'execution_manifest'):
            item = summary[label]
            if sha(item['path']) != item['sha256']:
                raise ValueError(f'World4 plan binding changed: {label}')
        if Path(summary['execution_manifest']['path']).resolve() != WORLD4_MANIFEST.resolve():
            raise ValueError('World4 plan does not bind manifest revision16')
        execution = read(WORLD4_MANIFEST)
        if (execution.get('immutable_revision') != 16
                or execution.get('status') != 'robotwin_world4_remaining_experts_authorized_capacity_passed'):
            raise ValueError('World4 execution manifest differs')
        return dict(
            kind='world4_revision16',
            plan_root=plan_root,
            checkpoint=Path(summary['run_output_dir']) / f'checkpoints/{step:06d}',
            train_config=Path(summary['train_config']['path']),
            sampling_plan=Path(summary['sampling_plan']['path']),
            source_files=[summary_path, WORLD4_MANIFEST, WORLD4_TRAINING / group / 'attempt-identity.json'],
            expected_world_size=4,
        )
    version = 'v2' if group == 'coordination' else 'v3'
    plan_root = TABLE5 / f'expert-plans/robotwin-pi05-train40-full-expert-plan-{version}' / group
    return dict(
        kind=f'legacy_world3_{version}',
        plan_root=plan_root,
        checkpoint=TABLE5 / f'expert-training/robotwin-pi05-train40-{version}' / group / f'run/checkpoints/{step:06d}',
        train_config=plan_root / 'train_config.json',
        sampling_plan=plan_root / 'sampling_plan.json',
        source_files=[],
        expected_world_size=None,
    )


def checkpoint_required_files(checkpoint, expected_world_size=None):
    checkpoint = Path(checkpoint)
    files = [
        checkpoint / 'pretrained_model/adapter_model.safetensors',
        checkpoint / 'pretrained_model/adapter_config.json',
        checkpoint / 'pretrained_model/config.json',
        checkpoint / 'pretrained_model/train_config.json',
        checkpoint / 'training_state/training_step.json',
    ]
    if expected_world_size is not None:
        files += [
            checkpoint / 'training_state/optimizer_state.safetensors',
            checkpoint / 'training_state/optimizer_param_groups.json',
            checkpoint / 'training_state/scheduler_state.json',
            checkpoint / 'training_state/rng_state.safetensors',
            *(checkpoint / f'training_state/rng_state_rank{rank}.safetensors'
              for rank in range(expected_world_size)),
        ]
    return files


def validate_checkpoint_complete(checkpoint, step, expected_world_size=None):
    checkpoint = Path(checkpoint)
    if checkpoint.name != f'{step:06d}':
        raise ValueError('Checkpoint path/name differs')
    required = checkpoint_required_files(checkpoint, expected_world_size)
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise ValueError(f'Checkpoint incomplete: {missing}')
    state = read(checkpoint / 'training_state/training_step.json')
    if state.get('step') != step:
        raise ValueError('Checkpoint training step mismatch')
    if expected_world_size is not None:
        if (state.get('dp_world_size') != expected_world_size
                or state.get('batch_size') != 32
                or state.get('grad_accum_steps') != 1
                or state.get('parallelism', {}).get('dp_replicate') != expected_world_size
                or state.get('parallelism', {}).get('dp_shard') != 1):
            raise ValueError('World4 checkpoint topology differs')
    return required


def build(args):
    if sha(SPLIT) != SPLIT_SHA:
        raise ValueError('Frozen split changed')
    source = source_paths(args.group, args.step)
    plan_root = source['plan_root']
    checkpoint = source['checkpoint']
    sampling = read(source['sampling_plan'])
    required_checkpoint = validate_checkpoint_complete(
        checkpoint, args.step, source['expected_world_size'])
    native = read(NATIVE)
    files = [SPLIT, source['sampling_plan'], source['train_config'], Path(__file__),
             ROOT / 'scripts/run_iclr2027_robotwin_native_pi05_smoke_v2.py',
             ROOT / 'scripts/train_pi05_robotwin_task_balanced.py']
    files += sorted((checkpoint / 'pretrained_model').glob('*'))
    files += required_checkpoint
    files += source['source_files']
    files += [Path(sampling['dataset_root']) / f'meta/{x}.json' for x in ('info', 'stats')]
    for key in ('robotwin_adapter', 'horizon_config', 'runtime_wrapper', 'demo_clean', 'camera_config'):
        old = native['bindings'][key]
        if sha(old['path']) != old['sha256']:
            raise ValueError(f'Accepted runtime component changed: {key}')
        files.append(Path(old['path']))
    panel = task_panel(read(SPLIT), sampling)
    manifest = dict(schema_version=1, purpose='development_checkpoint_audit', formal_result=False,
                    group=args.group, step=args.step, checkpoint=str(checkpoint),
                    training_source=source['kind'], expected_world_size=source['expected_world_size'],
                    train_config=str(source['train_config']), dataset_root=sampling['dataset_root'],
                    sampling_plan=str(source['sampling_plan']), tasks=panel,
                    heldout_loss=dict(frames='all development frames', batch_size=1,
                                      aggregation='frame mean within task, then macro over 10 tasks',
                                      random_seed='SHA256(namespace, converted_episode, frame_index)',
                                      image_augmentation=False),
                    simulator=dict(episodes_per_task=20, full_horizon=True, task_config='demo_clean',
                                   action_mode='joint', invalid_seed_policy='fail attempt; never replace seed'),
                    plateau_eligible=args.step != 7500,
                    candidate_steps=[5000, 10000, 15000], diagnostic_steps=[7500],
                    checkpoint_selection='five experts, common checkpoint, two consecutive intervals',
                    lerobot=native['lerobot'], robotwin=native['robotwin'],
                    runtime_environment=native['execution']['runtime_environment'],
                    bindings=[bind(p) for p in files if p.is_file()])
    validate(manifest)
    write(args.manifest, manifest)
    print(json.dumps(dict(status='manifest_ready_no_gpu', manifest=str(args.manifest), sha256=sha(args.manifest))))


def validate(m):
    if m['purpose'] != 'development_checkpoint_audit' or m['formal_result'] is not False or m['step'] not in STEPS:
        raise ValueError('Invalid development scope')
    for item in m['bindings']:
        if sha(item['path']) != item['sha256']:
            raise ValueError(f'Bound artifact changed: {item["path"]}')
    bound = {str(Path(b['path']).resolve()) for b in m['bindings']}
    checkpoint = Path(m['checkpoint']).resolve()
    required = [Path(__file__), SPLIT, Path(m['train_config']), Path(m['sampling_plan']),
                checkpoint / 'training_state/training_step.json']
    required += list((checkpoint / 'pretrained_model').glob('*'))
    required += [Path(m['dataset_root']) / f'meta/{x}.json' for x in ('info', 'stats')]
    if any(str(p.resolve()) not in bound for p in required if p.is_file()):
        raise ValueError('Runtime path is not bound')
    required_checkpoint = validate_checkpoint_complete(
        checkpoint, m['step'], m.get('expected_world_size'))
    if any(str(path.resolve()) not in bound for path in required_checkpoint):
        raise ValueError('Checkpoint completeness evidence is not bound')
    if m.get('training_source') == 'world4_revision16':
        source = source_paths(m['group'], m['step'])
        if checkpoint != source['checkpoint'].resolve() or m.get('expected_world_size') != 4:
            raise ValueError('World4 checkpoint source differs')
        for path in source['source_files']:
            if str(path.resolve()) not in bound:
                raise ValueError('World4 source evidence is not bound')
    if sha(SPLIT) != SPLIT_SHA or m['tasks'] != task_panel(read(SPLIT), read(m['sampling_plan'])):
        raise ValueError('Task panel changed')
    task_groups = {t['task_index']: t['group_id'] for t in read(SPLIT)['tasks']}
    if any(task_groups[t['task_index']] != m['group'] for t in m['tasks']):
        raise ValueError('Wrong own-group panel')
    native = read(NATIVE)
    for key in ('lerobot', 'robotwin'):
        if m[key] != native[key]:
            raise ValueError('Runtime root changed')
    if m['runtime_environment'] != native['execution']['runtime_environment']:
        raise ValueError('Runtime environment changed')
    if m['plateau_eligible'] != (m['step'] != 7500):
        raise ValueError('Diagnostic checkpoint cannot select budget')


def fixed_rng(torch, episode, frame):
    import random
    import numpy as np
    s = int(hashlib.sha256(f'robotwin-heldout-loss-v1\0{episode}\0{frame}'.encode()).hexdigest()[:8], 16)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def loss_audit(m, out, smoke):
    import torch
    from scripts.train_pi05_robotwin_task_balanced import install_frozen_training_entry, resolve_frozen_train_config
    install_frozen_training_entry()
    from lerobot.datasets.factory import make_dataset
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from lerobot.utils.collate import lerobot_collate_fn
    cfg, _ = resolve_frozen_train_config(Path(m['train_config']))
    dev_ids = sorted(e for t in m['tasks'] for e in t['episode_ids'])
    cfg.dataset.episodes = dev_ids
    cfg.dataset.exclude_episodes = None
    cfg.dataset.image_transforms.enable = False
    dataset = make_dataset(cfg)
    observed = {int(e) for e in dataset.hf_dataset['episode_index']}
    if observed != set(dev_ids):
        raise ValueError('Actual loader contains wrong episodes')
    checkpoint = Path(m['checkpoint']) / 'pretrained_model'
    pcfg = PreTrainedConfig.from_pretrained(checkpoint)
    pcfg.pretrained_path, pcfg.device = checkpoint, 'cuda'
    policy = make_policy(pcfg, ds_meta=dataset.meta, rename_map={})
    pre, _ = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=str(checkpoint),
                                      preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    policy.eval()
    sums, counts = {}, {}
    for index in range(len(dataset)):
        item = dataset[index]
        episode, frame = int(item['episode_index']), int(item['frame_index'])
        task = episode // 550
        fixed_rng(torch, episode, frame)
        batch = _preprocess_dataset_batch(lerobot_collate_fn([item]), dataset.meta.camera_keys, {}, pre)
        with torch.inference_mode():
            output = policy(batch)
        value = float((output[0] if isinstance(output, tuple) else output).detach().float().cpu())
        if not math.isfinite(value):
            raise ValueError('Nonfinite development loss')
        sums[task] = sums.get(task, 0.0) + value
        counts[task] = counts.get(task, 0) + 1
        if index % 100 == 0:
            print(json.dumps(dict(phase='heldout_loss', frames=index + 1, last_loss=value)), flush=True)
        if smoke:
            break
    means = {str(t): sums[t] / counts[t] for t in sums}
    if not smoke and set(sums) != {t['task_index'] for t in m['tasks']}:
        raise ValueError('Incomplete task loss coverage')
    result = dict(status='smoke_complete' if smoke else 'development_loss_complete', task_frame_counts=counts,
                  task_loss=means, macro_loss=sum(means.values()) / len(means), plateau_eligible=not smoke and m['plateau_eligible'])
    write(out / 'loss.json', result)
    return result


def simulator_audit(m, out, smoke):
    import torch
    import yaml
    from scripts import run_iclr2027_robotwin_native_pi05_smoke_v2 as native
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
    from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.utils.constants import ACTION
    horizon_file = Path(m['robotwin']['root']) / 'task_config/_eval_step_limit.yml'
    horizons = yaml.safe_load(horizon_file.read_text())
    checkpoint = Path(m['checkpoint']) / 'pretrained_model'
    pcfg = PreTrainedConfig.from_pretrained(checkpoint)
    native.apply_runtime_visual_override(pcfg)
    pcfg.pretrained_path, pcfg.device = checkpoint, 'cuda'
    first_cfg = RoboTwinEnvConfig(task=m['tasks'][0]['task'], episode_length=None, task_config='demo_clean', action_mode='joint')
    policy = make_policy(pcfg, env_cfg=first_cfg, rename_map={})
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=str(checkpoint),
                                         preprocessor_overrides={'device_processor': {'device': 'cuda'}})
    rows = []
    for task in m['tasks'][:1] if smoke else m['tasks']:
        horizon = int(horizons[task['task']])
        env_cfg = RoboTwinEnvConfig(task=task['task'], episode_length=None, task_config='demo_clean', action_mode='joint')
        ep, ap = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
        env = make_env(env_cfg, n_envs=1, use_async_envs=False)[task['task']][0]
        try:
            for seed in task['seeds'][:1] if smoke else task['seeds']:
                native.SEED, native.MAX_STEPS = seed, horizon
                fixed_rng(torch, task['task_index'], seed)
                journal = native.ProgressJournal(out / f'{task["task"]}-{seed}.jsonl')
                begin = time.monotonic()
                loop = native.run_control_loop(env=env, policy=policy, torch_module=torch,
                    preprocess_observation=preprocess_observation, env_preprocessor=ep,
                    preprocessor=pre, postprocessor=post, env_postprocessor=ap, action_key=ACTION, progress=journal)
                row = dict(task=task['task'], task_index=task['task_index'], seed=seed,
                           success=loop['success'], steps=loop['steps'], horizon=horizon,
                           seconds=time.monotonic() - begin, initial_observation_sha256=loop['initial_observation_sha256'])
                write(out / f'{task["task"]}-{seed}.json', row)
                rows.append(row)
        finally:
            env.close()
    means = {t: sum(r['success'] for r in rows if r['task'] == t) / sum(r['task'] == t for r in rows) for t in {r['task'] for r in rows}}
    result = dict(status='smoke_complete' if smoke else 'development_rollout_complete', episodes=len(rows),
                  task_success=means, macro_success=sum(means.values()) / len(means),
                  plateau_eligible=not smoke and m['plateau_eligible'])
    write(out / 'simulator.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'validate', 'run'])
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--group', choices=GROUPS, default='coordination')
    parser.add_argument('--step', type=int, choices=STEPS, default=5000)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=['loss', 'simulator'], default='simulator')
    parser.add_argument('--smoke', action='store_true', help='1 full-horizon rollout or 1 dev frame; no plateau claim')
    args = parser.parse_args()
    if args.action == 'build':
        return build(args)
    m = read(args.manifest)
    validate(m)
    if args.action == 'validate':
        print('development manifest validated; no GPU allocated')
        return
    if args.gpu is None or args.output is None or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('Explicit --gpu, --output and matching CUDA_VISIBLE_DEVICES required')
    from scripts.run_iclr2027_robotwin_native_pi05_smoke_v2 import exclusive_gpu_lock, check_gpu_admission
    with exclusive_gpu_lock(RUNTIME / f'resource-leases/gpu-{args.gpu}.lock'):
        admission = check_gpu_admission(args.gpu)
        args.output = args.output.resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'started.json', dict(manifest=bind(args.manifest), admission=admission,
              mode=args.mode, smoke=args.smoke, formal_result=False))
        os.environ.update(m['runtime_environment'])
        sys.path.insert(0, str(Path(m['lerobot']['root']) / 'src'))
        sys.path.insert(0, m['robotwin']['root'])
        os.chdir(m['robotwin']['root'])
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise ValueError('Exactly one visible CUDA device required')
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            result = (loss_audit if args.mode == 'loss' else simulator_audit)(m, args.output, args.smoke)
            write(args.output / 'complete.json', dict(result=result, seconds=time.monotonic()-start,
                  peak_cuda_memory_mib=torch.cuda.max_memory_allocated()/1024**2,
                  manifest_sha256=sha(args.manifest), formal_result=False))
        except BaseException as error:
            write(args.output / 'failure.json', dict(error_type=type(error).__name__, error=str(error)))
            raise


if __name__ == '__main__':
    main()
