#!/usr/bin/env python3
"""Fixed Figure 4 TIES subset adapter. All commands here use CPU only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

WORK = Path('/mnt/workspace/Wilson/parameter-fusion')
REPO = WORK / 'vla-merge'
ROOT = WORK / 'vla-merge-runtime/experiments/fig4-baselines-codex-20260925'
HERE = Path(__file__).resolve().parent
CHILD = ROOT / 'ties-adapter-v2'
PARENT_SHA = '62ff0891df02a75d20a8b724f70f93a8373f75852a7385c88034993c326d7875'
SUBSETS = {'spatial-goal': ['spatial', 'goal'], 'spatial-object-goal': ['spatial', 'object', 'goal']}
ALPHA, DENSITY, CHUNK = .9, .3, 4 * 1024 * 1024
sys.path[:0] = [str(HERE), str(REPO / 'scripts'), str(REPO / 'src'),
               str(WORK / 'pi05_lora_finetune_v2_20260826/src'),
               str(WORK / 'pi05_lora_finetune_v2_20260826/lerobot/src')]

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def require(ok, reason):
    if not ok: raise ValueError(reason)
def stamp(): return datetime.now(timezone.utc).isoformat()
def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream: stream.write(json.dumps(data, sort_keys=True, indent=2) + '\n')
def event(**data): print(json.dumps({'time': stamp(), **data}), flush=True)

def cpu_imports():
    # These commands must never allocate a GPU or inspect its runtime context.
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'Set CUDA_VISIBLE_DEVICES to the empty string for CPU commands')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    import ta_ties_core as core
    import materialize_iclr2027_table1_ta_ties_fastlane as fastlane
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    return np, torch, core, fastlane

def parent():
    require(sha(ROOT / 'plan.json') == PARENT_SHA, 'Parent Figure 4 plan changed')
    p = read(ROOT / 'plan.json')
    require(p['algorithms']['ties']['density'] == DENSITY and p['algorithms']['ties']['global_alpha'] == ALPHA,
            'Parent TIES constants differ')
    for row in p['algorithm_sources']:
        if Path(row['path']).name in ['ta_ties_core.py', 'materialize_iclr2027_table1_ta_ties_fastlane.py']:
            require(sha(row['path']) == row['sha256'], 'Frozen main-table TIES source changed')
    return p

def contract(large=False):
    p = parent()
    np, torch, core, fastlane = cpu_imports()
    # Registry validation is read-only; no old materializer CLI, search or export runs.
    c = fastlane.validate_inputs(
        Path(core.CANONICAL_OUTPUT_ROOT).parents[1] / 'manifest-revision7.json',
        Path(p['source_binding']['dense_bank']['path']), verify_large_files=False)
    segments = fastlane.build_segments(c)
    vectors = p['inputs']['ties_task_vectors']
    require(sha(vectors['manifest']['path']) == vectors['manifest']['sha256'], 'Task-vector cache manifest changed')
    manifest = read(vectors['manifest']['path'])
    require(manifest['identity'] == fastlane._cache_identity(c), 'Original task-vector/cache identity differs')
    for expert, row in vectors['vectors'].items():
        require(Path(row['path']).stat().st_size == row['bytes'] == fastlane.EXPECTED_COORDINATE_COUNT * 4, 'Vector size differs')
        require(manifest['vectors'][expert]['sha256'] == row['sha256'], 'Vector hash binding differs')
        if large:
            event(phase='verify_task_vector', expert=expert)
            require(sha(row['path']) == row['sha256'], 'Task vector bytes changed')
    if large:
        event(phase='verify_common_base')
        require(sha(c['base_model']) == core.BASE_MODEL_SHA256, 'Common base bytes changed')
    return p, c, segments, (np, torch, core, fastlane)

def prepare():
    require(not (CHILD / 'plan.json').exists(), 'TIES adapter plan already exists')
    p, c, segments, imported = contract()
    _, _, core, fastlane = imported
    jobs = []
    for subset, experts in SUBSETS.items():
        reference = next(x for x in p['new_builds'] if x['id'] == f'ties-{subset}')
        jobs.append({'id': reference['id'], 'subset': subset, 'experts': experts,
                     'checkpoint': reference['checkpoint'], 'density': DENSITY, 'alpha': ALPHA,
                     'direction_file': str(CHILD / subset / 'direction.f64.bin'),
                     'native_cpu_check_required': True,
                     'build_command': [str(WORK / 'pi05_lora_finetune_v2_20260826/.venv/bin/python'), '-u', str(HERE / 'ties_subset_v2.py'), 'build', '--subset', subset],
                     'native_load_command': [str(WORK / 'pi05_lora_finetune_v2_20260826/.venv/bin/python'), '-u', str(HERE / 'ties_subset_v2.py'), 'native-load', '--subset', subset]})
    plan = {'schema': 'fig4_ties_subset_adapter_v2', 'created_at': stamp(), 'parent_plan_sha256': PARENT_SHA,
            'adapter_sha256': sha(__file__), 'kernel': {'path': core.__file__, 'sha256': sha(core.__file__)},
            'export_helpers': {'path': fastlane.__file__, 'sha256': sha(fastlane.__file__)},
            'operator': 'unchanged ta_ties_core.ties_direction(selected original float32 vectors, keep_fraction=.3); base + .9*float64 direction cast to original tensor dtype',
            'global_trim': True, 'sign_election': 'mass', 'global_zero_sign_fallback': True, 'disjoint': 'mean',
            'coordinate_count': fastlane.EXPECTED_COORDINATE_COUNT, 'modified_tensor_count': len(segments),
            'segments': segments, 'base_model': str(c['base_model']), 'base_sha256': core.BASE_MODEL_SHA256,
            'sidecar_template': str(c['soup_root']), 'sidecar_rule': 'Original main-table runtime sidecars; only config.pretrained_path points at isolated new export, use_peft=false.',
            'jobs': jobs, 'cpu_only': True, 'gpu_jobs_launched': 0, 'no_parameter_search': True,
            'supersedes_cpu_import_failure': str(ROOT / 'ties-adapter-v1/CPU-PREFLIGHT-FAILED.json'),
            'no_automatic_retry': True, 'existing_four_expert_outputs_unchanged': True,
            'parent_port_gap_resolved': 'TIES subset vector selection/export only; RegMean++ and FeatCal remain separate missing ports.'}
    save(CHILD / 'plan.json', plan)
    save(CHILD / 'PLAN-SHA256.json', {'plan_sha256': sha(CHILD / 'plan.json')})
    return preflight()

def frozen():
    p = read(CHILD / 'plan.json')
    require(sha(CHILD / 'plan.json') == read(CHILD / 'PLAN-SHA256.json')['plan_sha256'], 'TIES adapter plan changed')
    require(p['parent_plan_sha256'] == PARENT_SHA and p['adapter_sha256'] == sha(__file__), 'Adapter source/parent changed')
    for key in ['kernel', 'export_helpers']:
        require(sha(p[key]['path']) == p[key]['sha256'], 'Imported source changed')
    return p

@contextmanager
def cpu_native_imports():
    # Same bounded CPU-validation workaround as the existing repository helper
    # train_pi05_robotwin_task_balanced_ddp_capacity.py:49-73. Dense PI05 never
    # uses TE; restoring discovery after import leaves all shared files intact.
    original = importlib.util.find_spec
    def find_spec(name, *args, **kwargs):
        if name == 'transformer_engine' or name.startswith('transformer_engine.'):
            return None
        return original(name, *args, **kwargs)
    importlib.util.find_spec = find_spec
    try:
        yield
    finally:
        importlib.util.find_spec = original

def preflight():
    plan = frozen()
    p, c, segments, imported = contract()
    np, torch, core, fastlane = imported
    require(segments == plan['segments'], '422-tensor adapted coordinate order differs')
    with cpu_native_imports():
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    configs = []
    for subset in SUBSETS:
        path = p['inputs']['subset_soups'][subset]['checkpoint']
        cfg = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        cfg.device = 'cpu'; cfg.compile_model = False; cfg.gradient_checkpointing = False; cfg.n_action_steps = 10
        require(cfg.type == 'pi05' and cfg.use_peft is False and cfg.n_action_steps == 10, 'Native PI05 config differs')
        cfg.validate_features()
        configs.append({'subset': subset, 'config_class': type(cfg).__name__, 'policy_class': PI05Policy.__name__,
                        'device': cfg.device, 'use_peft': cfg.use_peft, 'n_action_steps': cfg.n_action_steps})
    require(not torch.cuda.is_initialized(), 'CPU preflight initialized CUDA')
    return {'status': 'pass_cpu_preflight', 'plan_sha256': sha(CHILD / 'plan.json'),
            'subsets': list(SUBSETS), 'tensor_count': len(segments), 'coordinates': plan['coordinate_count'],
            'native_config_imports': configs,
            'native_model_weights_loaded': False,
            'cpu_import_workaround': 'Only optional transformer_engine discovery hidden during imports; original native dense modules and GPU evaluator unchanged',
            'native_load_pending_reason': 'New subset checkpoints have not been built; native-load is implemented and must run after each CPU build.',
            'full_task_vector_hashes_rechecked': False, 'full_hash_gate': 'Enforced by build before writing weights.',
            'gpu_initialized': False, 'builds_started': 0}

def patch_export(base_path, output_file, segments, direction, alpha, imported):
    np, torch, core, fastlane = imported
    from safetensors import safe_open
    layout = fastlane.parse_safetensors_layout(base_path)
    clone = fastlane.clone_file(base_path, output_file)
    descriptor = os.open(output_file, os.O_RDWR)
    try:
        with safe_open(base_path, framework='pt', device='cpu') as base:
            for index, segment in enumerate(segments, 1):
                key = segment['key']; loc = layout[key]
                base_value = base.get_tensor(key).detach().cpu()
                require(list(base_value.shape) == segment['shape'] == loc['shape'], 'Export shape differs')
                delta = torch.from_numpy(np.asarray(direction[segment['start']:segment['end']])).reshape(base_value.shape)
                value = (base_value.double() + float(alpha) * delta).to(base_value.dtype)
                require(torch.isfinite(value).all().item(), 'Nonfinite TIES export')
                payload = value.contiguous().view(torch.uint8).numpy().tobytes()
                require(len(payload) == loc['end'] - loc['start'], 'Serialized tensor size differs')
                fastlane.pwrite_all(descriptor, payload, loc['start'])
                if index % 25 == 0 or index == len(segments): event(phase='export', tensors=index)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return clone

def build(subset):
    plan = frozen()
    job = next(j for j in plan['jobs'] if j['subset'] == subset)
    output = Path(job['checkpoint']); run = CHILD / subset
    require(not output.exists() and not run.exists(), 'No overwrite or automatic retry of an existing subset build')
    p, c, segments, imported = contract(large=True)
    np, torch, core, fastlane = imported
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'STARTED.json', {'pid': os.getpid(), 'started_at': stamp(), 'cpu_only': True, 'subset': subset,
                              'plan_sha256': sha(CHILD / 'plan.json')})
    temporary = output.parent.with_name(output.parent.name + f'.partial-{os.getpid()}-{uuid.uuid4().hex}')
    started = time.monotonic()
    try:
        vectors = [np.memmap(p['inputs']['ties_task_vectors']['vectors'][e]['path'], mode='r', dtype=np.float32) for e in SUBSETS[subset]]
        direction = np.memmap(job['direction_file'], mode='w+', dtype=np.float64, shape=(plan['coordinate_count'],))
        event(phase='global_ties_direction', subset=subset, experts=SUBSETS[subset])
        _, metadata = core.ties_direction(vectors, keep_fraction=DENSITY, output=direction, chunk_size=CHUNK)
        direction.flush()
        policy = temporary / 'pretrained_model'; policy.mkdir(parents=True)
        fastlane.copy_policy_sidecars(c['soup_root'], policy, output)
        clone = patch_export(c['base_model'], policy / 'model.safetensors', segments, direction, ALPHA, imported)
        model_sha = sha(policy / 'model.safetensors')
        identity = core.full_policy_identity(policy, model_sha, expected_sidecars=core.RUNTIME_POLICY_SIDECARS,
                    model_sha256_already_verified=model_sha, reported_root=output)
        manifest = {'schema': 'fig4_ties_subset_checkpoint_v1', 'status': 'complete_cpu_export_native_load_pending',
                    'subset': subset, 'experts': SUBSETS[subset], 'method': 'ties', 'density': DENSITY, 'alpha': ALPHA,
                    'fusion_metadata': metadata, 'model_sha256': model_sha, 'checkpoint': str(output),
                    'modified_tensor_count': len(segments), 'coordinate_count': plan['coordinate_count'],
                    'base_sha256': core.BASE_MODEL_SHA256, 'parent_plan_sha256': PARENT_SHA,
                    'adapter_plan_sha256': sha(CHILD / 'plan.json'), 'source_vector_sha256': {e:p['inputs']['ties_task_vectors']['vectors'][e]['sha256'] for e in SUBSETS[subset]},
                    'direction_sha256': sha(job['direction_file']), 'full_policy_identity': identity,
                    'finite': True, 'copy_strategy': clone, 'cpu_only': True, 'wall_seconds': time.monotonic()-started}
        save(temporary / 'candidate-manifest.json', manifest)
        fastlane._fsync_tree(temporary)
        output.parent.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output.parent)
        save(run / 'BUILD-COMPLETE.json', manifest)
        return {'status': 'complete_cpu_export_native_load_pending', 'checkpoint': str(output), 'model_sha256': model_sha}
    except BaseException as exc:
        save(run / 'FAILED.json', {'status': 'failed_no_retry', 'error': repr(exc), 'partial_output_preserved': str(temporary)})
        raise

def native_load(subset):
    plan = frozen(); job = next(j for j in plan['jobs'] if j['subset'] == subset)
    root = Path(job['checkpoint']); manifest = read(root.parent / 'candidate-manifest.json')
    require(manifest['adapter_plan_sha256'] == sha(CHILD / 'plan.json') and manifest['experts'] == SUBSETS[subset], 'Export provenance differs')
    require(sha(root / 'model.safetensors') == manifest['model_sha256'], 'New checkpoint changed')
    np, torch, _, _ = cpu_imports()
    with cpu_native_imports():
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    cfg = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    cfg.device = 'cpu'; cfg.compile_model = False; cfg.gradient_checkpointing = False; cfg.n_action_steps = 10
    policy = PI05Policy.from_pretrained(root, config=cfg, local_files_only=True, strict=True)
    for name, tensor in policy.state_dict().items():
        require(tensor.device.type == 'cpu' and torch.isfinite(tensor).all().item(), f'Native tensor invalid: {name}')
    require(not torch.cuda.is_initialized(), 'Native CPU load initialized CUDA')
    receipt = {'status': 'pass_native_cpu_load', 'checkpoint': str(root), 'model_sha256': manifest['model_sha256'],
               'strict_native_loader': True, 'all_state_tensors_finite': True, 'gpu_initialized': False,
               'native_action_forward_checked': False,
               'remaining_before_formal': 'One native action/velocity finite check on an allocated GPU or CPU, plus exact model/sidecar/reset binding and owned resource lease.'}
    save(CHILD / subset / 'NATIVE-CPU-LOAD.json', receipt)
    return receipt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','preflight','build','native-load'])
    parser.add_argument('--subset', choices=list(SUBSETS))
    args = parser.parse_args()
    if args.action in ['build','native-load'] and args.subset is None: parser.error('--subset is required')
    result = prepare() if args.action == 'prepare' else preflight() if args.action == 'preflight' else build(args.subset) if args.action == 'build' else native_load(args.subset)
    print(json.dumps(result, indent=2))
