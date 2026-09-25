#!/usr/bin/env python3
"""Release preflight, CPU import smoke, and explicit guarded one-episode GPU smoke.

Never launches training, a formal panel, or a persistent experiment queue.
GPU mode is limited to the independently bound native expert/simulator smoke.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent

class Blocked(ValueError):
    pass

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def selection_identity(manifest):
    model = manifest.get('model')
    return model.get('sha256') if isinstance(model, dict) else manifest.get('model_sha256')

def validate_current_selection(method, manifest, catalog):
    """No override: an old selection is rejected even when its own hash is valid."""
    spec = catalog['methods'][method]
    want = spec.get('current_checkpoint_sha256')
    if not want:
        raise Blocked('No accepted checkpoint identity for this benchmark/method')
    observed = selection_identity(manifest)
    if observed != want:
        raise Blocked(f'Stale or missing checkpoint binding: expected {want}, observed {observed}')
    index = manifest.get('selection_index', {})
    if set(index) != {'repeat-01', 'repeat-02', 'repeat-03'}:
        raise Blocked('Require a regenerated three-repeat PRO selection index')
    if (manifest.get('jobs'), manifest.get('episodes')) != (480, 4800):
        raise Blocked('Require the full PRO reset panel: 480 jobs / 4800 episodes')
    return observed

def package_preflight():
    provenance = json.loads((HERE / 'PROVENANCE.json').read_text())
    failures = []
    for row in provenance['files']:
        path = HERE / row['path']
        if not path.is_file() or digest(path) != row['sha256']:
            failures.append(row['path'])
        elif path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
    if failures:
        raise ValueError('Source/provenance mismatch: ' + ', '.join(failures))
    return provenance

def method_preflight(key, spec, catalog, workspace):
    result = {'method': key, 'status': spec['status'], 'current_checkpoint_sha256': spec.get('current_checkpoint_sha256')}
    if spec.get('block_reason'):
        result['reason'] = spec['block_reason']
    if spec['benchmark'] == 'LIBERO-PRO':
        relative = spec['selection_manifest']
        path = (workspace / relative) if workspace else HERE / 'evidence' / relative
        result['selection_manifest'] = str(path)
        if not path.exists():
            result.update(status='BLOCKED_MISSING_SELECTION', reason='Selection manifest is unavailable; no implicit regeneration or substitute')
        else:
            try:
                identity = validate_current_selection(key, json.loads(path.read_text()), catalog)
                result['selection_checkpoint_sha256'] = identity
                result['selection_manifest_sha256'] = digest(path)
            except Blocked as exc:
                result.update(status='BLOCKED_STALE_SELECTION', reason=str(exc))
        # Current identity alone does not certify parity or make archived stale runners safe.
        if spec['status'] == 'BLOCKED_STALE_SELECTION':
            result['status'] = 'BLOCKED_STALE_SELECTION'
            result['next_step'] = 'Create a NEW current-model manifest and runner, validate all reset/task/config bindings, then review the catalog status; never edit a running historical selection in place.'
    return result

def native_help_smoke(workspace, provenance):
    """Exercise the native import/argparse path, with CUDA hidden and no run mode."""
    if workspace is None:
        return {'status': 'BLOCKED', 'reason': 'Native workspace/runtime not supplied; package checks still run'}
    native_repo = workspace / 'vla-merge'
    records = {x['path'].removeprefix('native_sources/'): x for x in provenance['files'] if x['path'].startswith('native_sources/')}
    requests = [
        ('libero_pro', 'scripts/run_pi05_libero_pro_evaluation.py', 'vla-merge-runtime/envs/iclr2027-libero-pro-py312-v1/bin/python'),
        ('robotwin_ties', 'scripts/watch_robotwin_model_soups_formal_v1.py', 'vla-merge-runtime/envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python'),
        ('robotwin_tcr', 'experiments/robotwin-tcr-local-codex-20260924/run_tcr_formal_local_v4.py', 'vla-merge-runtime/envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python'),
    ]
    results = []
    for name, source, python in requests:
        entry, runtime = native_repo / source, workspace / python
        row = {'name': name, 'command': [str(runtime), str(entry), '--help'], 'scope': 'Native imports and CLI only; no model, simulator episode, or CUDA request'}
        if not entry.is_file() or not runtime.is_file():
            row.update(status='BLOCKED', reason='Native entrypoint/runtime absent')
        elif digest(entry) != records[source]['sha256']:
            row.update(status='BLOCKED', reason='Native entrypoint changed since source snapshot')
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            if name == 'robotwin_ties':
                env['ROBOTWIN_FORMAL_MERGE_METHOD'] = 'ties'
            started = time.monotonic()
            try:
                proc = subprocess.run(row['command'], cwd=native_repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
                row.update(status='PASS' if proc.returncode == 0 and 'usage:' in proc.stdout.lower() else 'BLOCKED', exit_code=proc.returncode, seconds=time.monotonic()-started, output_sha256=hashlib.sha256(proc.stdout.encode()).hexdigest(), output_tail=proc.stdout[-1500:])
            except subprocess.TimeoutExpired:
                row.update(status='BLOCKED', reason='45 second timeout; only this smoke subprocess was terminated')
        results.append(row)
    return {'status': 'PASS' if all(r['status']=='PASS' for r in results) else 'BLOCKED', 'checks': results}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, help='Original workspace containing vla-merge/ and vla-merge-runtime/; optional for package-only checks')
    parser.add_argument('--mode', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--method', default='all', help='Catalog key such as libero_pro/regmean_pp')
    parser.add_argument('--json-output', type=Path)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--expected-gpu-uuid')
    parser.add_argument('--output', type=Path, help='Fresh isolated native GPU smoke output; never a formal directory')
    args = parser.parse_args(argv)
    try:
        provenance = package_preflight()
        catalog = json.loads((HERE/'config/catalog.json').read_text())
        if args.method != 'all' and args.method not in catalog['methods']:
            raise ValueError('Unknown method')
        keys = list(catalog['methods']) if args.method == 'all' else [args.method]
        methods = [method_preflight(k, catalog['methods'][k], catalog, args.workspace_root) for k in keys]
        report = {'schema': 'robotwin_pro_release_smoke_v1', 'status': 'PASS', 'scope': 'Package integrity and explicitly selected smoke mode; not a formal benchmark evaluation', 'package_files_verified': len(provenance['files']), 'methods': methods, 'gpu_tasks_started': 0, 'other_processes_signalled': 0}
        report['methods_catalog_snapshot'] = 'Archived 2026-09-24 bindings, retained for stale-run rejection; see README successor snapshots for later model acceptance and waiting queues.'
        if args.mode == 'gpu':
            if args.method != 'all':
                report.update(status='BLOCKED', reason='GPU smoke is explicitly an expert/native-interface test, not any unfinished or stale method. Use the default all scope; method-specific execution remains blocked.')
            else:
                from gpu_smoke import run
                native = run(args.workspace_root, args.gpu, args.expected_gpu_uuid, args.output)
                report.update(status=native['status'], native_gpu_smoke=native,
                              gpu_tasks_started=native['gpu_tasks_started'])
        elif not args.preflight_only:
            report['native_smoke'] = native_help_smoke(args.workspace_root, provenance)
            if args.workspace_root and report['native_smoke']['status'] != 'PASS':
                report['status'] = 'BLOCKED'
        if args.method != 'all' and methods[0]['status'].startswith('BLOCKED'):
            report['status'] = 'BLOCKED'
    except (ValueError, OSError, SyntaxError, KeyError) as exc:
        report = {'status': 'BLOCKED', 'reason': str(exc), 'gpu_tasks_started': 0, 'other_processes_signalled': 0}
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status']=='PASS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
