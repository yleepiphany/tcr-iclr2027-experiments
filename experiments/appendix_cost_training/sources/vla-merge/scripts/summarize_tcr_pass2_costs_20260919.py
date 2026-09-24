"""Account for explicit pass2 composite jobs, including original failed attempts.

Resource wall_seconds measures elapsed worker occupancy, not CUDA busy time.
The numeric composite must already be independently audited. Never launch jobs.
"""
import argparse
import json
import math
from pathlib import Path

import run_tcr_night_20260919 as night
from audit_tcr_pass2_composite_20260919 import ORIGINAL, REMOTE


def aggregate(items):
    if len({x['folder'] for x in items}) != len(items):
        raise ValueError('Repeated worker would double-count cost')
    for x in items:
        for key in ('worker_seconds', 'peak_allocator_gib'):
            value = x[key]
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid resource value')
    return dict(jobs=len(items), worker_seconds=sum(x['worker_seconds'] for x in items),
                peak_allocator_gib=max((x['peak_allocator_gib'] for x in items), default=0))


def record(folder, kind, completed):
    resource_path = folder / 'resources.json'
    resource = night.read(resource_path)
    if resource['mode'] != kind or resource['completed'] is not completed:
        raise ValueError(f'Resource outcome differs: {folder}')
    exit_path = folder / 'exit.json'
    code = night.read(exit_path)['return_code'] if exit_path.exists() else None
    if (completed and code not in (None, 0)) or (not completed and code in (None, 0)):
        raise ValueError(f'Exit evidence differs: {folder}')
    return dict(folder=str(folder), mode=kind, completed=completed, return_code=code,
                worker_seconds=resource['wall_seconds'],
                peak_allocator_gib=resource['peak_allocator_gib'],
                resource_sha256=night.sha(resource_path),
                launch_sha256=night.sha(folder / 'launch.json'),
                exit_sha256=night.sha(exit_path) if exit_path.exists() else None)


def summarize(audit_path):
    audit = night.read(audit_path)
    if not audit['numeric_evidence_complete'] or audit['evaluated_episodes'] != 1200:
        raise ValueError('Require the complete audited composite')
    sources = audit['episode_job_provenance']
    expected = {f'{a}/{o}' for a in ('reuse', 'alternate', 'soup') for o in range(30, 40)}
    if set(sources) != expected:
        raise ValueError('Wrong logical job allocation')
    evaluated = []
    for item in sources.values():
        folder = Path(item['folder'])
        for filename, key in (('eval/eval_info.json', 'raw_sha256'),
                              ('eval/paired-noise.jsonl', 'noise_sha256'),
                              ('verified.json', 'verified_sha256')):
            if night.sha(folder / filename) != item[key]:
                raise ValueError('Evidence changed since numeric audit')
        evaluated.append(record(folder, 'development', True))
    solved = [record(root / 'jobs' / f'solve-{arm}', 'solve', True)
              for root, arm in ((REMOTE, 'reuse'), (ORIGINAL, 'alternate'), (ORIGINAL, 'soup'))]
    failed = [record(ORIGINAL / 'jobs' / job, kind, False) for job, kind in
              (('solve-reuse', 'solve'), ('dev-alternate-30', 'development'),
               ('dev-soup-30', 'development'))]
    return dict(numeric_audit_sha256=night.sha(audit_path),
                successful_solves=aggregate(solved), successful_evaluations=aggregate(evaluated),
                original_failed_attempts=aggregate(failed),
                total_including_failures=aggregate(solved + evaluated + failed),
                jobs=solved + evaluated + failed,
                missing_exit_receipts=[x['folder'] for x in solved + evaluated if x['return_code'] is None],
                excluded='Earlier expert training, first-pass construction, historical cache collection, '
                         'reused unchanged evaluation, scheduling waits and unrelated campaigns.',
                measurement='Sum of worker elapsed wall_seconds, not device busy-hours, FLOPs, energy '
                            'or whole-study elapsed time. Concurrent workers may share a GPU. '
                            'Missing remote exits remain a lifecycle limitation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.audit)
    if args.output.exists():
        raise FileExistsError('Preserve the previous cost report')
    night.write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'jobs'}, indent=2))
