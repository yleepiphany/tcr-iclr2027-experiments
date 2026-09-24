"""Precommit four unused frames from the existing A/demo-0 trajectories."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'vla-merge/experiments/fastwam-local-diagnosis-20260924'))
import run_diagnosis as common

BASE = ROOT / 'vla-merge-runtime/experiments/claude-fastwam-tcr-20260923'
OUTPUT = ROOT / 'vla-merge-runtime/experiments/fastwam-interface-recalibration-codex-20260924/same-trajectory-extra-requests-plan-v1.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    import torch
    accepted = common.read(BASE / 'calibration-acceptance-v1.json')
    bank = common.read(BASE / 'calibration-bank-v1.json')
    if not accepted['accepted'] or accepted['jobs'] != 80:
        raise ValueError('Frozen calibration acceptance incomplete')
    rows = []
    for suite in common.SUITES:
        job = next(r for r in accepted['jobs_detail']
                   if r['round'] == 'A' and r['expert'] == suite and r['task_id'] == 0)
        source = next(r for r in bank['jobs'] if r['id'] == job['id'])
        if source['demo_index'] != 0:
            raise ValueError('Expected demo-0 trajectory')
        hashes = {r['index']: r['sha256'] for r in job['request_file_sha256']}
        extra = sorted(set(hashes) - set(job['selected_request_indices']))
        if not extra:
            raise ValueError('No unused frame available')
        index = extra[0]
        path = BASE / 'calibration-v1' / job['id'] / f'request-{index:06d}.pt'
        spec = common.frozen(path, hashes[index])
        common.check(spec)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['request_index'] != index or not {'native_request', 'trace'}.issubset(payload):
            raise ValueError('Native extra request payload differs')
        rows.append({'suite': suite, 'task_id': 0, 'round': 'A',
                     'demo_index': 0, 'calibration_job': job['id'],
                     'request_index': index, 'request': spec,
                     'calibration_selected_indices': job['selected_request_indices'],
                     'demo_file': source['demo_file'],
                     'expert_checkpoint_sha256': source['checkpoint']['sha256']})
    common.write(OUTPUT, {'schema': 'fastwam_same_trajectory_unused_frame_v1',
                 'created_at': common.now(), 'outcome_blind': True,
                 'selection_rule': 'A/task00/demo0: lowest available request index excluded from calibration selected indices',
                 'source_acceptance_sha256': common.sha(BASE / 'calibration-acceptance-v1.json'),
                 'candidate_build_plan_sha256': common.sha(OUTPUT.parent / 'attempt-01/plan.json'),
                 'requests': rows, 'request_count': 4, 'suite_count': 4,
                 'interpretation': 'Unused frames on the SAME trajectories as calibration; neither independent-trajectory holdout nor closed-loop evaluation.',
                 'environment_episodes': 0, 'training': False,
                 'run_condition': 'Only if first 12-request candidate comparison shows improvement; compare candidate and both baselines on every exact request.'})
    print({'prepared': True, 'requests': [(r['suite'], r['request_index']) for r in rows]})


if __name__ == '__main__':
    main()
