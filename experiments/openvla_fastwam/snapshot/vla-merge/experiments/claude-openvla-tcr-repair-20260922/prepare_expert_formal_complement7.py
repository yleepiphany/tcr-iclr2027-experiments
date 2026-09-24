#!/usr/bin/env python3
"""Freeze the exact seven-cell expert formal complement after one salvaged cell."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import run_expert_formal_repeats23 as original
import run_local as base

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
OLD = WORK / 'vla-merge-runtime/experiments/claude-openvla-tcr-repair-20260922/expert-formal-repeats23-attempt-02'
SALVAGE = WORK / 'coordination/2026-09-23/openvla-expert-spatial-r02-salvage.json'


def prepare(run: Path) -> None:
    if run.exists() or SALVAGE.exists():
        raise FileExistsError('New complement/salvage path must be unused')
    old_plan = json.loads((OLD / 'plan.json').read_text())
    old_terminal = json.loads((OLD / 'queue-ended.json').read_text())
    if (old_terminal.get('status') != 'incomplete' or old_terminal.get('accepted_jobs') != 0
            or old_terminal.get('expected_jobs') != 8):
        raise ValueError('Prior supervisor terminal differs')
    old_job = next(job for job in old_plan['jobs'] if job['id'] == 'expert-spatial-r02')
    exit_receipt = json.loads((Path(old_job['output']) / 'exit.json').read_text())
    if exit_receipt.get('returncode') != 0 or exit_receipt.get('outcome') != 'success':
        raise ValueError('Spatial child did not exit cleanly')
    audited = original.verify(old_job)
    if audited['episodes'] != 100:
        raise ValueError('Salvaged cell is not 100 complete episodes')
    base.BASE_SAVE(SALVAGE, {
        'schema': 'openvla_expert_formal_spatial_r02_independent_salvage_v1',
        'audited_utc': datetime.now(timezone.utc).isoformat(),
        'id': old_job['id'], 'episodes': 100,
        'raw_episodes_sha256': audited['episodes_sha256'],
        'selection_sha256': audited['selection_sha256'],
        'summary': base.file_identity(Path(old_job['output']) / 'eval/summary.json'),
        'exit': base.file_identity(Path(old_job['output']) / 'exit.json'),
        'prior_terminal': base.file_identity(OLD / 'queue-ended.json'),
        'verifier': base.file_identity(HERE / 'run_expert_formal_repeats23.py'),
        'supervisor_failure': 'verification environment lacked LIBERO Python path; child and raw identity valid',
        'success_values_not_used_for_complement_selection': True,
    })
    original.prepare(run)
    identity_path = run / 'identities.json'
    identity = json.loads(identity_path.read_text())
    identity['new_episodes'] = 700
    identity['salvaged_repeat2_spatial'] = str(SALVAGE)
    identity['files'][str(SALVAGE.resolve())] = base.file_identity(SALVAGE)
    identity['files'][str(Path(__file__).resolve())] = base.file_identity(Path(__file__).resolve())
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
    plan_path = run / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['jobs'] = [job for job in plan['jobs'] if job['id'] != 'expert-spatial-r02']
    if len(plan['jobs']) != 7 or len({job['id'] for job in plan['jobs']}) != 7:
        raise ValueError('Exact seven-cell complement differs')
    plan['episodes'] = 700
    plan['salvaged_accepted_cell'] = str(SALVAGE)
    plan['identity_audit_sha256'] = original.formal.sha(identity_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    claim_path = run / 'CLAIM.json'
    claim = json.loads(claim_path.read_text())
    claim['episodes'] = 700
    claim['scope'] = 'only seven missing official-expert procedural cells; repeat2/spatial independently salvaged'
    claim['salvage'] = str(SALVAGE)
    claim_path.write_text(json.dumps(claim, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'new_jobs': len(plan['jobs']), 'new_episodes': 700,
                      'salvaged_episodes': 100, 'prior_repeat1_episodes': 400}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    prepare(parser.parse_args().run.resolve())
