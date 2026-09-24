#!/usr/bin/env python3
"""Dispatch a pinned current controller in the mounted experiment runtime.

Default prints the exact command; --execute runs it with caller-supplied flags.
Controller-specific output, ownership, existing-job and GPU checks stay active.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from cli import HERE, read, sha

ENTRIES={
    'openvla-r2-formal':'vla-merge/experiments/claude-openvla-tcr-repair-20260922/run_repair_formal1200.py',
    'openvla-experts-r23':'vla-merge/experiments/claude-openvla-tcr-repair-20260922/run_expert_formal_repeats23.py',
    'fastwam-experts-v2':'vla-merge/experiments/fastwam-formal-20260924/fastwam_experts_formal_v2.py',
    'fastwam-fixed-interface-build':'vla-merge/experiments/fastwam-interface-recalibration-codex-20260924/run_fixed_build.py',
    'fastwam-fixed-interface-eval':'vla-merge/experiments/fastwam-interface-recalibration-codex-20260924/run_fixed_eval.py',
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--entry',choices=ENTRIES,required=True)
    p.add_argument('--execute',action='store_true')
    args,remaining=p.parse_known_args()
    if remaining[:1]==['--']:remaining=remaining[1:]
    relative=ENTRIES[args.entry];script=args.workspace_root/relative
    expected=next(r['sha256'] for r in read(HERE/'provenance.json')['files'] if r['path']==relative)
    if not script.is_file() or sha(script)!=expected:
        print(json.dumps({'status':'blocked','reason':'Host controller differs from packaged snapshot','script':str(script)}));return 2
    python=args.workspace_root/'vla-merge-runtime/envs/mergevla/bin/python'
    command=[str(python),str(script),*remaining]
    print(json.dumps({'status':'ready','execute':args.execute,'entry':args.entry,'command':command,'source_sha256':expected}),flush=True)
    if not args.execute:return 0
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    return subprocess.run(command,env=env,cwd=args.workspace_root).returncode

if __name__=='__main__':sys.exit(main())
