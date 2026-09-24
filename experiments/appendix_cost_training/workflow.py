"""Inspect or stage the historical appendix source; no GPU work by default."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE=Path(__file__).resolve().parent

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def stage(root):
    """Copy exact source bytes into the expected layout; never overwrite drift."""
    provenance=json.loads((HERE/'PROVENANCE.json').read_text())
    count=0
    for row in provenance['files']:
        if not row['workspace_relative_path'].endswith('.py'):continue
        source=HERE/row['packaged_path'];target=root/row['workspace_relative_path']
        if digest(source)!=row['sha256']:raise ValueError(f'Packaged source drift: {source}')
        if target.exists():
            if digest(target)!=row['sha256']:raise FileExistsError(f'Refusing to overwrite different source: {target}')
            continue
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);count+=1
    return {'status':'PASS','copied_source_files':count,'workspace_root':str(root),'gpu_started':False}

def command(root,name,phase):
    config=json.loads((HERE/'workflows.json').read_text())
    workflow=config['workflows'][name]
    script=root/config['appendix_source']/workflow['build' if phase.startswith('build') else 'evaluate']
    python=root/'pi05_lora_finetune_v2_20260826/.venv/bin/python'
    action='prepare' if phase.endswith('prepare') else 'run'
    return [str(python),'-u',str(script),action]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['list','stage','plan','run'])
    parser.add_argument('--workspace-root',type=Path)
    parser.add_argument('--workflow',choices=list(json.loads((HERE/'workflows.json').read_text())['workflows']))
    parser.add_argument('--phase',choices=['build-prepare','build-run','eval-prepare','eval-run'],default='build-prepare')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.action=='list':print((HERE/'workflows.json').read_text());return
    if args.workspace_root is None:parser.error('--workspace-root is required')
    root=args.workspace_root.resolve()
    if args.action=='stage':print(json.dumps(stage(root),indent=2));return
    if args.workflow is None:parser.error('--workflow is required')
    cmd=command(root,args.workflow,args.phase)
    absent=[p for p in (cmd[0],cmd[2]) if not Path(p).is_file()]
    result={'status':'BLOCKED' if absent else 'READY_FOR_ORIGINAL_RUNNER_PREFLIGHT','command':cmd,
            'shell_command':shlex.join(cmd),'missing_entrypoints':absent,
            'external_assets':'ASSETS.json; original source validates input hashes, plans and host/GPU leases',
            'gpu_started':False,'warning':'Preparing writes a new immutable plan; run may use GPUs. Historical output directories must not be overwritten.'}
    print(json.dumps(result,indent=2))
    if args.action=='run' and args.execute:
        if absent:raise SystemExit(2)
        raise SystemExit(subprocess.call(cmd,cwd=root/'vla-merge'))
    if args.action=='run' and not args.execute:print('Dry run only; --execute is required.')

if __name__=='__main__':main()
