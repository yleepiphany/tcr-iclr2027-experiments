"""Four native replay checks for the exported Soup prior, not success evaluation."""
import argparse
import json
from pathlib import Path
import socket
import run_local as base
from run_dynamic import environment, UUIDS

formal=base.formal
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[2]
SOURCE=WORK/'vla-merge-runtime/experiments/openvla-tcr-20260921/soup-prior-attempt-02'
OLD=WORK/'vla-merge-runtime/experiments/openvla-tcr-20260921/local-expert-dynamic-01'

def verify(job):
    result=base.verify(job)
    contract=json.loads((Path(job['output'])/'eval/contract.json').read_text())
    if contract['checkpoint']!=str((SOURCE/'checkpoint').resolve()):raise ValueError('Wrong Soup prior')
    return {**result,'is_tcr':False,'success_evaluation':False}

def prepare(run):
    if run.exists():raise FileExistsError(run)
    report=json.loads((SOURCE/'manifest.json').read_text())
    if not report['complete'] or report['is_tcr']:raise ValueError('Not a completed Soup prior')
    if json.loads((OLD/'queue-ended.json').read_text())['status']!='complete':raise ValueError('Expert queue not complete')
    ledger=json.loads((OLD/'identities.json').read_text());base.assert_unchanged(ledger['files'])
    for name,digest in report['files_sha256'].items():
        p=SOURCE/'checkpoint'/name;stat=p.stat()
        if formal.sha(p)!=digest:raise ValueError(f'Soup export changed: {name}')
        ledger['files'][str(p)]={'sha256':digest,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    for path in (SOURCE/'manifest.json',Path(__file__).resolve(),HERE/'run_dynamic.py'):
        ledger['files'][str(path)]=base.file_identity(path)
    run.mkdir(parents=True)
    formal.save(run/'identities.json',ledger)
    old=json.loads((OLD/'smoke/plan.json').read_text())
    jobs=[]
    for j in old['jobs']:
        out=run/'jobs'/('soup-'+j['id'])
        jobs.append({**j,'id':out.name,'output':str(out),'identity':str(run/'identities.json'),
                     'command':base.command(SOURCE/'checkpoint',j['suite'],out,True)})
    plan={**old,'schema':'oft_soup_prior_native_smoke_v1','run':str(run),'jobs':jobs,
          'host':socket.gethostname(),'identity_audit':str(run/'identities.json'),
          'identity_audit_sha256':formal.sha(run/'identities.json'),
          'gpus':[2,3],'max_workers':2,'gpu_uuids':UUIDS,'episodes':0,
          'is_tcr':False,'success_evaluation':False,'source_prior_manifest_sha256':formal.sha(SOURCE/'manifest.json')}
    formal.save(run/'plan.json',plan)
    formal.save(run/'CLAIM.json',{'owner':'Codex','host':socket.gethostname(),'gpus':[2,3],
                'task':'Soup prior native reload/replay only','success_episodes':0,'no_retry':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();run=a.run.resolve()
    if socket.gethostname()!=base.HOST:raise ValueError('Wrong host')
    if a.prepare:prepare(run);return
    plan=json.loads((run/'plan.json').read_text())
    base.assert_unchanged(json.loads((run/'identities.json').read_text())['files'])
    if any(formal.gpu_row(g)['uuid']!=u for g,u in UUIDS.items()):raise ValueError('GPU identity changed')
    formal.GPUS=(2,3);formal.FLOOR_MIB=12288
    formal.IDENTITIES=run/'identities.json';formal.EVALUATOR=HERE/'evaluate.py'
    formal.scientific_environment=environment;formal.verify_job=verify;formal.save=base.save_with_actual_counts
    formal.run_queue(plan)

if __name__=='__main__':main()
