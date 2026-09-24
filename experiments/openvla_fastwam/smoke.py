#!/usr/bin/env python3
"""Root release CLI interface: --workspace-root PATH --mode cpu|gpu."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import sys
import time
from cli import HERE, package_check, runtime_preflight, runtime_environment, smoke


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--backend',choices=['openvla','fastwam','both'],default='both')
    parser.add_argument('--gpu',type=int,default=5)
    parser.add_argument('--suite',choices=['spatial','object','goal','long'],default='spatial')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--wait-idle-seconds',type=int,default=0,
                        help='Wait bounded time for existing GPU leases; never signals their owners')
    args=parser.parse_args()
    report={'status':'pass','mode':args.mode,'scope':'Table4 OpenVLA-OFT/Fast-WAM','checks':[]}
    try:
        report['checks'].append(package_check())
        backends=['openvla','fastwam'] if args.backend=='both' else [args.backend]
        if args.mode=='gpu' and args.backend=='both':
            backends=['openvla']  # At most ONE native request per invocation.
        output=args.output or Path(tempfile.mkdtemp(prefix='tcr-table4-smoke-'))
        output.mkdir(parents=True,exist_ok=True)
        report['output']=str(output)
        for backend in backends:
            if args.mode=='gpu':
                call=argparse.Namespace(host_root=args.workspace_root,backend=backend,suite=args.suite,
                    gpu=args.gpu,output=output/backend)
                deadline=time.monotonic()+max(0,args.wait_idle_seconds)
                while True:
                    check=smoke(call)
                    if check['status']!='BLOCKED' or time.monotonic()>=deadline:break
                    time.sleep(1)
            else:
                check=runtime_preflight(backend,args.suite,args.workspace_root)
                if check['status']=='PASS':
                    command=[str(args.workspace_root/'vla-merge-runtime/envs/mergevla/bin/python'),
                             str(HERE/'worker.py'),'--backend',backend,'--host-root',str(args.workspace_root),
                             '--suite',args.suite,'--output',str(output/backend),'--preflight-only']
                    proc=subprocess.run(command,env=runtime_environment(backend,args.workspace_root),
                        capture_output=True,text=True,timeout=300)
                    (output/f'{backend}-cpu.log').write_text(proc.stdout+proc.stderr)
                    check.update(native_cpu_returncode=proc.returncode,native_cpu_log=str(output/f'{backend}-cpu.log'))
                    if proc.returncode:check['status']='FAILED'
            report['checks'].append(check)
            if check['status']!='PASS':report['status']='blocked' if check['status']=='BLOCKED' else 'failed'
        (output/'smoke-report.json').write_text(json.dumps(report,indent=2)+'\n')
    except Exception as exc:
        report.update(status='failed',error=str(exc),error_type=type(exc).__name__)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='pass' else 2


if __name__=='__main__':sys.exit(main())
