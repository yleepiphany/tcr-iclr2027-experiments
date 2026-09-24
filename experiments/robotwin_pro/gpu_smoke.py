"""Bounded native RoboTwin --smoke adapter; preserves the native idle-only gate."""
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent
PARENT = 'vla-merge-runtime/experiments/iclr2027-table5-20260910/expert-training/robotwin-three-expert-15k-sequence-v1/coordination/evaluation-015000/parent.json'
PARENT_SHA = 'ca92c96de997fa190bb204e690e6cac21ba98ebd60264b9ce089b6f9bcb84c84'
ENTRY = 'scripts/run_iclr2027_robotwin_checkpoint_development.py'
HOST = 'dsw-824375-57c745db88-n6tv9'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def resource_snapshot(gpu):
    line = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, used, free, utilization = [x.strip() for x in line.split(',')]
    raw = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return {'gpu': gpu, 'uuid': uuid, 'used_mib': float(used), 'free_mib': float(free), 'utilization': float(utilization), 'compute_pids': [int(x) for x in raw.split() if x.isdigit()]}

def run(workspace, gpu, expected_uuid, output):
    result = {'status': 'BLOCKED', 'scope': 'One native RoboTwin coordination-expert episode, fixed first task/seed, full native horizon; not a merging-method performance test', 'gpu_tasks_started': 0, 'other_processes_signalled': 0}
    if workspace is None or gpu is None or not expected_uuid or output is None:
        return {**result, 'reason': 'GPU mode requires --workspace-root, --gpu, --expected-gpu-uuid and a fresh --output directory'}
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', expected_uuid) or gpu < 0:
        return {**result, 'reason': 'Invalid explicit GPU index/UUID'}
    if socket.gethostname() != HOST:
        return {**result, 'reason': 'Native parent contract is bound to host1016; no implicit remapping'}
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output.exists() or any(x in output.parts for x in ('formal-v1', 'formal', 'tcr-formal-attempt-05')):
        return {**result, 'reason': 'Fresh isolated smoke output required; formal directories forbidden'}
    records = json.loads((HERE/'PROVENANCE.json').read_text())['files']
    wanted = next(x['sha256'] for x in records if x['path']=='native_sources/'+ENTRY)
    entry = workspace/'vla-merge'/ENTRY
    parent = workspace/PARENT
    runtime = workspace/'vla-merge-runtime/envs/iclr2027-robotwin2-py312-mplib-curobo-v3/bin/python'
    if not runtime.is_file() or not parent.is_file() or sha(parent)!=PARENT_SHA or sha(entry)!=wanted:
        return {**result, 'reason': 'Native runtime, parent, or source identity differs'}
    lease_dir = workspace/'vla-merge-runtime/resource-leases'
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease = lease_dir/f'release-smoke-{HOST}-{expected_uuid}.lock'
    with lease.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**result, 'reason': 'BLOCKED_RESOURCE_CONTENDED: host/UUID lease held'}
        board = resource_snapshot(gpu)
        result.update(resource_admission=board, host=HOST, lease=str(lease))
        # The existing native runner itself refuses >64 MiB, any utilization or PID.
        # Preserve its stricter contract; do not weaken it to share unidentified jobs.
        if board['uuid']!=expected_uuid or board['free_mib']<40960 or board['used_mib']>64 or board['utilization']!=0 or board['compute_pids']:
            return {**result, 'reason': 'BLOCKED_RESOURCE_CONTENDED: need >=40GiB and native idle-only admission; no process is stopped or preempted'}
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        command = [str(runtime), str(entry), 'run', '--manifest', str(parent), '--group', 'coordination', '--step', '15000', '--mode', 'simulator', '--smoke', '--gpu', str(gpu), '--output', str(output)]
        result.update(command=command, parent_sha256=PARENT_SHA, source_sha256=wanted)
        # Native runner acquires its legacy per-index lock and repeats admission
        # immediately before CUDA. --smoke selects tasks[:1] and seeds[:1].
        started = time.time()
        completed = subprocess.run(command, cwd=workspace/'vla-merge', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        result.update(gpu_tasks_started=1, exit_code=completed.returncode, seconds=time.time()-started, output_tail=completed.stdout[-2500:], native_output=str(output))
        proof = output/'complete.json'
        if completed.returncode or not proof.is_file():
            return {**result, 'status': 'BLOCKED', 'reason': 'Native one-episode smoke did not complete; retained output, no retry'}
        terminal = json.loads(proof.read_text())
        native = terminal.get('result', {})
        if native.get('status')!='smoke_complete' or native.get('episodes')!=1 or terminal.get('formal_result') is not False:
            return {**result, 'status': 'BLOCKED', 'reason': 'Native smoke result scope differs'}
        return {**result, 'status': 'PASS', 'native_result': terminal, 'receipt_sha256': sha(proof), 'performance_claim': False}
