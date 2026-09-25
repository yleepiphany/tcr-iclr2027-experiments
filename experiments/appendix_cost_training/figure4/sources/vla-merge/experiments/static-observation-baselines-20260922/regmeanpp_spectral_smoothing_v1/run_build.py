"""Freeze or launch one disclosed original-Gram spectral-smoothing extension."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from smoothing_contract import BANK_SHA,NAMES,RECIPE,SCHEMA,SMOOTHING,digest,validate_trace

HERE=Path(__file__).resolve().parent
WORK=Path('/mnt/workspace/Wilson/parameter-fusion')
REPO=WORK/'vla-merge';RUNTIME=WORK/'vla-merge-runtime';SOURCE=WORK/'pi05_lora_finetune_v2_20260826'
ROOT=RUNTIME/'experiments/static-observation-baselines-20260922/regmeanpp_spectral_smoothing/attempt-01'
BANK=RUNTIME/'experiments/iclr2027-table1-20260910/expert-dense-bank-peft-v2.json'
INPUT=RUNTIME/'experiments/claude-three-level-main-20260921/attempt-01/inputs/demo-A/repeat-01'
BASE=SOURCE/'ckpt/theta0/checkpoints/000200/pretrained_model'
PYTHON=SOURCE/'.venv/bin/python'

def prepare():
    if digest(BANK)!=BANK_SHA:raise ValueError('Expert bank changed')
    bank=json.loads(BANK.read_text());inputs={}
    for expert in bank['experts']:
        name=expert['name'];manifest=INPUT/name/'replay.json';replay=INPUT/name/'replay.safetensors'
        source=json.loads(manifest.read_text());samples=[x for x in source['samples'] if x['flow_index']==0]
        if source.get('paired_noise_verified')!={'flow0_pairs_checked':50,'flow0_equal':50,'max_abs_diff':0.0}:raise ValueError('Initial Gaussian noise identity not verified')
        view={**source,'samples':samples,'sample_count':50,'flow_indices':[0],
              'source_kind':'static_demo_observation_pure_noise_t1','static_repair_schema':SCHEMA}
        validate_trace(view,expert['dense_checkpoint']['path'],name)
        inputs[name]={'adapter_path':str(Path(expert['source_adapter']['path']).resolve()),
                      'dense_path':expert['dense_checkpoint']['path'],'dense_sha256':expert['dense_checkpoint']['model_sha256'],
                      'manifest_path':str(manifest.resolve()),'manifest_sha256':digest(manifest),
                      'replay_path':str(replay.resolve()),'replay_sha256':digest(replay),
                      'selected_source_indices':[x['index'] for x in samples]}
    implementations=[HERE/'materialize_smoothed.py',HERE/'smoothing_contract.py',HERE/'run_build.py',
        HERE/'spectral_smoothing.py',HERE/'smoothing_adapter.py',REPO/'src/vla_merge/original_regmeanpp_v2.py',REPO/'scripts/pi05_replay_prefix.py',
        REPO/'scripts/materialize_pi05_block_regmeanpp.py',REPO/'scripts/pi05_tcr_e_dense_contract.py']
    plan={'schema':SCHEMA,'recipe':RECIPE,'spectral_smoothing':SMOOTHING,'inputs':inputs,'output':str(ROOT/'checkpoint'),'base_model':str(BASE),
          'dense_bank':str(BANK),'dense_bank_sha256':BANK_SHA,
          'parameter_prior':'fixed alpha .3 from RegMean++ Appendix C.3.2 small language models and preexisting local v9; no outcome selection',
          'official_paper':'https://arxiv.org/html/2508.03121v3','official_commit':'a9fdd97629707062a5872653dd9b34f7e25c7243',
          'calibration_observations_total':200,'calibration_observations_per_expert':50,'timestep':1.0,'flow_indices':[0],
          'demonstration_action_labels_used':False,'native_denoising_trajectory_used':False,'environment_rollouts_used':False,
          'formal_outcomes_used_for_recipe_selection':False,'bias_rule':'uniform expert mean; no augmented bias regression',
          'row_normalization':'Equal row counts per expert make raw XTX and per-expert token normalization equivalent up to one common factor.',
          'implementation_sha256':{str(p):digest(p) for p in implementations},
          'resource_contract':{'hostname':'dsw-967394-56ffd4897d-42wft','physical_gpu':2,'minimum_free_mib':66560,'no_existing_compute_app':True,'allocator_fraction':.70}}
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen plan differs; refuse overwrite')
    if not path.exists():path.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    cmd=[str(PYTHON),'-u',str(HERE/'materialize_smoothed.py'),f'--dense-expert-bank={BANK}',
         f'--experiment-manifest={path}',f'--expected-experiment-sha256={digest(path)}',f'--base-model={BASE}',
         f'--output={ROOT/"checkpoint"}','--offdiag-scale=.3','--ridge-ratio=0','--max-correction-ratio=0',
         '--max-rows-per-sample=16','--replay-prefix=merged','--allow-mixed-calibration-policies','--device=cuda']
    for name in NAMES:
        row=inputs[name];cmd += [f'--expert={name}={row["adapter_path"]}',f'--calibration={name}={row["replay_path"]}',f'--manifest={name}={row["manifest_path"]}']
    return plan,path,cmd

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    plan,path,cmd=prepare()
    print(json.dumps({'plan':str(path),'plan_sha256':digest(path),'checkpoint':plan['output'],'command':cmd,'execute':args.execute}),flush=True)
    if not args.execute:return
    acceptance_path=HERE/'REAL-FIRST-LAYER-ACCEPTANCE.json'
    acceptance=json.loads(acceptance_path.read_text())
    if acceptance.get('accepted_for_full_build') is not True or acceptance.get('spectral_smoothing_sha256')!=digest(HERE/'spectral_smoothing.py'):
        raise ValueError('Missing matching independent real-first-layer acceptance')
    if socket.gethostname()!=plan['resource_contract']['hostname']:raise ValueError('Only reviewed host1023 may run this build')
    gpu=2;query=['nvidia-smi','-i',str(gpu)]
    free=int(subprocess.check_output(query+['--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    existing=subprocess.check_output(query+['--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if free<66560 or existing:raise RuntimeError(f'Dedicated GPU admission refused: free={free}, processes={existing!r}')
    if Path(plan['output']).exists():raise FileExistsError(plan['output'])
    fd=os.open(ROOT/'EXECUTION.claim',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname(),'gpu':gpu,'time':time.time()}).encode());os.close(fd)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',REGMEAN_ALLOCATOR_FRACTION='.70',TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=':'.join(map(str,[RUNTIME/'python-overlay',SOURCE/'src',REPO/'scripts',REPO/'src',HERE])),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='1',
        PALIGEMMA_TOKENIZER_PATH=str(SOURCE/'assets/paligemma-3b-pt-224-tokenizer'))
    env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
    started=time.time()
    with (ROOT/'build.log').open('x') as log:
        process=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        (ROOT/'BUILD-STARTED.json').write_text(json.dumps({'pid':process.pid,'supervisor_pid':os.getpid(),'gpu':2,'hostname':socket.gethostname(),'started_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':digest(path)},indent=2)+'\n')
        rc=process.wait()
    (ROOT/'BUILD-ENDED.json').write_text(json.dumps({'exit_code':rc,'seconds':time.time()-started,'gpu':2},indent=2)+'\n')
    raise SystemExit(rc)

if __name__=='__main__':main()
