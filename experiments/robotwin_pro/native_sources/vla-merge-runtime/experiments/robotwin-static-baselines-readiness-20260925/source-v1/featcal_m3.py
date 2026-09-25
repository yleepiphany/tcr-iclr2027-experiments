#!/usr/bin/env python3
"""Isolated M=3 RoboTwin adapter for the existing exact Table-1 FeatCal engine.

prepare is CPU-only. GPU stages require an explicit launch flag and are not run
by preparation/tests. Historical files, constants and checkpoints remain intact:
private module instances bind the three experts and the new static input bank.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

if __name__=='__main__' and ('--stage=prepare' in sys.argv or any(
        value=='--stage' and i+1<len(sys.argv) and sys.argv[i+1]=='prepare' for i,value in enumerate(sys.argv))):
    os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from prepare_readiness import sha
from static_bank import GROUPS,load_batch,validate_model_batch

FLOW_SLOTS=(0,5,9)  # Schema identifiers only: all physical times are one.
TOTAL_ROWS=3_330_900

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
def event(run,**data):
    value={'at':datetime.now(timezone.utc).isoformat(),**data}
    with (run/'progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(value),flush=True)

def clone_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

class API:
    def __init__(self,work,bank_path):
        self.work=Path(work).resolve();self.repo=self.work/'vla-merge';self.bank_path=Path(bank_path).resolve()
        source=self.work/'pi05_lora_finetune_v2_20260826'
        sys.path[:0]=[str(self.repo),str(self.repo/'src'),str(source/'src'),str(source/'lerobot/src')]
        self.baseline=clone_module('scripts._robotwin_featcal_m3_baseline',self.repo/'scripts/run_pi05_featcal_formal_full.py')
        self.cache=clone_module('vla_merge._robotwin_featcal_m3_cache_plan',self.repo/'src/vla_merge/featcal_cache_plan.py')
        self.cache.EXPERTS=GROUPS;self.cache.EXPECTED_TOTAL_ROWS=TOTAL_ROWS
        from vla_merge.featcal_prefix_chain import SingleStepSelectedRowCollector
        from vla_merge.featcal_execution import explicit_velocity_forward
        self.SingleStepSelectedRowCollector=SingleStepSelectedRowCollector;self.forward=explicit_velocity_forward
        self.bank=read(self.bank_path);self.selection=read(self.bank['selection_plan'])
        if self.bank['schema']!='robotwin_static_observation_initial_noise_bank_v1' or set(self.bank['groups'])!=set(GROUPS):
            raise ValueError('Need the new three-group static bank')
        self.models={e['name']:e['dense_checkpoint'] for e in self.selection['models']}
        self.soup=Path(self.selection['soup']['path']);self.soup_sha=self.selection['soup']['manifest']['model_sha256']
        self.base=Path(self.selection['base']['path']);self.bank_sha=sha(self.bank_path)
        self.active_gpu=None
        b=self.baseline;b.EXPERT_ORDER=list(GROUPS);b.STUDENT=self.soup;b.STUDENT_SHA256=self.soup_sha;b.CALL_STATE_MANIFEST_SHA256=self.bank_sha
        self.steps=b.build_pi05_adapted_linear_forward_plan(b.expected_pi05_adapted_weight_keys())

    def check_runtime_memory(self):
        if self.active_gpu is None:return
        free=int(subprocess.check_output(['nvidia-smi','-i',str(self.active_gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<12288:raise RuntimeError('Own FeatCal stage stopped: less than 12GiB runtime free; no other process signaled')

    def implementations(self):
        paths=[Path(__file__),Path(__file__).with_name('static_bank.py'),Path(__file__).with_name('prepare_readiness.py'),
            Path(self.baseline.__file__),Path(self.baseline.teacher_runner.__file__),Path(self.cache.__file__),
            self.repo/'src/vla_merge/featcal_forward_order.py',self.repo/'src/vla_merge/featcal_prefix_chain.py',
            self.repo/'src/vla_merge/featcal_execution.py',self.repo/'src/vla_merge/featcal_hybrid_solve.py',
            self.work/'pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/policies/pi05/modeling_pi05.py',
            self.work/'pi05_lora_finetune_v2_20260826/lerobot/src/lerobot/policies/pi_gemma.py']
        return {str(p.resolve()):sha(p) for p in paths}

    def quotas(self):
        keys=self.baseline.expected_pi05_adapted_weight_keys();logical=[]
        with safe_open(str(self.soup/'model.safetensors'),framework='pt',device='cpu') as f:
            for key in sorted(keys):logical.append({'base_key':key,'base_shape':f.get_slice(key).get_shape()})
            for path in ('model.action_in_proj','model.action_out_proj','model.time_mlp_in','model.time_mlp_out'):
                key=path+'.bias';logical.append({'base_key':key,'base_shape':f.get_slice(key).get_shape()})
        return self.cache.derive_module_quotas({'adaptation_domain':{'logical_tensors':logical}},self.steps)

    def spec_map(self,index):
        result={}
        for spec in index['shards']:
            key=(spec['role'],spec['expert'],int(spec['task']),int(spec['step']))
            if key in result:raise ValueError('Duplicate M3 shard specification')
            result[key]=spec
        expected={(role,g,t,s) for role in ('teacher','student') for g in GROUPS for t in range(10) for s in range(47)}
        if set(result)!=expected:raise ValueError('M3 shard role/group/task/stage coverage differs')
        return result

    def prepare(self,run):
        if run.exists():raise FileExistsError('New FeatCal run required')
        for group,entry in self.bank['groups'].items():
            if sha(self.bank_path.parent/entry['tensor_file'])!=entry['tensor_sha256']:raise ValueError('Input tensor bank changed')
            if entry['observations']!=50 or entry['noise_states']!=150:raise ValueError('Group quota differs')
        if sha(self.bank['selection_plan'])!=self.bank['selection_plan_sha256']:raise ValueError('Selection source changed')
        quotas=self.quotas();sources=[{'path':str(self.soup),'model_sha256':self.soup_sha},self.selection['base'],*self.models.values()]
        model_files=[]
        for model in sources:
            file=Path(model['path'])/'model.safetensors'
            before=file.stat()
            if sha(file)!=model['model_sha256']:raise ValueError('Model bytes differ from accepted bank')
            after=file.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Model changed while hashing')
            model_files.append({'path':str(file),'sha256':model['model_sha256'],'size':after.st_size,'mtime_ns':after.st_mtime_ns})
        deployment_files={}
        for folder in [self.soup,*[Path(v['path']) for v in self.models.values()]]:
            for file in folder.rglob('*'):
                if file.is_file() and file.name!='model.safetensors':deployment_files[str(file)]=sha(file)
        plan={'schema':'robotwin_featcal_static_m3_forward_order_v1','prepared_at':datetime.now(timezone.utc).isoformat(),
            'work_root':str(self.work),'bank':str(self.bank_path),'bank_sha256':self.bank_sha,
            'models':self.models,'soup':{'path':str(self.soup),'model_sha256':self.soup_sha},'base':self.selection['base'],'model_files':model_files,
            'groups':list(GROUPS),'stages':47,'linear_weights':418,'unchanged_biases':4,'regression_rows':TOTAL_ROWS,
            'observations':150,'noise_states':450,'physical_timestep':1.,'legacy_flow_slots_are_noise_replicas':True,
            'alpha':.3,'rho':2.,'lambda':.05,'eps':1e-8,'solve_dtype':'float64','workspace_cap_bytes':2*1024**3,
            'source_kind':'static calibration demonstration observations, no action labels or execution features',
            'initialization':'existing Table5 uniform task-delta Soup, PEFT-safe dense form',
            'implementations':self.implementations(),'deployment_files_sha256':deployment_files,
            'gpu_smoke_passed':False,'formal_evaluation_authorized_by_plan':False,
            'resource_proposal':{'minimum_free_mib':49152,'allocator_fraction':.5,'minimum_runtime_free_mib':12288},
            'status':'CPU_PREPARED_NATIVE_GPU_PARITY_REQUIRED'}
        run.mkdir(parents=True);write(run/'plan.json',plan);digest=sha(run/'plan.json')
        index=self.cache.build_shard_index(quotas,self.steps,self.cache.build_call_slots(),contract_sha256=digest,
            expert_model_sha256={g:self.models[g]['model_sha256'] for g in GROUPS},initial_student_sha256=self.soup_sha)
        if index['shard_count']!=2820 or sum(q.rows_per_expert for q in quotas)*3!=TOTAL_ROWS:raise ValueError('M3 dimensions incorrect')
        write(run/'shard-index.json',index);write(run/'quotas.json',{'quotas':[q.to_dict() for q in quotas]})
        event(run,event='cpu_prepared',stages=47,weights=418,calls=450,shards=2820,rows=TOTAL_ROWS,plan_sha256=digest)

    def contract(self,run):
        plan=read(run/'plan.json')
        if plan['implementations']!=self.implementations() or plan['bank_sha256']!=self.bank_sha:raise ValueError('Frozen implementation/bank changed')
        if plan['groups']!=list(GROUPS) or plan['regression_rows']!=TOTAL_ROWS or plan['stages']!=47:raise ValueError('M3 method contract changed')
        for item in plan['model_files']:
            stat=Path(item['path']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(item['size'],item['mtime_ns']):raise ValueError('Frozen model file changed since full-hash preparation')
        for filename,digest in plan['deployment_files_sha256'].items():
            if sha(filename)!=digest:raise ValueError('Frozen deployment sidecar changed')
        for group,entry in self.bank['groups'].items():
            if sha(self.bank_path.parent/entry['tensor_file'])!=entry['tensor_sha256']:raise ValueError('Input bank changed')
        digest=sha(run/'plan.json');index=read(run/'shard-index.json')
        if index['contract_sha256']!=digest:raise ValueError('Shard index belongs to another plan')
        specs=self.spec_map(index)
        return plan,digest,specs,self.quotas()

    def row_provider(self,batch):
        return self.baseline.teacher_runner.build_streaming_row_mask_provider(
            image_masks=[batch[f'image_mask_{i}'] for i in range(3)],token_mask=batch['masks'],
            action_valid_mask=torch.ones(batch['x_t'].shape[:2],dtype=torch.bool,device=batch['x_t'].device))

    @staticmethod
    def slots(group,task,replica):
        return [f'expert={group}/task={task:02d}/episode=00/request={r:02d}/flow={FLOW_SLOTS[replica]:02d}' for r in range(5)]

    def inputs(self,group,task,replica,device='cuda'):
        self.check_runtime_memory()
        cpu=load_batch(self.bank_path,group,task,replica,verify=False)
        return {k:v.to(device) for k,v in cpu.items()}

    @staticmethod
    def load_policy(path):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        cfg=PreTrainedConfig.from_pretrained(path);cfg.device='cuda';cfg.compile_model=False;cfg.gradient_checkpointing=False;cfg.pretrained_path=Path(path)
        return PI05Policy.from_pretrained(path,config=cfg).to('cuda').eval()

    def oracle(self,model,batch):
        from unittest.mock import patch
        from lerobot.policies.pi05 import modeling_pi05
        result=[];handle=model.model.action_out_proj.register_forward_hook(lambda m,a,out:result.append(out.detach().clone()))
        try:
            with patch.object(modeling_pi05,'_build_flow_matching_inputs',return_value=(batch['x_t'],batch['time'])):
                model.model.forward(images=[batch[f'image_{i}'] for i in range(3)],img_masks=[batch[f'image_mask_{i}'] for i in range(3)],
                    tokens=batch['tokens'],masks=batch['masks'],actions=torch.zeros_like(batch['x_t']),noise=torch.zeros_like(batch['x_t']),time=batch['time'])
        finally:handle.remove()
        if len(result)!=1:raise ValueError('Native action head did not execute once')
        return result[0]

    @torch.inference_mode()
    def smoke(self,run):
        plan,digest,specs,quotas=self.contract(run);reports=[]
        for group,path in [('soup',self.soup)]+[(g,Path(self.models[g]['path'])) for g in GROUPS]:
            model=self.load_policy(path);batch=self.inputs(group if group in GROUPS else 'coordination',0,0)
            collector=self.cache.MultiLayerStreamingRowCollector(model,self.steps,quotas,
                call_slot_ids=self.slots('coordination',0,0),call_ordinals_within_task=[r*3 for r in range(5)],
                row_mask_provider=self.row_provider(batch),selection_seed=self.baseline.teacher_runner.SELECTION_SEED)
            output={}
            def forward():output['v']=self.forward(model,batch);return output['v']
            captured=collector.capture(forward);oracle=self.oracle(model,batch)
            if not torch.equal(output['v'],oracle) or output['v'].shape!=(5,50,32):raise ValueError('Original full-joint graph parity failed')
            reports.append({'model':group,'hook_trace_events':len(captured.trace),'all_target_modules':len(captured.inputs_by_module),
                'native_full_joint_bitwise_equal':True,'velocity_shape':list(output['v'].shape)})
            del model,batch,captured,collector,oracle,output;gc.collect();torch.cuda.empty_cache()
        write(run/'native-smoke.json',{'status':'PASS','plan_sha256':digest,'models':reports,'formal_episodes':0,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})

    def require_smoke(self,run,digest):
        r=read(run/'native-smoke.json')
        if r['status']!='PASS' or r['plan_sha256']!=digest:raise ValueError('Matching native graph smoke required')

    @torch.inference_mode()
    def teachers(self,run,group):
        plan,digest,specs,quotas=self.contract(run);self.require_smoke(run,digest);cache=run/'cache'
        model=self.load_policy(self.models[group]['path'])
        for task in range(10):
            task_specs=[specs[('teacher',group,task,step.step)] for step in self.steps]
            existing=[self.baseline._validated_or_missing(cache,s,digest) for s in task_specs]
            if all(existing):continue
            captures={q.module_path:[] for q in quotas};keys={q.module_path:[] for q in quotas}
            for replica in range(3):
                batch=self.inputs(group,task,replica);collector=self.cache.MultiLayerStreamingRowCollector(model,self.steps,quotas,
                    call_slot_ids=self.slots(group,task,replica),call_ordinals_within_task=[r*3+replica for r in range(5)],
                    row_mask_provider=self.row_provider(batch),selection_seed=self.baseline.teacher_runner.SELECTION_SEED)
                captured=collector.capture(lambda:self.forward(model,batch))
                for path in captures:captures[path].append(captured.inputs_by_module[path]);keys[path].append(captured.row_keys_by_module[path])
                del batch,collector,captured
            for spec,old in zip(task_specs,existing,strict=True):
                if old is not None:continue
                tensors=self.baseline.teacher_runner.assemble_step_tensors(spec,captures,keys)
                self.cache.write_completed_shard_atomic(cache,spec,tensors,plan_sha256=digest,call_state_manifest_sha256=self.bank_sha,
                    source_checkpoint_sha256=self.models[group]['model_sha256'])
            event(run,event='teacher_task_complete',group=group,task=task);del captures,keys;gc.collect();torch.cuda.empty_cache()
        write(run/f'teachers-{group}.json',{'status':'complete','plan_sha256':digest,'group':group})

    def capture_student(self,model,step,quotas,spec,teacher_keys,group,task):
        captures={p:[] for p in step.target_module_paths};keys={p:[] for p in step.target_module_paths}
        for replica in range(3):
            batch=self.inputs(group,task,replica);collector=self.SingleStepSelectedRowCollector(model,step,quotas,
                teacher_row_keys=teacher_keys,call_slot_ids=self.slots(group,task,replica),
                call_ordinals_within_task=[r*3+replica for r in range(5)],row_mask_provider=self.row_provider(batch))
            captured=collector.capture(lambda:self.forward(model,batch))
            for path in captures:captures[path].append(captured.inputs_by_module[path]);keys[path].append(captured.row_keys_by_module[path])
        tensors=self.baseline.teacher_runner.assemble_step_tensors(spec,captures,keys)
        for target in spec['targets']:
            if not torch.equal(tensors[target['row_keys_tensor']],teacher_keys[target['module_path']]):raise ValueError('Unpaired teacher/student rows')
        return tensors

    def export(self,run,model,paths,prefix,digest):
        output=run/'checkpoint/pretrained_model'
        output.mkdir(parents=True,exist_ok=False)
        for p in self.soup.iterdir():
            if p.name=='model.safetensors' or 'manifest' in p.name:continue
            if p.is_dir():shutil.copytree(p,output/p.name)
            else:shutil.copy2(p,output/p.name)
        modules=dict(model.named_modules())
        with safe_open(str(self.soup/'model.safetensors'),framework='pt',device='cpu') as f:
            tensors={k:(modules[k[:-7]].weight.detach().cpu().contiguous() if k.endswith('.weight') and k[:-7] in paths else f.get_tensor(k).contiguous()) for k in f.keys()}
        save_file(tensors,str(output/'model.safetensors'),metadata={'format':'pt'})
        result={'path':str(output),'model_sha256':sha(output/'model.safetensors'),'method':'RoboTwin M3 static-observation FeatCal',
            'plan_sha256':digest,'input_bank_sha256':self.bank_sha,'final_prefix_sha256':prefix,'linear_weights':len(paths),
            'tensor_count':len(tensors),'biases_and_other_tensors':'preserve initial Soup','formal_result':False}
        write(output/'featcal_m3_manifest.json',result);return result

    @torch.inference_mode()
    def solve(self,run):
        plan,digest,specs,quotas=self.contract(run);self.require_smoke(run,digest);cache=run/'cache'
        for group in GROUPS:
            r=read(run/f'teachers-{group}.json')
            if r['status']!='complete' or r['plan_sha256']!=digest:raise ValueError('Teachers incomplete')
        model=self.load_policy(self.soup);next_step,prefix,previous=self.baseline._load_completed_prefix(model,cache);by_path={q.module_path:q for q in quotas}
        with ExitStack() as stack:
            soup=stack.enter_context(safe_open(str(self.soup/'model.safetensors'),framework='pt',device='cpu'))
            base=stack.enter_context(safe_open(str(self.base/'model.safetensors'),framework='pt',device='cpu'))
            experts={g:stack.enter_context(safe_open(str(Path(self.models[g]['path'])/'model.safetensors'),framework='pt',device='cpu')) for g in GROUPS}
            for step in self.steps[next_step:]:
                for group in GROUPS:
                    for task in range(10):
                        spec=specs[('student',group,task,step.step)];old=self.baseline._validated_or_missing(cache,spec,digest)
                        if old is not None:
                            if old['student_prefix_input_sha256']!=prefix:raise ValueError('Student prefix mismatch on resume')
                            continue
                        teacher_spec=specs[('teacher',group,task,step.step)]
                        _,teacher_keys,_=self.baseline._load_shard_inputs(cache,teacher_spec,digest)
                        tensors=self.capture_student(model,step,[by_path[p] for p in step.target_module_paths],spec,teacher_keys,group,task)
                        self.cache.write_completed_shard_atomic(cache,spec,tensors,plan_sha256=digest,call_state_manifest_sha256=self.bank_sha,
                            source_checkpoint_sha256=prefix,student_prefix_input_sha256=prefix,
                            previous_step_solve_receipt=previous['path'] if previous else None,
                            previous_step_solve_receipt_sha256=previous['sha256'] if previous else None)
                prefix,previous,_=self.baseline._solve_formal_step(cache_root=cache,student=model,step=step,specs=specs,
                    plan_sha256=digest,current_prefix=prefix,previous_solve=previous,soup_handle=soup,base_handle=base,expert_handles=experts)
                self.check_runtime_memory();event(run,event='step_complete',step=step.step,prefix=prefix);gc.collect();torch.cuda.empty_cache()
        result=self.export(run,model,{p for s in self.steps for p in s.target_module_paths},prefix,digest)
        batch=self.inputs('coordination',0,0);expected=self.forward(model,batch).cpu();del model;gc.collect();torch.cuda.empty_cache()
        reloaded=self.load_policy(result['path']);actual=self.forward(reloaded,batch).cpu()
        if not torch.equal(expected,actual):raise ValueError('Checkpoint reload differs from solved model')
        write(run/'complete.json',{'status':'complete','checkpoint':result,'reload_bitwise_equal':True,'formal_episodes':0})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','smoke','teachers','solve'],required=True);p.add_argument('--group',choices=GROUPS)
    p.add_argument('--gpu',type=int);p.add_argument('--authorize-gpu-run',action='store_true');p.add_argument('--allow-shared',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);api=API(args.workspace_root,args.bank);run=args.run_root.resolve()
    if args.stage=='prepare':
        api.prepare(run)
        if torch.cuda.is_initialized():raise RuntimeError('CPU preparation initialized CUDA')
        return
    if not args.authorize_gpu_run or args.gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.gpu):
        raise ValueError('GPU stages require explicit authorization flag, GPU and matching CUDA_VISIBLE_DEVICES')
    info=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
    uuid=info[0].strip();free=int(info[1]);processes=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if free<49152 or (processes and not args.allow_shared):raise RuntimeError('GPU requires >=48GiB free and explicit shared-card permission when occupied')
    with ExitStack() as stack:
        paths=[api.work/'vla-merge-runtime/resource-leases'/socket.gethostname()/f'gpu-{args.gpu}.lock',
               api.work/'vla-merge-runtime/experiments/claude-card-flocks'/socket.gethostname()/f'gpu-{args.gpu}-{uuid}.lock',run/f'{args.stage}-{args.group or "all"}.lock']
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True);f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        again=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        if again[0].strip()!=uuid or int(again[1])<49152:raise RuntimeError('GPU identity/free memory changed before launch')
        api.active_gpu=args.gpu
        native_kernel=api.baseline.featcal_hybrid_linear_weight
        def checked_kernel(*a,**kw):
            api.check_runtime_memory();result=native_kernel(*a,**kw);api.check_runtime_memory();return result
        api.baseline.featcal_hybrid_linear_weight=checked_kernel
        torch.cuda.set_per_process_memory_fraction(.5,0)
        if args.stage=='smoke':api.smoke(run)
        elif args.stage=='teachers':
            if args.group is None:raise ValueError('A teacher group is required')
            api.teachers(run,args.group)
        else:api.solve(run)

if __name__=='__main__':main()
