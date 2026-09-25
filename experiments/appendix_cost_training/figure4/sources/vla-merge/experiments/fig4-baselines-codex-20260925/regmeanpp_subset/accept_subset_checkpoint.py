"""Hash/finite/export and one fixed native action request; never scores success."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from fig4_regmeanpp_contract import digest,require,validate_rows_for_names

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--queue-plan',type=Path,required=True);parser.add_argument('--queue-plan-sha256',required=True);parser.add_argument('--build-id',required=True);parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args()
    require(digest(args.queue_plan)==args.queue_plan_sha256,'Queue plan changed')
    queue=json.loads(args.queue_plan.read_text());job=next(j for j in queue['builds'] if j['id']==args.build_id)
    require(not args.receipt.exists(),'Native acceptance output already exists')
    checkpoint=Path(job['checkpoint']);plan=json.loads(Path(job['build_plan']['path']).read_text());manifest=json.loads((checkpoint/'block_regmeanpp_manifest.json').read_text())
    require(manifest['method']=='pi05_fig4_regmeanpp_subset_alpha03_static_t1_spectral_smoothing_v1','Wrong exported method')
    require(manifest['selected_expert_names']==plan['experts'] and manifest['expert_count']==len(plan['experts']),'Exported subset differs')
    require(manifest['figure4_parent_plan_sha256']==queue['fig4_plan_sha256'] and manifest['figure4_subset']==plan['subset'],'Exported parent/subset differs')
    require(manifest['experiment_manifest']['sha256']==job['build_plan']['sha256'],'Export does not bind build plan')
    require(manifest['replay_prefix']=='merged' and manifest['modified_tensor_count']==422 and manifest['module_count']==418,'Scope/prefix differs')
    require(manifest['calibration_observations_total']==50*len(plan['experts']),'Observation count differs')
    require(validate_rows_for_names(manifest['modules'],plan['experts'])==manifest['realized_row_total']==plan['expected_realized_rows'],'Row budget differs')
    require(manifest['manual_native_block_max_error']<=.02,'Native block replay parity failed')
    model_sha=digest(checkpoint/'model.safetensors');require(model_sha==manifest['model_sha256'],'Exported model bytes differ')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as handle:
        require(len(handle.keys())==813,'Unexpected serialized tensor count')
        for key in handle.keys():require(bool(torch.isfinite(handle.get_tensor(key)).all()),'Nonfinite export: '+key)
    request=queue['native_request']
    for label in ('observations','manifest'):require(digest(request[label]['path'])==request[label]['sha256'],'Native smoke input changed')
    row=json.loads(Path(request['manifest']['path']).read_text())['samples'][0]
    require(row==request['sample'],'Native smoke sample identity changed')
    noise=torch.randn((1,50,32),generator=torch.Generator(device='cpu').manual_seed(row['noise_seed']),dtype=torch.float32)
    header=json.dumps({'dtype':str(noise.dtype),'shape':list(noise.shape)},sort_keys=True).encode()
    require(hashlib.sha256(header+b'\0'+noise.view(torch.uint8).numpy().tobytes()).hexdigest()==row['x_t_sha256'],'Native smoke initial noise changed')
    with safe_open(request['observations']['path'],framework='pt',device='cpu') as handle:
        prefix=f"observation_{row['observation_index']:03d}."
        values={key[len(prefix):]:handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}
    import eval_with_local_tokenizer  # existing native compatibility layer
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    require(torch.cuda.device_count()==1,'Need exactly one reserved visible GPU')
    torch.cuda.set_per_process_memory_fraction(.35,0)
    policy=PI05Policy.from_pretrained(checkpoint,device='cuda');policy.eval()
    require(not policy.config.use_peft,'Export must reload as dense checkpoint')
    batch={k:v.to('cuda') for k,v in values.items()}
    with torch.inference_mode():
        action=policy.model.sample_actions(images=[batch[f'image_{i}'] for i in range(3)],img_masks=[batch[f'image_mask_{i}'] for i in range(3)],tokens=batch['tokens'],masks=batch['masks'],noise=noise.to('cuda'),num_steps=10)
    require(tuple(action.shape)==(1,50,32) and bool(torch.isfinite(action).all()),'Native action shape/finite check failed')
    action=action.detach().cpu().float().contiguous()
    result={'status':'accepted_export_and_one_native_request','build_id':job['id'],'model_sha256':model_sha,'manifest_sha256':digest(checkpoint/'block_regmeanpp_manifest.json'),'queue_plan_sha256':args.queue_plan_sha256,'build_plan_sha256':job['build_plan']['sha256'],'selected_experts':plan['experts'],'serialized_tensors':813,'all_finite':True,'dense_reload':True,'native_requests':1,'generation_steps':10,'action_shape':[1,50,32],'action_sha256':hashlib.sha256(action.numpy().tobytes()).hexdigest(),'native_input':'Frozen Figure4 Spatial observation_000, no action label; acceptance only, not calibration or success measurement','success_measured':False,'weights_modified':False}
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
