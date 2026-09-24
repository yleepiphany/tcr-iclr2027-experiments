"""Five demonstration observation slots/task, with paired execution initial noise."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--expert', required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from collect_pi05_demo_observation_replay import DATASET, select_demo_frames
    records=select_demo_frames(args.expert, episode_rank=0, request_stride=10)
    import torch
    torch.cuda.set_per_process_memory_fraction(.40,0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from safetensors import safe_open
    from collect_pi05_block_regmeanpp_calibration import BlockCalibrationCollector
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    template=json.loads((args.template/'replay.json').read_text())
    indices={}
    for sample in template['samples']:
        if sample['flow_index']==0:
            text=template['prompt_texts'][str(sample['prompt_signature'])]
            indices[(text,sample['selected_request_slot'])]=sample['index']
    settings=dict(TASK=args.expert,TENSOR_OUTPUT=str(args.output/'replay.safetensors'),
                  MANIFEST_OUTPUT=str(args.output/'replay.json'),CALIBRATION_POLICY=str(args.policy),
                  MAX_CALLS='150',MAX_CALLS_PER_PROMPT='1',REQUESTS_PER_EPISODE='5',
                  EPISODE_AWARE='1',FULL_PREFIX='1',FLOW_INDICES='0,5,9',REQUEST_MODE='initial',START_SEED='271001')
    os.environ.update({f'PI05_BLOCK_REGMEANPP_{k}':v for k,v in settings.items()})
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    dataset=LeRobotDataset('lerobot/libero',root=DATASET,episodes=[r['episode_index'] for r in records],video_backend='pyav')
    cfg=PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path=args.policy
    cfg.device='cuda'; cfg.compile_model=False; cfg.n_action_steps=10
    policy=make_policy(cfg,ds_meta=dataset.meta).eval()
    pre,_=make_pre_post_processors(policy_cfg=policy.config,pretrained_path=args.policy,
                                  preprocessor_overrides={'device_processor':{'device':'cuda'}})
    collector=BlockCalibrationCollector(policy); collector.install()
    frame_rows=dataset.hf_dataset.select_columns(['episode_index','frame_index']).to_pandas()
    provenance={}
    with safe_open(args.template/'replay.safetensors',framework='pt',device='cpu') as handle:
        for record in records:
            rows=frame_rows[frame_rows.episode_index==record['episode_index']].sort_values('frame_index')
            candidates=list(rows.index)[::10]
            selected=[candidates[(len(candidates)-1)*k//4] for k in range(5)]
            if len(set(selected))!=5:
                raise ValueError('Demo too short; no replacement based on success')
            policy.reset()
            for slot,index in enumerate(selected):
                sample=dataset[int(index)]
                if sample['task']!=record['task']:
                    raise ValueError('Demo task mismatch')
                noise=handle.get_tensor(f"sample_{indices[(sample['task'],slot)]:03d}.action_input")
                def paired_noise(shape,device,value=noise):
                    if tuple(shape)!=tuple(value.shape):
                        raise ValueError('Paired native noise shape mismatch')
                    return value.to(device).clone()
                policy.model.sample_noise=paired_noise
                obs={k:v for k,v in sample.items() if k.startswith('observation.')}
                obs['task']=sample['task']
                collector.capture_task_texts([sample['task']])
                with torch.inference_mode():
                    policy.predict_action_chunk(pre(obs))
                provenance[(collector.latest_prompt_signature,slot)]=dict(demo_episode_index=record['episode_index'],demo_frame_index=int(frame_rows.loc[index,'frame_index']))
                print(f'Captured {args.expert} demo task={record["task_index"]} slot={slot}',flush=True)
    collector.save()
    manifest_path=args.output/'replay.json'
    manifest=json.loads(manifest_path.read_text())
    for sample in manifest['samples']:
        sample['selected_request_slot']=sample['request_index']
        sample.update(provenance[(sample['prompt_signature'],sample['request_index'])])
        sample.update(init_state_id=None,simulator_seed=None,initial_observation_sha256=None)
    manifest.update(table3_capture_version=1,source_kind='demonstration_observations',
                    paired_noise_template=str(args.template),demonstration_actions_used=False,
                    table3_request_selection='across',single_repeat=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
