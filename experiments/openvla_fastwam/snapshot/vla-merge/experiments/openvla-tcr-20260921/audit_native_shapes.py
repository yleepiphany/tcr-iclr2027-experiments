"""Meta-device native model/head shape audit; never allocates GPU/model weights."""
import json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from native_oft import configure_runtime, load_processor


def main():
    configure_runtime()
    from experiments.robot import openvla_utils as u
    from transformers import AutoProcessor
    from PIL import Image
    import numpy as np
    root = Path(__file__).resolve().parents[3] / 'vla-merge_table4/OpenVLA-OFT'
    result = []
    for expert in ('spatial','object','goal','long'):
        models = list((root / 'weights' / expert).glob('*/model.safetensors.index.json'))
        if len(models) != 1: raise ValueError('Ambiguous expert')
        model = models[0].parent
        config = u.OpenVLAConfig.from_pretrained(model, local_files_only=True)
        with init_empty_weights():
            native = u.OpenVLAForActionPrediction(config)
            head = u.L1RegressionActionHead(input_dim=native.llm_dim, hidden_dim=native.llm_dim, action_dim=7)
            proprio = u.ProprioProjector(llm_dim=native.llm_dim, proprio_dim=8)
        expected = {k:tuple(v.shape) for k,v in native.state_dict().items()}
        index = json.loads(models[0].read_text())['weight_map']
        observed = {}
        for shard in set(index.values()):
            with safe_open(str(model/shard),framework='pt',device='cpu') as handle:
                observed.update({k:tuple(handle.get_slice(k).get_shape()) for k in handle.keys()})
        if observed != expected:
            raise ValueError({'expert':expert,'missing': sorted(expected.keys()-observed.keys()),
                              'unexpected':sorted(observed.keys()-expected.keys()),
                              'wrong_shapes':[k for k in expected.keys() & observed.keys() if expected[k]!=observed[k]]})
        counts = {}
        for name, module in (('action_head',head),('proprio_projector',proprio)):
            path, = model.glob(f'{name}--*_checkpoint.pt')
            states = torch.load(str(path),map_location='cpu',weights_only=True,mmap=True)
            shapes = {k.removeprefix('module.'):tuple(v.shape) for k,v in states.items()}
            if shapes != {k:tuple(v.shape) for k,v in module.state_dict().items()}:
                raise ValueError(f'Native {name} shape mismatch for {expert}')
            counts[name] = len(shapes)
            del states
        direct = load_processor(model)
        # Official processor loader is read-only; unlike get_vla, no checkpoint
        # config/source synchronization is performed. HF cache writes are expected.
        original = AutoProcessor.from_pretrained(model,trust_remote_code=True,local_files_only=True)
        pixels = (np.arange(224*224*3,dtype=np.uint32)%251).astype(np.uint8).reshape(224,224,3)
        prompt = 'In: What action should the robot take to move the object?\nOut:'
        a,b = direct(prompt,Image.fromarray(pixels)), original(prompt,Image.fromarray(pixels))
        if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
            raise ValueError(f'Native processor parity failed for {expert}')
        result.append({'expert':expert,'backbone_keys':len(expected),**counts,
                       'processor_tensor_parity':True,'allocated_model_weights':False})
        print(json.dumps(result[-1]),flush=True)
        del native,head,proprio
    out=Path(__file__).resolve().parents[3]/'coordination/2026-09-21/openvla-native-cpu-shape-parity.json'
    with out.open('x') as f: json.dump({'accepted':True,'experts':result},f,indent=2)


if __name__=='__main__': main()
