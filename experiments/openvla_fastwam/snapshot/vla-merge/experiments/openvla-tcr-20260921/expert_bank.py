"""Read-only expert tensors for native block-atomic OFT regression."""
from contextlib import ExitStack
import json
from pathlib import Path
import torch
from safetensors import safe_open
from materialize_soup import ORDER,verify_sources

def resolve_block(policy,name):
    component,sep,suffix=name.partition('.')
    roots={'backbone':policy.model,'action_head':policy.head,'proprio_projector':policy.proprio}
    if component not in roots:raise ValueError(f'Unknown native component: {component}')
    return roots[component].get_submodule(suffix) if sep else roots[component]

class ExpertBank:
    """Memory-map all four experts on CPU; copy only the requested block.

    Expert floating tensors are converted to each native loaded module's dtype,
    matching NativePolicy's inference representation. Original files stay intact.
    This is explicit: the calibration matrices are native effective parameters,
    not a claim that an FP32 component and its BF16 runtime value are identical.
    """
    def __init__(self,ledger_path):
        self.ledger_path=Path(ledger_path)
        self.ledger=json.loads(self.ledger_path.read_text())
        if not self.ledger.get('complete') or set(self.ledger['experts'])!=set(ORDER):
            raise ValueError('Incomplete expert identity ledger')
        verify_sources(self.ledger)
        self.stack=ExitStack();self.backbones={};self.components={}
        try:
            for expert in ORDER:
                root=Path(self.ledger['experts'][expert]['local_path'])
                index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
                handles={}
                for shard in set(index.values()):
                    path=(root/shard).resolve()
                    if path.parent!=root.resolve():raise ValueError('Invalid expert shard path')
                    handles[shard]=self.stack.enter_context(safe_open(str(path),framework='pt',device='cpu'))
                self.backbones[expert]=(index,handles)
                self.components[expert]={}
                for component in ('action_head','proprio_projector'):
                    paths=list(root.glob(component+'--*_checkpoint.pt'))
                    if len(paths)!=1:raise ValueError('Ambiguous expert component')
                    state=torch.load(str(paths[0]),map_location='cpu',weights_only=True,mmap=True)
                    normalized={}
                    for key,value in state.items():
                        key=key.removeprefix('module.')
                        if key in normalized:raise ValueError('Duplicate normalized component key')
                        normalized[key]=value
                    self.components[expert][component]=normalized
        except BaseException:
            self.close();raise
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.stack.close()
    def tensor(self,expert,name):
        component,sep,key=name.partition('.')
        if expert not in ORDER or not sep:raise ValueError('Invalid expert tensor identity')
        if component=='backbone':
            index,handles=self.backbones[expert]
            if key not in index:raise KeyError(name)
            return handles[index[key]].get_tensor(key)
        if component not in self.components[expert]:raise ValueError('Unknown component')
        return self.components[expert][component][key]
    def block_state(self,expert,block_name,block):
        result={};conversions=[]
        for key,current in block.state_dict().items():
            full_name=block_name+'.'+key
            source=self.tensor(expert,full_name)
            if source.shape!=current.shape:raise ValueError(f'Expert/native block shape mismatch: {full_name}')
            if source.is_floating_point()!=current.is_floating_point():raise ValueError('Incompatible tensor kinds')
            if source.dtype!=current.dtype:
                if not source.is_floating_point():raise ValueError('Nonfloating native dtype mismatch')
                conversions.append({'tensor':full_name,'source_dtype':str(source.dtype),'native_dtype':str(current.dtype)})
            value=source.to(device='cpu',dtype=current.dtype).clone()
            if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('Nonfinite native expert state')
            result[key]=value
        return result,conversions
    def affine(self,expert,module_name,module):
        state,conversions=self.block_state(expert,module_name,module)
        expected={'weight','bias'} if module.bias is not None else {'weight'}
        if set(state)!=expected:raise ValueError('Not a plain native Linear state')
        weight=state['weight']
        if module.bias is not None:weight=torch.cat((weight,state['bias'][:,None]),dim=1)
        return weight,conversions
