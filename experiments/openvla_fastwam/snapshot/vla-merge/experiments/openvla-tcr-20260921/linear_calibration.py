"""Native-replay Linear calibration primitives; no collector or full-model claim.

Callers freeze the block order, request IDs, row caps, numerical rules, and
export scope. The paper uses a merged prefix and EXPERT computations within the
current block. Use expert_block_state while capturing a block's features; restore
the block before solving from its fixed pass prior. capture_rows alone does not
perform this block swap and must not be treated as the complete TCR procedure.
"""
from contextlib import contextmanager
import hashlib
import math
import torch
from ridge import solve

@contextmanager
def expert_block_state(block, expert_state):
    """Temporarily replace all current-block tensors, including its norms.

Only the current block may be passed here, never the entire policy. Its input
producers remain merged. Restore even if capture/replay raises an exception.
"""
    current=block.state_dict()
    if set(current)!=set(expert_state):raise ValueError('Expert block tensor keys differ')
    for key,value in current.items():
        other=expert_state[key]
        if value.shape!=other.shape or value.dtype!=other.dtype:
            raise ValueError(f'Expert block tensor structure differs: {key}')
        if value.is_floating_point() and (not torch.isfinite(value).all() or not torch.isfinite(other).all()):
            raise ValueError('Nonfinite expert/current block state')
    saved={key:value.detach().cpu().clone() for key,value in current.items()}
    try:
        block.load_state_dict(expert_state,strict=True)
        yield
    finally:
        block.load_state_dict(saved,strict=True)

def capture_expert_block(policy,block,expert_state,module_names,request,*,request_id,cap,seed,expected_calls):
    """One native replay captures all Linears inside a temporarily expert block."""
    with expert_block_state(block,expert_state):
        return capture_block_rows(policy,block,module_names,request,request_id=request_id,
                                  cap=cap,seed=seed,expected_calls=expected_calls)

def capture_block_rows(policy,block,module_names,request,*,request_id,cap,seed,expected_calls):
    """Capture at current block weights; caller owns the expert-block context.

    A collector may hold one expert_block_state context across all that expert's
    requests, avoiding repeated whole-block CPU/GPU copies for every request.
    This function alone is not a complete TCR calibration step.
    """
    modules=dict(policy.linear_modules())
    if not module_names or len(set(module_names))!=len(module_names):raise ValueError('Empty or duplicate block scope')
    members={id(m) for m in block.modules()}
    if any(name not in modules or id(modules[name]) not in members for name in module_names):
        raise ValueError('Requested Linear is outside the current block')
    mask=request.get('attention_mask')
    if mask is None or mask.ndim!=2 or mask.shape[0]!=1 or not bool(mask.all()):
        raise ValueError('Expected one unpadded native request')
    rows={name:[] for name in module_names}; receipts={name:[] for name in module_names}; handles=[]
    for name in module_names:
        calls=expected_calls.get(name)
        if type(calls) is not int or calls<1 or type(cap) is not int or cap<calls:
            raise ValueError('Insufficient cap or missing call contract')
    try:
        for name in module_names:
            def hook(mod,args,name=name):
                ordinal=len(rows[name]);calls=expected_calls[name]
                if ordinal>=calls or len(args)!=1 or args[0].shape[-1]!=mod.in_features:
                    raise ValueError('Block Linear call/input contract differs')
                x=args[0].detach().reshape(-1,mod.in_features)
                quota=cap//calls+(ordinal<cap%calls)
                ids=row_indices(x.shape[0],quota,seed=seed,request_id=request_id,module_name=name,call_index=ordinal)
                chosen=x.index_select(0,ids.to(x.device)).cpu().clone()
                if not torch.isfinite(chosen).all():raise ValueError('Nonfinite expert-in-block features')
                rows[name].append(chosen)
                receipts[name].append({'call_index':ordinal,'available_rows':len(x),'selected_rows':ids.tolist()})
            handles.append(modules[name].register_forward_pre_hook(hook))
        policy.replay(request)
    finally:
        for h in handles:h.remove()
    features={}
    for name in module_names:
        if len(rows[name])!=expected_calls[name]:raise ValueError('Missing native block calls')
        x=torch.cat(rows[name])
        if modules[name].bias is not None:x=torch.cat((x,torch.ones((len(x),1),dtype=x.dtype)),dim=1)
        features[name]=x
    return features,receipts

def row_indices(count, cap, *, seed, request_id, module_name, call_index):
    if type(count) is not int or type(cap) is not int or min(count,cap)<=0:
        raise ValueError('Positive row count and cap required')
    identity=f'{seed}\0{request_id}\0{module_name}\0{call_index}'.encode()
    value=int.from_bytes(hashlib.sha256(identity).digest()[:8],'little') % (2**63-1)
    generator=torch.Generator(device='cpu').manual_seed(value)
    return torch.randperm(count,generator=generator)[:min(count,cap)].sort().values

def capture_rows(policy, module_name, request, *, request_id, cap, seed, expected_calls):
    modules=dict(policy.linear_modules())
    if module_name not in modules or type(expected_calls) is not int or expected_calls<1:
        raise ValueError('Unknown module or invalid call count')
    if type(cap) is not int or cap<expected_calls:
        raise ValueError('Row cap cannot cover each native call once')
    # The native OFT single-request path has no prompt padding. Do not silently
    # sample padded tokens if another batching/tokenizer path is introduced.
    mask=request.get('attention_mask')
    if mask is None or mask.ndim!=2 or mask.shape[0]!=1 or not bool(mask.all()):
        raise ValueError('This row selector requires an unpadded single native request')
    module=modules[module_name]; rows=[]; receipts=[]
    def hook(mod,args):
        ordinal=len(rows)
        if ordinal>=expected_calls:raise ValueError('Unexpected extra native module call')
        if len(args)!=1 or args[0].shape[-1]!=mod.in_features:raise ValueError('Linear input contract differs')
        x=args[0].detach().reshape(-1,mod.in_features)
        quota=cap//expected_calls+(ordinal<cap%expected_calls)
        ids=row_indices(x.shape[0],quota,seed=seed,request_id=request_id,module_name=module_name,call_index=ordinal)
        selected=x.index_select(0,ids.to(x.device)).cpu().clone()
        if not torch.isfinite(selected).all():raise ValueError('Nonfinite native features')
        rows.append(selected)
        receipts.append({'call_index':ordinal,'available_rows':x.shape[0],'selected_rows':ids.tolist()})
    handle=module.register_forward_pre_hook(hook)
    try:policy.replay(request)
    finally:handle.remove()
    if len(rows)!=expected_calls:raise ValueError('Native module call coverage differs')
    x=torch.cat(rows)
    if module.bias is not None:x=torch.cat((x,torch.ones((len(x),1),dtype=x.dtype)),dim=1)
    return x,{'module':module_name,'request_id':request_id,'rows':len(x),
              'bias_augmented':module.bias is not None,'calls':receipts}

def affine_parameters(module):
    weight=module.weight.detach().cpu().clone()
    if module.bias is not None:weight=torch.cat((weight,module.bias.detach().cpu()[:,None]),dim=1)
    if not torch.isfinite(weight).all():raise ValueError('Nonfinite module parameters')
    return weight

def expert_masses(xs,experts,prior,rule):
    if rule not in ('uniform','relative') or len(xs)!=4 or len(experts)!=4:
        raise ValueError('Expected four experts and a frozen weighting rule')
    losses=[]
    for x,w in zip(xs,experts):
        if x.ndim!=2 or len(x)==0 or w.shape!=prior.shape or x.shape[1]!=prior.shape[1]:
            raise ValueError('Feature/parameter shapes differ')
        if not torch.isfinite(x).all() or not torch.isfinite(w).all() or not torch.isfinite(prior).all():
            raise ValueError('Nonfinite calibration data')
        residual=x.double()@(w.to(x.device).double()-prior.to(x.device).double()).T
        loss=float(residual.square().mean())
        if not math.isfinite(loss):raise ValueError('Nonfinite relative-prior loss')
        losses.append(loss)
    floor=max(sum(losses)/4*1e-6,1e-12)
    inverse=[1/max(v,floor) for v in losses]
    masses=[.25]*4 if rule=='uniform' else [v/sum(inverse) for v in inverse]
    return masses,{'rule':rule,'relative_losses':losses,'floor':floor,'masses':masses}

def solve_module(module,xs,experts,prior,*,mass_rule,ridge_multiplier,max_correction_ratio,device='cpu'):
    if not math.isfinite(ridge_multiplier) or ridge_multiplier<=0:
        raise ValueError('Positive ridge multiplier required')
    if not math.isfinite(max_correction_ratio) or max_correction_ratio<=0:
        raise ValueError('Positive fixed trust ratio required')
    current=affine_parameters(module)
    if current.shape!=prior.shape or not torch.equal(current,prior.cpu().to(current.dtype)):
        raise ValueError('Unsolved module is not at its frozen pass prior')
    masses,details=expert_masses(xs,experts,prior,mass_rule)
    trace=sum(a*float(x.double().square().sum()/len(x)) for a,x in zip(masses,xs))
    ridge=max(ridge_multiplier*trace/prior.shape[1],1e-12)
    if not math.isfinite(ridge):raise ValueError('Nonfinite numerical ridge')
    deltas=[w.to(device).double()-prior.to(device).double() for w in experts]
    correction,solved=solve([x.to(device) for x in xs],deltas,masses,ridge)
    norm=float(prior.double().norm());delta_norm=float(correction.norm())
    limit=max_correction_ratio*max(norm,1e-12)
    scale=min(1.,limit/delta_norm) if delta_norm else 1.
    candidate=prior.to(device).double()+correction*scale
    candidate=candidate.to(module.weight.dtype).to(module.weight.device)
    if not torch.isfinite(candidate).all():raise ValueError('Nonfinite exported parameters')
    # Validate everything before mutating the one module; no other module changes.
    with torch.no_grad():
        module.weight.copy_(candidate[:,:module.in_features])
        if module.bias is not None:module.bias.copy_(candidate[:,-1])
    return {**details,**solved,'weighted_trace':trace,'ridge_multiplier':ridge_multiplier,
            'prior_norm':norm,'unclipped_correction_norm':delta_norm,'trust_scale':scale,
            'max_correction_ratio':max_correction_ratio,'bias_augmented':module.bias is not None}
