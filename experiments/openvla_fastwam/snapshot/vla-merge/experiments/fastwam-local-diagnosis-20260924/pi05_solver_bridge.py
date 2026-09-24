"""Call the exact source-verified pi0.5 main-result solver for Fast-WAM.

Only extract its two pure numerical functions to avoid importing pi0.5 model
dependencies into the Fast-WAM environment. Their AST is not rewritten.
"""
import ast
import hashlib
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'vla-merge/scripts/materialize_pi05_tcr_e_peft_safe.py'
SOURCE_SHA = 'f2b2b1c1373343f00b4238bd349f06185ee08a59153503bf31dc54924eb1f740'


def load_reference():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError('pi0.5 main-result source hash differs from recorded build origin')
    tree = ast.parse(raw, filename=str(SOURCE))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('task_loss', 'solve_weight_multi')]
    if {n.name for n in functions} != {'task_loss', 'solve_weight_multi'}:
        raise ValueError('Missing main-result numerical functions')
    scope = {'torch': torch, 'math': math, 'mean': mean, 'median': median, 'Any': Any}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['solve_weight_multi']


REFERENCE = load_reference()


def solve_module(module, xs, experts, prior, *, mass_rule, ridge_multiplier,
                 max_correction_ratio, fixed_ridge=None,
                 ridge_source='computed_pass_A', device='cpu'):
    if len(xs) != 4 or len(experts) != 4 or mass_rule not in ('relative', 'uniform'):
        raise ValueError('Four experts and explicit mass rule required')
    if ridge_multiplier != .05 or max_correction_ratio != 3.:
        raise ValueError('Main-result numerical parameters changed')
    if fixed_ridge is None:
        if ridge_source != 'computed_pass_A':
            raise ValueError('Missing pass-A ridge')
    elif not math.isfinite(fixed_ridge) or fixed_ridge <= 0 or ridge_source != 'fixed_from_same_candidate_pass_A':
        raise ValueError('Invalid fixed pass-A ridge')
    current = module.weight.detach().cpu()
    if module.bias is not None:
        current = torch.cat((current, module.bias.detach().cpu()[:, None]), dim=1)
    if not torch.equal(current, prior.cpu().to(current.dtype)):
        raise ValueError('Unsolved module differs from its immutable pass prior')
    if any(len(x) == 0 or x.shape[1] != prior.shape[1] for x in xs):
        raise ValueError('Invalid feature coverage')
    if any(not torch.isfinite(t).all() for t in [prior, *xs, *experts]):
        raise ValueError('Nonfinite solver input')
    names = ('spatial', 'object', 'goal', 'long')
    inputs = {k: x.to(device) for k, x in zip(names, xs)}
    weights = {k: w.to(device) for k, w in zip(names, experts)}
    p = prior.to(device=device, dtype=torch.float32)
    merged, receipt = REFERENCE(inputs, weights, p,
        ridge_ratio=.05, ridge_scale='feature_energy', max_correction_ratio=3.,
        expert_loss_normalization='prior' if mass_rule == 'relative' else 'none',
        expert_loss_normalization_power=1., require_objective_improvement=False,
        expert_aggregation='mean', fixed_ridge=fixed_ridge)
    if not torch.isfinite(merged).all():
        raise ValueError('Nonfinite merged weight')
    delta_norm = sum(float((w.float() - p).norm()) for w in weights.values()) / 4
    candidate = merged.to(device=module.weight.device, dtype=module.weight.dtype)
    if not torch.isfinite(candidate).all():
        raise ValueError('Nonfinite cast weight')
    with torch.no_grad():
        module.weight.copy_(candidate[:, :module.in_features])
        if module.bias is not None:
            module.bias.copy_(candidate[:, -1])
    return {**receipt, 'ridge_source': ridge_source, 'mean_expert_delta_norm': delta_norm,
            'trust_limit': 3 * max(delta_norm, 1e-12), 'bias_augmented': module.bias is not None,
            'solver_source': str(SOURCE), 'solver_source_sha256': SOURCE_SHA,
            'solver_function': 'solve_weight_multi', 'solver_dtype': 'float32',
            'rule': mass_rule, 'max_correction_ratio': 3.}
