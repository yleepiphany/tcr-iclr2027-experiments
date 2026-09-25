"""Mean-centered weak-spectrum smoothing of the original RegMean equation.

This is an explicitly regularized extension, not an algebraically identical
implementation of RegMean++. No reward, teacher output, trust cap, clipping,
or floor on nonzero feature energies is used. All arithmetic is CPU by default.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch


def _relative(numerator, denominator):
    numerator=float(numerator);denominator=float(denominator)
    if denominator==0:return 0. if numerator==0 else math.inf
    return numerator/denominator


def _power_scale(diagonal,z,iterations=64):
    """Deterministic estimate of lambda_max(D+Z.T Z); no dense D-by-D allocation."""
    if not bool(torch.any(diagonal>0)) and not bool(torch.any(z!=0)):
        return 0.,{'method':'exact_zero_operator','iterations':0,'not_exact':False}
    generator=torch.Generator(device='cpu').manual_seed(20260922)
    v=torch.randn(diagonal.numel(),generator=generator,dtype=torch.float64).to(diagonal.device)
    v/=torch.linalg.vector_norm(v);history=[]
    for _ in range(iterations):
        av=diagonal*v+z.T@(z@v);length=torch.linalg.vector_norm(av)
        if float(length)==0:raise RuntimeError('Nonzero PSD operator annihilated the fixed power start')
        v=av/length
        av=diagonal*v+z.T@(z@v);history.append(float(torch.dot(v,av)))
        if len(history)>=4 and all(abs(history[-j]-history[-j-1])<=1e-12*history[-1] for j in (1,2,3)):break
    if not math.isfinite(history[-1]) or history[-1]<=0:raise RuntimeError('Invalid spectral scale estimate')
    return history[-1],{'method':'deterministic_PSD_power_iteration','iterations':len(history),'requested_iterations':iterations,
        'not_exact':True,'not_certified_upper_bound':True,'seed':20260922,'last_estimates':history[-5:]}


def prepare_original_equation(inputs:Mapping[str,torch.Tensor],weights:Mapping[str,torch.Tensor],*,alpha=.3,device='cpu'):
    if not 0<alpha<1:raise ValueError('Alpha must lie strictly between zero and one')
    names=list(inputs)
    if not names or set(names)!=set(weights):raise ValueError('Input/expert identities differ')
    shape=tuple(weights[names[0]].shape)
    if len(shape)!=2:raise ValueError('Weights must be matrices')
    xs={n:inputs[n].to(device=device,dtype=torch.float64) for n in names}
    ws={n:weights[n].to(device=device,dtype=torch.float64) for n in names}
    counts={n:int(xs[n].shape[0]) for n in names}
    if len(set(counts.values()))!=1 or not all(n>0 for n in counts.values()):raise ValueError('Equal positive row counts required for official common-scale Gram')
    for n in names:
        if xs[n].ndim!=2 or xs[n].shape[1]!=shape[1] or tuple(ws[n].shape)!=shape:raise ValueError('Input/weight shape differs')
        if not bool(torch.isfinite(xs[n]).all()) or not bool(torch.isfinite(ws[n]).all()):raise ValueError('Nonfinite source')
    mean=sum(ws.values())/len(ws)
    energy={n:xs[n].square().sum(0) for n in names}
    diagonal=(1-alpha)*sum(energy.values())
    z=math.sqrt(alpha)*torch.cat(list(xs.values()),dim=0)
    rhs=torch.zeros(shape[1],shape[0],device=device,dtype=torch.float64)
    for n in names:
        delta=(ws[n]-mean).T
        rhs.add_(alpha*(xs[n].T@(xs[n]@delta)))
        rhs.add_((1-alpha)*energy[n].unsqueeze(1)*delta)
    return diagonal,z,rhs,mean,counts


def solve_smoothed_weight(inputs,weights,metric_reference=None,*,offdiag_scale=.3,
                          strategy='auto',filter_kind='smooth',power_iterations=64,
                          numerical_rank_dtype=torch.float32,output_chunk_size=128,
                          device=None):
    """Solve delta = Re[(A+i*tau I)^-1 R], then return Wmean + delta.

    tau = d * eps(float32) * estimated_lambda_max(A). Float32 is the effective
    output precision of the existing original kernel. The dimension multiplier
    is the standard square-matrix numerical-rank scale, frozen independently of
    success. `hard_rank` is a small-matrix diagnostic comparator, not the default.
    Dense and Woodbury strategies compute the same smooth filter, in complex128.
    """
    if numerical_rank_dtype!=torch.float32:raise ValueError('v1 numerical-rank precision is frozen to float32')
    if strategy not in ('auto','dense','woodbury','eigh_reference'):raise ValueError('Unknown strategy')
    if filter_kind not in ('smooth','hard_rank'):raise ValueError('Unknown filter')
    if not 1<=power_iterations<=256 or not 1<=output_chunk_size<=1024:raise ValueError('Invalid bounded numerical budget')
    device=device or (metric_reference.device if metric_reference is not None else 'cpu')
    d,z,r,mean,counts=prepare_original_equation(inputs,weights,alpha=offdiag_scale,device=device)
    width=d.numel();rows=z.shape[0];scale,scale_info=_power_scale(d,z,power_iterations)
    rank_epsilon=torch.finfo(numerical_rank_dtype).eps
    tau=width*rank_epsilon*scale
    chosen=('dense' if width<=rows else 'woodbury') if strategy=='auto' else strategy
    if filter_kind=='hard_rank':chosen='eigh_reference'
    delta=torch.zeros_like(r);eigen_info=None
    if scale>0 and bool(torch.any(r!=0)):
        if tau<=0 or not math.isfinite(tau):raise RuntimeError('Invalid derived spectral threshold')
        if chosen=='eigh_reference':
            if width>4608:raise ValueError('Dense eigensolver is only a bounded reference')
            a=torch.diag(d)+z.T@z;a=(a+a.T)*.5
            eigenvalues,u=torch.linalg.eigh(a)
            if float(eigenvalues[0]) < -64*torch.finfo(torch.float64).eps*width*scale:raise RuntimeError('Gram is not PSD within FP64 roundoff')
            if filter_kind=='smooth':inv=eigenvalues/(eigenvalues.square()+tau*tau)
            else:
                inv=torch.zeros_like(eigenvalues);active=eigenvalues>tau;inv[active]=1/eigenvalues[active]
            delta=u@(inv.unsqueeze(1)*(u.T@r))
            eigen_info={'minimum_eigenvalue':float(eigenvalues[0]),'maximum_eigenvalue':float(eigenvalues[-1]),
                        'eigenvalues_at_or_below_tau':int((eigenvalues<=tau).sum())}
        elif chosen=='dense':
            a=(torch.diag(d)+z.T@z).to(torch.complex128)
            a.diagonal().add_(complex(0,tau));lu,piv=torch.linalg.lu_factor(a);del a
            for start in range(0,r.shape[1],output_chunk_size):
                stop=min(start+output_chunk_size,r.shape[1])
                delta[:,start:stop]=torch.linalg.lu_solve(lu,piv,r[:,start:stop].to(torch.complex128)).real
            del lu,piv
        else:
            inverse_diagonal=1/(d.to(torch.complex128)+complex(0,tau))
            complex_z=z.to(torch.complex128)
            kernel=(complex_z*inverse_diagonal.unsqueeze(0))@complex_z.T
            kernel.diagonal().add_(1);lu,piv=torch.linalg.lu_factor(kernel);del kernel
            for start in range(0,r.shape[1],output_chunk_size):
                stop=min(start+output_chunk_size,r.shape[1])
                base=inverse_diagonal.unsqueeze(1)*r[:,start:stop]
                dual=torch.linalg.lu_solve(lu,piv,complex_z@base)
                delta[:,start:stop]=(base-inverse_diagonal.unsqueeze(1)*(complex_z.T@dual)).real
            del complex_z,lu,piv
    apply=lambda v:d.unsqueeze(1)*v+z.T@(z@v)
    ar=apply(r);equation_residual=apply(delta)-r
    regularized_residual=apply(equation_residual)+(tau*tau)*delta
    smooth_residual=_relative(torch.linalg.vector_norm(regularized_residual),torch.linalg.vector_norm(ar))
    if not torch.isfinite(delta).all() or not math.isfinite(smooth_residual):raise RuntimeError('Nonfinite smoothed solution')
    if filter_kind=='smooth' and smooth_residual>1e-7:raise RuntimeError(f'Smoothed normal-equation residual too large: {smooth_residual}')
    result=(mean+delta.T).to(torch.float32).detach().cpu()
    if not torch.isfinite(result).all():raise RuntimeError('Nonfinite float32 output')
    return result,{'method':'original_gram_mean_centered_weak_spectrum_smoothing_v1',
        'strict_original_equivalence':False,'filter':filter_kind,'strategy':chosen,'offdiag_scale':offdiag_scale,
        'tau':tau,'tau_rule':'input_width * eps(float32) * deterministic_power_lambda_max_estimate',
        'numerical_rank_dtype':'torch.float32','rank_epsilon':rank_epsilon,'spectral_scale_estimate':scale,
        'spectral_scale_estimator':scale_info,'input_width':width,'output_width':r.shape[1],'rows_by_expert':counts,
        'solve_dtype':'float64/complex128','eigen_reference':eigen_info,
        'smooth_normal_equation_relative_residual':smooth_residual,
        'original_centered_equation_relative_residual':_relative(torch.linalg.vector_norm(equation_residual),torch.linalg.vector_norm(r)),
        'exact_zero_energy_coordinates':int((d==0).sum()),
        'common_mean_nullspace_fallback':True,'single_or_identical_expert_exact_by_centering':True,
        'teacher_output_targets_used':False,'success_based_selection':False,
        'final_weight_clipping':False,'nonzero_feature_energy_floor':False,'trust_cap':False,
        'claim_boundary':'Spectral smoothing changes the inverse only; it is a disclosed extension and cannot guarantee policy performance.'}
