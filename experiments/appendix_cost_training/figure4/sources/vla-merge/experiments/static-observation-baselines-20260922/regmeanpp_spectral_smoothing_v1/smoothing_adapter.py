"""Full418 adapter metadata for the disclosed spectral-smoothing extension."""
import torch
from spectral_smoothing import solve_smoothed_weight

def solve_smoothing_adapter(inputs,weights,metric_reference,*,offdiag_scale):
    if offdiag_scale!=.3:raise ValueError('Frozen alpha differs')
    result,diagnostics=solve_smoothed_weight(inputs,weights,metric_reference,
        offdiag_scale=.3,strategy='auto',filter_kind='smooth',power_iterations=64,
        numerical_rank_dtype=torch.float32,output_chunk_size=128)
    # Match the original runner's offline metric only for descriptive reporting.
    # This metric neither accepts/rejects a target nor selects any regularizer.
    names=list(inputs);device=metric_reference.device
    merged=result.to(device=device,dtype=torch.float32)
    reference=metric_reference.to(device=device,dtype=torch.float32)
    losses={};reference_losses={}
    for name in names:
        x=inputs[name].to(device=device,dtype=torch.float32)
        weight=weights[name].to(device=device,dtype=torch.float32)
        losses[name]=float((x@(merged-weight).T).square().mean())
        reference_losses[name]=float((x@(reference-weight).T).square().mean())
    mean_loss=sum(losses.values())/len(names);mean_reference=sum(reference_losses.values())/len(names)
    return result,{**diagnostics,'equal_expert_weighting':True,'raw_xtx_equal_rows':True,
        'soup_centered_ridge':False,'spectral_regularization_applied':True,
        'correction_cap':False,'pseudo_inverse':False,'relative_weighting':False,
        'rejection_gate':False,'expert_objective_weights':{n:1./len(names) for n in names},
        'prior_loss':mean_reference,'dense_regmean_loss':mean_loss,
        'prior_loss_by_expert':reference_losses,'dense_regmean_loss_by_expert':losses,
        'improvement_vs_prior':(mean_reference-mean_loss)/max(mean_reference,1e-12),
        'prior_metric_semantics':'Arithmetic expert mean reference for reporting only; no Soup prior penalty',
        'trust_scale':1.,'rejected_nonimproving':False,
        'relative_residual_field_policy':'No generic relative_residual key: smoothed-system and original-equation residuals are separate.'}
