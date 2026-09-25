#!/usr/bin/env python3
"""Verify existing Table-1 kernels support M=3 against small independent oracles.

This does not certify an M=3 model feature collector or checkpoint materializer.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from prepare_readiness import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import torch
    torch.set_num_threads(2)
    root=args.workspace_root/'vla-merge';sys.path[:0]=[str(root/'src')]
    from vla_merge.featcal_hybrid_solve import featcal_hybrid_linear_weight
    from vla_merge.featcal_forward_order import FeatCalRowFactors
    spectral=root/'experiments/static-observation-baselines-20260922/regmeanpp_spectral_smoothing_v1/spectral_smoothing.py'
    spec=importlib.util.spec_from_file_location('frozen_spectral',spectral);reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(reg)
    gen=torch.Generator().manual_seed(72591);names=('coordination','receptacle','precision');results=[]
    for rank_deficient in (False,True):
        xs={n:torch.randn(7,6,generator=gen,dtype=torch.float64) for n in names}
        if rank_deficient:
            for x in xs.values():x[:,-1]=0
        weights={n:torch.randn(2,6,generator=gen,dtype=torch.float64) for n in names}
        actual,meta=reg.solve_smoothed_weight(xs,weights,device='cpu')
        mean=sum(weights.values())/3;grams={}
        for n,x in xs.items():
            gram=x.T@x;grams[n]=.3*gram+.7*torch.diag(gram.diag())
        matrix=sum(grams.values());rhs=sum(grams[n]@(weights[n]-mean).T for n in names)
        vals,u=torch.linalg.eigh(matrix);filtered=vals/(vals.square()+meta['tau']**2)
        expected=(mean+(u@(filtered[:,None]*(u.T@rhs))).T).float()
        error=float((actual-expected).abs().max())
        if not torch.allclose(actual,expected,atol=2e-6,rtol=2e-6):raise ValueError('M3 RegMean++ differs from independent eigen-filter oracle')
        if rank_deficient and not torch.equal(actual[:,-1],mean.float()[:,-1]):raise ValueError('Zero-energy coordinate does not preserve mean')
        results.append({'method':'RegMean++ disclosed spectral smoothing','experts':3,'rank_deficient':rank_deficient,'max_abs':error})
        expert_x={n:xs[n]+.1*torch.randn(7,6,generator=gen,dtype=torch.float64) for n in names}
        soup=mean; base=torch.zeros_like(soup);factors=[];matrix=(.05+1e-8)*torch.eye(6,dtype=torch.float64);rhs=.05*(2*soup-base).T
        for n in names:
            target=.3*expert_x[n]+.7*xs[n]
            factors.append(FeatCalRowFactors(student=xs[n],target=target,row_weights=torch.ones(7,dtype=torch.float64),forward_identity=n))
            gram=xs[n].T@xs[n]/7;cross=xs[n].T@target/7;norm=gram.norm().clamp_min(1e-8)
            matrix+=gram/norm;rhs+=cross@weights[n].T/norm
        expected=torch.linalg.solve(matrix,rhs).T
        actual,meta=featcal_hybrid_linear_weight(list(weights.values()),factors,soup_weight=soup,base_weight=base,
            ridge_lambda=.05,anchor_blend_rho=2.,covariance_eps=1e-8,solve_device='cpu',force_strategy='primal')
        error=float((actual-expected).abs().max())
        if not torch.allclose(actual,expected,atol=1e-10,rtol=1e-10):raise ValueError('M3 FeatCal differs from independent normal-equation oracle')
        results.append({'method':'FeatCal','experts':3,'rank_deficient':rank_deficient,'max_abs':error})
    if torch.cuda.is_initialized():raise RuntimeError('CPU test initialized CUDA')
    output={'status':'PASS','cases':results,'gpu_used':False,'scope':'M3 matrix kernels only, not model graph/feature collection',
        'source_sha256':{str(spectral):sha(spectral),str(root/'src/vla_merge/featcal_hybrid_solve.py'):sha(root/'src/vla_merge/featcal_hybrid_solve.py')}}
    with args.output.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output))

if __name__=='__main__':main()
