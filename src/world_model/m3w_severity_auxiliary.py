"""Severity-weighted auxiliary supervision; causal prediction stays unchanged."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from src.world_model.m3w_membership_auxiliary import AuxiliaryCostHead, predict, restore, standardized
from src.world_model.m3w_native_forecast import draw_batch


def objective(pred, logits, target, easy, scales, mean_harm):
    if not np.isfinite(mean_harm) or mean_harm <= 0:
        raise ValueError('Positive fitting-only mean harm required')
    cost = (((pred-target.detach())/scales)**2).mean()
    bce = F.binary_cross_entropy_with_logits(logits, easy.detach(), reduction='none')
    weighted = (bce * target[:, 1].detach()/mean_harm).mean()
    return cost+weighted, dict(cost_loss=float(cost.detach()), membership_BCE=float(bce.mean().detach()),
        severity_BCE=float(weighted.detach()))


def support(y, easy, pr, sites, recordings, agents):
    known = np.isfinite(y).all(1); w = np.asarray(pr['weights']); h = np.nan_to_num(y[:, 1], nan=0.)
    if (not np.array_equal(known, pr['known']) or not np.array_equal(np.isfinite(easy), known)
            or not np.isin(easy[known], [0,1]).all() or (h < 0).any() or not np.isfinite(h).all()
            or any(np.asarray(v).shape != (len(y),) for v in (sites, recordings, agents))):
        raise ValueError('Aligned known fitting support required')
    mean = float(w@h); total = w*h; out = {}
    for name, mask in (('all', known), ('easy', known & (easy==1)), ('not_easy', known & (easy==0))):
        mass = np.where(mask, total, 0); pos = mass > 0
        def group_mass(keys):
            _, inv = np.unique(keys, axis=0, return_inverse=True)
            return np.bincount(inv, weights=mass)
        track = group_mass(np.column_stack((recordings, agents)))
        record = group_mass(np.asarray(recordings)[:,None])
        locality = [float(mass[sites==s].sum()) for s in sorted(set(sites))]
        def concentration(v):
            v = np.asarray(v); s = float(v.sum())
            return dict(positive_groups=int((v>0).sum()),
                mass_ESS=float(s*s/(v@v)) if s else None,
                maximum_share=float(v.max()/s) if s else None)
        out[name] = dict(known_rows=int(mask.sum()), positive_harm_rows=int(pos.sum()),
            weighted_mass=float(mass.sum()), windows=concentration(mass),
            tracks=concentration(track), recordings=concentration(record), localities=concentration(locality))
    return dict(mean_harm=mean, easy_prevalence=float(w@np.nan_to_num(easy, nan=0.)),
        harm_weighted_easy_fraction=float(total@np.nan_to_num(easy, nan=0.)/mean) if mean else None,
        strata=out, mean_identifiable=mean>0,
        two_sided_identifiable=out['easy']['weighted_mass']>0 and out['not_easy']['weighted_mass']>0,
        weak_support=any(out[k]['tracks']['positive_groups']<20 or out[k]['localities']['positive_groups']<2
                         for k in ('easy','not_easy')),
        ESS_is_mass_concentration_not_independent_sample_size=True)


def fit(x, y, easy, sites, env, pr, *, seed, settings, identity, directory,
        heartbeat, resume=False, stop_at=None):
    known=np.isfinite(y).all(1)
    if (y.shape!=(len(x),4) or not np.array_equal(np.isnan(y).all(1),~known)
            or not np.array_equal(known,pr['known']) or not np.array_equal(np.isfinite(easy),known)
            or not np.isin(easy[known],[0,1]).all() or not np.isfinite(x).all()
            or env.shape!=(len(x),) or not np.isfinite(env).all() or (env<0).any()
            or (y[known]<0).any() or (y[known,1]>env[known]+1e-5).any()
            or (y[known,2:]>y[known,:2]+1e-5).any()):
        raise ValueError('Known nested costs and membership supervision required')
    np.testing.assert_allclose(y[known,3],y[known,1]*easy[known],atol=1e-8)
    w,scale=pr['weights'],pr['cost_scale']; e=np.nan_to_num(easy,nan=0.)
    prevalence=float(w@e)
    if not 0<prevalence<1: raise ValueError('Both fitting strata required')
    target=torch.from_numpy(np.where(known[:,None],y/scale,0).astype(np.float32))
    mean_harm=float(w@target.numpy()[:,1].astype(float))
    if not mean_harm>0: raise ValueError('No harm support; do not invent a weighted objective')
    means=np.sum(w[:,None]*target.numpy(),axis=0)
    rms=np.sqrt(np.sum(w[:,None]*target.numpy().astype(float)**2,axis=0)).clip(1e-4)
    rt=torch.tensor(rms,dtype=torch.float32); et=torch.tensor(e,dtype=torch.float32)
    torch.manual_seed(seed); model=AuxiliaryCostHead(x.shape[1],settings['width'])
    def logit(v):
        v=np.clip(v,1e-6,1-1e-6); return np.log(v/(1-v))
    ref=max(means[0],1e-6); mean_env=float(w@(env/scale))
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([ref+np.log(-np.expm1(-ref)),
            logit(means[1]/max(mean_env,1e-6)),logit(means[2]/max(means[0],1e-6)),
            logit(means[3]/max(means[1],1e-6))],dtype=torch.float32))
        # Keep ordinary-membership initialization to isolate the changed loss.
        model.membership.weight.zero_(); model.membership.bias.fill_(logit(prevalence))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(x,pr)); d=torch.tensor(env/scale,dtype=torch.float32)
    groups=[np.flatnonzero(known & (sites==s)) for s in sorted(set(sites))]
    if any(not len(g) for g in groups): raise ValueError('Each fitting locality needs labels')
    rng=torch.Generator().manual_seed(seed+7919)
    fixed_rng=torch.Generator().set_state(rng.get_state()); fixed=draw_batch(groups,settings['batch_size'],fixed_rng)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step,seconds,trace,draws=0,0.,[],np.zeros(len(x),np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        s=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in dict(identity=identity,seed=seed,arm='severity_aux',settings=settings,
                        prevalence=prevalence,mean_harm=mean_harm).items():
            if s[k]!=v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean','std','known','weights'): np.testing.assert_array_equal(s['preprocess'][k],pr[k])
        assert s['preprocess']['cost_scale']==scale
        np.testing.assert_array_equal(s['loss_scales'],rms); np.testing.assert_array_equal(s['fixed_ids'],fixed)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step,seconds,trace,draws=s['step'],s['seconds'],s['trace'],s['draws']
    limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<=0 or limit<step: raise ValueError('Nondecreasing budget required')
    first=step; started=time.monotonic()
    def loss(ids):
        pred,logits=model(z[ids],d[ids]); return objective(pred,logits,target[ids],et[ids],rt,mean_harm)
    def diagnostic():
        with torch.no_grad(): value,parts=loss(fixed)
        return dict(step=step,objective=float(value),**parts)
    def save():
        s=dict(identity=identity,seed=seed,arm='severity_aux',settings=settings,prevalence=prevalence,
            mean_harm=mean_harm,preprocess=pr,loss_scales=rms,fixed_ids=fixed,step=step,
            seconds=seconds+time.monotonic()-started,trace=trace,draws=draws,model=model.state_dict(),
            optimizer=optimizer.state_dict(),sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state())
        tmp=path.with_suffix('.tmp'); torch.save(s,tmp); os.replace(tmp,path)
    if not trace: trace.append(diagnostic())
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng); optimizer.zero_grad(set_to_none=True)
            value,parts=loss(ids)
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite weighted auxiliary loss')
            value.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); step+=1; np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(diagnostic(),batch_loss=float(value.detach()),gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint',step=step); raise
    model.eval()
    return model,dict(step=step,complete=step==settings['steps'],new_updates=step-first,
        seconds=seconds+time.monotonic()-started,parameters=sum(p.numel() for p in model.parameters()),
        unknown_rows_sampled=int(draws[~known].sum()),prevalence=prevalence,mean_harm=mean_harm,
        loss_scales=rms.tolist(),trace=trace)
