"""Matched frozen-representation probes for bounded all/easy harm moments."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch


def extract(encoder,x,source_preprocess):
    encoder.eval(); outputs=[]
    with torch.no_grad():
        for start in range(0,len(x),4096):
            z=torch.from_numpy(standardized(x[start:start+4096],source_preprocess))
            outputs.append(encoder.network[:2](z).numpy())
    return np.concatenate(outputs)


def preprocess(h,source):
    h=np.asarray(h,float); w=source['weights']; known=source['known']
    if h.ndim!=2 or len(h)!=len(w) or not np.isfinite(h).all() or (w<0).any() or not np.isclose(w.sum(),1):
        raise ValueError('Finite fitting features and normalized fitting-only weights required')
    mean=w@h; std=np.sqrt(w@((h-mean)**2)).clip(1e-6)
    return dict(mean=mean,std=std,known=known.copy(),weights=w.copy(),
                cost_scale=source['cost_scale'],training_sites=source['training_sites'])


class HarmReadout(nn.Module):
    def __init__(self,width):
        super().__init__(); self.linear=nn.Linear(width,2)

    def forward(self,h,envelope):
        raw=self.linear(h); all_harm=envelope*torch.sigmoid(raw[:,0])
        return torch.stack((all_harm,all_harm*torch.sigmoid(raw[:,1])),1)


def fit(h,y,sites,env,pr,*,seed,settings,identity,directory,heartbeat,resume=False,stop_at=None):
    known=np.isfinite(y).all(1); np.testing.assert_array_equal(known,pr['known'])
    if y.shape!=(len(h),4) or not np.array_equal(np.isnan(y).all(1),~known):
        raise ValueError('Four fully known or fully unknown supervised moments required')
    if (y[known]<0).any() or (y[known,3]>y[known,1]+1e-5).any() or (y[known,1]>env[known]+1e-5).any():
        raise ValueError('Valid nested harm labels required')
    w,scale=pr['weights'],pr['cost_scale']
    t=np.where(known[:,None],y[:,[1,3]]/scale,0).astype(np.float32)
    means=w@t; rms=np.sqrt(w@(t.astype(float)**2)).clip(1e-4)
    torch.manual_seed(seed); model=HarmReadout(h.shape[1])
    fractions=np.array([means[0]/max(float(w@(env/scale)),1e-6),means[1]/max(means[0],1e-6)])
    fractions=np.clip(fractions,1e-6,1-1e-6)
    with torch.no_grad():
        model.linear.weight.zero_(); model.linear.bias.copy_(torch.tensor(np.log(fractions/(1-fractions)),dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    z=torch.from_numpy(standardized(h,pr)); d=torch.tensor(env/scale,dtype=torch.float32)
    target=torch.tensor(t); scales=torch.tensor(rms,dtype=torch.float32)
    groups=[np.flatnonzero(known&(sites==s)) for s in sorted(set(sites))]
    if any(len(v)==0 for v in groups): raise ValueError('Every fitting locality needs known targets')
    rng=torch.Generator().manual_seed(seed+7919); fixed_rng=torch.Generator().set_state(rng.get_state())
    fixed=draw_batch(groups,settings['batch_size'],fixed_rng)
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True); path=directory/'checkpoint.pt'
    step,seconds,trace,draws=0,0.,[],np.zeros(len(h),np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        for k,v in dict(identity=identity,seed=seed,settings=settings).items():
            if saved[k]!=v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean','std','known','weights'): np.testing.assert_array_equal(saved['preprocess'][k],pr[k])
        assert saved['preprocess']['cost_scale']==scale
        np.testing.assert_array_equal(saved['loss_scales'],rms); np.testing.assert_array_equal(saved['fixed_ids'],fixed)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step,seconds,trace,draws=saved['step'],saved['seconds'],saved['trace'],saved['draws']
    limit=settings['steps'] if stop_at is None else min(settings['steps'],stop_at)
    if limit<=0 or limit<step: raise ValueError('Nondecreasing budget required')
    started=time.monotonic(); first=step
    def loss(ids): return (((model(z[ids],d[ids])-target[ids])/scales)**2).mean()
    def diagnostic():
        with torch.no_grad(): value=loss(fixed)
        return dict(step=step,harm_mse=float(value))
    def save():
        s=dict(identity=identity,seed=seed,settings=settings,preprocess=pr,loss_scales=rms,fixed_ids=fixed,
            step=step,seconds=seconds+time.monotonic()-started,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),draws=draws,trace=trace)
        tmp=path.with_suffix('.tmp'); torch.save(s,tmp); os.replace(tmp,path)
    if not trace: trace.append(diagnostic())
    try:
        while step<limit:
            ids=draw_batch(groups,settings['batch_size'],rng); optimizer.zero_grad(set_to_none=True)
            value=loss(ids)
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite frozen harm loss')
            value.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
            optimizer.step(); step+=1; np.add.at(draws,ids,1)
            if step==1 or step%settings['heartbeat_every']==0 or step==limit:
                row=dict(diagnostic(),batch_loss=float(value.detach()),gradient_norm=float(grad)); trace.append(row)
                heartbeat(state='training',**row)
            if step%settings['checkpoint_every']==0 or step==limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_checkpoint',step=step); raise
    model.eval()
    return model,dict(step=step,complete=step==settings['steps'],new_updates=step-first,
        seconds=seconds+time.monotonic()-started,parameters=sum(v.numel() for v in model.parameters()),
        unknown_rows_sampled=int(draws[~known].sum()),trace=trace,loss_scales=rms.tolist())


def predict(model,h,env,pr,reference):
    out=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(h),4096):
            z=torch.from_numpy(standardized(h[start:start+4096],pr))
            d=torch.tensor(env[start:start+4096]/pr['cost_scale'],dtype=torch.float32)
            out.append(model(z,d).numpy()*pr['cost_scale'])
    value=np.array(reference,copy=True); value[:,[1,3]]=np.concatenate(out)
    if not np.isfinite(value).all() or (value<0).any(): raise FloatingPointError('Invalid predicted moments')
    np.testing.assert_array_equal(value[:,[0,2]],reference[:,[0,2]])
    return value


def restore(directory):
    s=torch.load(Path(directory)/'checkpoint.pt',map_location='cpu',weights_only=False)
    model=HarmReadout(len(s['preprocess']['mean'])); model.load_state_dict(s['model']); model.eval()
    return model,s
