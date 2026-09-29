"""Training-source-only checkpoint selection; no transfer-source tuning."""
import copy
import hashlib
from pathlib import Path
import platform
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
import numpy as np
import torch
from src.world_model import m3w_inner_separability as core


def source_partition(recordings,frames,source,train_fraction=.7,obs=8,pred=12,stride=12):
    r,f=np.asarray(recordings).astype(str),np.asarray(frames)
    if r.ndim!=1 or f.shape!=r.shape or not len(r) or not np.issubdtype(f.dtype,np.integer):
        raise ValueError('One training source, recording IDs and raw integer frames required')
    names=sorted(np.unique(r),key=lambda k:hashlib.sha256(('source-checkpoint-v1|'+source+'|'+k).encode()).hexdigest())
    if len(names)>=2:
        count=min(len(names)-1,max(1,int(np.floor(train_fraction*len(names)))))
        train=np.isin(r,names[:count]);val=~train
        info=dict(method='whole_recording_hash_split',train_recordings=names[:count],validation_recordings=names[count:])
        assert set(r[train]).isdisjoint(r[val])
    else:
        unique=np.unique(f);cut=int(unique[min(len(unique)-1,int(np.floor(train_fraction*len(unique))))])
        train=f<cut-pred*stride;val=f>=cut+(obs-1)*stride
        info=dict(method='single_recording_purged_time',boundary_frame=cut,observed_span=(obs-1)*stride,future_span=pred*stride)
        if train.any() and val.any():assert f[train].max()+pred*stride < f[val].min()-(obs-1)*stride
    if not train.any() or not val.any():raise ValueError('Training source lacks a disjoint optimization/validation partition')
    assert not (train&val).any()
    info.update(train_rows=int(train.sum()),validation_rows=int(val.sum()),purged_rows=int((~train&~val).sum()))
    return train,val,info


def validation_score(model,x,env,y,pr,sites,recordings,frames):
    known=np.isfinite(y).all(1)
    if not known.any():raise ValueError('No known validation labels in training source')
    w,_=core.weights(sites,recordings,frames,known)
    total=0.
    with torch.no_grad():
        for start in range(0,len(x),4096):
            end=min(len(x),start+4096);mask=known[start:end]
            if not mask.any():continue
            z=torch.from_numpy(np.clip((x[start:end][mask]-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32))
            e=torch.from_numpy((env[start:end][mask]/pr['scale']).astype(np.float32))
            p=model(z,e).numpy().astype(float)
            err=(core.signed(p)-core.signed(y[start:end][mask]/pr['scale']))/pr['rms'][5:]
            total+=float((w[start:end][mask,None]*err**2).sum()/3)
    return total


def fit(x,env,y,sites,recordings,frames,*,source,settings,seed,path,identity,heartbeat,resume=False,stop_at=None):
    train,val,partition=source_partition(recordings,frames,source)
    if set(sites)!={source}:raise ValueError('Only original training locality may select a checkpoint')
    pr=core.preprocess(x[train],env[train],y[train],sites[train],recordings[train],frames[train],training_site=source)
    known=np.isfinite(y[train]).all(1)
    if not np.isfinite(y[val]).all(1).any():raise ValueError('No known source-validation target')
    groups,source_groups,_=core.sampling.query_groups(sites[train],recordings[train],frames[train],known)
    z=torch.from_numpy(np.clip((x[train]-pr['mean'])/pr['std'],-pr['clip'],pr['clip']).astype(np.float32))
    e=torch.from_numpy((env[train]/pr['scale']).astype(np.float32))
    truth=torch.from_numpy(np.where(known[:,None],y[train]/pr['scale'],0).astype(np.float32))
    rms=torch.from_numpy(pr['rms']);model=core.initialize(pr,'nonlinear',settings['width'],seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+7919);initial=copy.deepcopy(model.state_dict())
    def score():return validation_score(model,x[val],env[val],y[val],pr,sites[val],recordings[val],frames[val])
    step=0;best_step=0;best_score=score();best_model=copy.deepcopy(initial)
    trace=[dict(step=0,validation_signed_MSE=best_score)];draw_hash='0'*64;seconds=0.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if not resume:raise ValueError('Resume required for existing checkpoint')
        old=core.read_checkpoint(path)
        for k,v in dict(identity=identity,settings=settings,preprocess=pr,seed=seed,partition=partition).items():core.exact(old[k],v)
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']);torch.set_rng_state(old['torch_rng'])
        step,best_step,best_score,best_model,trace,draw_hash,seconds=(old[k] for k in ('step','best_step','best_score','best_model','trace','draw_hash','seconds'))
    limit=settings['steps'] if stop_at is None else min(stop_at,settings['steps'])
    if limit<step:raise ValueError('Cannot decrease completed update count')
    began=time.monotonic()
    def save():
        core.save_checkpoint(path,dict(identity=identity,settings=settings,preprocess=pr,seed=seed,partition=partition,
            arm='nonlinear',initial_model=initial,model=model.state_dict(),optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(),torch_rng=torch.get_rng_state(),step=step,best_step=best_step,
            best_score=best_score,best_model=best_model,trace=trace,draw_hash=draw_hash,
            seconds=seconds+time.monotonic()-began,known_training_rows=int(known.sum()),unknown_training_rows=int((~known).sum())))
    while step<limit:
        ix,seg,qids=core.sampling.draw_queries(groups,source_groups,settings['query_batch_size'],rng)
        optimizer.zero_grad(set_to_none=True)
        terms=core.losses(model(z[ix],e[ix]),truth[ix],torch.from_numpy(seg),settings['query_batch_size'],rms)
        if not torch.isfinite(terms['total']):raise FloatingPointError('Nonfinite training objective')
        terms['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();step+=1;draw_hash=hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
        if step%settings['validate_every']==0 or step==settings['steps']:
            current=score()
            if current<best_score:
                best_score=current;best_step=step;best_model=copy.deepcopy(model.state_dict())
            trace.append(dict(step=step,validation_signed_MSE=current,training_loss=float(terms['total'].detach())))
            heartbeat(step=step,best_step=best_step,validation_signed_MSE=current,training_loss=trace[-1]['training_loss'])
        if step%settings['checkpoint_every']==0 or step==limit:save()
    return core.read_checkpoint(path)


def paired_predictions(state,x,env):
    final,support=core.predict(state,x,env)
    chosen={**state,'model':state['best_model']}
    selected,other=core.predict(chosen,x,env);np.testing.assert_array_equal(support,other)
    return dict(final=final,validation=selected),support


def decisions(p,moving,support,recordings,frames,ids):
    # Reuse identical registered decision mathematics, not affine/nonlinear model labels.
    take=core.decisions(dict(affine=p['final'],nonlinear=p['validation']),moving,support,recordings,frames,ids)
    return {name:take[key] for name,key in [('final','affine'),('validation','nonlinear'),
                                         ('final_matched','affine_matched'),('validation_matched','nonlinear_matched')]}
