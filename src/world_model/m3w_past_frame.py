"""Past-only frame conditioning over a typed, version-locked feature layout."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_objective_alignment import objective_loss

# Matches geometry_features and the bound seven-pair motion summaries.
VECTOR_BLOCKS = ((0,16), (24,38), (38,166), (308,476))
FEATURE_DIM = 581


def validate(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
        raise ValueError('Finite version-locked 581-column features required')
    if not np.isin(x[:,230:294], [0,1]).all():
        raise ValueError('Neighbor mask must be binary')
    return x


def rotate_features(features, rotation):
    """Re-express vector summaries, not RGB pixels or scalar/mask columns."""
    x, r = validate(features), np.asarray(rotation, dtype=np.float64)
    if (r.shape != (len(x),2,2) or not np.isfinite(r).all()
            or not np.allclose(r @ r.transpose(0,2,1), np.eye(2), atol=1e-10, rtol=0)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-10, rtol=0)):
        raise ValueError('One proper planar rotation per query required')
    out = x.copy()
    for start, end in VECTOR_BLOCKS:
        out[:,start:end] = np.einsum('nvd,ndk->nvk', x[:,start:end].reshape(len(x),-1,2), r).reshape(len(x),-1)
    motion = out[:,511:].reshape(len(x),7,10)
    motion[:,:,:6] = np.einsum('nvd,ndk->nvk', x[:,511:].reshape(len(x),7,10)[:,:,:6].reshape(len(x),21,2), r).reshape(len(x),7,6)
    return out


def past_frame(features):
    """Use latest ego velocity, nearest moving neighbor, then relative position."""
    x = validate(features)
    velocities = x[:,24:38].reshape(len(x),7,2)
    neighbors = x[:,38:166].reshape(len(x),8,8,2)
    times = x[:,166:230].reshape(len(x),8,8)
    masks = x[:,230:294].reshape(len(x),8,8).astype(bool)
    q = np.broadcast_to(np.eye(2),(len(x),2,2)).copy()
    valid = np.zeros(len(x),bool)
    kinds = np.full(len(x),'no_anchor',dtype='<U24')
    for i in range(len(x)):
        vector, kind = None, 'no_anchor'
        nz = np.flatnonzero(np.linalg.norm(velocities[i],axis=1) > 1e-8)
        if len(nz):
            vector, kind = velocities[i,nz[-1]], 'ego_past_velocity'
        else:
            for n in range(8):
                ids = np.flatnonzero(masks[i,n])
                if len(ids) < 2:
                    continue
                dt = times[i,n,ids[-1]]-times[i,n,ids[-2]]
                if dt <= 0:
                    raise ValueError('Valid neighbor observations must increase in time')
                delta = neighbors[i,n,ids[-1]]-neighbors[i,n,ids[-2]]
                if np.linalg.norm(delta) > 1e-8:
                    vector, kind = delta/dt, 'neighbor_past_velocity'
                    break
            if vector is None:
                for n in range(8):
                    ids = np.flatnonzero(masks[i,n])
                    if len(ids) and np.linalg.norm(neighbors[i,n,ids[-1]]) > 1e-8:
                        vector, kind = neighbors[i,n,ids[-1]], 'neighbor_position'
                        break
        if vector is not None:
            c, s = vector/np.linalg.norm(vector)
            q[i] = [[c,-s],[s,c]]
            valid[i], kinds[i] = True, kind
    return q, valid, kinds


def frame_prediction(model, features, observed, baseline, rotation, valid):
    if (observed.shape != (len(features),8) or baseline.shape != (len(features),12,2)
            or rotation.shape != (len(features),2,2) or valid.shape != (len(features),)):
        raise ValueError('Complete query-aligned features/frame/baseline required')
    latent = features.new_zeros((len(features),128))
    residual = model.output(model.fusion(torch.cat((features,observed,latent),1))).reshape(-1,12,2)
    restored = torch.bmm(residual,rotation.transpose(1,2))
    return baseline + torch.where(valid[:,None,None],restored,torch.zeros_like(restored))


def fit_frame(model, batch, train_ids, *, config, seed, identity, checkpoint, heartbeat, stop_at=None):
    weights = torch.ones(len(train_ids),dtype=torch.float64)/len(train_ids)
    generator = torch.Generator().manual_seed(seed+7919)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    step, losses, elapsed = 0, [], 0.
    if checkpoint.exists():
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['identity'] != identity or state['config'] != config or not torch.equal(state['train_ids'],train_ids):
            raise ValueError('Resume identity/config/train membership changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, losses, elapsed = state['step'], state['losses'], state['fit_seconds']
    initial, start = step, time.monotonic()
    limit = config['updates'] if stop_at is None else min(stop_at,config['updates'])
    model.train()
    while step < limit:
        chosen = torch.multinomial(weights,config['batch_size'],replacement=True,generator=generator)
        x, observed, baseline, target, rotation, valid = batch(train_ids[chosen])
        optimizer.zero_grad(set_to_none=True)
        loss, detail = objective_loss(frame_prediction(model,x,observed,baseline,rotation,valid),target,baseline,'log')
        loss.backward()
        grad = nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step(); step += 1
        if step == 1 or step % 100 == 0:
            losses.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(grad),**detail))
        if step % config['checkpoint_every'] == 0 or step == limit:
            seconds = elapsed+time.monotonic()-start
            state = dict(identity=identity,config=config,train_ids=train_ids,model=model.state_dict(),
                optimizer=optimizer.state_dict(),sampler_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                step=step,losses=losses,fit_seconds=seconds)
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            tmp = checkpoint.with_suffix('.tmp'); torch.save(state,tmp); os.replace(tmp,checkpoint)
            heartbeat(dict(state='training' if step < config['updates'] else 'fit_complete',step=step,
                           fit_seconds=seconds,loss=float(loss.detach())))
    model.eval()
    return dict(step=step,complete=step == config['updates'],new_updates_this_invocation=step-initial,
                fit_seconds=elapsed+time.monotonic()-start,losses=losses)
