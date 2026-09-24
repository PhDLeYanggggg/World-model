"""Verified source adaptation for the existing fixed-budget Torch predictor."""
import hashlib
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import numpy as np
import torch
from torch import nn
from src.world_model.m3w_supervised_intervention import build_forecaster

BASELINES = ('constant_position','constant_velocity_causal_fd','damped_velocity_090',
             'damped_velocity_097','history_ols4_velocity','history_ols8_velocity')


def source_folds(support, salt):
    if len(support) != 12 or any(v < 0 for v in support.values()):
        raise ValueError('Exactly twelve admitted source localities required')
    ordered = sorted(support,key=lambda k:(-support[k],k))
    folds = {}
    for start in range(0,12,3):
        block = sorted(ordered[start:start+3],key=lambda k:hashlib.sha256((salt+'|'+k).encode()).hexdigest())
        for fold,key in enumerate(block):
            folds[key] = fold
    return folds


def baseline_numpy(history, index):
    h = np.asarray(history,dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (8,2) or not np.isfinite(h).all() or not 0 <= index < len(BASELINES):
        raise ValueError('Finite complete eight-step causal history required')
    t = np.arange(1,13,dtype=np.float64)
    v = h[:,-1]-h[:,-2]
    if index == 0:
        v = np.zeros_like(v)
    elif index in (2,3):
        t = np.cumsum((.90 if index==2 else .97)**np.arange(12))
    elif index in (4,5):
        k = 4 if index==4 else 8
        x = np.arange(k,dtype=np.float64)-(k-1)/2
        v = np.sum(h[:,-k:]*x[None,:,None],axis=1)/np.sum(x*x)
    return h[:,-1,None]+t[None,:,None]*v[:,None]


def baseline_torch(history, index):
    if not 0 <= index < len(BASELINES):
        raise ValueError('Unknown causal baseline')
    h = history[...,:2]
    t = torch.arange(1,13,dtype=h.dtype,device=h.device)
    v = h[:,-1]-h[:,-2]
    if index == 0:
        v = torch.zeros_like(v)
    elif index in (2,3):
        t = torch.cumsum((.90 if index==2 else .97)**torch.arange(12,dtype=h.dtype,device=h.device),0)
    elif index in (4,5):
        k = 4 if index==4 else 8
        x = torch.arange(k,dtype=h.dtype,device=h.device)-(k-1)/2
        v = torch.sum(h[:,-k:]*x[None,:,None],1)/torch.sum(x*x)
    return h[:,-1,None]+t[None,:,None]*v[:,None]


class SourceForecaster(nn.Module):
    def __init__(self, architecture, baseline_index):
        super().__init__()
        self.predictor = build_forecaster(architecture)
        self.baseline_index = baseline_index

    def forward(self, inputs):
        safe = dict(inputs)
        safe['baseline'] = baseline_torch(inputs['history'],self.baseline_index)
        return self.predictor(safe)


def pack_scene(inputs):
    required = {'history_xy','history_valid','agent_id','target_eligible','baseline_cv'}
    if not required.issubset(inputs) or any(k.startswith('future') or k.startswith('label') for k in inputs):
        raise ValueError('Past input fields only')
    h = np.asarray(inputs['history_xy'],float)
    valid = np.asarray(inputs['history_valid'])
    eligible = np.asarray(inputs['target_eligible'])
    agents = np.asarray(inputs['agent_id'])
    if valid.dtype != bool or eligible.dtype != bool or not np.array_equal(eligible,valid.all(1)):
        raise ValueError('Target eligibility must equal past support')
    if len(np.unique(agents)) != len(agents) or h.shape != (len(agents),8,2):
        raise ValueError('One query with unique current-visible agents required')
    targets = np.flatnonzero(eligible)
    g = np.zeros((len(targets),476),dtype=np.float32)
    times = np.arange(-7,1,dtype=np.float32)/12
    for j,i in enumerate(targets):
        origin = h[i,-1]
        g[j,:16] = (h[i]-origin).reshape(-1)
        g[j,16:24] = times
        candidates = targets[targets!=i]
        order = np.lexsort((agents[candidates],np.sum((h[candidates,-1]-origin)**2,axis=1)))
        near = candidates[order[:8]]
        nxy = np.zeros((8,8,2),np.float32)
        nm = np.zeros((8,8),np.float32)
        nt = np.zeros((8,8),np.float32)
        nxy[:len(near)] = h[near]-origin
        nm[:len(near)] = 1
        nt[:len(near)] = times
        g[j,38:166] = nxy.reshape(-1)
        g[j,166:230] = nt.reshape(-1)
        g[j,230:294] = nm.reshape(-1)
        g[j,332:356] = (np.asarray(inputs['baseline_cv'])[i]-origin).reshape(-1)
    return g,targets


def fit_design(data, fit_sites, baseline_ade):
    sites = np.asarray(data['sites'])
    fit_sites = sorted(fit_sites)
    if not fit_sites or not set(fit_sites) < set(sites):
        raise ValueError('Proper nonempty subset of source fitting sites required')
    reference_scores = []
    for site in fit_sites:
        use = (sites==site)&np.isfinite(baseline_ade[:,1])
        mean = baseline_ade[use].mean(0)
        if not use.any() or mean[1] <= 0:
            raise ValueError('Positive CV supported training reference required')
        reference_scores.append(mean/mean[1])
    selected = int(np.argmin(np.mean(reference_scores,axis=0)))
    train_ids = np.flatnonzero(np.isin(sites,fit_sites))
    held_ids = np.flatnonzero(~np.isin(sites,fit_sites))
    factors = np.zeros(len(sites),dtype=np.float64)
    groups,normalizers = [],{}
    cv = []
    for site in fit_sites:
        ids = np.flatnonzero(sites==site)
        supported = np.isfinite(baseline_ade[ids,selected])
        mean = float(baseline_ade[ids[supported],selected].mean())
        if mean <= 0:
            raise ValueError('Positive selected-baseline training mean required')
        correction = len(ids)/int(supported.sum())
        factors[ids] = correction/mean
        groups.append(ids)
        normalizers[site] = dict(indexed=len(ids),supported=int(supported.sum()),
                                selected_baseline_mean_ADE=mean,correction=correction)
        cv.extend(baseline_ade[ids[supported],1].tolist())
    cv = np.asarray(cv)
    return dict(train_ids=train_ids,held_ids=held_ids,groups=groups,factors=factors,
        normalizers=normalizers,baseline_index=selected,selection_relative_scores=np.mean(reference_scores,axis=0).tolist(),
        hard_cut=float(np.quantile(cv,.75)),easy_cut=float(np.quantile(cv[cv>0],.25)))
