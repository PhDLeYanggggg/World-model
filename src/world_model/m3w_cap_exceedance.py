"""Causal producer-relative cap-event probes, not a deployment controller."""
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_risk_conditioned_residual import risk_features
from src.evaluation.m3w_harm_tail_diagnostics import weights_for_sites

ARMS = ('linear', 'mlp')


def causal_features(native, context, prediction, envelope):
    native, context = np.asarray(native, float), np.asarray(context, float)
    if (native.ndim != 2 or not np.isfinite(native).all()
            or context.shape != (len(native), 7) or np.isinf(context).any()):
        raise ValueError('Finite native causal inputs and seven context summaries required')
    risk = risk_features(prediction, envelope)
    # Missingness is a causal observation, not a future-derived imputation.
    return np.column_stack((native, context, np.isnan(context).astype(float), risk))


def event_target(y, prediction, envelope):
    y, prediction, envelope = map(np.asarray, (y, prediction, envelope))
    risk_features(prediction, envelope)
    if (y.shape != prediction.shape or np.isinf(y).any()
            or not np.array_equal(np.isnan(y).all(1), ~np.isfinite(y).all(1))):
        raise ValueError('Paired unknown or finite nested cost labels required')
    known = np.isfinite(y).all(1)
    if ((y[known] < 0).any() or (y[known, 3] > y[known, 1]+1e-6).any()
            or (y[known, 1] > envelope[known]+1e-4).any()):
        raise ValueError('Invalid nested cost labels')
    valid = known & (envelope > 0)
    target = np.full(len(y), np.nan)
    target[valid] = y[valid, 3] > prediction[valid, 1]
    return target


def preprocess(x, target, sites, outer):
    x, target, sites = map(np.asarray, (x, target, sites))
    if (x.ndim != 2 or target.shape != (len(x),) or sites.shape != target.shape
            or np.isinf(x).any() or np.isinf(target).any() or outer in sites
            or len(set(sites)) != 3):
        raise ValueError('Exactly three fitting localities with outer excluded required')
    known = np.isfinite(target)
    if not np.isin(target[known], [0, 1]).all():
        raise ValueError('Binary event supervision required')
    w = weights_for_sites(known, sites)
    prevalence = float(w @ np.nan_to_num(target))
    if not 0 < prevalence < 1:
        raise ValueError('Both event classes required across fitting localities')
    finite = np.isfinite(x)
    mass = (w[:, None]*finite).sum(0)
    mean = np.divide((w[:, None]*np.where(finite, x, 0)).sum(0), mass,
                     out=np.zeros(x.shape[1]), where=mass > 0)
    filled = np.where(finite, x, mean)
    std = np.sqrt((w[:, None]*(filled-mean)**2).sum(0))
    std[std < 1e-6] = 1.
    return dict(mean=mean, std=std, weights=w, known=known, prevalence=prevalence,
                training_sites=sorted(set(sites)), outer=outer)


def transform(x, pr):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != len(pr['mean']) or np.isinf(x).any():
        raise ValueError('Matched causal feature schema required')
    return np.clip((np.where(np.isfinite(x), x, pr['mean'])-pr['mean'])/pr['std'], -20, 20).astype(np.float32)


class CapEventHead(nn.Module):
    def __init__(self, dimension, arm, width):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Registered arm required')
        self.network = (nn.Sequential(nn.Linear(dimension, 1)) if arm == 'linear' else
                        nn.Sequential(nn.Linear(dimension, width), nn.SiLU(), nn.Linear(width, 1)))

    def forward(self, x):
        return self.network(x)[:, 0]


def fit(x, target, sites, outer, *, arm, seed, settings, identity, directory,
        heartbeat, resume=False, stop_at=None):
    pr = preprocess(x, target, sites, outer)
    torch.manual_seed(seed)
    model = CapEventHead(x.shape[1], arm, settings['width'])
    with torch.no_grad():
        model.network[-1].weight.zero_()
        p = pr['prevalence']
        model.network[-1].bias.fill_(np.log(p/(1-p)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(transform(x, pr))
    labels = torch.tensor(np.nan_to_num(target), dtype=torch.float32)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in pr['training_sites']]
    rng = torch.Generator().manual_seed(seed+7919)
    fixed_rng = torch.Generator().set_state(rng.get_state())
    fixed = draw_batch(groups, settings['batch_size'], fixed_rng)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace = 0, 0., []
    draws = np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in dict(identity=identity, seed=seed, arm=arm, settings=settings).items():
            if state[key] != value:
                raise ValueError('Resume identity mismatch: '+key)
        for key in ('mean', 'std', 'known', 'weights'):
            np.testing.assert_array_equal(state['preprocess'][key], pr[key])
        assert state['preprocess']['prevalence'] == p
        np.testing.assert_array_equal(state['fixed'], fixed)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draws = state['step'], state['seconds'], state['trace'], state['draws']
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing training budget required')
    started = time.monotonic(); first = step

    def diagnostic():
        with torch.no_grad():
            logits = model(z[fixed]); loss = F.binary_cross_entropy_with_logits(logits, labels[fixed])
            brier = ((logits.sigmoid()-labels[fixed])**2).mean()
        return dict(step=step, fixed_BCE=float(loss), fixed_Brier=float(brier))

    def save():
        state = dict(identity=identity, seed=seed, arm=arm, settings=settings, preprocess=pr,
                     fixed=fixed, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
                     seconds=seconds+time.monotonic()-started, trace=trace, draws=draws)
        tmp = path.with_suffix('.tmp'); torch.save(state, tmp); os.replace(tmp, path)

    if not trace:
        trace.append(diagnostic())
    while step < limit:
        ids = draw_batch(groups, settings['batch_size'], rng)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(z[ids]), labels[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite binary-event loss; resume last checkpoint')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; np.add.at(draws, ids, 1)
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
            row = diagnostic(); trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    model.eval()
    return model, pr, dict(step=step, complete=step == settings['steps'], new_updates=step-first,
                          seconds=seconds+time.monotonic()-started, trace=trace,
                          unknown_rows_sampled=int(draws[~pr['known']].sum()),
                          parameters=sum(p.numel() for p in model.parameters()))


def predict(model, x, pr):
    model.eval(); result = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            result.append(model(torch.from_numpy(transform(x[start:start+4096], pr))).sigmoid().numpy())
    result = np.concatenate(result)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite event prediction')
    return result


def restore(directory):
    state = torch.load(Path(directory)/'checkpoint.pt', map_location='cpu', weights_only=False)
    model = CapEventHead(len(state['preprocess']['mean']), state['arm'], state['settings']['width'])
    model.load_state_dict(state['model']); model.eval()
    return model, state
