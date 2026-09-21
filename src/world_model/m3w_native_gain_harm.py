"""Native benefit/harm regression with fixed, source-only fitting semantics."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import numpy as np
import torch

from src.world_model.m3w_native_forecast import pack_geometry, draw_batch
from src.world_model.m3w_supervised_intervention import risk_features, GainHarmHead


def cost_features(geometry, prediction, scale):
    g, p, s = np.asarray(geometry), np.asarray(prediction), np.asarray(scale)
    if (p.shape != (len(g), 12, 2) or s.shape != (len(g),)
            or not np.isfinite(p).all() or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError('Causal geometry, finite candidate and positive observed scale required')
    pieces = []
    with torch.no_grad():
        for start in range(0, len(g), 4096):
            end = start+4096
            payload = pack_geometry(g[start:end])
            candidate = torch.from_numpy(p[start:end].astype(np.float32))
            pieces.append(np.column_stack((risk_features(payload, candidate).numpy(),
                payload['baseline'].numpy().reshape(-1, 24), candidate.numpy().reshape(-1, 24),
                np.log(s[start:end]))).astype(np.float32))
    x = np.concatenate(pieces)
    if x.shape != (len(g), 355) or not np.isfinite(x).all():
        raise ValueError('Invalid native cost feature schema')
    same = np.all(p == g[:, 332:356].reshape(-1, 12, 2), axis=(1, 2))
    return x, same


def preprocess(x, y, cv, sites, outer):
    x, y, cv, sites = np.asarray(x), np.asarray(y), np.asarray(cv), np.asarray(sites)
    if (x.ndim != 2 or y.shape != (len(x), 2) or cv.shape != (len(x),)
            or sites.shape != (len(x),) or outer in sites or len(set(sites)) < 2
            or not np.isfinite(x).all() or np.isinf(y).any() or np.isinf(cv).any()
            or not np.array_equal(np.isnan(y[:, 0]), np.isnan(y[:, 1]))
            or not np.array_equal(np.isnan(y[:, 0]), np.isnan(cv))):
        raise ValueError('Aligned training-only features and paired cost support required')
    known = np.isfinite(cv)
    if np.any(y[known] < 0) or np.any(cv[known] < 0):
        raise ValueError('Nonnegative supported costs required')
    w = np.zeros(len(x), float)
    for site in sorted(set(sites)):
        use = known & (sites == site)
        if not use.any():
            raise ValueError('Every training scene needs supported cost labels')
        w[use] = 1/(len(set(sites))*int(use.sum()))
    # Sufficient statistics use supported training rows only, never outer rows.
    mean = np.zeros(x.shape[1]); second = mean.copy()
    for start in range(0, len(x), 4096):
        z = x[start:start+4096].astype(float); a = w[start:start+4096, None]
        mean += (a*z).sum(0); second += (a*z*z).sum(0)
    std = np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6)
    scale = float(np.dot(w[known], cv[known]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Positive training-only native CV cost scale required')
    constant = (w[known, None]*y[known]).sum(0)
    positive = cv[known & (cv > 0)]
    return dict(mean=mean, std=std, cost_scale=scale, constant=constant,
        known=known, weights=w, training_sites=sorted(set(sites)),
        positive_easy_cut=float(np.quantile(positive, .25)), hard_cut=float(np.quantile(cv[known], .75)))


def standardized(x, pr):
    return ((np.asarray(x, float)-pr['mean'])/pr['std']).astype(np.float32)


def fit_ridge(x, y, pr, *, alpha):
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Positive fixed ridge penalty required')
    width = x.shape[1]+1
    gram, rhs = np.zeros((width, width)), np.zeros((width, 2))
    for start in range(0, len(x), 4096):
        use = pr['known'][start:start+4096]
        z = np.column_stack((standardized(x[start:start+4096], pr)[use], np.ones(use.sum())))
        w = pr['weights'][start:start+4096][use]
        target = y[start:start+4096][use]/pr['cost_scale']
        gram += z.T@(w[:, None]*z); rhs += z.T@(w[:, None]*target)
    penalty = np.eye(width)*alpha; penalty[-1, -1] = 0
    return dict(coef=np.linalg.solve(gram+penalty, rhs), alpha=alpha)


def predict_ridge(head, x, same, pr):
    z = np.column_stack((standardized(x, pr), np.ones(len(x))))
    result = (z@head['coef'])*pr['cost_scale']
    result[same] = 0
    return result


def cost_loss(prediction, target, arm):
    if (arm not in ('mse', 'underharm4') or prediction.shape != target.shape
            or target.ndim != 2 or target.shape[1] != 2 or not torch.isfinite(target).all()
            or (target < 0).any()):
        raise ValueError('Fixed continuous benefit/harm objective required')
    error = (prediction-target.detach()).square()
    if arm == 'underharm4':
        harm_weight = torch.where(prediction[:, 1] < target[:, 1], 4., 1.)
        error = torch.stack((error[:, 0], error[:, 1]*harm_weight), 1)
    return error.mean()


def build_head(width, pr, seed):
    torch.manual_seed(seed)
    model = GainHarmHead(len(pr['mean']), width=width)
    with torch.no_grad():
        layer = model.network[-2]
        layer.weight.zero_()
        initial = torch.as_tensor(pr['constant']/pr['cost_scale'], dtype=torch.float32).clamp_min(1e-6)
        layer.bias.copy_(initial+torch.log(-torch.expm1(-initial)))
    return model


def fit_neural(x, y, sites, same, pr, *, seed, arm, settings, identity, directory,
               heartbeat, resume=False, stop_at=None):
    if not np.all(y[pr['known'] & same] == 0):
        raise ValueError('Identical causal predictions must have zero supported cost')
    model = build_head(settings['width'], pr, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    target = torch.from_numpy(np.where(pr['known'][:, None], y/pr['cost_scale'], 0).astype(np.float32))
    same_t = torch.from_numpy(same)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['settings'] != settings
                or state['arm'] != arm or state['seed'] != seed):
            raise ValueError('Cost-head resume identity changed')
        for k in ('mean', 'std', 'constant', 'weights', 'known'):
            np.testing.assert_array_equal(pr[k], state['preprocess'][k])
        if pr['cost_scale'] != state['preprocess']['cost_scale']:
            raise ValueError('Cost scale changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draws = state['step'], state['seconds'], state['trace'], state['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing fixed training budget required')
    def save():
        state = dict(identity=identity, settings=settings, arm=arm, seed=seed,
            preprocess=pr, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), draws=draws,
            step=step, seconds=seconds+time.monotonic()-started, trace=trace)
        tmp = path.with_suffix('.tmp'); torch.save(state, tmp); os.replace(tmp, path)
    model.train()
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            scores = model(z[ids]); p = torch.stack((scores['benefit'], scores['harm']), 1)
            p = torch.where(same_t[ids, None], 0., p)
            loss = cost_loss(p, target[ids], arm)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite cost regression loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step)
        raise
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unique_training_rows=int((draws > 0).sum()), unknown_rows_sampled=int(draws[~pr['known']].sum()),
        parameters=sum(p.numel() for p in model.parameters()), trace=trace)


def predict_neural(model, x, same, pr):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            score = model(torch.from_numpy(standardized(x[start:start+4096], pr)))
            p = torch.stack((score['benefit'], score['harm']), 1).numpy()*pr['cost_scale']
            p[same[start:start+4096]] = 0
            out.append(p)
    return np.concatenate(out)
