"""Separate event and expected-cost learning for baseline-relative protection."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import numpy as np
import torch
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import cost_features, standardized


def risk_targets(cv, harm, complete, arm):
    cv, h, full = np.asarray(cv), np.asarray(harm), np.asarray(complete)
    if (arm not in ('all_harm', 'zero_reference_harm') or cv.ndim != 1 or h.shape != cv.shape
            or full.shape != cv.shape or full.dtype != bool or np.isinf(cv).any() or np.isinf(h).any()
            or not np.array_equal(np.isnan(cv), np.isnan(h))
            or not np.isfinite(cv[full]).all() or not np.isfinite(h[full]).all()
            or np.any(cv[np.isfinite(cv)] < 0) or np.any(h[np.isfinite(h)] < 0)):
        raise ValueError('Complete, supported nonnegative supervision required')
    out = np.full((len(cv), 2), np.nan)
    value = h[full].copy()
    if arm == 'zero_reference_harm':
        value *= cv[full] == 0
    out[full, 0], out[full, 1] = value > 0, value
    return out


def risk_features(geometry, prediction, scale):
    x, same = cost_features(geometry, prediction, scale)
    xy = np.asarray(geometry)[:, :16].reshape(-1, 8, 2)
    current_stop = np.all(xy[:, -1] == xy[:, -2], axis=1)
    stationary = np.all(xy == xy[:, :1], axis=(1, 2))
    return np.column_stack((x, current_stop, stationary)).astype(np.float32), same


class RiskHead(torch.nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(features, width), torch.nn.GELU(), torch.nn.Linear(width, 2))

    def forward(self, x):
        z = self.network(x)
        return torch.stack((torch.sigmoid(z[:, 0]), torch.nn.functional.softplus(z[:, 1])), 1)


def build_head(features, width, seed):
    torch.manual_seed(seed)
    model = RiskHead(features, width)
    with torch.no_grad():
        last = model.network[-1]
        last.weight.zero_(); last.bias.zero_()
        last.bias[1] = float(np.log(np.expm1(1.)))
    return model


def fit_head(x, y, sites, same, pr, *, seed, arm, settings, identity, directory,
             heartbeat, resume=False, stop_at=None):
    if arm not in ('all_harm', 'zero_reference_harm') or not np.all(y[pr['known'] & same] == 0):
        raise ValueError('Fixed target arm and deterministic zero-risk identity required')
    model = build_head(x.shape[1], settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    targets = np.where(pr['known'][:, None], y, 0).astype(np.float32)
    targets[:, 1] /= pr['cost_scale']
    targets = torch.from_numpy(targets)
    same_t = torch.from_numpy(same)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if cp['identity'] != identity or cp['settings'] != settings or cp['arm'] != arm or cp['seed'] != seed:
            raise ValueError('Changed risk-head resume identity')
        for name in ('mean', 'std', 'constant', 'weights', 'known'):
            np.testing.assert_array_equal(pr[name], cp['preprocess'][name])
        if cp['preprocess']['cost_scale'] != pr['cost_scale']:
            raise ValueError('Changed training risk scale')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng']); torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing fixed budget required')
    def save():
        cp = dict(identity=identity, settings=settings, seed=seed, arm=arm, preprocess=pr,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            step=step, seconds=seconds+time.monotonic()-started, trace=trace, draws=draws)
        tmp = path.with_suffix('.tmp'); torch.save(cp, tmp); os.replace(tmp, path)
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            prediction = torch.where(same_t[ids, None], 0., model(z[ids]))
            bce = torch.nn.functional.binary_cross_entropy(prediction[:, 0], targets[ids, 0])
            mse = torch.nn.functional.mse_loss(prediction[:, 1], targets[ids, 1])
            loss = bce+mse
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite event/cost loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                record = dict(step=step, loss=float(loss.detach()), bce=float(bce.detach()), cost_mse=float(mse.detach()), gradient_norm=float(grad))
                trace.append(record); heartbeat(state='training', **record)
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


def predict_head(model, x, same, pr):
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            p = model(torch.from_numpy(standardized(x[start:start+4096], pr))).numpy()
            p[:, 1] *= pr['cost_scale']
            p[same[start:start+4096]] = 0
            parts.append(p)
    return np.concatenate(parts)
