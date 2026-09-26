"""Conditional harm experts; membership probabilities enter composition, never fitting."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch


class CostHead(nn.Module):
    def __init__(self, dimension, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, 2))

    def forward(self, x, envelope):
        return envelope[:, None] * self.network(x).sigmoid()


def moments(values, arm, probability):
    if arm == 'direct':
        return torch.stack((values[:, 0], values[:, 0] * probability), 1)
    if arm != 'conditional':
        raise ValueError('Registered arm required')
    easy = values[:, 0] * probability
    return torch.stack((easy + values[:, 1] * (1-probability), easy), 1)


def objective(model, z, envelope, harm, easy, arm, rms, prevalence):
    raw = model.network(z)
    if arm == 'direct':
        h = envelope * raw[:, 0].sigmoid()
        pred = torch.stack((h, h * raw[:, 1].sigmoid()), 1)
        target = torch.stack((harm, harm * easy), 1)
        return (((pred-target) / rms) ** 2).mean()
    if arm != 'conditional':
        raise ValueError('Registered arm required')
    error = ((envelope[:, None] * raw.sigmoid() - harm[:, None]) / rms) ** 2
    # Fitting prevalence normalizes conditional losses without changing sampler draws.
    return .5 * ((error[:, 0]*easy).mean()/prevalence
                 + (error[:, 1]*(1-easy)).mean()/(1-prevalence))


def fit(x, y, easy, sites, env, pr, *, arm, seed, settings, identity, directory,
        heartbeat, resume=False, stop_at=None):
    known = np.isfinite(y).all(1)
    if (arm not in ('direct', 'conditional') or y.shape != (len(x), 4)
            or not np.array_equal(np.isnan(y).all(1), ~known)
            or not np.array_equal(np.isfinite(easy), known)
            or not np.array_equal(known, pr['known'])
            or not np.isin(easy[known], [0, 1]).all()
            or not np.isfinite(x).all() or not np.isfinite(env).all() or (env < 0).any()
            or (y[known] < 0).any() or (y[known, 1] > env[known]+1e-5).any()):
        raise ValueError('Aligned known/unknown costs and binary supervision required')
    np.testing.assert_allclose(y[known, 3], y[known, 1]*easy[known], atol=1e-8)
    w, scale = pr['weights'], pr['cost_scale']
    e = np.nan_to_num(easy, nan=0.)
    h = np.where(known, y[:, 1]/scale, 0.)
    prevalence = float(w@e)
    if not 0 < prevalence < 1:
        raise ValueError('Both fitting membership strata required')
    if arm == 'direct':
        target = np.column_stack((h, h*e))
        rms = np.sqrt(w@(target**2)).clip(1e-4)
        fractions = [float(w@h)/max(float(w@(env/scale)), 1e-6),
                     float(w@(h*e))/max(float(w@h), 1e-6)]
    else:
        rms = np.sqrt([float(w@(h*h*e))/prevalence,
                       float(w@(h*h*(1-e)))/(1-prevalence)]).clip(1e-4)
        fractions = [float(w@(h*mask))/max(float(w@(env/scale*mask)), 1e-6)
                     for mask in (e, 1-e)]
    fractions = np.clip(fractions, 1e-6, 1-1e-6)
    torch.manual_seed(seed); model = CostHead(x.shape[1], settings['width'])
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor(np.log(fractions/(1-fractions)), dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr)); d = torch.tensor(env/scale, dtype=torch.float32)
    ht, et, rt = [torch.tensor(v, dtype=torch.float32) for v in (h, e, rms)]
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    if any(not len(g) for g in groups): raise ValueError('Each fitting locality needs labels')
    rng = torch.Generator().manual_seed(seed+7919)
    fixed_rng = torch.Generator().set_state(rng.get_state()); fixed = draw_batch(groups, settings['batch_size'], fixed_rng)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        s = torch.load(path, map_location='cpu', weights_only=False)
        for k, v in dict(identity=identity, seed=seed, arm=arm, settings=settings, prevalence=prevalence).items():
            if s[k] != v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean', 'std', 'known', 'weights'): np.testing.assert_array_equal(s['preprocess'][k], pr[k])
        assert s['preprocess']['cost_scale'] == scale
        np.testing.assert_array_equal(s['loss_scales'], rms); np.testing.assert_array_equal(s['fixed_ids'], fixed)
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, seconds, trace, draws = s['step'], s['seconds'], s['trace'], s['draws']
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit <= 0 or limit < step: raise ValueError('Nondecreasing budget required')
    first = step; started = time.monotonic()
    def loss(ids): return objective(model, z[ids], d[ids], ht[ids], et[ids], arm, rt, prevalence)
    def diagnostic():
        with torch.no_grad(): value = loss(fixed)
        return dict(step=step, objective=float(value))
    def save():
        s = dict(identity=identity, seed=seed, arm=arm, settings=settings, prevalence=prevalence,
                 preprocess=pr, loss_scales=rms, fixed_ids=fixed, step=step,
                 seconds=seconds+time.monotonic()-started, trace=trace, draws=draws,
                 model=model.state_dict(), optimizer=optimizer.state_dict(),
                 sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state())
        tmp = path.with_suffix('.tmp'); torch.save(s, tmp); os.replace(tmp, path)
    if not trace: trace.append(diagnostic())
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng); optimizer.zero_grad(set_to_none=True)
            value = loss(ids)
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite cost objective')
            value.backward(); grad = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(diagnostic(), batch_loss=float(value.detach()), gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint', step=step); raise
    model.eval()
    return model, dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
        unknown_rows_sampled=int(draws[~known].sum()), prevalence=prevalence, loss_scales=rms.tolist(), trace=trace)


def predict(model, x, env, pr, *, arm, probability=None):
    if (not np.isfinite(x).all() or not np.isfinite(env).all() or (env < 0).any()
            or env.shape != (len(x),)):
        raise ValueError('Aligned finite causal features and envelope required')
    if arm == 'conditional' and (probability is None or np.shape(probability) != (len(x),)
            or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any()):
        raise ValueError('Aligned causal membership probabilities required')
    if arm not in ('direct', 'conditional'): raise ValueError('Registered arm required')
    out = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], pr))
            d = torch.tensor(env[start:start+4096]/pr['cost_scale'], dtype=torch.float32)
            raw = model.network(z).sigmoid()
            if arm == 'direct':
                value = moments(d[:, None]*raw, arm, raw[:, 1])
            else:
                p = torch.tensor(probability[start:start+4096], dtype=torch.float32)
                value = moments(d[:, None]*raw, arm, p)
            out.append(value.numpy()*pr['cost_scale'])
    pred = np.concatenate(out)
    if (not np.isfinite(pred).all() or (pred < 0).any()
            or (pred[:, 1] > pred[:, 0]+1e-5).any() or (pred[:, 0] > env+1e-4).any()):
        raise FloatingPointError('Nested bounded harm violated')
    return pred


def restore(directory):
    s = torch.load(Path(directory)/'checkpoint.pt', map_location='cpu', weights_only=False)
    model = CostHead(len(s['preprocess']['mean']), s['settings']['width'])
    model.load_state_dict(s['model']); model.eval(); return model, s
