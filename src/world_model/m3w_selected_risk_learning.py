"""Nested event moments and fitting-only selected-population mean constraints."""
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch

ARMS = ('mean', 'selected')
POLICIES = ('reference', 'raw_neural', 'raw_ridge', 'mean_all', 'mean_dual',
            'mean_scene', 'mean_joint', 'selected_all', 'selected_dual',
            'selected_scene', 'selected_joint', 'selected_hash_matched')


def event_targets(cv, reference, candidate, easy_cut):
    cv, r, p = map(np.asarray, (cv, reference, candidate))
    if (cv.shape != r.shape or r.shape != p.shape or r.ndim != 1
            or not np.array_equal(np.isnan(cv), np.isnan(r))
            or not np.array_equal(np.isnan(r), np.isnan(p)) or not np.isfinite(easy_cut) or easy_cut <= 0
            or any(np.isinf(v).any() or (v[np.isfinite(v)] < 0).any() for v in (cv, r, p))):
        raise ValueError('Aligned known/unknown cost support required')
    easy = (cv > 0) & (cv <= easy_cut); h = np.maximum(p-r, 0)
    y = np.column_stack((r, h, np.where(easy, r, 0), np.where(easy, h, 0)))
    y[~np.isfinite(cv)] = np.nan
    return y


class EventMomentHead(nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, x, envelope):
        if envelope.shape != (len(x),) or not torch.isfinite(envelope).all() or (envelope < 0).any():
            raise ValueError('Finite nonnegative past-only envelope required')
        raw = self.network(x)
        den, harm = F.softplus(raw[:, 0]), envelope*torch.sigmoid(raw[:, 1])
        return torch.stack((den, harm, den*torch.sigmoid(raw[:, 2]), harm*torch.sigmoid(raw[:, 3])), 1)


def objective(pred, target, scales, masks, sites, arm):
    if (arm not in ARMS or pred.shape != target.shape or pred.shape[1] != 4
            or scales.shape != (4,) or (scales <= 0).any() or not torch.isfinite(target).all()
            or not torch.isfinite(pred).all() or (target < 0).any()
            or masks.dtype != torch.bool or masks.shape != (len(target), 3)):
        raise ValueError('Supported labels, positive fitting scales and causal subgroup masks required')
    error = (pred-target.detach())/scales
    mse = error.square().mean(); terms = []
    for site in sorted(set(sites)):
        pop = torch.as_tensor(sites == site)
        for j in range(3):
            use = pop & masks[:, j]
            if use.any(): terms.append(error[use].mean(0).square().mean())
    group = torch.stack(terms).mean() if terms else pred.sum()*0
    loss = mse + (group if arm == 'selected' else 0)
    return loss, dict(moment_mse=float(mse.detach()), selected_group_mse=float(group.detach()),
                      nonempty_groups=len(terms))


def fit(x, y, sites, env, masks, pr, *, arm, seed, settings, identity, directory,
        heartbeat, resume=False, stop_at=None):
    known = np.isfinite(y).all(1)
    np.testing.assert_array_equal(known, pr['known'])
    if (y.shape != (len(x), 4) or masks.dtype != bool or masks.shape != (len(x), 3)
            or not np.array_equal(np.isnan(y).all(1), ~known)
            or (y[known, 1] > env[known]+1e-5).any()
            or (y[known, 2:] > y[known, :2]+1e-5).any()):
        raise ValueError('Invalid nested targets or causal envelope')
    w = pr['weights']; scale = pr['cost_scale']
    target = torch.from_numpy(np.where(known[:, None], y/scale, 0).astype(np.float32))
    means = np.sum(w[:, None]*target.numpy(), axis=0)
    rms = np.sqrt(np.sum(w[:, None]*target.numpy().astype(float)**2, axis=0)).clip(1e-4)
    loss_scales = torch.tensor(rms, dtype=torch.float32)
    torch.manual_seed(seed); model = EventMomentHead(x.shape[1], settings['width'])
    mean_env = float(np.dot(w, env/scale))
    def logit(v):
        v = np.clip(v, 1e-6, 1-1e-6); return np.log(v/(1-v))
    ref = max(means[0], 1e-6)
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([ref+np.log(-np.expm1(-ref)),
            logit(means[1]/max(mean_env, 1e-6)), logit(means[2]/max(means[0], 1e-6)),
            logit(means[3]/max(means[1], 1e-6))], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr)); d = torch.tensor(env/scale, dtype=torch.float32)
    mt = torch.as_tensor(masks, dtype=torch.bool)
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    fixed_rng = torch.Generator().set_state(rng.get_state())
    fixed = draw_batch(groups, settings['batch_size'], fixed_rng)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for k, v in dict(identity=identity, arm=arm, seed=seed, settings=settings).items():
            if saved[k] != v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean', 'std', 'known', 'weights'):
            np.testing.assert_array_equal(saved['preprocess'][k], pr[k])
        np.testing.assert_array_equal(saved['loss_scales'], rms)
        np.testing.assert_array_equal(saved['fixed_ids'], fixed)
        assert saved['preprocess']['cost_scale'] == scale
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, draws = saved['step'], saved['seconds'], saved['trace'], saved['draws']
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0: raise ValueError('Nondecreasing training budget required')
    started = time.monotonic(); first = step
    def diagnostic():
        with torch.no_grad():
            loss, parts = objective(model(z[fixed], d[fixed]), target[fixed], loss_scales,
                                    mt[fixed], sites[fixed], arm)
        return dict(step=step, loss=float(loss), **parts)
    def save():
        value = dict(identity=identity, arm=arm, seed=seed, settings=settings, preprocess=pr,
            loss_scales=rms, fixed_ids=fixed, step=step, seconds=seconds+time.monotonic()-started,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), draws=draws, trace=trace)
        tmp = path.with_suffix('.tmp'); torch.save(value, tmp); os.replace(tmp, path)
    if not trace: trace.append(diagnostic())
    model.train()
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng); optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(model(z[ids], d[ids]), target[ids], loss_scales, mt[ids], sites[ids], arm)
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(diagnostic(), batch_loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_checkpoint', step=step); raise
    model.eval()
    return model, dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, parameters=sum(v.numel() for v in model.parameters()),
        unknown_rows_sampled=int(draws[~known].sum()), total_draws=int(draws.sum()),
        unique_training_rows=int((draws > 0).sum()), trace=trace, loss_scales=rms.tolist())


def predict(model, x, env, pr):
    out = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], pr))
            d = torch.tensor(env[start:start+4096]/pr['cost_scale'], dtype=torch.float32)
            out.append(model(z, d).numpy()*pr['cost_scale'])
    value = np.concatenate(out)
    if not np.isfinite(value).all() or (value < 0).any(): raise FloatingPointError('Invalid inferred moments')
    return value


def decisions(utility, moments, moving, envelope, groups, ids, mode):
    u, move, env, ids = map(np.asarray, (utility, moving, envelope, ids))
    m = np.asarray(moments, float)
    n = len(u)
    if (u.shape != (n, 2) or m.shape != (n, 4) or move.dtype != bool or move.shape != (n,)
            or env.shape != (n,) or not np.isfinite(u).all() or not np.isfinite(m).all()
            or (u < 0).any() or (m < 0).any() or not np.isfinite(env).all() or (env < 0).any()
            or ids.shape != (n,) or len(set(ids)) != n
            or not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n))
            or mode not in ('all', 'dual', 'scene', 'joint')):
        raise ValueError('Finite causal inputs and complete query partition required')
    eligible = move & (env > 0) & (u[:, 0] > u[:, 1]); out = np.zeros(n, bool)
    if mode in ('all', 'dual'):
        out = eligible & (m[:, 0] > 0) & (m[:, 1] <= .02*m[:, 0])
        if mode == 'dual': out &= (m[:, 2] > 0) & (m[:, 3] <= .02*m[:, 2])
        return out
    for query in groups:
        pool = query[eligible[query]]; budget = .02*m[query][:, [0, 2]].sum(0)
        if (budget <= 0).any(): continue
        if mode == 'scene':
            if (m[pool][:, [1, 3]].sum(0) <= budget).all(): out[pool] = True
        else:
            order = pool[np.lexsort((ids[pool], -(u[pool, 0]-u[pool, 1])))]
            used = np.zeros(2)
            for i in order:
                cost = m[i, [1, 3]]
                if (used+cost <= budget).all(): out[i] = True; used += cost
    return out


def scalar_decisions(u, moments, moving, env, groups, ids, mode):
    m = np.asarray(moments, float); out = np.zeros(len(ids), bool)
    eligible = [bool(moving[i] and env[i] > 0 and u[i, 0] > u[i, 1]) for i in range(len(ids))]
    if mode in ('all', 'dual'):
        for i in range(len(ids)):
            out[i] = eligible[i] and m[i, 0] > 0 and m[i, 1] <= .02*m[i, 0]
            if mode == 'dual': out[i] &= m[i, 2] > 0 and m[i, 3] <= .02*m[i, 2]
        return out
    for query in groups:
        pool = [int(i) for i in query if eligible[i]]
        budget = [.02*sum(float(m[i, col]) for i in query) for col in (0, 2)]
        if min(budget) <= 0: continue
        if mode == 'scene':
            if all(sum(float(m[i, col]) for i in pool) <= limit for col, limit in zip((1, 3), budget)):
                out[pool] = True
        else:
            spent = [0., 0.]
            for i in sorted(pool, key=lambda j: (-float(u[j, 0]-u[j, 1]), int(ids[j]))):
                if all(spent[k]+m[i, col] <= budget[k] for k, col in enumerate((1, 3))):
                    out[i] = True
                    for k, col in enumerate((1, 3)): spent[k] += float(m[i, col])
    return out


def matched_hash(utility, moving, env, groups, ids, reference):
    out = np.zeros(len(ids), bool); eligible = moving & (env > 0) & (utility[:, 0] > utility[:, 1])
    for query in groups:
        count = int(reference[query].sum()); pool = query[eligible[query]]
        order = sorted(pool, key=lambda i: hashlib.sha256(f'selected-risk-v1|{ids[i]}'.encode()).hexdigest())
        out[order[:count]] = True
        assert out[query].sum() == count
    return out
