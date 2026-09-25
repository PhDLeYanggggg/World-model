"""Fixed B-only harm-mass sampling with an unchanged expected moment objective."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_selected_risk_learning import EventMomentHead, objective
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized


def probabilities(target, sites, p):
    y, sites, p = np.asarray(target), np.asarray(sites), np.asarray(p, float)
    if (y.shape != (len(p), 4) or sites.shape != p.shape or not np.isfinite(p).all()
            or (p < 0).any() or not np.isclose(p.sum(), 1.) or np.isinf(y).any()):
        raise ValueError('Supported B targets and normalized base probabilities required')
    known = np.isfinite(y).all(1)
    if (not np.array_equal(np.isnan(y).all(1), ~known) or (y[known] < 0).any()
            or (p[known] <= 0).any() or (p[~known] != 0).any()):
        raise ValueError('Unknown future labels are not a sampling stratum')
    q = p.copy()
    for site in sorted(set(sites)):
        take = known & (sites == site); mass = p[take].sum()
        if not take.any() or not np.isclose(mass, 1/len(set(sites))):
            raise ValueError('The unchanged base estimand is equal-site supported rows')
        harm_mass = p[take]*y[take, 3]
        if harm_mass.sum() > 0:
            q[take] = .5*p[take]+.5*mass*harm_mass/harm_mass.sum()
    ratio = np.divide(p, q, out=np.zeros_like(p), where=q > 0)
    assert np.isclose(q.sum(), 1.) and (ratio <= 2+1e-12).all()
    np.testing.assert_allclose(q*ratio, p, rtol=1e-12, atol=1e-15)
    return q, ratio


def per_row_loss(pred, target, scales):
    if (pred.shape != target.shape or pred.shape[1] != 4 or scales.shape != (4,)
            or not torch.isfinite(pred).all() or not torch.isfinite(target).all()
            or not torch.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError('Finite aligned costs and B-only scales required')
    return ((pred-target.detach())/scales).square().mean(1)


def draw(cdf, size, generator):
    u = torch.rand(size, dtype=torch.float64, generator=generator)
    return torch.searchsorted(cdf, u, right=True).numpy()


def initialized(features, target, env, weights, width, seed):
    torch.manual_seed(seed); model = EventMomentHead(features, width)
    means = np.sum(weights[:, None]*target.numpy(), axis=0)
    mean_env = float(np.dot(weights, env)); ref = max(means[0], 1e-6)
    def logit(v):
        v = np.clip(v, 1e-6, 1-1e-6); return np.log(v/(1-v))
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([ref+np.log(-np.expm1(-ref)),
            logit(means[1]/max(mean_env, 1e-6)), logit(means[2]/max(means[0], 1e-6)),
            logit(means[3]/max(means[1], 1e-6))], dtype=torch.float32))
    return model


def fit(x, y, sites, env, masks, pr, *, seed, settings, identity, directory,
        heartbeat, resume=False, stop_at=None, control=None):
    known = pr['known']; np.testing.assert_array_equal(known, np.isfinite(y).all(1))
    if (y[known, 1] > env[known]+1e-5).any() or (y[known, 2:] > y[known, :2]+1e-5).any():
        raise ValueError('Targets must respect nested causal bounds')
    p = pr['weights']; q, ratio = probabilities(y, sites, p); scale = pr['cost_scale']
    target = torch.tensor(np.where(known[:, None], y/scale, 0), dtype=torch.float32)
    rms = np.sqrt(np.sum(p[:, None]*target.numpy().astype(float)**2, axis=0)).clip(1e-4)
    scales = torch.tensor(rms, dtype=torch.float32)
    model = initialized(x.shape[1], target, env/scale, p, settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr)); d = torch.tensor(env/scale, dtype=torch.float32)
    w = torch.tensor(ratio, dtype=torch.float32); mt = torch.as_tensor(masks)
    rng = torch.Generator().manual_seed(seed+7919); fixed_rng = torch.Generator().set_state(rng.get_state())
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    fixed = draw_batch(groups, settings['batch_size'], fixed_rng)
    cdf = torch.tensor(q.cumsum(), dtype=torch.float64); cdf[-1] = 1.
    if control is not None:
        for k in ('mean', 'std', 'known', 'weights'):
            np.testing.assert_array_equal(control['preprocess'][k], pr[k])
        np.testing.assert_array_equal(control['loss_scales'], rms)
        np.testing.assert_array_equal(control['fixed_ids'], fixed)
        with torch.no_grad():
            value, _ = objective(model(z[fixed], d[fixed]), target[fixed], scales,
                mt[fixed], sites[fixed], 'mean')
        assert float(value) == control['trace'][0]['moment_mse']
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Checkpoint requires explicit resume')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for k,v in dict(identity=identity, settings=settings, seed=seed).items(): assert saved[k] == v
        for k,v in dict(probabilities=q, importance_ratio=ratio, loss_scales=rms, fixed_ids=fixed).items():
            np.testing.assert_array_equal(saved[k], v)
        for k in ('mean', 'std', 'known', 'weights'): np.testing.assert_array_equal(saved['preprocess'][k], pr[k])
        assert saved['preprocess']['cost_scale'] == scale
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampler_rng']); torch.set_rng_state(saved['torch_rng'])
        step, seconds, trace, draws = saved['step'], saved['seconds'], saved['trace'], saved['draws']
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0: raise ValueError('Nondecreasing registered budget required')
    first = step; started = time.monotonic()
    def diagnostic():
        with torch.no_grad():
            loss, parts = objective(model(z[fixed], d[fixed]), target[fixed], scales, mt[fixed], sites[fixed], 'mean')
        return dict(step=step, loss=float(loss), **parts)
    def save():
        saved = dict(identity=identity, settings=settings, seed=seed, preprocess=pr, probabilities=q,
            importance_ratio=ratio, loss_scales=rms, fixed_ids=fixed, model=model.state_dict(),
            optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            step=step, seconds=seconds+time.monotonic()-started, draws=draws, trace=trace)
        temp = path.with_suffix('.tmp'); torch.save(saved, temp); os.replace(temp, path)
    if not trace: trace.append(diagnostic())
    model.train()
    try:
        while step < limit:
            ids = draw(cdf, settings['batch_size'], rng); assert known[ids].all()
            optimizer.zero_grad(set_to_none=True)
            loss = (w[ids]*per_row_loss(model(z[ids], d[ids]), target[ids], scales)).mean()
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(diagnostic(), weighted_batch_loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint', step=step); raise
    model.eval(); positive = known & (y[:,3] > 0)
    return model, dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, trace=trace,
        parameters=sum(v.numel() for v in model.parameters()), total_draws=int(draws.sum()),
        unknown_rows_sampled=int(draws[~known].sum()), positive_easy_harm_draws=int(draws[positive].sum()),
        base_positive_probability=float(p[positive].sum()), sampled_positive_probability=float(q[positive].sum()),
        max_importance_ratio=float(ratio.max()), loss_scales=rms.tolist())
