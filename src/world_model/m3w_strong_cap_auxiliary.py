"""Cap-event supervision retaining the original four-cost estimator and sampler."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_membership_auxiliary as original
from src.world_model.m3w_cap_auxiliary_cost import auxiliary_target, ARMS
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch


def objective(pred, logits, target, event, scales, arm):
    if arm not in ARMS:
        raise ValueError('Registered arm required')
    known = torch.isfinite(event)
    cost = (((pred-target.detach())/scales)**2).mean()
    binary = (F.binary_cross_entropy_with_logits(logits[known], event[known].detach())
              if known.any() else logits.sum()*0)
    value = cost + (binary if arm != 'cost_only' else 0*binary)
    return value, dict(cost_loss=float(cost.detach()), auxiliary_BCE=float(binary.detach()))


def validate(x, y, easy, event, sites, outer, env, pr):
    known = np.isfinite(y).all(1)
    if (y.shape != (len(x), 4) or easy.shape != (len(x),) or event.shape != (len(x),)
            or env.shape != (len(x),) or sites.shape != (len(x),)
            or outer in sites or sorted(set(sites)) != sorted(pr['training_sites'])
            or not np.array_equal(np.isnan(y).all(1), ~known)
            or not np.array_equal(known, pr['known'])
            or not np.array_equal(np.isfinite(easy), known)
            or not np.array_equal(np.isfinite(event), known & (env > 0))
            or not np.isin(easy[known], [0, 1]).all()
            or not np.isin(event[np.isfinite(event)], [0, 1]).all()
            or not np.isfinite(x).all() or not np.isfinite(env).all() or (env < 0).any()
            or (y[known] < 0).any() or (y[known, 1] > env[known]+1e-5).any()
            or (y[known, 2:] > y[known, :2]+1e-5).any()):
        raise ValueError('Original known cost support and positive-envelope auxiliary labels required')
    np.testing.assert_allclose(y[known, 3], y[known, 1]*easy[known], atol=1e-8)
    w = pr['weights']
    if (not np.isfinite(w).all() or (w < 0).any() or w[~known].any()
            or not np.isclose(w.sum(), 1) or not np.isfinite(pr['cost_scale']) or pr['cost_scale'] <= 0):
        raise ValueError('Original fitting-only weights and scale required')
    return known


def fit(x, y, easy, event, sites, outer, env, pr, *, arm, seed, settings, identity,
        directory, heartbeat, resume=False, stop_at=None):
    known = validate(x, y, easy, event, sites, outer, env, pr)
    labels = auxiliary_target(event, sites, arm, seed)
    w, scale = pr['weights'], pr['cost_scale']
    prevalence = float(w@np.nan_to_num(easy, nan=0.))
    if not 0 < prevalence < 1:
        raise ValueError('Both original fitting strata required')
    target = torch.from_numpy(np.where(known[:, None], y/scale, 0).astype(np.float32))
    means = np.sum(w[:, None]*target.numpy(), axis=0)
    rms = np.sqrt(np.sum(w[:, None]*target.numpy().astype(float)**2, axis=0)).clip(1e-4)
    rt = torch.tensor(rms, dtype=torch.float32)
    et, true_event = torch.tensor(labels, dtype=torch.float32), torch.tensor(event, dtype=torch.float32)
    torch.manual_seed(seed); model = original.AuxiliaryCostHead(x.shape[1], settings['width'])
    def logit(v):
        v = np.clip(v, 1e-6, 1-1e-6); return np.log(v/(1-v))
    ref = max(means[0], 1e-6); mean_env = float(w@(env/scale))
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.copy_(torch.tensor([ref+np.log(-np.expm1(-ref)),
            logit(means[1]/max(mean_env, 1e-6)), logit(means[2]/max(means[0], 1e-6)),
            logit(means[3]/max(means[1], 1e-6))], dtype=torch.float32))
        # Preserve the original initialization, including its unused auxiliary intercept.
        model.membership.weight.zero_(); model.membership.bias.fill_(logit(prevalence))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr)); d = torch.tensor(env/scale, dtype=torch.float32)
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    if any(not len(g) for g in groups): raise ValueError('Every fitting locality needs costs')
    rng = torch.Generator().manual_seed(seed+7919)
    fixed_rng = torch.Generator().set_state(rng.get_state())
    fixed = draw_batch(groups, settings['batch_size'], fixed_rng)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        s = torch.load(path, map_location='cpu', weights_only=False)
        for k, v in dict(identity=identity, seed=seed, arm=arm, settings=settings, prevalence=prevalence).items():
            if s[k] != v: raise ValueError('Resume identity changed: '+k)
        for k in ('mean', 'std', 'known', 'weights'):
            np.testing.assert_array_equal(s['preprocess'][k], pr[k])
        assert s['preprocess']['cost_scale'] == scale
        for k, a in (('loss_scales', rms), ('fixed_ids', fixed), ('auxiliary_target', labels)):
            np.testing.assert_array_equal(s[k], a)
        for k, a in initial.items(): assert torch.equal(a, s['initial_model'][k])
        model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, seconds, trace, draws = s['step'], s['seconds'], s['trace'], s['draws']
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit <= 0 or limit < step: raise ValueError('Nondecreasing budget required')
    first = step; started = time.monotonic()
    def diagnostic():
        with torch.no_grad():
            pred, logits = model(z[fixed], d[fixed])
            value, parts = objective(pred, logits, target[fixed], et[fixed], rt, arm)
            _, actual = objective(pred, logits, target[fixed], true_event[fixed], rt, arm)
        return dict(step=step, objective=float(value), true_cap_BCE=actual['auxiliary_BCE'], **parts)
    def save():
        s = dict(identity=identity, seed=seed, arm=arm, settings=settings, prevalence=prevalence,
            preprocess=pr, loss_scales=rms, fixed_ids=fixed, step=step,
            auxiliary_target=labels, initial_model=initial,
            seconds=seconds+time.monotonic()-started, trace=trace, draws=draws,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state())
        tmp = path.with_suffix('.tmp'); torch.save(s, tmp); os.replace(tmp, path)
    if not trace: trace.append(diagnostic())
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng); optimizer.zero_grad(set_to_none=True)
            pred, logits = model(z[ids], d[ids])
            value, _ = objective(pred, logits, target[ids], et[ids], rt, arm)
            if not torch.isfinite(value): raise FloatingPointError('Nonfinite strong auxiliary loss')
            value.backward(); grad = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(diagnostic(), gradient_norm=float(grad)); trace.append(row)
                heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint', step=step); raise
    model.eval()
    return model, dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
        unknown_rows_sampled=int(draws[~known].sum()), zero_envelope_rows_sampled=int(draws[env == 0].sum()),
        prevalence=prevalence, loss_scales=rms.tolist(), trace=trace)


predict, restore = original.predict, original.restore
