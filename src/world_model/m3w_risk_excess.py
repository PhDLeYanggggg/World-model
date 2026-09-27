"""Matched cost-head architecture, directly supervised signed budget score."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_geometric_cost_head import initialize_head, predict
from src.world_model.m3w_native_gain_harm import standardized
from src.world_model.m3w_native_forecast import draw_batch


def excess(parts, budget=.02):
    p = np.asarray(parts, float)
    if budget != .02 or p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Two nonnegative score-basis components and fixed2% budget required')
    return p[:, 1]-budget*p[:, 0]


def loss(prediction, target):
    if (prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 2
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Paired nonnegative supervised moments required')
    # Only their signed combination is identified; components are not calibrated moments.
    q = prediction[:, 1]-.02*prediction[:, 0]
    target_q = (target[:, 1]-.02*target[:, 0]).detach()
    return (q-target_q).square().mean()


def fit(x, y, sites, envelope, pr, *, seed, settings, identity, directory, heartbeat,
        resume=False, stop_at=None):
    d = np.asarray(envelope, float)
    if d.shape != (len(x),) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Aligned causal envelope required')
    known = pr['known']
    if np.any(y[known, 1] > d[known]+1e-5):
        raise ValueError('Positive-harm supervision exceeds causal rollout envelope')
    mean_envelope = float(np.dot(pr['weights'], d))
    model = initialize_head(settings['width'], pr, seed, 'all', mean_envelope)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    dt = torch.from_numpy((d/pr['cost_scale']).astype(np.float32))
    target = torch.from_numpy(np.where(known[:, None], y/pr['cost_scale'], 0).astype(np.float32))
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires explicit resume')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['identity'] != identity or state['settings'] != settings
                or state['task'] != 'signed_budget_excess' or state['seed'] != seed):
            raise ValueError('Resume identity mismatch')
        for key in ('mean','std','constant','weights','known'):
            np.testing.assert_array_equal(pr[key], state['preprocess'][key])
        if mean_envelope != state['mean_envelope'] or pr['cost_scale'] != state['preprocess']['cost_scale']:
            raise ValueError('Fitting-only normalizers changed')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draws = state['step'], state['seconds'], state['trace'], state['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0: raise ValueError('Nondecreasing fixed update budget required')
    def save():
        state = dict(identity=identity, settings=settings, task='signed_budget_excess', seed=seed,
            preprocess=pr, mean_envelope=mean_envelope, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), draws=draws, step=step,
            seconds=seconds+time.monotonic()-started, trace=trace)
        temp = path.with_suffix('.tmp'); torch.save(state, temp); os.replace(temp, path)
    model.train()
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            p = model(z[ids], dt[ids]); objective = loss(p, target[ids])
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite risk-excess loss')
            objective.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(objective.detach()), gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step)
        raise
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unique_training_rows=int((draws > 0).sum()), unknown_rows_sampled=int(draws[~known].sum()),
        parameters=sum(p.numel() for p in model.parameters()), mean_envelope=mean_envelope, trace=trace,
        loss_name='MSE_of_signed_budget_excess', component_moments_identified=False)
