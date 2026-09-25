"""Supported-event-first pairing; frozen model, original losses and batch draws."""
import os
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from src.world_model.m3w_ranked_hurdle import ranking_loss as original_ranking_loss

from src.world_model.m3w_hurdle_risk import factor_targets, initialize, objective
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized


def ranking_loss(moments, target, sites, *, epsilon):
    """Apply the frozen loss after removing only undefined event-mass rows."""
    if (moments.ndim != 2 or moments.shape[1] != 2 or target.shape != moments.shape
            or len(sites) != len(target) or np.asarray(sites).ndim != 1
            or not np.isfinite(epsilon) or epsilon <= 0
            or not torch.isfinite(moments).all() or not torch.isfinite(target).all()
            or (moments < 0).any() or (target < 0).any()):
        raise ValueError('Finite nonnegative moments, known training labels and epsilon required')
    supported = target.sum(1) > 0
    loss, info = original_ranking_loss(moments[supported], target[supported],
        np.asarray(sites)[supported.cpu().numpy()], epsilon=epsilon)
    return loss, dict(info, rank_supported_rows=int(supported.sum()),
                      rank_batch_rows=len(target))


def fit(x, y, sites, envelope, pr, *, seed, settings, identity, directory, heartbeat,
        rank_weight, epsilon, resume=False, stop_at=None):
    if not np.isfinite(rank_weight) or rank_weight < 0 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Nonnegative fixed auxiliary weight and positive epsilon required')
    d = np.asarray(envelope, float); f = factor_targets(y, d); known = f['known']
    np.testing.assert_array_equal(known, pr['known'])
    w = np.asarray(pr['weights']); p = float(np.dot(w, f['positive']))
    prior = dict(probability=p, severity=float(np.dot(w, f['fraction'])/p) if p > 0 else 0.)
    model = initialize(pr, prior, settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    dt = torch.from_numpy((d/pr['cost_scale']).astype(np.float32))
    target = torch.from_numpy(np.where(known[:, None], y/pr['cost_scale'], 0).astype(np.float32))
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    fixed_rng = torch.Generator().set_state(rng.get_state())
    fixed_ids = draw_batch(groups, settings['batch_size'], fixed_rng)
    fixed_sha = hashlib.sha256(np.asarray(fixed_ids, dtype=np.int64).tobytes()).hexdigest()
    fixed_trace = []
    pairs_seen = 0
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in dict(identity=identity, settings=settings, seed=seed, prior=prior,
                               rank_weight=rank_weight, epsilon=epsilon).items():
            if state[key] != value:
                raise ValueError('Resume identity mismatch: ' + key)
        for key in ('mean', 'std', 'constant', 'weights', 'known'):
            np.testing.assert_array_equal(state['preprocess'][key], pr[key])
        if state['preprocess']['cost_scale'] != pr['cost_scale']:
            raise ValueError('Changed training scale')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draws = state['step'], state['seconds'], state['trace'], state['draws']
        pairs_seen = state['rank_pairs_seen']
        np.testing.assert_array_equal(state['fixed_ids'], fixed_ids)
        fixed_trace = state['fixed_trace']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0:
        raise ValueError('Nondecreasing fixed training budget required')

    def save():
        state = dict(identity=identity, settings=settings, seed=seed, preprocess=pr, prior=prior,
            rank_weight=rank_weight, epsilon=epsilon, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), draws=draws, step=step,
            seconds=seconds+time.monotonic()-started, trace=trace, rank_pairs_seen=pairs_seen,
            fixed_ids=fixed_ids, fixed_trace=fixed_trace)
        tmp = path.with_suffix('.tmp'); torch.save(state, tmp); os.replace(tmp, path)

    def fixed_diagnostics():
        was_training = model.training
        model.eval()
        with torch.no_grad():
            parts = model.components(z[fixed_ids], dt[fixed_ids])
            base, diagnostics = objective(parts, target[fixed_ids], dt[fixed_ids], 'hurdle')
            rank, info = ranking_loss(parts['moments'], target[fixed_ids], sites[fixed_ids], epsilon=epsilon)
            loss = base if rank_weight == 0 else base + rank_weight * rank
            row = dict(step=step, batch_sha256=fixed_sha, loss=float(loss),
                       ranking_loss=float(rank), **diagnostics, **info)
        model.train(was_training)
        return row

    if not fixed_trace:
        fixed_trace.append(fixed_diagnostics())
    model.train()
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            parts = model.components(z[ids], dt[ids])
            base, diagnostics = objective(parts, target[ids], dt[ids], 'hurdle')
            rank, rdiag = ranking_loss(parts['moments'], target[ids], sites[ids], epsilon=epsilon)
            loss = base if rank_weight == 0 else base + rank_weight * rank
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite ranked hurdle objective')
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1); pairs_seen += rdiag['rank_pairs']
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad),
                           ranking_loss=float(rank.detach()), **diagnostics, **rdiag)
                trace.append(row)
                fixed_trace.append(fixed_diagnostics())
                heartbeat(state='training', fixed_batch_loss=fixed_trace[-1]['loss'], **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_atomic_checkpoint', step=step)
        raise
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()), rank_pairs_seen=pairs_seen,
        unique_training_rows=int((draws > 0).sum()), unknown_rows_sampled=int(draws[~known].sum()),
        parameters=sum(p.numel() for p in model.parameters()), prior=prior, trace=trace, fixed_trace=fixed_trace)
