"""Fit-only decision-region weighting for a frozen bounded cost architecture."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_bounded_cost_head import build
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized


def region_weights(selected, pr, multiplier=4.):
    selected = np.asarray(selected)
    known, sampling = np.asarray(pr['known']), np.asarray(pr['weights'])
    if (selected.shape != known.shape or selected.dtype.kind != 'b'
            or known.dtype.kind != 'b' or sampling.shape != known.shape
            or not np.isfinite(sampling).all() or (sampling < 0).any()
            or np.any(sampling[~known] != 0) or not np.isclose(sampling.sum(), 1.)
            or not np.isfinite(multiplier) or multiplier < 1):
        raise ValueError('Frozen causal selections and supported fitting distribution required')
    w = np.where(selected, multiplier, 1.)
    w /= np.dot(sampling, w)
    w[~known] = 0
    return w.astype(np.float32)


def loss_value(score, target, distance, weights):
    if (score.shape != target.shape or target.shape != (len(score), 2)
            or distance.shape != (len(score),) or weights.shape != distance.shape
            or not torch.isfinite(target).all() or not torch.isfinite(distance).all()
            or not torch.isfinite(weights).all() or (target < 0).any()
            or (distance < 0).any() or (weights <= 0).any()):
        raise ValueError('Finite supported costs and positive fixed loss weights required')
    den = torch.where(distance > 0, distance, 1.)
    per_row = (score-target.detach()).square().mean(1)/den
    return (weights.detach()*per_row).mean()


def fit(x, y, distance, sites, pr, loss_weights, *, seed, settings, identity,
        directory, heartbeat, resume=False, stop_at=None):
    loss_weights = np.asarray(loss_weights, np.float32)
    if (loss_weights.shape != (len(x),) or not np.isfinite(loss_weights).all()
            or (loss_weights[pr['known']] <= 0).any() or (loss_weights[~pr['known']] != 0).any()):
        raise ValueError('Fixed supported loss weights required')
    model = build(x.shape[1], settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    d = torch.from_numpy((distance/pr['cost_scale']).astype(np.float32))
    target = torch.from_numpy(np.where(pr['known'][:, None], y/pr['cost_scale'], 0).astype(np.float32))
    weights = torch.from_numpy(loss_weights)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    path = Path(directory)/'checkpoint.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if (cp['identity'] != identity or cp['settings'] != settings or cp['seed'] != seed
                or cp['loss_exponent'] != 1 or cp['forward_arm'] != 'bounded_native'):
            raise ValueError('Changed resume identity')
        np.testing.assert_array_equal(cp['loss_weights'], loss_weights)
        for k in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(cp['preprocess'][k], pr[k])
        for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
            assert cp['preprocess'][k] == pr[k]
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng'])
        torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit <= 0 or limit < step:
        raise ValueError('Nondecreasing fixed budget required')
    def save():
        cp = dict(identity=identity, settings=settings, seed=seed, loss_exponent=1,
            forward_arm='bounded_native', loss_weights=loss_weights, preprocess=pr,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), step=step, seconds=seconds+time.monotonic()-started,
            trace=trace, draws=draws)
        temp = path.with_suffix('.tmp')
        torch.save(cp, temp)
        os.replace(temp, path)
    try:
        model.train()
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            score = model(z[ids], d[ids], 'bounded_native')
            loss = loss_value(score, target[ids], d[ids], weights[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite conditional cost loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row)
                heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step)
        raise
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unknown_rows_sampled=int(draws[~pr['known']].sum()), trace=trace,
        parameters=sum(p.numel() for p in model.parameters()))
