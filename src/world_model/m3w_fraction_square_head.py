"""Forest-objective neural control: distance-weighted two-cost fraction MSE."""
import os
from pathlib import Path
import time
import numpy as np
import torch
from src.world_model.m3w_bounded_cost_head import build
from src.world_model.m3w_forest_cost_head import fit_targets
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized

OBJECTIVE = 'distance_weighted_two_cost_fraction_squared'


def fraction(logits):
    if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
        raise ValueError('Finite paired logits required')
    positive = torch.nn.functional.softplus(logits)
    return positive/(1+positive.sum(1, keepdim=True))


def loss_value(logits, targets, scaled_distance, weights):
    q = fraction(logits)
    if (targets.shape != q.shape or not torch.isfinite(targets).all() or (targets < 0).any()
            or (targets.sum(1) > 1+1e-6).any() or scaled_distance.shape != (len(q),)
            or not torch.isfinite(scaled_distance).all() or (scaled_distance < 0).any()
            or weights.shape != (len(q),) or not torch.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError('Complete fraction targets and positive fixed weights required')
    return (weights.detach()*scaled_distance.detach()*(q-targets.detach()).square().mean(1)).mean()


def fit(x, labels, distance, sites, pr, loss_weights, reference_draws, *, seed, settings,
        identity, directory, heartbeat, resume=False, stop_at=None):
    targets, expected_weight = fit_targets(labels, distance, pr['known'], reference_draws, loss_weights, pr['cost_scale'])
    model = build(x.shape[1], settings['width'], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    d = torch.from_numpy((distance/pr['cost_scale']).astype(np.float32))
    q = torch.from_numpy(targets.astype(np.float32)); weights = torch.from_numpy(np.asarray(loss_weights, np.float32))
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    path = Path(directory)/'checkpoint.pt'; path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing checkpoint requires resume')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if (cp['identity'] != identity or cp['settings'] != settings or cp['seed'] != seed or cp['objective'] != OBJECTIVE):
            raise ValueError('Changed resume identity')
        for key in ('mean', 'std', 'known', 'constant', 'weights'):
            np.testing.assert_array_equal(cp['preprocess'][key], pr[key])
        for key in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
            assert cp['preprocess'][key] == pr[key]
        np.testing.assert_array_equal(cp['loss_weights'], loss_weights)
        np.testing.assert_array_equal(cp['reference_draws'], reference_draws)
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng']); torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 < limit <= settings['steps'] or limit < step:
        raise ValueError('Nondecreasing fixed training budget required')
    def save():
        cp = dict(identity=identity, settings=settings, seed=seed, objective=OBJECTIVE,
            forward_arm='bounded_native', model=model.state_dict(), optimizer=optimizer.state_dict(),
            preprocess=pr, loss_weights=loss_weights, reference_draws=reference_draws, draws=draws,
            expected_empirical_weight=expected_weight, sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            step=step, seconds=seconds+time.monotonic()-started, trace=trace)
        temp = path.with_suffix('.tmp'); torch.save(cp, temp); os.replace(temp, path)
    try:
        model.train()
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value(model.network(z[ids]), q[ids], d[ids], weights[ids])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite fraction-square loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(step=step, loss=float(loss.detach()), gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step)
        raise
    if step == settings['steps']:
        np.testing.assert_array_equal(draws, reference_draws)
    model.eval()
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-started, total_draws=int(draws.sum()),
        unknown_rows_sampled=int(draws[~pr['known']].sum()), parameters=sum(p.numel() for p in model.parameters()), trace=trace)
