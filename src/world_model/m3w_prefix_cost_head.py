"""Capacity-matched terminal-repeat versus prefix-profile risk supervision."""
import os
from pathlib import Path
import time

import numpy as np
import torch

from src.world_model.m3w_log_cost_head import composition, log_composition
from src.world_model.m3w_native_forecast import draw_batch
from src.world_model.m3w_native_gain_harm import standardized

ARMS = ('terminal_repeat', 'prefix')
OBJECTIVE = 'distance_weighted_prefix_compositional_log'


class PrefixHead(torch.nn.Module):
    def __init__(self, features, width):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(features, width), torch.nn.GELU(),
                                           torch.nn.Linear(width, 24))

    def forward(self, x, distance):
        if distance.shape != (len(x), 12):
            raise ValueError('Twelve causal disagreement scales required')
        positive = torch.nn.functional.softplus(self.network(x).reshape(-1, 12, 2))
        return distance[..., None] * positive / (1 + positive.sum(-1, keepdim=True))


def build(features, width, seed):
    torch.manual_seed(seed)
    model = PrefixHead(features, width)
    with torch.no_grad():
        model.network[-1].weight.zero_()
        model.network[-1].bias.zero_()
    return model


def arm_arrays(costs, distance, arm):
    if arm not in ARMS or costs.shape != (len(distance), 12, 2) or distance.shape != (len(distance), 12):
        raise ValueError('Registered arm and aligned prefix arrays required')
    if arm == 'terminal_repeat':
        return np.repeat(costs[:, -1:, :], 12, 1), np.repeat(distance[:, -1:], 12, 1)
    return costs, distance


def loss_value(logits, target, distance, weights):
    if (logits.shape != target.shape or logits.shape != (len(weights), 12, 2)
            or distance.shape != logits.shape[:2] or weights.shape != (len(logits),)
            or not torch.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError('Aligned supported prefix supervision and positive weights required')
    q = composition(target.reshape(-1, 2), distance.reshape(-1))
    lp = log_composition(logits.reshape(-1, 2))
    ce = -(q * lp).sum(-1).reshape(-1, 12)
    return (weights.detach()[:, None] * distance.detach() * ce).mean()


def predict(model, x, distance, pr):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(standardized(x[start:start+4096], pr))
            d = torch.from_numpy((distance[start:start+4096] / pr['cost_scale']).astype(np.float32))
            out.append(model(z, d).numpy().astype(float) * pr['cost_scale'])
    return np.concatenate(out)


def fit(x, costs, distance, sites, pr, weights, *, arm, seed, settings, identity,
        directory, heartbeat, resume=False, stop_at=None):
    costs, distance = arm_arrays(costs, distance, arm)
    weights = np.asarray(weights, np.float32)
    if (weights.shape != (len(x),) or not np.isfinite(weights).all()
            or (weights[pr['known']] <= 0).any() or (weights[~pr['known']] != 0).any()):
        raise ValueError('Complete-label fitting support and frozen weights required')
    model = build(x.shape[1], settings['width'], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(standardized(x, pr))
    d = torch.from_numpy((distance / pr['cost_scale']).astype(np.float32))
    y = torch.from_numpy(np.where(pr['known'][:, None, None], costs / pr['cost_scale'], 0).astype(np.float32))
    composition(y[pr['known']].reshape(-1, 2), d[pr['known']].reshape(-1))
    w = torch.from_numpy(weights)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in sorted(set(sites))]
    if any(len(g) == 0 for g in groups):
        raise ValueError('Every admitted fitting source needs complete supervision')
    rng = torch.Generator().manual_seed(seed + 7919)
    path = Path(directory) / 'checkpoint.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires resume')
        cp = torch.load(path, map_location='cpu', weights_only=False)
        if any(cp[k] != v for k, v in dict(identity=identity, settings=settings, seed=seed,
                                          arm=arm, objective=OBJECTIVE).items()):
            raise ValueError('Changed resume identity')
        np.testing.assert_array_equal(cp['loss_weights'], weights)
        for k in ('mean', 'std', 'known', 'weights', 'constant'):
            np.testing.assert_array_equal(cp['preprocess'][k], pr[k])
        for k in ('cost_scale', 'hard_cut', 'positive_easy_cut'):
            assert cp['preprocess'][k] == pr[k]
        model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer'])
        rng.set_state(cp['sampler_rng']); torch.set_rng_state(cp['torch_rng'])
        step, seconds, trace, draws = cp['step'], cp['seconds'], cp['trace'], cp['draws']
    first, started = step, time.monotonic()
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit <= 0 or limit < step:
        raise ValueError('Nondecreasing registered budget required')
    def save():
        cp = dict(identity=identity, settings=settings, seed=seed, arm=arm, objective=OBJECTIVE,
                  loss_weights=weights, preprocess=pr, model=model.state_dict(), optimizer=opt.state_dict(),
                  sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step,
                  seconds=seconds+time.monotonic()-started, trace=trace, draws=draws)
        tmp = path.with_suffix('.tmp'); torch.save(cp, tmp); os.replace(tmp, path)
    try:
        model.train()
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            opt.zero_grad(set_to_none=True)
            loss = loss_value(model.network(z[ids]).reshape(-1, 12, 2), y[ids], d[ids], w[ids])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite prefix cost loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1; np.add.at(draws, ids, 1)
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
                      unknown_rows_sampled=int(draws[~pr['known']].sum()), trace=trace,
                      parameters=sum(p.numel() for p in model.parameters()))
