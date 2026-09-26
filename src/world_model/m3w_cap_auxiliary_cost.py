"""Nested forecasting-cost heads with cap-event supervision, not a policy."""
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_cap_exceedance as cap
from src.world_model.m3w_native_forecast import draw_batch

ARMS = ('cost_only', 'cap_aux', 'shuffled_aux')


def auxiliary_target(event, sites, arm, seed):
    if arm not in ARMS:
        raise ValueError('Registered arm required')
    event, sites = np.asarray(event, float), np.asarray(sites)
    if event.shape != sites.shape or np.isinf(event).any():
        raise ValueError('Aligned event labels and fitting localities required')
    target = event.copy()
    if arm == 'shuffled_aux':
        rng = np.random.default_rng(seed+104729)
        for site in sorted(set(sites)):
            ids = np.flatnonzero((sites == site) & np.isfinite(target))
            target[ids] = event[rng.permutation(ids)]
    return target


def prepare(x, y, event, sites, outer, envelope, cost_scale):
    y, event, envelope = map(np.asarray, (y, event, envelope))
    if (y.shape != (len(x), 4) or envelope.shape != (len(x),)
            or not np.isfinite(envelope).all() or (envelope < 0).any()
            or not np.isfinite(cost_scale) or cost_scale <= 0):
        raise ValueError('Aligned labels, nonnegative envelope and fitting scale required')
    known = np.isfinite(y).all(1)
    if (not np.array_equal(np.isnan(y).all(1), ~known)
            or not np.array_equal(np.isfinite(event), known & (envelope > 0))
            or (y[known] < 0).any() or (y[known, 3] > y[known, 1]+1e-6).any()
            or (y[known, 1] > envelope[known]+1e-4).any()):
        raise ValueError('Nested labels and known positive-disagreement event required')
    pr = cap.preprocess(x, event, sites, outer)
    target = np.nan_to_num(y[:, [1, 3]]/cost_scale).astype(np.float32)
    pr['cost_scale'] = float(cost_scale)
    pr['loss_scales'] = np.sqrt((pr['weights'][:, None]*target.astype(float)**2).sum(0)).clip(1e-4)
    pr['cost_means'] = (pr['weights'][:, None]*target).sum(0)
    pr['mean_envelope'] = float(pr['weights'] @ (envelope/cost_scale))
    return pr


class CapAuxiliaryCostHead(nn.Module):
    def __init__(self, dimension, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dimension, width), nn.SiLU())
        self.harm = nn.Linear(width, 2)
        self.event = nn.Linear(width, 1)

    def forward(self, x, envelope):
        hidden = self.encoder(x)
        raw = self.harm(hidden)
        harm = envelope*raw[:, 0].sigmoid()
        costs = torch.stack((harm, harm*raw[:, 1].sigmoid()), 1)
        return costs, self.event(hidden)[:, 0]


def objective(costs, logits, target, event, scales, arm):
    if arm not in ARMS:
        raise ValueError('Registered arm required')
    cost = (((costs-target.detach())/scales)**2).mean()
    binary = F.binary_cross_entropy_with_logits(logits, event.detach())
    value = cost + (binary if arm != 'cost_only' else 0*binary)
    return value, dict(cost_loss=float(cost.detach()), auxiliary_BCE=float(binary.detach()))


def fit(x, y, event, sites, outer, envelope, cost_scale, *, arm, seed,
        settings, identity, directory, heartbeat, resume=False, stop_at=None):
    pr = prepare(x, y, event, sites, outer, envelope, cost_scale)
    labels = auxiliary_target(event, sites, arm, seed)
    torch.manual_seed(seed)
    model = CapAuxiliaryCostHead(x.shape[1], settings['width'])
    def logit(p):
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p/(1-p))
    with torch.no_grad():
        model.harm.weight.zero_()
        a, b = pr['cost_means']
        model.harm.bias.copy_(torch.tensor([logit(a/max(pr['mean_envelope'], 1e-6)),
                                            logit(b/max(a, 1e-6))], dtype=torch.float32))
        model.event.weight.zero_()
        model.event.bias.fill_(logit(pr['prevalence']))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(cap.transform(x, pr))
    target = torch.from_numpy(np.nan_to_num(y[:, [1, 3]]/cost_scale).astype(np.float32))
    et = torch.tensor(np.nan_to_num(labels), dtype=torch.float32)
    true_event = torch.tensor(np.nan_to_num(event), dtype=torch.float32)
    env = torch.tensor(envelope/cost_scale, dtype=torch.float32)
    scales = torch.tensor(pr['loss_scales'], dtype=torch.float32)
    groups = [np.flatnonzero(pr['known'] & (sites == s)) for s in pr['training_sites']]
    rng = torch.Generator().manual_seed(seed+7919)
    fixed_rng = torch.Generator().set_state(rng.get_state())
    fixed = draw_batch(groups, settings['batch_size'], fixed_rng)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'checkpoint.pt'
    step, seconds, trace, draws = 0, 0., [], np.zeros(len(x), np.int64)
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        state = torch.load(path, map_location='cpu', weights_only=False)
        for key, value in dict(identity=identity, seed=seed, arm=arm, settings=settings).items():
            if state[key] != value:
                raise ValueError('Resume identity changed: '+key)
        for key in ('mean', 'std', 'weights', 'known', 'loss_scales', 'cost_means'):
            np.testing.assert_array_equal(state['preprocess'][key], pr[key])
        for key in ('cost_scale', 'prevalence', 'mean_envelope'):
            assert state['preprocess'][key] == pr[key]
        np.testing.assert_array_equal(state['fixed'], fixed)
        np.testing.assert_array_equal(state['auxiliary_target'], labels)
        for key in initial:
            assert torch.equal(state['initial_model'][key], initial[key])
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, seconds, trace, draws = state['step'], state['seconds'], state['trace'], state['draws']
    limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit <= 0 or limit < step:
        raise ValueError('Nondecreasing budget required')
    started = time.monotonic(); first = step

    def diagnostic():
        with torch.no_grad():
            cost, logits = model(z[fixed], env[fixed])
            value, parts = objective(cost, logits, target[fixed], et[fixed], scales, arm)
            true_bce = F.binary_cross_entropy_with_logits(logits, true_event[fixed])
        return dict(step=step, objective=float(value), true_cap_BCE=float(true_bce), **parts)

    def save():
        state = dict(identity=identity, seed=seed, arm=arm, settings=settings, preprocess=pr,
            fixed=fixed, auxiliary_target=labels, initial_model=initial, model=model.state_dict(),
            optimizer=optimizer.state_dict(), sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(),
            step=step, seconds=seconds+time.monotonic()-started, trace=trace, draws=draws)
        tmp = path.with_suffix('.tmp'); torch.save(state, tmp); os.replace(tmp, path)

    if not trace:
        trace.append(diagnostic())
    try:
        while step < limit:
            ids = draw_batch(groups, settings['batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            costs, logits = model(z[ids], env[ids])
            loss, _ = objective(costs, logits, target[ids], et[ids], scales, arm)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite auxiliary-cost objective')
            loss.backward()
            grad = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; np.add.at(draws, ids, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == limit:
                row = dict(diagnostic(), gradient_norm=float(grad)); trace.append(row)
                heartbeat(state='training', arm=arm, **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_checkpoint', step=step, arm=arm)
        raise
    model.eval()
    return model, pr, dict(step=step, complete=step == settings['steps'], new_updates=step-first,
        seconds=seconds+time.monotonic()-started, unknown_rows_sampled=int(draws[~pr['known']].sum()),
        trace=trace, parameters=sum(p.numel() for p in model.parameters()))


def predict(model, x, envelope, pr):
    envelope = np.asarray(envelope, float)
    if envelope.shape != (len(x),) or not np.isfinite(envelope).all() or (envelope < 0).any():
        raise ValueError('Aligned causal disagreement envelope required')
    costs, probabilities = [], []; model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(cap.transform(x[start:start+4096], pr))
            env = torch.tensor(envelope[start:start+4096]/pr['cost_scale'], dtype=torch.float32)
            cost, event = model(z, env)
            costs.append(cost.numpy()*pr['cost_scale']); probabilities.append(event.sigmoid().numpy())
    cost, probability = np.concatenate(costs), np.concatenate(probabilities)
    if (not np.isfinite(cost).all() or not np.isfinite(probability).all()
            or (cost < 0).any() or (cost[:, 1] > cost[:, 0]).any()
            or (cost[:, 0] > envelope+1e-4).any()):
        raise FloatingPointError('Finite nested cost invariants violated')
    return cost, probability


def restore(directory):
    state = torch.load(Path(directory)/'checkpoint.pt', map_location='cpu', weights_only=False)
    model = CapAuxiliaryCostHead(len(state['preprocess']['mean']), state['settings']['width'])
    model.load_state_dict(state['model']); model.eval()
    return model, state
