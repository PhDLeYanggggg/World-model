"""Matched temporal auxiliary training with the unchanged primary moment loss."""
import copy
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_inner_separability as core
from src.world_model import m3w_source_forest as forest
from src.world_model import m3w_temporal_target_audit as temporal

ARMS = ('none', 'rowmean', 'temporal')


class AuxiliaryHead(nn.Module):
    def __init__(self, pr, width):
        super().__init__()
        self.primary = core.MomentHead(len(pr['mean']), 'nonlinear', width, pr['bias'])
        self.auxiliary = nn.Linear(width, 24)
        nn.init.zeros_(self.auxiliary.weight)
        nn.init.zeros_(self.auxiliary.bias)

    def forward(self, z, envelope):
        # Both objectives reach the same encoder; the primary decoder is unchanged.
        return self.primary(z, envelope), self.auxiliary(self.primary.encoder(z)).reshape(-1, 12, 2)


def initialize(pr, settings, seed):
    torch.manual_seed(seed)
    return AuxiliaryHead(pr, settings['width'])


def auxiliary_targets(series, arm):
    if arm not in ARMS:
        raise ValueError('Registered auxiliary arm required')
    v, valid = temporal.validate_series(series)
    if arm != 'rowmean':
        return v.copy()
    means = np.divide(np.where(valid[..., None], v, 0).sum(1), valid.sum(1)[:, None],
                      out=np.zeros((len(v), 2)), where=valid.sum(1)[:, None] > 0)
    return np.where(valid[..., None], means[:, None], np.nan)


def storage_status(path, *, reserve, remaining, free=None):
    if reserve < 0 or remaining < 0:
        raise ValueError('Nonnegative storage budget required')
    available = shutil.disk_usage(path).free if free is None else free
    return dict(free_bytes=int(available), reserve_bytes=int(reserve),
                remaining_cap_bytes=int(remaining), required_bytes=int(reserve+remaining),
                shortfall_bytes=int(max(0, reserve+remaining-available)),
                allowed=available >= reserve+remaining)


def require_storage(path, *, reserve, remaining, free=None):
    status = storage_status(path, reserve=reserve, remaining=remaining, free=free)
    if not status['allowed']:
        raise OSError('Checkpoint storage below registered reserve: '+str(status))
    return status


def fit(x, envelope, target, series, sites, recordings, frames, pr, *, arm,
        settings, seed, identity, path, heartbeat, resume=False, stop_at=None,
        checkpoint_guard=None):
    if checkpoint_guard is not None:
        checkpoint_guard()
    if arm not in ARMS or not 0 <= settings['auxiliary_weight'] <= 1:
        raise ValueError('Registered arm and bounded fixed auxiliary weight required')
    fresh = core.preprocess(x, envelope, target, sites, recordings, frames, training_site=pr['training_site'])
    core.exact(pr, fresh)
    v, valid = temporal.validate_series(series)
    known = np.isfinite(target).all(1)
    if len(v) != len(x) or not np.array_equal(known, valid.any(1)):
        raise ValueError('Temporal and primary supervision must have identical row support')
    means = np.where(valid[..., None], v, 0).sum(1)/valid.sum(1).clip(1)[:, None]
    if (not np.allclose(means[known, 0], target[known, 1]-target[known, 0], rtol=1e-7, atol=1e-7)
            or not np.allclose(means[known, 1], target[known, 2], rtol=1e-7, atol=1e-7)):
        raise ValueError('Temporal labels do not reconstruct primary labels')
    hashes = {k: forest.fingerprint(np.asarray(a)) for k, a in dict(
        x=x, envelope=envelope, primary=target, temporal=v, sites=sites,
        recordings=recordings, frames=frames).items()}
    groups, source_groups, keys = core.sampling.query_groups(sites, recordings, frames, known)
    z = torch.from_numpy(np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    env = torch.from_numpy((envelope/pr['scale']).astype(np.float32))
    truth = torch.from_numpy(np.where(known[:, None], target/pr['scale'], 0).astype(np.float32))
    aux_truth = torch.from_numpy((auxiliary_targets(v, arm)/pr['scale']).astype(np.float32))
    mask = torch.from_numpy(valid)
    rms = torch.from_numpy(pr['rms'])
    model = initialize(pr, settings, seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = core.sampling.draw_queries(groups, source_groups, 128, torch.Generator().manual_seed(seed+9137))
    coefficient = 0. if arm == 'none' else settings['auxiliary_weight']
    path = Path(path)
    step, seconds, trace, draw_hash, draws = 0, 0., [], '0'*64, 0
    if path.exists():
        if not resume:
            raise ValueError('Existing checkpoint requires explicit resume')
        old = core.read_checkpoint(path)
        for k, value in dict(identity=identity, settings=settings, arm=arm, preprocess=pr,
                             seed=seed, initial_model=initial, input_hashes=hashes).items():
            core.exact(old[k], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        rng.set_state(old['sampler_rng']); torch.set_rng_state(old['torch_rng'])
        step, seconds, trace, draw_hash, draws = (old[k] for k in ('step', 'seconds', 'trace', 'draw_hash', 'row_draws'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not step <= limit <= settings['steps'] or limit <= 0:
        raise ValueError('Nondecreasing registered training budget required')
    began = time.monotonic()

    def terms(ix, seg, count):
        p, a = model(z[ix], env[ix])
        primary = core.losses(p, truth[ix], torch.from_numpy(seg), count, rms)
        auxiliary = temporal.auxiliary_loss(a, aux_truth[ix], mask[ix], torch.from_numpy(seg), count)
        return dict(primary_total=primary['total'], moments=primary['moments'],
                    decision_scores=primary['decision_scores'], auxiliary=auxiliary,
                    total=primary['total']+coefficient*auxiliary)

    def monitor_loss():
        ix, seg, _ = monitor
        with torch.no_grad():
            return {k:float(v) for k,v in terms(ix, seg, 128).items()}

    def save():
        if checkpoint_guard is not None:
            checkpoint_guard()
        path.parent.mkdir(parents=True, exist_ok=True)
        core.save_checkpoint(path, dict(identity=identity, settings=settings, arm=arm, preprocess=pr,
            seed=seed, initial_model=initial, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), step=step, trace=trace,
            seconds=seconds+time.monotonic()-began, draw_hash=draw_hash, row_draws=draws,
            queries=len(keys), unknown_rows_sampled=0, input_hashes=hashes,
            coefficient=coefficient, primary_objective_unchanged=True))

    if not trace:
        trace.append(dict(step=0, monitor=monitor_loss()))
        save()
    try:
        while step < limit:
            ix, seg, qids = core.sampling.draw_queries(groups, source_groups, settings['query_batch_size'], rng)
            optimizer.zero_grad(set_to_none=True)
            loss = terms(ix, seg, settings['query_batch_size'])
            if not torch.isfinite(loss['total']):
                raise FloatingPointError('Nonfinite matched training loss')
            loss['total'].backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1; draws += len(ix)
            draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
                row = dict(step=step, loss=float(loss['total'].detach()), gradient_norm=float(gradient), monitor=monitor_loss())
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except KeyboardInterrupt:
        # A signal may interrupt AdamW midway. Retain the last atomic snapshot,
        # rather than overwriting it with a partially applied optimizer update.
        raise
    return core.read_checkpoint(path)


def predict(state, x, envelope):
    pr = state['preprocess']
    x, envelope = np.asarray(x), np.asarray(envelope)
    _, support = forest.causal_inputs(x, envelope, pr)
    model = initialize(pr, state['settings'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    primary, aux = [], []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = torch.from_numpy(np.clip((x[start:start+4096]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
            e = torch.from_numpy((envelope[start:start+4096]/pr['scale']).astype(np.float32))
            p, a = model(z, e)
            primary.append(p.numpy().astype(float)*pr['scale'])
            aux.append(a.numpy().astype(float)*pr['scale'])
    if not primary:
        return np.empty((0, 5)), np.empty((0, 12, 2)), support
    return np.concatenate(primary), np.concatenate(aux), support


def assert_matched(states):
    if tuple(s['arm'] for s in states) != ARMS:
        raise ValueError('All three controls in registered order required')
    for other in states[1:]:
        for key in ('identity', 'settings', 'preprocess', 'seed', 'step', 'initial_model',
                    'input_hashes', 'sampler_rng', 'draw_hash', 'row_draws', 'queries'):
            core.exact(states[0][key], other[key])
