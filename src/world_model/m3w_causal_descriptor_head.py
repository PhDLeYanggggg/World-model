"""Causal descriptor branch for a matched frozen-floor signed-risk experiment."""
import gzip
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.world_model import m3w_fixed_floor_excess as parent
from src.world_model.m3w_fixed_floor_slices import causal_axes
from src.world_model.m3w_native_forecast import draw_batch
from scripts.replay_m3w_dimensionless_training import exact

FEATURES = ('mean_step_over_extent', 'last_step_over_extent', 'path_nonlinearity',
    'mean_turn_radians', 'neighbor_occupancy', 'rollout_disagreement_over_extent')


def descriptors(geometry, floor, candidate, x, pr):
    axes = causal_axes(geometry, floor, candidate, x, pr, 1.)
    return np.column_stack([axes[k] for k in FEATURES]).astype(np.float32)


def descriptor_preprocess(u, pr):
    u = np.asarray(u, float); w = np.asarray(pr['weights'])
    if (u.shape != (len(w), len(FEATURES)) or not np.isfinite(u).all() or
            not np.isclose(w.sum(), 1) or np.any(w[~pr['known']] != 0)):
        raise ValueError('Six finite causal fitting descriptors and source-balanced weights required')
    mean = (u*w[:, None]).sum(0)
    std = np.sqrt(((u-mean)**2*w[:, None]).sum(0)).clip(1e-6)
    return dict(mean=mean, std=std, clip=pr['clip'], training_sites=pr['training_sites'])


class DescriptorHead(nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.network = base.network
        self.descriptor = nn.Linear(len(FEATURES), width, bias=False)
        with torch.no_grad(): self.descriptor.weight.zero_()

    def forward(self, x, envelope, u):
        h = self.network[0](x)+self.descriptor(u)
        raw = self.network[2](self.network[1](h))
        return torch.stack((F.softplus(raw[:, 0]), envelope*torch.sigmoid(raw[:, 1]),
            F.softplus(raw[:, 2]), envelope*torch.sigmoid(raw[:, 3])), 1)


def initialize(pr, width, seed, mean_envelope):
    base = parent.initialize(pr, width, seed, mean_envelope)
    rng = torch.get_rng_state()
    model = DescriptorHead(base, width)
    torch.set_rng_state(rng)
    return model


def read_checkpoint(path):
    with gzip.open(path, 'rb') as f: return torch.load(f, map_location='cpu', weights_only=False)


def save_checkpoint(path, state):
    path = Path(path); tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as compressed:
            torch.save(state, compressed)
        raw.flush(); os.fsync(raw.fileno())
    os.replace(tmp, path)


def predict(model, x, u, envelope, pr, upr):
    x, u, env = map(np.asarray, (x, u, envelope))
    if (u.shape != (len(x), len(FEATURES)) or env.shape != (len(x),) or
            not all(np.isfinite(a).all() for a in (x, u, env)) or (env < 0).any()):
        raise ValueError('Only finite causal inputs and envelope required')
    out = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            z = np.clip((x[start:start+4096].astype(float)-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
            v = np.clip((u[start:start+4096].astype(float)-upr['mean'])/upr['std'], -upr['clip'], upr['clip']).astype(np.float32)
            p = model(torch.from_numpy(z), torch.from_numpy((env[start:start+4096]/pr['cost_scale']).astype(np.float32)), torch.from_numpy(v))
            out.append((p.numpy()*pr['cost_scale']).astype(np.float32))
    return np.concatenate(out)


def assert_matched(state, control):
    for k in ('settings', 'preprocess', 'seed', 'step', 'draws', 'sampler_rng', 'torch_rng', 'mean_envelope'):
        exact(state[k], control[k])
    for k, value in control['initial_model'].items(): exact(state['initial_model'][k], value)
    assert torch.count_nonzero(state['initial_model']['descriptor.weight']) == 0
    assert control['objective'] == 'fixed_floor_signed_excess'


def fit(x, u, y, sites, envelope, pr, *, seed, settings, identity, directory, heartbeat, resume=False, stop_at=None):
    x, u, y, sites, env = map(np.asarray, (x, u, y, sites, envelope)); known = pr['known']
    if (not np.array_equal(np.isfinite(y).all(1), known) or set(sites) != set(pr['training_sites']) or
            not np.isfinite(env).all() or (env < 0).any() or (y[known][:, [1, 3]] > env[known, None]+1e-5).any()):
        raise ValueError('Matched fitting-only targets and causal envelope required')
    upr = descriptor_preprocess(u, pr)
    mean_envelope = float(np.dot(pr['weights'], env))
    model = initialize(pr, settings['width'], seed, mean_envelope)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    z = torch.from_numpy(np.clip((x.astype(float)-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32))
    v = torch.from_numpy(np.clip((u.astype(float)-upr['mean'])/upr['std'], -upr['clip'], upr['clip']).astype(np.float32))
    target = torch.from_numpy(np.where(known[:, None], (y/pr['cost_scale']).astype(np.float32), 0).astype(np.float32))
    dt = torch.from_numpy((env/pr['cost_scale']).astype(np.float32))
    groups = [np.flatnonzero(known & (sites == s)) for s in sorted(set(sites))]
    rng = torch.Generator().manual_seed(seed+7919)
    monitor = draw_batch(groups, 1024, torch.Generator().manual_seed(seed+9137))
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); path = directory/'checkpoint.pt.gz'
    step = 0; seconds = 0.; trace = []; draws = np.zeros(len(x), np.int64)
    if path.exists():
        if not resume: raise ValueError('Existing head requires --resume')
        s = read_checkpoint(path)
        if s['identity'] != identity or s['settings'] != settings or s['seed'] != seed:
            raise ValueError('Resume changes the registered experiment')
        for a, b in ((s['preprocess'], pr), (s['descriptor_preprocess'], upr), (s['initial_model'], initial), (s['mean_envelope'], mean_envelope)): exact(a, b)
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        rng.set_state(s['sampler_rng']); torch.set_rng_state(s['torch_rng'])
        step, seconds, trace, draws = s['step'], s['seconds'], s['trace'], s['draws']
    first = step; start = time.monotonic(); limit = settings['steps'] if stop_at is None else min(stop_at, settings['steps'])
    if limit < step or limit <= 0: raise ValueError('Nondecreasing training budget required')
    def save():
        save_checkpoint(path, dict(identity=identity, settings=settings, seed=seed,
            objective='fixed_floor_signed_excess_plus_causal_descriptors', preprocess=pr, descriptor_preprocess=upr,
            model=model.state_dict(), initial_model=initial, optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), draws=draws, step=step, seconds=seconds+time.monotonic()-start,
            trace=trace, mean_envelope=mean_envelope))
    model.train()
    try:
        while step < limit:
            ix = draw_batch(groups, settings['batch_size'], rng)
            opt.zero_grad(set_to_none=True)
            objective = parent.loss(model(z[ix], dt[ix], v[ix]), target[ix])
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite matched descriptor objective')
            objective.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            opt.step(); step += 1; np.add.at(draws, ix, 1)
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
                with torch.no_grad(): fixed = float(parent.loss(model(z[monitor], dt[monitor], v[monitor]), target[monitor]))
                row = dict(step=step, loss=float(objective.detach()), fixed_training_excess_MSE=fixed, gradient_norm=float(grad))
                trace.append(row); heartbeat(state='training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit: save()
    except BaseException:
        heartbeat(state='interrupted_resume_last_atomic_checkpoint', step=step); raise
    return model, dict(step=step, new_updates=step-first, complete=step == settings['steps'],
        seconds=seconds+time.monotonic()-start, total_draws=int(draws.sum()), unknown_rows_sampled=int(draws[~known].sum()),
        trace=trace, parameters=sum(p.numel() for p in model.parameters()), component_moments_identified=False)
