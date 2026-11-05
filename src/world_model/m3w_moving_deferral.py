"""Causal multi-expert routing using Mao et al. (ICML 2024), Eq. (3).

Task adaptation on a fixed forecast bank, not a reproduction of their datasets
or a proof of relative-risk control. No target or target mask enters inference.
"""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_selection_controller as base
from src.world_model.m3w_regression_deferral_loss import two_stage_logistic


class DeferralRouter(nn.Module):
    def __init__(self, dimension, width, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(dimension + 7, width)
        self.action = nn.Linear(width, 1)
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)

    def forward(self, x, segments, count):
        if x.ndim != 3 or x.shape[1] != 7 or segments.shape != (len(x),):
            raise ValueError('Complete aligned causal queries required')
        code = torch.eye(7, dtype=x.dtype, device=x.device).expand(len(x), -1, -1)
        hidden = F.silu(self.encoder(torch.cat((x, code), -1)))
        membership = F.one_hot(segments, count).to(hidden.dtype).T
        total = membership @ hidden.flatten(1)
        sizes = membership.T @ membership.sum(1)
        others = ((membership.T @ total).reshape_as(hidden)-hidden)/(sizes-1).clamp_min(1)[:, None, None]
        return self.action(F.silu(hidden + others)).squeeze(-1)


def legal_actions(allowed):
    legal = np.asarray(allowed).copy()
    if legal.ndim != 2 or legal.shape[1] != 7 or legal.dtype != bool:
        raise ValueError('Seven causal action flags required')
    legal[:, 0] = True
    return legal


def objective(logits, data, ix, segments, count):
    truth = data['truth_by_arm']['bound_aware'][ix]
    observed = torch.from_numpy(data['observed_future'][ix])
    # R - B + H equals the available-label ADE, scaled by past-only geometry.
    costs = truth[..., 2]-truth[..., 0]+truth[..., 1]
    costs = costs.masked_fill(~observed[:, None], float('nan'))
    loss, supervised = two_stage_logistic(logits, costs,
        torch.from_numpy(legal_actions(data['causal']['allowed'][ix])), observed)
    membership = F.one_hot(segments, count).to(logits.dtype).T
    mass = membership @ supervised.float()
    total = ((membership @ loss)/mass.clamp_min(1)).sum()/(mass > 0).sum().clamp_min(1)
    return dict(total=total, supervised_rows=int(supervised.sum()),
        supervised_queries=int((mass > 0).sum()), unknown_outcomes_used=0)


def fit(data, *, seed, settings, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    c, pr = data['causal'], data['preprocess']
    model = DeferralRouter(c['x'].shape[-1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    sampler = base.CyclicQueries(c['groups'], c['source_groups'], settings['query_batch_size'], seed)
    budget = base.steps_for(c, settings)
    binding = dict(identity=identity, settings=settings, seed=seed, preprocess=pr,
        initial_model=initial, parameters=sum(p.numel() for p in model.parameters()), budget=budget,
        source_sites=sorted(set(c['sites'])), fit_ids=c['ids'].copy(), loss='Mao2024_Eq3_logistic')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draws = 0, 0., [], '0'*64
    if path.exists():
        if not resume:
            raise ValueError('Explicit unchanged resume required')
        old = base.read_checkpoint(path)
        for key, value in binding.items():
            base.exact(old[key], value)
        model.load_state_dict(old['model']); optimizer.load_state_dict(old['optimizer'])
        sampler.restore(old['sampler']); torch.set_rng_state(old['torch_rng'])
        step, elapsed, trace, draws = (old[k] for k in ('step', 'seconds', 'trace', 'draw_hash'))
    limit = budget if stop_at is None else min(budget, stop_at)
    if not 0 <= step <= limit or limit <= 0:
        raise ValueError('No budget rollback')
    started = time.monotonic()
    def save():
        guard()
        base.save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=optimizer.state_dict(), sampler=sampler.state(),
            torch_rng=torch.get_rng_state(), trace=trace, draw_hash=draws))
    if step == 0:
        save()
    try:
        while step < limit:
            guard(); ix, seg, qids = sampler.draw(); segments = torch.from_numpy(seg)
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model(data['z'][ix], segments, len(qids)), data, ix, segments, len(qids))
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('Nonfinite deferral objective')
            terms['total'].backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); step += 1
            draws = hashlib.sha256(bytes.fromhex(draws)+qids.tobytes()+ix.tobytes()).hexdigest()
            if step == 1 or step % settings['heartbeat_every'] == 0 or step == budget:
                row = dict(step=step, budget=budget, gradient_norm=float(norm),
                    **{k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in terms.items()},
                    minimum_query_visits=int(sampler.counts.min()), never_drawn_queries=int((sampler.counts == 0).sum()))
                trace.append(row); heartbeat(state='public_deferral_training', **row)
            if step % settings['checkpoint_every'] == 0 or step == limit:
                save()
    except (KeyboardInterrupt, TimeoutError):
        save(); raise
    if step == budget and sampler.counts.min() < settings['largest_source_cycles']:
        raise ValueError('Every original fitting query must receive full coverage')
    return base.read_checkpoint(path)


def exact_state(a, b):
    for key in ('model', 'optimizer', 'sampler', 'torch_rng', 'step', 'trace', 'draw_hash',
                'fit_ids', 'source_sites', 'preprocess', 'initial_model', 'parameters', 'budget'):
        base.exact(a[key], b[key])


def predict(state, causal):
    if set(causal) != base.CAUSAL_KEYS:
        raise ValueError('Only declared causal inputs, never future targets or masks')
    pr = state['preprocess']
    model = DeferralRouter(causal['x'].shape[-1], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval()
    result = np.empty((len(causal['x']), 7), float)
    pending, size = [], 0
    def flush():
        ix = np.concatenate(pending); seg = np.repeat(np.arange(len(pending)), [len(v) for v in pending])
        z = np.clip((causal['x'][ix]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
        z[..., 380:] = 0
        with torch.no_grad():
            result[ix] = model(torch.from_numpy(z), torch.from_numpy(seg), len(pending)).numpy()
    for group in causal['groups']:
        if pending and size+len(group) > 1024:
            flush(); pending, size = [], 0
        pending.append(group); size += len(group)
    if pending:
        flush()
    if not np.isfinite(result).all():
        raise FloatingPointError('Finite logits required')
    return result


def choose(logits, allowed):
    legal = legal_actions(allowed)
    logits = np.asarray(logits)
    if logits.shape != legal.shape or not np.isfinite(logits).all():
        raise ValueError('Finite logits aligned with causal actions required')
    selected = np.where(legal, logits, -np.inf).argmax(1)
    margin = logits[np.arange(len(logits)), selected]-logits[:, 0]
    return np.where(selected == 0, -1, selected), margin
