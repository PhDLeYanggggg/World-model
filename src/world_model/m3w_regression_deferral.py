"""Mao et al. ICML2024 Eq.(3) logistic surrogate on frozen forecast costs.

Application adapter, not a reproduction of their tabular experiments or theorem
verification for native partial-label ADE. There is no risk guarantee here.
"""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_multiaction_cost import ARMS, read_checkpoint, save_checkpoint, exact
from src.world_model.m3w_native_forecast import draw_batch


def deferral_loss(logits, costs):
    if (logits.shape != costs.shape or costs.ndim != 2 or costs.shape[1] < 2
            or not torch.isfinite(costs).all() or (costs < 0).any()):
        raise ValueError('Known nonnegative continuous expert costs, not best-class labels, required')
    # Including floor as class0 makes both Eq.(3) coefficients total_cost-cost_j.
    coefficient = costs.sum(1, keepdim=True)-costs
    return -(coefficient.detach()*F.log_softmax(logits, dim=1)).sum(1).mean()/np.log(2.)


class DeferralHead(nn.Module):
    def __init__(self, dimension, width, seed):
        super().__init__(); torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(dimension+8, width), nn.SiLU(), nn.Linear(width, 1))
        with torch.no_grad():
            self.network[-1].weight.zero_(); self.network[-1].bias.zero_()

    def forward(self, features, classes):
        if features.ndim != 3 or features.shape[1] != len(classes):
            raise ValueError('Fixed causal expert feature rows required')
        code = F.one_hot(classes, 8).to(features.dtype).expand(len(features), -1, -1)
        return self.network(torch.cat((features, code), -1)).squeeze(-1)


def choices(scores, allowed, *, arm, risk_allowed=None):
    s, ok = np.asarray(scores, float), np.asarray(allowed)
    if (arm not in ARMS or s.ndim != 2 or s.shape[1] != 8 or ok.shape != (len(s), 7)
            or ok.dtype != bool or not np.isfinite(s).all()):
        raise ValueError('Causal eight-expert scores and real-action eligibility required')
    actions = list(ARMS[arm]); class_ids = np.asarray(actions)+1
    # Merge genuine analytic floor aliases rather than crediting fake interventions.
    floor_score = np.maximum(s[:, 0], np.where(~ok[:, actions], s[:, class_ids], -np.inf).max(1))
    use = ok[:, actions].copy()
    if risk_allowed is not None:
        if np.shape(risk_allowed) != ok.shape or np.asarray(risk_allowed).dtype != bool:
            raise ValueError('Optional fixed predicted-cost guard must be causal Boolean')
        use &= risk_allowed[:, actions]
    available = np.where(use, s[:, class_ids], -np.inf)
    local = available.argmax(1); best = available[np.arange(len(s)), local]
    active = use.any(1) & (best > floor_score)
    return np.where(active, np.asarray(actions)[local], -1).astype(np.int64), np.where(active, best-floor_score, 0.)


def fit(data, *, arm, settings, seed, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    classes = torch.tensor([0, *[k+1 for k in ARMS[arm]]])
    model = DeferralHead(data['z'].shape[-1], settings['width'], seed)
    initial = copy.deepcopy(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    binding = dict(arm=arm, settings=settings, seed=seed, identity=identity,
        preprocess=data['preprocess'], input_hashes=data['input_hashes'], initial_model=initial,
        dimension=data['z'].shape[-1], parameters=sum(p.numel() for p in model.parameters()))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    step, elapsed, trace, draw_hash, row_draws = 0, 0., [], '0'*64, 0
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        state = read_checkpoint(path)
        for key, value in binding.items():
            exact(state[key], value)
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, elapsed, trace, draw_hash, row_draws = (state[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit == 0:
        raise ValueError('Fixed nondecreasing surrogate update budget required')
    started = time.monotonic()

    def save():
        guard()
        save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            model=model.state_dict(), optimizer=opt.state_dict(), sampler_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(), trace=trace, draw_hash=draw_hash, row_draws=row_draws,
            unknown_rows_sampled=0, inner_held_used=False))

    if step == 0:
        save()
    while step < limit:
        ids = draw_batch(data['groups'], settings['batch_size'], rng)
        score = model(data['z'][ids][:, classes], classes)
        loss = deferral_loss(score, data['cost'][ids][:, classes])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite regression-deferral objective')
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        opt.step(); step += 1; row_draws += len(ids)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+ids.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            event = dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm))
            trace.append(event); heartbeat(state='training', arm=arm, **event)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return read_checkpoint(path)


def predict(state, features, *, chunk_size=1024):
    if features.ndim != 3 or features.shape[1] != 8 or not np.isfinite(features).all():
        raise ValueError('Eight finite causal expert features required')
    model = DeferralHead(state['dimension'], state['settings']['width'], state['seed'])
    model.load_state_dict(state['model']); model.eval(); output = []
    pr = state['preprocess']
    with torch.no_grad():
        for start in range(0, len(features), chunk_size):
            z = np.clip((features[start:start+chunk_size]-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)
            output.append(model(torch.from_numpy(z), torch.arange(8)).numpy().astype(float))
    return np.concatenate(output)
