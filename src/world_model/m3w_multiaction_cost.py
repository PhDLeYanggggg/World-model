"""Past-conditioned action costs; labels are accepted only by training functions."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_causal_descriptor_head import read_checkpoint, save_checkpoint
from src.world_model.m3w_native_forecast import draw_batch
from scripts.replay_m3w_dimensionless_training import exact

ARMS = dict(physical_only=tuple(range(6)), physical_neural=tuple(range(7)), neural_only=(6,))


def validate_arrays(x, envelope, allowed):
    x, envelope, allowed = map(np.asarray, (x, envelope, allowed))
    if (x.ndim != 3 or x.shape[1] != 7 or envelope.shape != x.shape[:2]
            or allowed.shape != envelope.shape or allowed.dtype != bool
            or not np.isfinite(x).all() or not np.isfinite(envelope).all()
            or (envelope < 0).any()):
        raise ValueError('Seven aligned finite causal actions and Boolean eligibility required')


def preprocessing(x, envelope, y, scales, sites):
    """Only physical, fitting-row features/labels set common normalization."""
    x, envelope, y, scales, sites = map(np.asarray, (x, envelope, y, scales, sites))
    validate_arrays(x, envelope, np.ones(envelope.shape, bool))
    if (y.shape != (*envelope.shape, 5) or scales.shape != (len(x),)
            or sites.shape != scales.shape or (scales <= 0).any()
            or not np.isfinite(scales).all()):
        raise ValueError('Fitting-only action moments and frozen positive cost scales required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any():
        raise ValueError('Unknown outcomes must remain wholly unknown')
    names = sorted(set(sites))
    mean, second = np.zeros(x.shape[-1]), np.zeros(x.shape[-1])
    constants, rms, mean_env = [], [], []
    for name in names:
        ix = np.flatnonzero(sites == name)
        for start in range(0, len(ix), 1024):
            at = ix[start:start+1024]
            v = x[at, :6].astype(float)
            mean += v.sum((0, 1))/(len(names)*len(ix)*6)
            second += np.square(v).sum((0, 1))/(len(names)*len(ix)*6)
        take = (sites == name) & known
        if not take.any():
            raise ValueError('Every fitting source needs supported training outcomes')
        values = y[take, :6]/scales[take, None, None]
        constants.append(values.mean((0, 1)))
        rms.append(np.square(values).mean((0, 1)))
        mean_env.append((envelope[take, :6]/scales[take, None]).mean())
    constant = np.mean(constants, axis=0)
    fractions = np.maximum(np.r_[constant[:2], max(0., np.mean(mean_env)-constant[:2].sum())], 1e-5)
    ref = max(float(constant[2]), 1e-5)
    cond = np.clip([constant[3]/ref, constant[4]/max(constant[1], 1e-5)], 1e-5, 1-1e-5)
    bias = np.r_[np.log(fractions[:2]/fractions[2]),
                 ref+np.log(-np.expm1(-ref)), np.log(cond/(1-cond))].astype(np.float32)
    return dict(mean=mean, std=np.sqrt(np.maximum(second-mean*mean, 0)).clip(1e-6),
                rms=np.sqrt(np.mean(rms, axis=0)).clip(.05).astype(np.float32),
                bias=bias, clip=8., source_names=names, fitting_rows=len(x),
                unknown_fitting_rows=int((~known).sum()), neural_fields_set_normalization=False)


def standardized(x, pr):
    return np.clip((x-pr['mean'])/pr['std'], -pr['clip'], pr['clip']).astype(np.float32)


class ActionCostHead(nn.Module):
    def __init__(self, dimension, width, seed, bias):
        super().__init__()
        torch.manual_seed(seed)
        self.network = nn.Sequential(nn.Linear(dimension+7, width), nn.SiLU(), nn.Linear(width, 5))
        with torch.no_grad():
            self.network[-1].weight.zero_()
            self.network[-1].bias.copy_(torch.as_tensor(bias))

    def forward(self, x, envelope, action_ids):
        if x.ndim != 3 or envelope.shape != x.shape[:2] or len(action_ids) != x.shape[1]:
            raise ValueError('Candidate-conditioned causal feature and envelope shapes required')
        code = F.one_hot(action_ids, 7).to(x.dtype).expand(len(x), -1, -1)
        raw = self.network(torch.cat((x, code), dim=-1))
        fractions = torch.softmax(torch.cat((raw[..., :2], torch.zeros_like(raw[..., :1])), -1), -1)
        benefit, harm = (envelope[..., None]*fractions[..., :2]).unbind(-1)
        reference = F.softplus(raw[..., 2])
        benefit = torch.minimum(benefit, reference)
        easy_reference = reference*torch.sigmoid(raw[..., 3])
        easy_harm = harm*torch.sigmoid(raw[..., 4])
        return torch.stack((benefit, harm, reference, easy_reference, easy_harm), -1)


def continuous_loss(prediction, target, real, rms, *, underharm_weight, ranking_weight):
    if (prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 5
            or real.shape != target.shape[:2] or real.dtype != torch.bool
            or not torch.isfinite(target).all() or (target < 0).any()):
        raise ValueError('Known supervised action moments required, never classification labels')
    error = (prediction-target.detach())/rms
    weight = torch.ones_like(error)
    for k in (1, 4):
        weight[..., k] = torch.where(prediction[..., k] < target[..., k], underharm_weight, 1.)
    count = real.sum(1).clamp_min(1)
    regression = ((error.square()*weight).mean(-1)*real).sum(1)/count
    difference = ((prediction[..., 0]-prediction[..., 1])
                  -(target[..., 0]-target[..., 1]))/torch.linalg.vector_norm(rms[:2])
    center = (difference*real).sum(1)/count
    ranking = (torch.square(difference-center[:, None])*real).sum(1)/count
    return dict(total=(regression+ranking_weight*ranking).mean(),
                regression=regression.mean(), ranking=ranking.mean())


def select_actions(prediction, allowed, *, arm, harm_multiplier=1.):
    """Inference has no labels, recording outcomes or future support arguments."""
    p, allowed = np.asarray(prediction, float), np.asarray(allowed)
    if (arm not in ARMS or p.ndim != 3 or p.shape[1:] != (7, 5)
            or allowed.shape != p.shape[:2] or allowed.dtype != bool
            or not np.isfinite(p).all() or (p < 0).any() or harm_multiplier < 1):
        raise ValueError('Finite predicted moments and causal action eligibility required')
    use = allowed.copy()
    use[:, sorted(set(range(7))-set(ARMS[arm]))] = False
    gain = p[..., 0]-p[..., 1]
    use &= ((gain > 0) & (p[..., 2] > 0) & (p[..., 3] > 0)
            & (harm_multiplier*p[..., 1] <= .02*p[..., 2])
            & (harm_multiplier*p[..., 4] <= .02*p[..., 3]))
    scores = np.where(use, gain, -np.inf)
    best = scores.argmax(1)
    chosen = np.where(use.any(1), best, -1)
    return chosen.astype(np.int64), np.where(use.any(1), scores[np.arange(len(p)), best], 0.)


def fit(data, *, arm, settings, seed, identity, path, heartbeat=lambda **_: None,
        guard=lambda: None, resume=False, stop_at=None):
    if arm not in ARMS:
        raise ValueError('Fixed action-bank arm required')
    pr = data['preprocess']; actions = torch.tensor(ARMS[arm])
    model = ActionCostHead(data['z'].shape[-1], settings['width'], seed, pr['bias'])
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+7919)
    binding = dict(arm=arm, settings=settings, seed=seed, identity=identity,
                   preprocess=pr, input_hashes=data['input_hashes'], initial_model=initial,
                   dimension=data['z'].shape[-1], parameters=sum(p.numel() for p in model.parameters()))
    step, elapsed, trace, draw_hash, row_draws = 0, 0., [], '0'*64, 0
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not resume:
            raise ValueError('Explicit resume required')
        state = read_checkpoint(path)
        for key, value in binding.items():
            exact(state[key], value)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampler_rng']); torch.set_rng_state(state['torch_rng'])
        step, elapsed, trace, draw_hash, row_draws = (state[k] for k in
            ('step', 'seconds', 'trace', 'draw_hash', 'row_draws'))
    limit = settings['steps'] if stop_at is None else stop_at
    if not 0 <= step <= limit <= settings['steps'] or limit == 0:
        raise ValueError('Fixed nondecreasing update budget required')
    started = time.monotonic(); rms = torch.from_numpy(pr['rms'])

    def save():
        guard()
        save_checkpoint(path, dict(**binding, step=step, seconds=elapsed+time.monotonic()-started,
            trace=trace, model=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler_rng=rng.get_state(), torch_rng=torch.get_rng_state(), draw_hash=draw_hash,
            row_draws=row_draws, unknown_rows_sampled=0, inner_held_used_for_training=False))

    if step == 0:
        save()
    while step < limit:
        ix = draw_batch(data['groups'], settings['batch_size'], rng)
        prediction = model(data['z'][ix][:, actions], data['envelope'][ix][:, actions], actions)
        terms = continuous_loss(prediction, data['truth'][ix][:, actions], data['allowed'][ix][:, actions], rms,
            underharm_weight=settings['underharm_weight'], ranking_weight=settings['ranking_weight'])
        if not torch.isfinite(terms['total']):
            raise FloatingPointError('Nonfinite action-cost objective')
        optimizer.zero_grad(set_to_none=True); terms['total'].backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); step += 1; row_draws += len(ix)
        draw_hash = hashlib.sha256(bytes.fromhex(draw_hash)+ix.tobytes()).hexdigest()
        if step == 1 or step % settings['heartbeat_every'] == 0 or step == settings['steps']:
            row = dict(step=step, loss=float(terms['total'].detach()), regression=float(terms['regression'].detach()),
                       ranking=float(terms['ranking'].detach()), gradient_norm=float(norm))
            trace.append(row); heartbeat(state='training', arm=arm, **row)
        if step % settings['checkpoint_every'] == 0 or step == limit:
            save()
    return read_checkpoint(path)


def predict(state, x, envelope, scales, *, chunk_size=1024):
    validate_arrays(x, envelope, np.ones(envelope.shape, bool))
    if np.shape(scales) != (len(x),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Frozen fitting-only source scales required')
    model = ActionCostHead(state['dimension'], state['settings']['width'], state['seed'], state['preprocess']['bias'])
    model.load_state_dict(state['model']); model.eval(); parts = []
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            end = start+chunk_size
            z = torch.from_numpy(standardized(x[start:end], state['preprocess']))
            env = torch.from_numpy((envelope[start:end]/scales[start:end, None]).astype(np.float32))
            p = model(z, env, torch.arange(7)).numpy()*scales[start:end, None, None]
            parts.append(p.astype(float))
    return np.concatenate(parts)
