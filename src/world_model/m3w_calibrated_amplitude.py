"""Predict additive utility and harm for strictly positive amplitude actions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation.m3w_frozen_amplitude import shrink
from src.evaluation.m3w_recording_forecast_comparison import errors

ALPHA = np.array([.125, .25, .375, .5, 1.])
QUARTER = 1
PRICES = (0., .1, .25, .5, 1., 2., 4.)
STEPS, BATCH = 220, 128


def realized(floor, path, target, valid, selected, weights, easy_cut):
    ref, _ = errors(floor, target, valid)
    known = valid.any(1)
    easy = known & (ref > 0.) & (ref <= easy_cut)
    i = selected
    radius = np.linalg.norm(path-floor[i], axis=-1).max(1)
    gains = np.column_stack([ref[i]-errors(shrink(floor[i], path, a), target[i], valid[i])[0] for a in ALPHA])
    gain = np.where(known[i, None], gains, 0.)
    unknown = np.where(known[i], 0., radius)
    miss = unknown[:, None]*ALPHA
    harm = (-gain).clip(0.)
    target = np.stack((gain-miss, harm+miss, harm*easy[i, None]+miss), axis=-1)
    return dict(target=target, gain=gain, unknown=unknown, signed=-gain*easy[i, None], weights=weights[i],
        reference=np.array([ref[i[known[i]]].sum(), ref[i[easy[i]]].sum(), ref[easy].sum()]))


def observe(real, choice):
    i = np.arange(len(choice))
    all_h, easy_h = real['target'][i, choice, 1:].sum(0)
    signed = real['signed'][i, choice].sum()
    ref = real['reference']
    passes = bool((ref > 0.).all() and (np.array([all_h, easy_h, signed]) <= .02*ref).all())
    easy_bad = bool(ref[2] > 0. and signed > .02*ref[2])
    lower = real['gain'][i, choice]-real['gain'][i, QUARTER] - real['unknown']*np.abs(ALPHA[choice]-.25)
    return passes, easy_bad, float(real['weights']@lower)


def value(pred, price):
    return pred[..., 0]-price*pred[..., 1:].clip(0.).sum(-1)


def choose(pred, price):
    return value(pred, price).argmax(1)


def matched(pred, groups):
    u = value(pred, 1.)
    chosen = np.full(len(pred), QUARTER, dtype=int)
    for q in groups:
        if len(q) == 2:
            pairs = np.array([[1, 1], [0, 2], [2, 0]])
            chosen[q] = pairs[u[q, pairs].sum(1).argmax()]
        elif len(q) != 1:
            raise ValueError('Matched amplitude requires the original at-most-two query budget')
    return chosen


def train(x, y, w, seed, checkpoint, beat):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, len(ALPHA)*3))
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    identity = [seed, *x.shape, STEPS, BATCH]
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Incompatible amplitude checkpoint')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start = state['step']
    plan = np.random.default_rng(seed).choice(len(x), (STEPS, BATCH), p=w/w.sum())
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y.clip(-10., 10.).reshape(len(y), -1), dtype=torch.float32)
    for i in range(start, STEPS):
        j = plan[i]
        loss = F.smooth_l1_loss(net(x[j]), y[j], beta=.1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite amplitude objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1) % 55 == 0:
            tmp = checkpoint.with_suffix('.pending')
            torch.save(dict(identity=identity, step=i+1, model=net.state_dict(),
                            optimizer=opt.state_dict()), tmp)
            tmp.replace(checkpoint)
            beat(step=i+1, loss=float(loss.detach()))
    return net.eval()


@torch.no_grad()
def predict(net, x):
    return net(torch.from_numpy(x)).numpy().reshape(len(x), len(ALPHA), 3)
