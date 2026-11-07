"""Listwise easy-reference training with matched early-query weighting."""
import numpy as np
import torch
from torch import nn

STEPS = 220
BATCH = 16
EARLY_WEIGHT = 8.


def loss(logits, valid, positive, early, multiplier):
    all_mass = logits.masked_fill(~valid, -torch.inf).logsumexp(-1)
    positive_mass = logits.masked_fill(~positive, -torch.inf).logsumexp(-1)
    w = 1. + (multiplier-1.)*early
    return ((all_mass-positive_mass)*w).sum()/w.sum()


def sample_plan(sources, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(STEPS):
        rows = []
        for s in rng.integers(len(sources), size=BATCH):
            r = int(rng.integers(len(sources[s])))
            rows.append((int(s), r, int(rng.integers(len(sources[s][r])))))
        out.append(rows)
    return out


def batch(rows):
    n, d = max(len(x) for x, _, _ in rows), rows[0][0].shape[1]
    x = np.zeros((len(rows), n, d), np.float32)
    mask = np.zeros((len(rows), n), bool)
    y = mask.copy()
    for i, (v, label, _) in enumerate(rows):
        x[i, :len(v)] = v
        mask[i, :len(v)] = True
        y[i, :len(v)] = label
    early = np.asarray([v[2] for v in rows], np.float32)
    return tuple(torch.from_numpy(a) for a in (x, mask, y, early))


def train(sources, plan, width, seed, multiplier, checkpoint, beat):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(width, 24), nn.Tanh(), nn.Linear(24, 1))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (state['seed'], state['weight'], state['updates']) != (seed, multiplier, len(plan)):
            raise ValueError('Incompatible checkpoint')
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
    for k in range(start, len(plan)):
        x, mask, y, e = batch([sources[s][r][q] for s, r, q in plan[k]])
        objective = loss(net(x).squeeze(-1), mask, y, e, multiplier)
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite listwise loss')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if (k+1) % 55 == 0 or k+1 == len(plan):
            tmp = checkpoint.with_suffix('.pending')
            torch.save(dict(seed=seed, weight=multiplier, updates=len(plan), step=k+1,
                model=net.state_dict(), optimizer=optimizer.state_dict()), tmp)
            tmp.replace(checkpoint)
            beat(step=k+1, loss=float(objective.detach()))
    return net.eval()


def predict(net, x):
    with torch.no_grad():
        return net(torch.from_numpy(x.astype(np.float32))).squeeze(-1).numpy()
