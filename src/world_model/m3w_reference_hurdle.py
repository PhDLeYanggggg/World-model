"""Decompose insertion value without changing the recording risk contract."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STEPS, BATCH = 220, 256
ARMS = ('support', 'direct_value', 'hurdle')
CUTS = tuple((p, u) for p in (0., .1, .25, .5) for u in (0., .005, .02, .05))


def targets(atoms, scale):
    observed = atoms[..., 1] > 0
    utility = atoms[..., -1] / scale[:, None]
    easy_gain = np.clip(utility, -3., 3.)
    no_support_harm = np.clip(-utility, 0., 3.)
    return np.stack((observed, easy_gain, no_support_harm), axis=-1)


def objective(out, y):
    e, gain, harm = y.T
    conditional = (e * (out[:, 1]-gain).square()).sum() / e.sum().clamp(min=1.)
    other = ((1-e) * (F.softplus(out[:, 2])-harm).square()).sum() / (1-e).sum().clamp(min=1.)
    direct = F.mse_loss(out[:, 3], e * gain - (1-e) * harm)
    return F.binary_cross_entropy_with_logits(out[:, 0], e) + conditional + other + direct


def predict(net, x):
    with torch.no_grad():
        z = net(torch.from_numpy(x.astype(np.float32, copy=False)))
        p = z[:, 0].sigmoid()
        value = p * z[:, 1] - (1-p) * F.softplus(z[:, 2])
        return torch.stack((p, z[:, 3], value), dim=-1).numpy()


def train(x, y, weight, seed, checkpoint, beat):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, 4))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != [seed, *x.shape, STEPS, BATCH]:
            raise ValueError('Incompatible hurdle checkpoint')
        net.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
    rng = np.random.default_rng(seed)
    plan = rng.choice(len(x), (STEPS, BATCH), p=weight/weight.sum())
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    for i in range(start, STEPS):
        j = plan[i]
        loss = objective(net(x[j]), y[j])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite hurdle objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (i+1) % 55 == 0:
            temp = checkpoint.with_suffix('.pending')
            torch.save(dict(identity=[seed, *x.shape, STEPS, BATCH], step=i+1,
                optimizer=optimizer.state_dict(), model=net.state_dict()), temp)
            temp.replace(checkpoint)
            beat(step=i+1, loss=float(loss.detach()))
    return net.eval()


def options(direct, groups, legal, scores, arm):
    """Return one score-ranked candidate per eligible query, in stream order."""
    rank = ARMS.index(arm)
    for q in groups:
        if (direct[q] >= 0).sum() >= min(2, len(q)-1):
            continue
        row, family = np.where(legal[q])
        if len(row):
            row = q[row]
            s = scores[row, family]
            # The support-only control uses direct value only to break ties.
            j = np.lexsort((family, row, -s[:, 1], -s[:, rank]))[0]
            yield int(row[j]), int(family[j]), float(s[j, 0]), float(s[j, max(rank, 1)])


def first(options, cut):
    if cut is None:
        return None
    return next(((i, k) for i, k, p, u in options if p >= cut[0] and u > cut[1]), None)
