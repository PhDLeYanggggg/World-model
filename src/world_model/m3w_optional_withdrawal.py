"""Past-only insertion and optional, conditional withdrawal."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_record_rescue import supported

STEPS, BATCH = 220, 256
INSERT_CUTS = (.02, .05, .1, .2, .35, .5, .65, .8)
DROP_CUTS = (None, .1, .2, .35, .5, .65, .8)


def teach(add, drop, drop_gain, add_q, drop_q, add_crowd, drop_crowd, base, full, rng):
    """Return reachable insertions and sampled conditional teacher pairs."""
    old = bool(supported(base, full))
    observed = add[:, 1] > 0.
    solo = observed & supported(base+add, full) & (add[:, -1] >= 0.) & (not old)
    reach = solo.copy()
    pairs, labels = [], []
    if not len(drop):
        return reach, np.empty((0, 2), dtype=int), np.empty(0, dtype=bool)
    for batch in np.array_split(np.flatnonzero(observed), 1+int(observed.sum())//128):
        legal = drop_q[None, :] >= add_q[batch, None]
        gain = add[batch, None, -1]/add_crowd[batch, None] + (drop_gain/drop_crowd)[None, :]
        after = supported(base+add[batch, None, :]-drop[None, :, :], full)
        yes = after & legal & (gain >= 0.) & ~solo[batch, None] & (not old)
        reach[batch] |= yes.any(1)
        for row, allow, pos in zip(batch, legal, yes):
            for candidates, label in ((np.flatnonzero(pos), True), (np.flatnonzero(allow & ~pos), False)):
                if len(candidates):
                    pairs.append((row, int(rng.choice(candidates))))
                    labels.append(label)
    for row in np.flatnonzero(~observed)[::16]:
        legal = np.flatnonzero(drop_q >= add_q[row])
        if len(legal):
            pairs.append((row, int(rng.choice(legal))))
            labels.append(False)
    return reach, np.asarray(pairs, dtype=int).reshape(-1, 2), np.asarray(labels, dtype=bool)


def pair_features(insert, withdraw, gap):
    return np.column_stack((np.broadcast_to(insert, (len(withdraw), len(insert))),
                            withdraw, np.log1p(gap)))


def plan(y, weight, seed):
    rng = np.random.default_rng(seed)
    draw = rng.choice(len(y), (STEPS, BATCH), p=weight/weight.sum())
    positive = np.flatnonzero(y)
    if len(positive):
        w = weight[positive]
        draw[:, :BATCH//4] = rng.choice(positive, (STEPS, BATCH//4), p=w/w.sum())
    return draw


def train(x, y, w, seed, checkpoint, beat):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    identity = [seed, *x.shape, STEPS, BATCH]
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Incompatible withdrawal checkpoint')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start = state['step']
    draw = plan(y, w, seed)
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    for k in range(start, STEPS):
        i = draw[k]
        loss = F.binary_cross_entropy_with_logits(net(x[i]).squeeze(-1), y[i])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite continuation objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (k+1) % 55 == 0:
            partial = checkpoint.with_suffix('.pending')
            torch.save(dict(identity=identity, step=k+1, optimizer=opt.state_dict(),
                            model=net.state_dict()), partial)
            partial.replace(checkpoint)
            beat(step=k+1, loss=float(loss.detach()))
    return net.eval()


@torch.no_grad()
def predict(net, x):
    return net(torch.from_numpy(x)).squeeze(-1).sigmoid().numpy()


def first(rows, query, score, cut):
    """One top action per query, then the first exceeding the fixed cutoff."""
    if cut is None:
        return None
    bounds = np.r_[0, np.flatnonzero(np.diff(query))+1, len(query)]
    for a, b in zip(bounds, bounds[1:]):
        if a == b:
            continue
        best = a+np.argmax(score[a:b])
        if score[best] >= cut:
            return int(rows[best])
    return None
