"""Dense candidate costs for fixed-count, quarter-amplitude allocation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation.m3w_frozen_amplitude import shrink
from src.evaluation.m3w_recording_forecast_comparison import errors

PRICES = (0., .25, 1., 4.)
STEPS, BATCH = 440, 256


def targets(floor, candidate, target, valid, rows, families, easy_cut):
    base, _ = errors(floor, target, valid)
    c = floor[rows]
    p = shrink(c, candidate[rows, families], .25)
    known = valid[rows].any(1)
    ref = np.where(known, base[rows], 0.)
    err, _ = errors(p, target[rows], valid[rows])
    gain = np.where(known, ref-err, 0.)
    easy = known & (ref > 0.) & (ref <= easy_cut)
    unknown = np.where(known, 0., np.linalg.norm(p-c, axis=-1).max(1))
    harm = (-gain).clip(0.)
    return np.column_stack((gain-unknown, harm+unknown, harm*easy+unknown, ref, ref*easy))


def allocate(groups, original, costs, legal, price):
    """Exact maximizer of the additive score at the original query count."""
    utility, all_harm, easy_harm, ref, easy_ref = costs.transpose(2, 0, 1)
    score = utility-price*((all_harm.clip(0.)-.02*ref.clip(0.))
                           +(easy_harm.clip(0.)-.02*easy_ref.clip(0.)))
    if not np.isfinite(score[legal]).all():
        raise ValueError('Nonfinite legal allocation score')
    score = np.where(legal, score, -np.inf)
    best = score.argmax(1)
    value = score[np.arange(len(best)), best]
    existing = np.flatnonzero(original >= 0)
    best[existing] = np.where(score[existing, original[existing]] == value[existing],
                              original[existing], best[existing])
    result = np.full_like(original, -1)
    for q in groups:
        k = int((original[q] >= 0).sum())
        if k:
            i = q[np.lexsort((q, original[q] < 0, -value[q]))[:k]]
            if not np.isfinite(value[i]).all():
                raise ValueError('Original action count exceeds legal alternatives')
            result[i] = best[i]
    return result


def train(x, y, weight, seed, checkpoint, beat):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, 5))
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    identity = [seed, *x.shape, STEPS, BATCH]
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Incompatible allocation checkpoint')
        net.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        start = saved['step']
    plan = np.random.default_rng(seed).choice(len(x), (STEPS, BATCH), p=weight/weight.sum())
    x = torch.from_numpy(x)
    y = torch.from_numpy(y.clip(-10., 10.).astype(np.float32))
    for i in range(start, STEPS):
        j = plan[i]
        loss = F.smooth_l1_loss(net(x[j]), y[j], beta=.1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite candidate cost loss')
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
    return net(torch.from_numpy(x)).numpy()
