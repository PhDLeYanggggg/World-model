"""Learn single-insertion feasibility from fitting recordings, not future inputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STEPS, BATCH = 220, 256
ARMS = ('local', 'record', 'prefix_record')


def supported(mass, full, budget=.02):
    ref, easy, harm, easy_harm, signed, _ = np.moveaxis(mass, -1, 0)
    return ((ref > 0) & (easy > 0) & (full > 0) & (harm <= budget*ref)
            & (easy_harm <= budget*easy) & (signed <= budget*full))


def supervision(atoms, direct):
    i = np.flatnonzero(direct >= 0)
    mass = atoms[i, direct[i]].sum(0)
    full = float(atoms[:, 0, 1].sum())
    old = supported(mass, full)
    nonloss = atoms[..., -1] >= 0
    local = nonloss & (atoms[..., 1] > 0)
    rescue = nonloss & ~old & supported(mass+atoms, full)
    return np.stack((local, rescue), axis=-1), mass, full


def prefix(groups, direct, estimated):
    """Each query sees only its earlier original substitutions."""
    n = len(direct)
    out = np.empty((n, 5), dtype=np.float32)
    total = np.zeros(4)
    for j, q in enumerate(groups):
        out[q] = np.r_[np.log1p(j), np.arcsinh(total)]
        old = q[direct[q] >= 0]
        total[0] += len(old)
        total[1:] += estimated[old, direct[old]].sum(0)
    return out


def train(x, y, w, seed, checkpoint, beat):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x.shape[1], 32), nn.Tanh(), nn.Linear(32, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    identity = [seed, *x.shape, STEPS, BATCH]
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Incompatible record-rescue checkpoint')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start = state['step']
    plan = np.random.default_rng(seed).choice(len(x), (STEPS, BATCH), p=w/w.sum())
    x, y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    for i in range(start, STEPS):
        j = plan[i]
        loss = F.binary_cross_entropy_with_logits(net(x[j]), y[j], pos_weight=torch.tensor(8.))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite rescue objective')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i+1) % 55 == 0:
            partial = checkpoint.with_suffix('.pending')
            torch.save(dict(identity=identity, step=i+1, model=net.state_dict(),
                            optimizer=opt.state_dict()), partial)
            partial.replace(checkpoint)
            beat(step=i+1, loss=float(loss.detach()))
    return net.eval()


@torch.no_grad()
def predict(net, x):
    return net(torch.from_numpy(x)).sigmoid().numpy()


def options(groups, legal, scores, arm):
    """One ranked option per query with a common local-score guard."""
    for q in groups:
        i, k = np.where(legal[q])
        if not len(i):
            continue
        i = q[i]
        v = scores[i, k]
        at = np.lexsort((k, i, -v[:, 0], -v[:, arm]))[0]
        yield int(i[at]), int(k[at]), float(v[at, 0]), float(v[at, arm])
