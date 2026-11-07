"""Matched-count insertion and deletion in the same causal query."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STEPS, BATCH = 220, 256
CUTS = (.02, .05, .1, .2, .35, .5, .65, .8)


def roster(groups, direct, legal):
    for q in groups:
        drop = q[direct[q] >= 0]
        i, f = np.where(legal[q] & (direct[q, None] < 0))
        add = np.column_stack((q[i], f))
        yield np.column_stack((np.repeat(add, len(drop), axis=0), np.tile(drop, len(add))))


def edits(atoms, direct, known, pairs):
    i, f, j = pairs.T
    add, drop = atoms[i, f], atoms[j, direct[j]]
    # Removing an unknown forecast still costs its disagreement in the paired bound.
    lower = add[:, -1] + np.where(known[j], -drop[:, -1], drop[:, -1])
    return add-drop, lower


def first(options, cut):
    if cut is None:
        return None
    return next((key for key, s in options if s >= cut), None)


def ranked(roster_, scores):
    offset = 0
    for pairs in roster_:
        if len(pairs):
            s = scores[offset:offset+len(pairs)]
            offset += len(pairs)
            j = int(np.argmax(s))
            yield tuple(int(i) for i in pairs[j]), float(s[j])
    if offset != len(scores):
        raise ValueError('Pair scores do not align to the query roster')


class SwapHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.add = nn.Sequential(nn.Linear(width, 24), nn.Tanh())
        self.drop = nn.Sequential(nn.Linear(width, 24), nn.Tanh())
        self.value = nn.Linear(48, 1)
        self.cross = nn.Linear(24, 1, bias=False)
        nn.init.zeros_(self.cross.weight)

    def forward(self, x):
        a, b = self.add(x[:, 0]), self.drop(x[:, 1])
        independent = self.value(torch.cat((a, b), dim=-1)).squeeze(-1)
        joint = independent + self.cross(a*b).squeeze(-1)
        return torch.stack((independent, joint), dim=-1)


def train(x, y, weight, seed, checkpoint, beat):
    torch.manual_seed(seed)
    net = SwapHead(x.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.001)
    identity = [seed, *x.shape, STEPS, BATCH]
    start = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise ValueError('Incompatible swap checkpoint')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start = state['step']
    plan = np.random.default_rng(seed).choice(len(x), (STEPS, BATCH), p=weight/weight.sum())
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    for i in range(start, STEPS):
        j = plan[i]
        loss = F.binary_cross_entropy_with_logits(net(x[j]), y[j, None].expand(-1, 2),
                                                  pos_weight=torch.tensor(8.))
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite swap objective')
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
