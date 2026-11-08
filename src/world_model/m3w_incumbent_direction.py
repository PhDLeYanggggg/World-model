"""Score a change of forecast direction without changing the edited agent."""

import numpy as np
import torch
from torch import nn


class PairedGain(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2*width, 40), nn.Tanh(), nn.Linear(40, 24),
                                   nn.Tanh(), nn.Linear(24, 1))

    def forward(self, a, b):
        # Signed known ADE gain must reverse when the two paths are swapped.
        return (self.score(torch.cat((a, b), -1))-self.score(torch.cat((b, a), -1))).squeeze(-1)*.5


def loss(net, a, b, target):
    return (net(a, b)-target).square().mean()


def lower_gain(known_gain, missing, disagreement):
    return (1.-missing)*known_gain-missing*disagreement


def single_swaps(scores, legal, groups, initial, requested=None):
    """Change at most one direction per query; matched controls may force it."""
    if scores.shape != legal.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid direction scores")
    out, used = initial.copy(), []
    for i, q in enumerate(groups):
        z = np.where(legal[q], scores[q], -np.inf)
        if not len(q) or (requested is not None and not requested[i]):
            used.append(False)
            continue
        flat = int(z.argmax())
        r, a = np.unravel_index(flat, z.shape)
        take = np.isfinite(z[r, a]) and (requested is not None or z[r, a] > 0.)
        if take:
            out[q[r]] = a
        elif requested is not None:
            raise ValueError("Cannot match a swap without legal alternatives")
        used.append(bool(take))
    return out, np.asarray(used)
