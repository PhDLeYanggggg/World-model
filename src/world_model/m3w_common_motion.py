"""Learn bounded scene-common motion without changing the relative forecast."""
import numpy as np
import torch
from torch import nn


STEPS, BATCH, CHECKPOINT = 480, 12, 80
VALIDATE = (160, 320, 480)
FIXED = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1./3]*3])


def describe(history, causal_paths):
    """Return shared inputs and a convex bank in OLS4-centered coordinates."""
    v = np.diff(history, axis=1)
    mean = v.mean(0)
    spread = np.sqrt(np.mean(np.sum((v-mean)**2, axis=-1), axis=0))
    speed = np.mean(np.linalg.norm(v, axis=-1), axis=0)
    centres = causal_paths.mean(1)
    offset = centres-centres[:1]
    crowd = history[:, -1] - history[:, -1].mean(0)
    feature = np.r_[np.log1p(len(history)), np.log1p(np.linalg.norm(crowd, axis=-1).mean()),
                    np.log1p(np.linalg.norm(mean, axis=-1)), np.log1p(speed), np.log1p(spread),
                    np.log1p(np.linalg.norm(np.diff(mean, axis=0), axis=-1)),
                    np.log1p(np.linalg.norm(offset[:, [2, 5, 11]], axis=-1)).ravel()]
    return feature, offset


def shifted(ols, offset, weight):
    w = np.asarray(weight)
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1., atol=1e-7):
        raise ValueError('Require a convex common-motion mixture')
    return ols+np.einsum('a,atd->td', w, offset)[None]


def make_net(width):
    return nn.Sequential(nn.Linear(width, 24), nn.Tanh(), nn.Linear(24, 3))


def batch_loss(net, x, offsets, ols, target, valid, query):
    weight = net(x).softmax(-1)
    delta = (weight[:, :, None, None]*offsets).sum(1)[query]
    observed = valid.sum(1)
    error = torch.linalg.vector_norm(ols+delta-target, dim=-1)
    known = (error*valid).sum(1)/observed.clamp_min(1)
    radius = torch.linalg.vector_norm(delta, dim=-1).max(1).values
    row = torch.where(observed > 0, known, radius)
    qsum = row.new_zeros(len(x)).index_add(0, query, row)
    counts = torch.bincount(query, minlength=len(x))
    return (qsum/counts).mean()


@torch.no_grad()
def predict(net, x):
    return net(torch.as_tensor(x, dtype=torch.float32)).softmax(-1).numpy()
