"""Chunked fixed-region moments; supervision metadata never enters the network."""
import numpy as np
import torch
from src.world_model import m3w_policy_region_cost as old


def prediction(model, data, actions, rows):
    return model(data['z'][rows][:, actions], data['envelope'][rows][:, actions], actions)


def gradient_vector(model):
    return np.concatenate([np.zeros(p.numel()) if p.grad is None else
        p.grad.detach().double().numpy().ravel().copy() for p in model.parameters()])


def exact_region_pass(model, data, regions, actions, rms, *, chunk_size=512,
                      pool_id=None, channels=old.MOMENTS, backward=False, zero_grad=True):
    """Two-pass chain rule is exact at the current weights, without a large graph."""
    pool, local = regions['pool'], regions['local_action']
    selected = torch.nonzero(pool >= 0 if pool_id is None else pool == pool_id,
                             as_tuple=False).flatten().numpy()
    if not len(selected) or chunk_size <= 0:
        raise ValueError('A nonempty fixed known-label region and positive chunk size are required')
    ids = torch.unique(pool[selected], sorted=True)
    g = len(regions['target'])
    sums = torch.zeros((g, 5), dtype=torch.float64)
    counts = torch.bincount(pool[selected], minlength=g)
    with torch.no_grad():
        for start in range(0, len(selected), chunk_size):
            rows = selected[start:start+chunk_size]
            p = prediction(model, data, actions, rows)
            chosen = p[torch.arange(len(rows)), local[rows]].double()
            sums.index_add_(0, pool[rows], chosen)
    means = sums[ids]/counts[ids, None]
    target = regions['target'][ids].double()
    scale = rms[list(channels)].double()
    normalized = means[:, channels]/scale
    truth = target[:, channels]/scale
    safe = normalized+1e-12
    terms = safe-truth+torch.xlogy(truth, truth)-truth*torch.log(safe)
    if not torch.isfinite(terms).all():
        raise FloatingPointError('Nonfinite exact regional divergence')
    coefficients = torch.zeros((g, 5), dtype=torch.float64)
    derivative = (1-truth/safe)/(len(ids)*len(channels)*scale)
    for k, channel in enumerate(channels):
        coefficients[ids, channel] = derivative[:, k]/counts[ids]
    if backward:
        if zero_grad:
            model.zero_grad(set_to_none=True)
        for start in range(0, len(selected), chunk_size):
            rows = selected[start:start+chunk_size]
            p = prediction(model, data, actions, rows)
            chosen = p[torch.arange(len(rows)), local[rows]]
            (chosen.double()*coefficients[pool[rows]]).sum().backward()
    return dict(loss=float(terms.mean()), pool_ids=ids.tolist(), rows=len(selected),
        means=means.numpy().tolist(), targets=target.numpy().tolist(),
        channels=list(channels), counts=counts[ids].tolist(),
        gradient=gradient_vector(model) if backward else None)


def cosine(a, b):
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/denominator) if denominator > 0 else None
