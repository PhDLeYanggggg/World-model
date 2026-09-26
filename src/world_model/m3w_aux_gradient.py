"""Fitting-only task-gradient and isolated AdamW intervention diagnostics."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from src.world_model.m3w_native_forecast import draw_batch

VARIANTS = ('cost_only', 'true_aux', 'shuffled_aux', 'projected_true', 'projected_shuffled')


def sample_rows(known, sites, seed, repeat, batch_size, probe_count):
    """Disjoint row IDs, not independent trajectories or unexposed evaluation data."""
    rng = np.random.default_rng(seed + 104729 * repeat)
    probes, groups = {}, []
    for site in sorted(set(sites)):
        ids = np.flatnonzero(known & (sites == site))
        if len(ids) <= probe_count:
            raise ValueError('Insufficient known fitting rows for disjoint probes')
        chosen = rng.permutation(ids)
        probes[str(site)] = np.sort(chosen[:probe_count])
        groups.append(chosen[probe_count:])
    generator = torch.Generator().manual_seed(seed + 104729 * repeat)
    batch = draw_batch(groups, batch_size, generator)
    assert not np.intersect1d(batch, np.concatenate(list(probes.values()))).size
    return batch, probes


def flat(gradients, indices):
    return torch.cat([gradients[i].detach().reshape(-1).double() for i in indices])


def geometry(main, aux):
    nm, na = float(main.norm()), float(aux.norm())
    dot = float(main @ aux)
    return dict(main_norm=nm, auxiliary_norm=na, dot=dot,
                cosine=dot/(nm*na) if nm > 0 and na > 0 else None,
                norm_ratio=na/nm if nm > 0 else None,
                conflict=dot < 0 if nm > 0 and na > 0 else None)


def projected(main, auxiliary, shared):
    """Remove only the negative shared auxiliary component; never change main."""
    result = [g.clone() for g in auxiliary]
    g, a = flat(main, shared), flat(auxiliary, shared)
    square, dot = float(g @ g), float(g @ a)
    if square > 0 and dot < 0:
        coefficient = dot / square
        for i in shared:
            result[i] = auxiliary[i] - coefficient * main[i]
    return result


def gradients(model, x, envelope, target, event, loss_scales):
    if not torch.isfinite(target).all():
        raise ValueError('Unknown targets must not enter gradient batches')
    pred, logits = model(x, envelope)
    error = ((pred - target.detach()) / loss_scales)**2
    cost = error.mean()
    positive = envelope > 0
    he = error[positive, 3].mean() if positive.any() else pred.sum() * 0
    known = torch.isfinite(event)
    bce = (F.binary_cross_entropy_with_logits(logits[known], event[known].detach())
           if known.any() else logits.sum() * 0)
    params = list(model.parameters())
    out = []
    for loss in (cost, he, bce):
        grad = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        out.append([torch.zeros_like(p) if g is None else g.detach().clone()
                    for p, g in zip(params, grad)])
    return out, dict(cost4=float(cost.detach()), easy_harm_positive=float(he.detach()) if positive.any() else None,
                     auxiliary_BCE=float(bce.detach()) if known.any() else None,
                     known_events=int(known.sum()), positive_events=int((event[known] == 1).sum()))


def probe_losses(model, x, envelope, target, scales):
    with torch.no_grad():
        pred = model(x, envelope)[0]
        error = ((pred.double() - target.double()) / scales.double())**2
    positive = envelope > 0
    return dict(cost4=float(error.mean()), easy_harm_all=float(error[:, 3].mean()),
                easy_harm_positive=float(error[positive, 3].mean()) if positive.any() else None,
                rows=len(x), positive_rows=int(positive.sum()))


def virtual_step(model, optimizer_state, settings, main, auxiliary, shared, variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered intervention')
    out = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(out.parameters(), lr=settings['learning_rate'], weight_decay=.0001)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    aux = projected(main, auxiliary, shared) if variant.startswith('projected_') else auxiliary
    for p, g, a in zip(out.parameters(), main, aux):
        p.grad = g.clone() if variant == 'cost_only' else g + a
    norm = torch.nn.utils.clip_grad_norm_(out.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in out.parameters()):
        raise FloatingPointError('Nonfinite virtual update')
    return out, float(norm)
