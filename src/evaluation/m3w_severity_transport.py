"""Frozen cost-error accounting, radial support and fitting gradient concentration."""
import numpy as np
import torch
from torch.nn import functional as F
from src.evaluation.m3w_source_gradient_diagnostic import flat_gradient


def concentration(mass):
    v = np.asarray(mass, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError('Finite nonnegative group masses required')
    total = float(v.sum()); ordered = np.sort(v)[::-1]
    return dict(groups=len(v), positive_groups=int((v > 0).sum()), total=total,
        top1_share=float(ordered[0]/total) if total else None,
        top5_share=float(ordered[:5].sum()/total) if total else None,
        mass_ESS=float(total**2/(v@v)) if total else None)


def grouped_mass(values, keys):
    keys = np.asarray(keys)
    if keys.ndim == 1: keys = keys[:, None]
    if keys.ndim != 2 or len(keys) != len(values): raise ValueError('Aligned group keys required')
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return np.bincount(inverse, weights=values)


def error_accounting(new, old, target, recordings, agents, easy, outside):
    n = len(target)
    if any(np.asarray(v).shape != (n,) for v in (new, old, target, recordings, agents, easy, outside)):
        raise ValueError('Aligned one-dimensional rows required')
    known = np.isfinite(target)
    if not np.isfinite(new).all() or not np.isfinite(old).all(): raise ValueError('Finite predictions required')
    if outside.dtype != bool or not np.isin(easy[known], [0, 1]).all(): raise ValueError('Known partitions required')
    if not known.any(): return dict(status='not_estimable', reason='no_known_labels')
    new, old, target, recordings, agents, easy, outside = [v[known] for v in
        (new, old, target, recordings, agents, easy, outside)]
    a, b = (new-target)**2, (old-target)**2
    delta = a-b; pos, neg = np.maximum(delta, 0), np.maximum(-delta, 0)
    def totals(mask):
        return dict(rows=int(mask.sum()), row_share=float(mask.mean()),
            new_MSE=float(a[mask].mean()) if mask.any() else None,
            old_MSE=float(b[mask].mean()) if mask.any() else None,
            positive_excess_mass=float(pos[mask].sum()), negative_excess_mass=float(neg[mask].sum()),
            positive_excess_share=float(pos[mask].sum()/pos.sum()) if pos.sum() else None,
            signed_excess_MSE=float(delta[mask].mean()) if mask.any() else None)
    units = dict(window=np.arange(len(delta)), recording=recordings,
                 track=np.column_stack((recordings, agents)))
    groups = {unit: {name: concentration(grouped_mass(v, keys)) for name, v in
        (('positive_row_excess', pos), ('negative_row_excess', neg), ('new_squared_error', a))}
        for unit, keys in units.items()}
    parts = {k: totals(mask) for k, mask in dict(all=np.ones(len(delta), bool), easy=easy == 1,
        outside_easy=easy == 0, radial_outside=outside, radial_inside=~outside).items()}
    np.testing.assert_allclose(parts['all']['signed_excess_MSE'], (pos.sum()-neg.sum())/len(delta), atol=1e-10)
    return dict(status='computed', unknown_rows=int(n-known.sum()), partitions=parts, groups=groups,
                mass_ESS_is_not_independent_sample_size=True)


def radial_support(x, held_x, pr, sites, held):
    """A coarse causal-feature extrapolation proxy, not a density/support test."""
    if (held in sites or sorted(set(sites)) != pr['training_sites'] or len(x) != len(pr['known'])
            or x.ndim != 2 or held_x.ndim != 2 or x.shape[1] != held_x.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(held_x).all()):
        raise ValueError('Excluded held locality and finite fitting features required')
    if (pr['mean'].shape != (x.shape[1],) or pr['std'].shape != (x.shape[1],)
            or not np.isfinite(pr['mean']).all() or not np.isfinite(pr['std']).all()
            or (pr['std'] <= 0).any() or not len(held_x)):
        raise ValueError('Valid fitting normalization and nonempty query required')
    def scores(a):
        radius, extreme = [], []
        for start in range(0, len(a), 4096):
            z = (a[start:start+4096].astype(float)-pr['mean'])/pr['std']
            radius.extend(np.sqrt(np.mean(z*z, axis=1))); extreme.extend(np.mean(np.abs(z)>8, axis=1))
        return np.asarray(radius), np.asarray(extreme)
    r, _ = scores(x); q, extreme = scores(held_x)
    ids = np.flatnonzero(pr['known']); order = ids[np.argsort(r[ids], kind='stable')]
    w = pr['weights'][order]; assert w.sum() > 0
    edge = float(r[order[min(np.searchsorted(np.cumsum(w)/w.sum(), .95), len(order)-1)]])
    outside = q > edge
    return outside, dict(training_radius_q95=edge, fitting_fraction_outside=float(pr['weights']@(r>edge)),
        held_fraction_outside=float(outside.mean()), held_radius_median=float(np.median(q)),
        held_fraction_abs_standardized_feature_above8=float(extreme.mean()),
        threshold_uses_fitting_only=True, conditional_support_proven=False)


def gradient_concentration(model, batch, recordings, mean_harm):
    """Group gradients keep the full-batch denominator, so their vectors add."""
    z, env, y, easy, scales = batch
    if len(recordings) != len(z) or not np.isfinite(mean_harm) or mean_harm <= 0:
        raise ValueError('Aligned fitting batch and positive fitting mean required')
    pred, logit = model(z, env)
    bce = F.binary_cross_entropy_with_logits(logit, easy.detach(), reduction='none')
    weights = y[:, 1].detach()/mean_harm
    terms = dict(ordinary_BCE=bce, weighted_BCE=bce*weights,
                 easy_harm_cost=((pred[:, 3]-y[:, 3].detach())/scales[3])**2/4)
    parameters = list(model.network[0].parameters()); out = {}
    keys, inverse = np.unique(recordings, return_inverse=True)
    for name, loss in terms.items():
        total = flat_gradient(loss.mean(), parameters, retain_graph=True); rows = []
        for i in range(len(keys)):
            mask = torch.tensor(inverse == i, dtype=loss.dtype)
            rows.append(flat_gradient((loss*mask).mean(), parameters, retain_graph=True))
        rows = np.asarray(rows); np.testing.assert_allclose(rows.sum(0), total, rtol=2e-4, atol=2e-7)
        norm = np.linalg.norm(rows, axis=1); total_norm = float(np.linalg.norm(total))
        projection = rows@total/(total@total) if total_norm else None
        out[name] = dict(value=float(loss.mean().detach()), recording_gradient_norm_mass=concentration(norm),
            total_shared_norm=total_norm, cancellation_ratio=total_norm/float(norm.sum()) if norm.sum() else None,
            projection_sum=float(projection.sum()) if projection is not None else None,
            negative_projection_recordings=int((projection < 0).sum()) if projection is not None else None,
            min_projection=float(projection.min()) if projection is not None else None,
            max_projection=float(projection.max()) if projection is not None else None,
            group_gradient_additivity_pass=True)
    out['label_weight_recording_mass'] = concentration(grouped_mass(weights.numpy(), recordings))
    out['fixed_fitting_rows'] = len(z)
    out['not_training_path_or_optimizer_influence'] = True
    return out
