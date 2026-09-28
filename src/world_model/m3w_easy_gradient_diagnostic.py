"""Fitting-only loss geometry for frozen paired easy-risk heads."""
import numpy as np
import torch
from src.world_model import m3w_easy_hurdle as api


def geometry(vectors):
    vectors = {k: v.detach().to(torch.float64).flatten() for k, v in vectors.items()}
    if set(vectors) != {'marginal', 'occurrence', 'conditional'}:
        raise ValueError('Exactly the three registered objective components required')
    if not all(torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError('Nonfinite gradients')
    vectors['auxiliary'] = vectors['occurrence']+vectors['conditional']
    vectors['supervised'] = vectors['marginal']+vectors['auxiliary']
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    risk = vectors['marginal']; norm = norms['marginal']
    cosines, projections = {}, {}
    for k in ('occurrence', 'conditional', 'auxiliary', 'supervised'):
        dot = float(torch.dot(risk, vectors[k]))
        cosines[k] = dot/(norm*norms[k]) if min(norm, norms[k]) > 1e-14 else None
        projections[k] = dot/(norm*norm) if norm > 1e-14 else None
    return dict(norms=norms, risk_cosine=cosines, risk_projection=projections,
                auxiliary_to_risk_norm=norms['auxiliary']/norm if norm > 1e-14 else None)


def gradients(model, x, envelope, target, segments, query_count):
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    terms = api.losses(model(x, envelope), target, segments, query_count)
    params = list(model.named_parameters())
    components = {}
    for name in ('marginal', 'occurrence', 'conditional'):
        values = torch.autograd.grad(terms[name], [p for _, p in params], retain_graph=True)
        components[name] = {k: v for (k, _), v in zip(params, values)}
    out = {}
    for block, prefix in (('all', ''), ('shared', 'network.0.'), ('output', 'network.2.')):
        vectors = {c: torch.cat([v.flatten() for k, v in values.items() if k.startswith(prefix)])
                   for c, values in components.items()}
        out[block] = geometry(vectors)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), 'Diagnostic must not update any parameter'
    assert all(p.grad is None for _, p in params), 'No accumulated optimizer gradients'
    return dict(losses={k: float(v.detach()) for k, v in terms.items()}, gradients=out)


def fitting_signal(target, groups, source_groups):
    target = np.asarray(target, dtype=float)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError('Aligned easy/reference/harm target required')
    rows = []
    for group in groups:
        y = target[group]
        if not np.isfinite(y).all() or (y < 0).any() or not np.isin(y[:, 0], [0, 1]).all():
            raise ValueError('Known fitting labels only')
        easy, reference, harm = y.T
        rows.append([easy.mean(), (easy*reference).mean(), (easy*harm).mean(),
                     (easy*(harm-api.BUDGET*reference)).mean(),
                     (easy*(harm <= api.BUDGET*reference)).mean(), (easy*(harm == 0)).mean()])
    values = np.asarray(rows)
    sources = {}
    for site, indices in source_groups.items():
        mean = values[indices].mean(0)
        sources[str(site)] = dict(queries=len(indices), easy_probability=mean[0],
            easy_reference_mass=mean[1], easy_harm_mass=mean[2], easy_signed_risk=mean[3],
            easy_realized_within_budget_fraction=mean[4]/mean[0] if mean[0] else None,
            easy_zero_harm_fraction=mean[5]/mean[0] if mean[0] else None,
            easy_positive_harm_over_reference=mean[2]/mean[1] if mean[1] else None)
    return sources
