"""Exact fitting-only additive signed-score fit; not a risk certificate."""
import numpy as np


def groups_and_weights(sites, recordings, frames, known, bank):
    sites, recordings, frames, known, bank = map(np.asarray, (sites, recordings, frames, known, bank))
    if (known.dtype != bool or bank.dtype != bool or bank.shape != (len(known), 3)
            or any(x.shape != known.shape for x in (sites, recordings, frames)) or len(set(sites)) != 2):
        raise ValueError('Two fitting sources and three fixed causal banks required')
    groups = {}
    for i in np.flatnonzero(known):
        groups.setdefault((str(sites[i]), str(recordings[i]), int(frames[i])), []).append(i)
    counts = {s:sum(k[0] == str(s) for k in groups) for s in set(sites)}
    if any(v == 0 for v in counts.values()):raise ValueError('Known queries in each fitting source required')
    weights = np.zeros(len(known)); queries = []
    for key, positions in sorted(groups.items()):
        ix = np.asarray(positions); qw = 1/(2*counts[key[0]])
        parts = [ix[bank[ix, j]] for j in range(3)]
        weights[ix] += .5*qw/len(ix)
        for p in parts:
            if len(p):weights[p] += qw/(6*len(p))
        queries.append((ix, parts, qw))
    return queries, weights


def objective(error, queries, aggregate):
    value = 0.
    for ix, subsets, qw in queries:
        value += .5*qw*float(np.mean(error[ix]**2))
        for p in subsets:
            if len(p):value += qw/6*float(np.mean(error[p].mean(0)**2) if aggregate else np.mean(error[p]**2))
    return value


def fit_intercept(prediction, target, sites, recordings, frames, bank, aggregate):
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or np.isinf(y).any():
        raise ValueError('Finite signed predictions and aligned two-axis targets required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(known, np.isfinite(y[:,0])) or not np.array_equal(known, np.isfinite(y[:,1])):
        raise ValueError('Both targets share unknown support')
    queries, weights = groups_and_weights(sites, recordings, frames, known, bank)
    error = np.zeros_like(y); error[known] = y[known]-p[known]
    mass = float(weights.sum()); numerator = (error*weights[:,None]).sum(0)
    optimum = numerator/mass
    conservative = np.maximum(optimum, 0.)
    before = objective(error, queries, aggregate)
    after = objective(error-conservative, queries, aggregate)
    predicted_reduction = float((numerator*conservative-.5*mass*conservative**2).sum())
    np.testing.assert_allclose(before-after, predicted_reduction, rtol=1e-9, atol=1e-12)
    if after > before+1e-12:raise AssertionError('Projected exact quadratic fit cannot increase its fitting loss')
    return dict(unconstrained_offset=optimum.tolist(), nonnegative_offset=conservative.tolist(),
        fitting_objective_before=before, fitting_objective_after=after,
        fitting_loss_reduction=before-after, expected_quadratic_reduction=predicted_reduction,
        curvature_mass=mass, weighted_residual_sum=numerator.tolist(),
        row_residual_mean=error[known].mean(0).tolist(), known_rows=int(known.sum()),
        unknown_rows=int((~known).sum()), fitting_queries=len(queries),
        selected_policy_changed=False, held_labels_used=False,
        fitting_loss_reduction_is_downstream_lift=False)
