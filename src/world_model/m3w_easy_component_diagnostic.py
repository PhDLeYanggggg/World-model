"""Descriptive factor accounting on fitting labels; never produce actions."""
import itertools
import numpy as np

COMPONENTS = ('occurrence', 'reference', 'harm')
BUDGET = .02


def compose(pred):
    p = np.asarray(pred, float)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all()
            or ((p[:, 0] < 0) | (p[:, 0] > 1)).any() or (p[:, 1:] < 0).any()):
        raise ValueError('Finite occurrence probability and nonnegative conditional costs required')
    return p[:, 0]*(p[:, 2]-BUDGET*p[:, 1])


def weights(sites, recordings, frames, known):
    sites, recordings, frames, known = map(np.asarray, (sites, recordings, frames, known))
    if not all(a.shape == sites.shape and a.ndim == 1 for a in (sites, recordings, frames, known)):
        raise ValueError('Aligned one-dimensional query keys required')
    if known.dtype != bool or not known.any():
        raise ValueError('Nonempty known-label mask required')
    ix = np.flatnonzero(known)
    keys = np.rec.fromarrays([sites[ix].astype(str), recordings[ix].astype(str), frames[ix]], names='site,recording,frame')
    _, inv = np.unique(keys, return_inverse=True)
    sizes = np.bincount(inv)
    w = np.zeros(len(sites)); q = np.full(len(sites), -1, dtype=int); q[ix] = inv
    source_names = np.unique(sites[ix])
    for site in source_names:
        at = sites[ix] == site; queries = len(np.unique(inv[at]))
        w[ix[at]] = 1/(len(source_names)*queries*sizes[inv[at]])
    np.testing.assert_allclose(w.sum(), 1., rtol=1e-12)
    return w, q


def objective(error, w, q):
    error, w, q = map(np.asarray, (error, w, q))
    valid = w > 0
    if error.shape != w.shape or q.shape != w.shape or not valid.any() or (q[valid] < 0).any():
        raise ValueError('Valid error and query weights required')
    if not np.isfinite(error[valid]).all() or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Finite weighted error required')
    mass = np.bincount(q[valid], weights=w[valid]); used = mass > 0
    mean = np.bincount(q[valid], weights=w[valid]*error[valid])[used]/mass[used]
    row_mse = float(np.dot(w[valid], error[valid]**2))
    query_mse = float(np.dot(mass[used], mean**2))
    return dict(row_MSE=row_mse, query_mean_MSE=query_mse, marginal=.5*(row_mse+query_mse))


def attribution(values):
    if set(values) != {str(i) for i in range(8)} or not np.isfinite(list(values.values())).all():
        raise ValueError('All eight finite factor combinations required')
    contributions = np.zeros(3)
    for order in itertools.permutations(range(3)):
        mask = 0
        for component in order:
            after = mask | (1 << component)
            contributions[component] += (values[str(after)]-values[str(mask)])/6
            mask = after
    np.testing.assert_allclose(contributions.sum(), values['7']-values['0'], atol=1e-10, rtol=1e-10)
    return contributions.tolist()


def partition(error, parts, w, mask):
    at = mask & (w > 0); wm = w[at]; mass = float(wm.sum())
    squared = [float(np.dot(wm, parts[at, i]**2)) for i in range(3)]
    cross = [float(2*np.dot(wm, parts[at, i]*parts[at, j])) for i, j in ((0, 1), (0, 2), (1, 2))]
    mse = float(np.dot(wm, error[at]**2))
    np.testing.assert_allclose(sum(squared)+sum(cross), mse, rtol=1e-9, atol=1e-10)
    return dict(rows=int(at.sum()), weight_mass=mass, MSE_contribution=mse,
                conditional_MSE=mse/mass if mass else None,
                squared_components=squared, cross_components=cross)


def source_readout(left, right, target, sites, recordings, frames):
    w, q = weights(sites, recordings, frames, np.ones(len(target), bool))
    easy, reference, harm = target.T
    truth = easy*(harm-BUDGET*reference)
    partitions = dict(all=np.ones(len(target), bool), not_easy=easy == 0,
        easy_zero_harm=(easy == 1) & (harm == 0), easy_positive_harm=(easy == 1) & (harm > 0))
    arms = {}
    for name, pred in [('uncapped', left), ('risk_priority', right)]:
        probability, r, h = pred.T
        error = compose(pred)-truth
        # Exact telescoping identity, not a causal or uniquely identified decomposition.
        parts = np.column_stack(((probability-easy)*(harm-BUDGET*reference),
                                 -BUDGET*probability*(r-reference), probability*(h-harm)))
        np.testing.assert_allclose(parts.sum(1), error, rtol=1e-9, atol=1e-10)
        easy_mass = float(w@easy)
        arms[name] = dict(objective=objective(error, w, q), Brier=float(w@((probability-easy)**2)),
            signed_bias=float(w@error), easy_mass=easy_mass,
            conditional_reference_MSE=float(w@(easy*(r-reference)**2))/easy_mass if easy_mass else None,
            conditional_harm_MSE=float(w@(easy*(h-harm)**2))/easy_mass if easy_mass else None,
            partitions={key: partition(error, parts, w, mask) for key, mask in partitions.items()})
    transplants = {}
    for mask in range(8):
        pred = np.column_stack([right[:, i] if mask & (1 << i) else left[:, i] for i in range(3)])
        transplants[str(mask)] = objective(compose(pred)-truth, w, q)
    result = dict(known_rows=len(target), queries=len(np.unique(q)), arms=arms, transplants=transplants,
        attribution={key: attribution({mask: row[key] for mask, row in transplants.items()})
                     for key in ('row_MSE', 'query_mean_MSE', 'marginal')})
    for arm, mask in [('uncapped', '0'), ('risk_priority', '7')]:
        assert arms[arm]['objective'] == transplants[mask]
    return result


def diagnose(left, right, target, sites, recordings, frames):
    left, right, target = map(lambda a: np.asarray(a, float), (left, right, target))
    compose(left); compose(right)
    n = len(left)
    keys = list(map(np.asarray, (sites, recordings, frames)))
    if right.shape != left.shape or target.shape != left.shape or any(a.shape != (n,) for a in keys):
        raise ValueError('Aligned three-component predictions, labels and keys required')
    known = np.isfinite(target).all(1)
    if not known.any() or not (known | np.isnan(target).all(1)).all():
        raise ValueError('Label rows must be entirely known or entirely missing')
    if (target[known] < 0).any() or not np.isin(target[known, 0], (0., 1.)).all():
        raise ValueError('Binary easy labels and nonnegative label-only costs required')
    sites, recordings, frames = keys
    sources = {}
    for source in np.unique(sites):
        at = known & (sites == source)
        if not at.any():
            raise ValueError('A fitting source has no known labels')
        sources[str(source)] = source_readout(left[at], right[at], target[at], sites[at], recordings[at], frames[at])
    all_sources = source_readout(left[known], right[known], target[known], sites[known], recordings[known], frames[known])
    for arm in ('uncapped', 'risk_priority'):
        for key, value in all_sources['arms'][arm]['objective'].items():
            np.testing.assert_allclose(value, np.mean([s['arms'][arm]['objective'][key] for s in sources.values()]), rtol=1e-12, atol=1e-12)
    return dict(known_rows=int(known.sum()), unknown_rows_excluded=int((~known).sum()), sources=sources,
        source_balanced=all_sources, components=list(COMPONENTS),
        result_kind='fitting_only_descriptive_factor_accounting', parameter_updates=0,
        policy_actions_computed=False, useful_switch_ranking='not_run_no_signed_benefit_in_packet',
        counterfactual_mixtures_are_not_trained_models=True)
