"""Metrics on aligned displacements; this module never opens a dataset."""

import numpy as np

from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_selection_forecast_readout import errors, fraction, policy

ARMS = ('independent', 'additive', 'same_family', 'joint')
SEEDS = (17, 29, 43)


def coverage(valid, offsets):
    complete = valid.all(axis=1)
    full = np.zeros(len(valid), dtype=bool)
    pair_windows = roster_windows = 0
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        count = complete[lo:hi].sum()
        pair_windows += count >= 2
        if count == hi-lo:
            full[lo:hi] = True
            roster_windows += 1
    return full, dict(windows=len(offsets)-1, agents=len(valid), complete_agents=int(complete.sum()),
                      complete_pair_windows=int(pair_windows), complete_roster_windows=roster_windows)


def absolute(pred, y, valid, full):
    ade, fde = errors(pred, y, valid)
    available = ade[np.isfinite(ade)]
    end = fde[np.isfinite(fde)]
    return dict(
        available_ade_native=float(available.mean()) if len(available) else None,
        available_fde_native=float(end.mean()) if len(end) else None,
        available_ade_p95_native=float(np.quantile(available, .95)) if len(available) else None,
        complete_roster_ade_native=float(ade[full].mean()) if full.any() else None,
        complete_roster_fde_native=float(fde[full].mean()) if full.any() else None,
        available_agent_count=len(available), endpoint_count=len(end), complete_roster_agent_count=int(full.sum()))


def compare(pred, ref, y, valid, weight, cv_mass, bounds):
    known, lower = paired(ref, pred, y, valid)
    rel = None
    if bounds is not None:
        lo, hi = bounds
        rel = pair_reduction(pred[None, lo:hi], ref[lo:hi], y[lo:hi], valid[lo:hi])[0]
    return dict(known_pp=fraction(float(100.*weight @ known.sum(1)), cv_mass),
                lower_pp=fraction(float(100.*weight @ lower.sum(1)), cv_mass),
                relative_available_native=float(rel[0]) if rel is not None and (
                    valid[lo:hi].sum(axis=0) >= 2).any() else None,
                relative_lower_native=float(rel[1]) if rel is not None else None)


def matched_paths(cv, offsets, proposals):
    if proposals is None:
        return None, {}
    bounds, radius, edits = proposals['bounds'], proposals['radius'], proposals['arms']
    lo, hi = bounds
    idx = np.searchsorted(offsets, lo)
    if (idx >= len(offsets)-1 or offsets[idx] != lo or offsets[idx+1] != hi or hi-lo < 3
            or not np.isfinite(radius) or radius <= 0. or set(edits) != set(ARMS)):
        raise ValueError('Expected an eligible full query and common edit radius')
    result = {}
    for arm, (rows, paths) in edits.items():
        rows, paths = np.asarray(rows), np.asarray(paths)
        if (rows.shape != (2,) or not np.issubdtype(rows.dtype, np.integer) or rows[0] == rows[1]
                or not ((lo <= rows) & (rows < hi)).all() or paths.shape != (2, 12, 2)
                or not np.isfinite(paths).all()):
            raise ValueError('Expected two finite, distinct edits within the common query')
        actual = np.linalg.norm(paths-cv[rows], axis=-1).max(axis=1)
        if not np.allclose(actual, radius, atol=1e-9, rtol=1e-9):
            raise ValueError('Edit radius differs from the matched budget')
        pred = cv.copy()
        pred[rows] = paths
        result[arm] = pred
    return bounds, result


def score_flight(cv, ols, y, valid, offsets, proposals, *, easy_cut, hard_cut):
    if (cv.shape != ols.shape or cv.shape != y.shape or cv.ndim != 3 or cv.shape[1:] != (12, 2)
            or valid.shape != cv.shape[:2] or valid.dtype != bool or not np.isfinite(y[valid]).all()
            or not np.isfinite(cv).all() or not np.isfinite(ols).all() or offsets.ndim != 1
            or not len(offsets) or offsets[0] != 0 or offsets[-1] != len(cv) or np.any(np.diff(offsets) < 2)
            or not (0 < easy_cut <= hard_cut < np.inf)):
        raise ValueError('Invalid aligned predictions, label mask, roster, or prespecified cutoffs')
    full, counts = coverage(valid, offsets)
    bounds, pred = matched_paths(cv, offsets, proposals)
    weight = np.repeat(1./np.diff(offsets), np.diff(offsets))
    cv_mass = float(weight @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1))
    stats = {}
    for arm, p in dict(cv=cv, ols4=ols, **pred).items():
        row = absolute(p, y, valid, full)
        row['vs_cv'] = compare(p, cv, y, valid, weight, cv_mass, bounds)
        if arm in pred:
            r = policy(p, cv, cv, y, valid, np.repeat('flight', len(y)), easy_cut=easy_cut, hard_cut=hard_cut)
            row['risk'] = r['recordings']['flight']
            row['easy_known_gain_percent'] = r['subsets']['easy']['gain_percent']
            row['hard_known_gain_percent'] = r['subsets']['hard']['gain_percent']
            row['vs_ols4'] = compare(p, ols, y, valid, weight, cv_mass, bounds)
            lo, hi = bounds
            changed = (p != cv).any(axis=(1, 2))
            row['intervention_rate'] = float(changed.mean())
            row['query_intervention_rate'] = float(changed[lo:hi].mean())
            row['dual_pass'] = (row['risk']['supported'] and row['vs_cv']['lower_pp'] is not None
                                and row['vs_cv']['lower_pp'] > 0. and row['vs_ols4']['lower_pp'] > 0.)
        stats[arm] = row
    contrasts = {arm: compare(pred['joint'], pred[arm], y, valid, weight, cv_mass, bounds)
                 for arm in ARMS[:-1]} if pred else {}
    return dict(coverage=counts, arms=stats, matched_contrasts=contrasts, matched_eligible=bool(pred))


def component_interval(views, flight_components, control, metric, *, draws=3000, seed=1439):
    """Average seeds within flight, flights within source, then resample sources."""
    if (set(views) != set(flight_components) or draws < 1
            or any(set(v) != set(SEEDS) for v in views.values())):
        raise ValueError('Expected the fixed roster and all three seed views')
    groups = {g: [] for g in sorted(set(flight_components.values()))}
    for flight, values in views.items():
        x = [r['matched_contrasts'].get(control, {}).get(metric) for r in values.values()]
        if all(v is not None for v in x):
            groups[flight_components[flight]].append(float(np.mean(x)))
    counts = {g: len(x) for g, x in groups.items()}
    means = {g: float(np.mean(x)) if x else None for g, x in groups.items()}
    if len(groups) < 2 or not all(counts.values()):
        return dict(point=None, ci95=None, by_component=means, flights_by_component=counts,
                    status='required_component_not_estimable')
    sample = np.array(list(means.values()))
    b = np.random.default_rng(seed).choice(sample, size=(draws, len(sample))).mean(1)
    return dict(point=float(sample.mean()), ci95=np.quantile(b, [.025, .975]).tolist(),
                by_component=means, flights_by_component=counts, status='estimable')
