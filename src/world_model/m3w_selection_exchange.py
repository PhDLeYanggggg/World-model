"""Evaluation-only benefit/harm accounting for two already frozen policies."""
import numpy as np


def partition(new, control, eligible):
    new, control, eligible = map(np.asarray, (new, control, eligible))
    if (new.ndim != 1 or any(a.shape != new.shape or a.dtype != bool for a in (new, control, eligible))
            or ((new | control) & ~eligible).any()):
        raise ValueError('Aligned eligible Boolean frozen actions required')
    return dict(common=new & control, new_only=new & ~control,
        control_only=control & ~new, unselected_eligible=eligible & ~new & ~control,
        excluded=~eligible)


def account(floor, neural, new, control, eligible, utility, scale):
    groups = partition(new, control, eligible)
    floor, neural, utility = [np.asarray(v, float) for v in (floor, neural, utility)]
    known = np.isfinite(floor)
    if (floor.ndim != 1 or any(a.shape != floor.shape for a in (neural, utility, new)) or
            not np.array_equal(known, np.isfinite(neural)) or np.isinf(floor).any() or
            np.isinf(neural).any() or (floor[known] < 0).any() or (neural[known] < 0).any() or
            not np.isfinite(utility).all() or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Aligned evaluation labels and causal utility required')
    f, n, u = floor/scale, neural/scale, utility/scale
    benefit, harm = np.maximum(f-n, 0), np.maximum(n-f, 0)
    sums = {}
    for name, mask in groups.items():
        at = mask & known
        sums[name] = dict(rows=int(mask.sum()), known=int(at.sum()), unknown=int((mask & ~known).sum()),
            reference_sum=float(f[at].sum()), benefit_sum=float(benefit[at].sum()),
            harm_sum=float(harm[at].sum()), harmful_rows=int((at & (harm > 0)).sum()),
            useful_rows=int((at & (benefit > 0)).sum()), zero_gain_rows=int((at & (f == n)).sum()),
            predicted_utility_sum=float(u[at].sum()))
    a, b = sums['new_only'], sums['control_only']
    delta_b = a['benefit_sum']-b['benefit_sum']
    delta_h = a['harm_sum']-b['harm_sum']
    old_error = float(np.where(control, n, f)[known].sum())
    new_error = float(np.where(new, n, f)[known].sum())
    np.testing.assert_allclose(old_error-new_error, delta_b-delta_h, rtol=1e-10, atol=1e-8)
    factor = 100/old_error if old_error > 0 else None
    metric = dict(benefit_change_percent=None if factor is None else factor*delta_b,
        harm_change_percent=None if factor is None else factor*delta_h,
        net_ADE_gain_percent=None if factor is None else factor*(delta_b-delta_h),
        predicted_utility_change_percent=None if factor is None else factor*(a['predicted_utility_sum']-b['predicted_utility_sum']),
        new_only_rows=a['rows'], control_only_rows=b['rows'], common_rows=sums['common']['rows'],
        new_only_unknown=a['unknown'], control_only_unknown=b['unknown'])
    return dict(sums=sums, old_error_sum=old_error, new_error_sum=new_error, metric=metric)


def check_queries(new, control, eligible, recordings, frames, ids):
    partition(new, control, eligible)
    recordings, frames, ids = map(np.asarray, (recordings, frames, ids))
    if any(a.shape != new.shape for a in (recordings, frames, ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique aligned rows and current query identity required')
    queries = {}
    for i, key in enumerate(zip(recordings.tolist(), frames.tolist())):
        queries.setdefault(key, []).append(i)
    changed = 0
    for positions in queries.values():
        if new[positions].sum() != control[positions].sum():
            raise ValueError('Unequal current-query intervention counts')
        changed += bool((new[positions] != control[positions]).any())
    return dict(queries=len(queries), changed_queries=changed)
