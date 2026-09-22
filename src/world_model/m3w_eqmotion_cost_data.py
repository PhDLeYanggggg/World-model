"""Assemble source-excluded EqMotion predictions and separate cost labels."""
import numpy as np


def validate_training_labels(targets, complete):
    """Check supervision support separately; it never alters inference inputs."""
    complete = np.asarray(complete)
    declared = np.asarray(targets['complete_future'])
    if complete.dtype.kind != 'b' or declared.dtype.kind != 'b':
        raise ValueError('Boolean future-supervision masks required')
    np.testing.assert_array_equal(declared, complete)
    values = np.column_stack([targets[k] for k in ('benefit', 'harm', 'baseline_ade')])
    if (values.shape != (len(complete), 3) or np.isinf(values).any()
            or not np.isfinite(values[complete]).all()
            or np.any(values[np.isfinite(values)] < 0)
            or not np.array_equal(np.isnan(values[:, 0]), np.isnan(values[:, 1]))
            or not np.array_equal(np.isnan(values[:, 0]), np.isnan(values[:, 2]))):
        raise ValueError('Nonnegative aligned cost support required')
    if np.any((values[:, 0] > 0) & (values[:, 1] > 0)):
        raise ValueError('Benefit and harm cannot both be positive for one realized outcome')


def assemble_view(sites, outer, groups):
    sites = np.asarray(sites)
    expected = np.flatnonzero(sites != outer)
    if outer not in sites or {g['inner_site'] for g in groups} != set(sites)-{outer}:
        raise ValueError('Exactly the other source sites are required')
    inputs, labels, seen = [], [], np.zeros(len(sites), int)
    for group in groups:
        inner, ids = group['inner_site'], np.asarray(group['ids'])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
                or np.any(ids < 0) or np.any(ids >= len(sites)) or np.any(np.diff(ids) <= 0)):
            raise ValueError('Sorted unique in-range global IDs required')
        # Pair caches also contain outer rows, which must not enter this head's fit.
        pair = np.flatnonzero(np.isin(sites, [outer, inner]))
        np.testing.assert_array_equal(ids, pair)
        use = sites[ids] == inner; selected = ids[use]
        pred = np.asarray(group['prediction'])
        costs = group['labels']
        if pred.shape != (len(ids), 12, 2) or not np.isfinite(pred).all():
            raise ValueError('Finite fixed-grid predictions required')
        wanted = ('benefit', 'harm', 'baseline_ade', 'complete_future')
        if any(np.asarray(costs[k]).shape != (len(ids),) for k in wanted):
            raise ValueError('Aligned separate cost labels required')
        seen[selected] += 1
        inputs.append((selected, pred[use]))
        labels.append({k:np.asarray(costs[k])[use] for k in wanted})
    np.testing.assert_array_equal(seen, (sites != outer).astype(int))
    ids = np.concatenate([a for a, _ in inputs]); order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], expected)
    x = dict(ids=ids[order], prediction=np.concatenate([b for _, b in inputs])[order])
    y = dict(ids=ids[order], **{k:np.concatenate([a[k] for a in labels])[order]
                              for k in ('benefit', 'harm', 'baseline_ade', 'complete_future')})
    return x, y
