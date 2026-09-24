"""Source-only zero-reference readout on frozen forest partitions, not calibration."""
import numpy as np


def zero_event(targets, known):
    q, k = np.asarray(targets), np.asarray(known)
    if (q.shape != (len(k), 6) or k.dtype != bool or not np.isfinite(q).all()
            or not np.isin(q[:, 5], [0, 1]).all() or (q < 0).any() or (q > 1).any()
            or q[~k].any()):
        raise ValueError("Unchanged supported six-moment training targets required")
    return k & (q[:, 5] == 1) & (q[:, 4] == 0)


def source_weights(known, draws, distance):
    k, w, d = map(np.asarray, (known, draws, distance))
    if (k.ndim != 1 or k.dtype != bool or w.shape != k.shape or d.shape != k.shape
            or w.dtype.kind not in "iu" or (w < 0).any() or w[~k].any()
            or not np.isfinite(d).all() or (d < 0).any()):
        raise ValueError("The original known-source draws and causal distance are required")
    return w.astype(float)*(d > 0)


def fit_tree(tree, x, event, weight):
    x, y, w = np.asarray(x, np.float32), np.asarray(event), np.asarray(weight)
    if (x.ndim != 2 or y.shape != (len(x),) or y.dtype != bool or w.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(w).all() or (w < 0).any()
            or not (w > 0).any()):
        raise ValueError("Finite source features, binary labels and positive support required")
    use = w > 0; leaves = tree.apply(x[use]); size = tree.tree_.node_count
    total = np.bincount(leaves, weights=w[use], minlength=size)
    positive = np.bincount(leaves, weights=w[use]*y[use], minlength=size)
    unique = np.bincount(leaves, minlength=size)
    leaf = tree.tree_.children_left == -1
    np.testing.assert_array_equal(total[leaf], tree.tree_.weighted_n_node_samples[leaf])
    if (positive > total).any():
        raise ValueError("Invalid zero-reference event counts")
    return dict(total=total, positive=positive, unique=unique)


def predict(forest, readouts, x):
    x = np.asarray(x, np.float32)
    if len(readouts) != len(forest.estimators_) or not readouts or not np.isfinite(x).all():
        raise ValueError("A complete source readout and finite causal inputs are required")
    probability = np.zeros(len(x)); support = np.ones(len(x), bool)
    minimum = np.full(len(x), np.iinfo(np.int64).max, np.int64)
    for tree, readout in zip(forest.estimators_, readouts):
        leaf = tree.apply(x); total, positive = readout["total"][leaf], readout["positive"][leaf]
        support &= total > 0
        probability += np.divide(positive, total, out=np.zeros(len(x)), where=total > 0)
        minimum = np.minimum(minimum, readout["unique"][leaf])
    probability /= len(readouts)
    return dict(probability=probability, supported=support, minimum_unique_rows=minimum)


def allowed(probability, supported):
    p, s = np.asarray(probability), np.asarray(supported)
    if (p.ndim != 1 or s.shape != p.shape or s.dtype != bool or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any()):
        raise ValueError("Finite event frequencies and support mask required")
    # Zero is the pre-existing zero-reference tolerance, not a swept confidence cutoff.
    return s & (p == 0)


def matched_with_incumbent(gain, risk, eligible, budget, incumbent, allocator):
    import math
    g, r, ok, inc = map(np.asarray, (gain, risk, eligible, incumbent))
    if (g.shape != r.shape or ok.shape != g.shape or inc.shape != g.shape
            or inc.dtype != bool or ok.dtype != bool or (inc & ~ok).any()
            or not np.isfinite(g).all() or not np.isfinite(r).all()
            or not np.isfinite(budget) or budget < 0 or math.fsum(r[inc]) > budget):
        raise ValueError("A genuinely feasible same-query incumbent is required")
    bits, report = allocator(g, r, ok, budget, count=int(inc.sum()))
    valid = (bits.dtype == bool and bits.shape == inc.shape and not (bits & ~ok).any()
             and int(bits.sum()) == int(inc.sum()) and math.fsum(r[bits]) <= budget
             and math.fsum(g[bits]) >= math.fsum(g[inc]))
    if not valid:
        return inc.copy(), dict(status="feasible_incumbent_retained", proposal=report,
            optimal=False, selected=int(inc.sum()), constraint_pass=True, exact_count_pass=True)
    return bits, dict(report, incumbent_retained=False)
