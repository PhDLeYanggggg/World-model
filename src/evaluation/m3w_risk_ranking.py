"""Fixed-count causal ranking controls, never a deployable calibrated policy."""
import numpy as np

POLICIES = ('forest_strict', 'neural_strict', 'forest_ratio', 'forest_gain',
            'neural_ratio', 'neural_gain')
MATCHED = ('forest_ratio', 'forest_gain', 'neural_ratio', 'neural_gain')


def risk_fraction(score):
    p = np.asarray(score, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Finite nonnegative benefit/harm estimates required')
    # Equivalent ordering to harm/benefit; rescaling avoids overflow and no epsilon is fitted.
    largest = p.max(1)
    q = np.divide(p, largest[:, None], out=np.zeros_like(p), where=largest[:, None] > 0)
    return np.divide(q[:, 1], q.sum(1), out=np.ones(len(q)), where=q.sum(1) > 0)


def fixed_count(score, eligible, ids, count, ranking):
    p, e, ids = np.asarray(score, float), np.asarray(eligible), np.asarray(ids)
    fraction = risk_fraction(p)
    if (e.shape != (len(p),) or e.dtype != bool or ids.shape != e.shape
            or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
            or isinstance(count, (bool, np.bool_)) or int(count) != count
            or not 0 <= count <= int(e.sum()) or ranking not in ('ratio', 'gain')):
        raise ValueError('Aligned causal support, unique integer ids and feasible fixed count required')
    value = fraction if ranking == 'ratio' else p[:, 1]-p[:, 0]
    pool = np.flatnonzero(e)
    selected = np.zeros(len(p), bool)
    selected[pool[np.lexsort((ids[pool], value[pool]))[:int(count)]]] = True
    return selected


def choices(forest, neural, past, distance, ids, frozen_forest, frozen_neural):
    f, nn, h, d = map(np.asarray, (forest, neural, past, distance))
    n = len(ids)
    if (f.shape != (n, 2) or nn.shape != f.shape or h.shape != (n, 8, 2)
            or not np.isfinite(h).all() or d.shape != (n,) or not np.isfinite(d).all()
            or (d < 0).any()):
        raise ValueError('Aligned causal scores, past and disagreement required')
    risk_fraction(f); risk_fraction(nn)
    eligible = (d > 0) & np.any(h[:, -1] != h[:, -2], axis=1)
    out = {}
    for name, p, frozen in (('forest', f, frozen_forest), ('neural', nn, frozen_neural)):
        frozen = np.asarray(frozen)
        strict = eligible & (p[:, 0] > p[:, 1]) & (p[:, 1] <= .1*p[:, 0])
        if frozen.shape != (n,) or frozen.dtype != bool or not np.array_equal(strict, frozen):
            raise ValueError('Frozen parent choices changed')
        out[name+'_strict'] = frozen.copy()
    k = int(out['forest_strict'].sum())
    for name, p in (('forest', f), ('neural', nn)):
        for rule in ('ratio', 'gain'):
            out[name+'_'+rule] = fixed_count(p, eligible, ids, k, rule)
    if not np.array_equal(out['forest_strict'], out['forest_ratio']):
        raise ValueError('Forest ratio must exactly recover its strict count boundary')
    return {key: out[key] for key in POLICIES}


def overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != bool or b.dtype != bool or a.shape != b.shape or a.ndim != 1:
        raise ValueError('Aligned binary selections required')
    union, common = int((a | b).sum()), int((a & b).sum())
    return dict(a=int(a.sum()), b=int(b.sum()), common=common,
                a_only=int((a & ~b).sum()), b_only=int((b & ~a).sum()),
                jaccard=None if union == 0 else common/union)
