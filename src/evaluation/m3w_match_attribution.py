"""Descriptive fixed-reference attribution, not an inference-time label filter."""
import numpy as np

BLOCKS = {'ego_history': (0, 42), 'neighbor_history': (42, 50),
          'rollout_summary': (50, 64), 'rollout_history': (64, 96),
          'candidate_neighbor': (96, 120)}


def summarize(query, donors, donor_weights, truth, predicted, weights):
    q, d, w, y, p, qw = map(np.asarray, (query, donors, donor_weights, truth, predicted, weights))
    n = len(q)
    if (q.shape != (n, 120) or d.shape != (n, 32, 120) or w.shape != (n, 32)
            or y.shape != (n, 2) or p.shape != y.shape or qw.shape != (n,)
            or not all(np.isfinite(a).all() for a in (q, d, w, y, p, qw))
            or (w < 0).any() or (qw < 0).any() or (w.sum(1) <= 0).any() or qw.sum() <= 0):
        raise ValueError('Aligned causal matches and separate known evaluation labels required')
    w = w / w.sum(1, keepdims=True)
    delta = q[:, None] - d
    squared = np.einsum('nk,nkd->nd', w, delta**2)
    signed = np.einsum('nk,nkd->nd', w, delta)
    admitted = (p <= 0).all(1)
    unsafe = (y > 0).any(1)
    groups = {'all': np.ones(n, bool), 'admitted_unsafe': admitted & unsafe,
              'admitted_nonpositive': admitted & ~unsafe, 'rejected': ~admitted}
    out = {}
    for name, at in groups.items():
        mass = qw[at].sum()
        if not mass:
            out[name] = dict(rows=int(at.sum()), weight=0., blocks=None, label_bias=None)
            continue
        ww = qw[at] / mass
        ms, md = ww @ squared[at], ww @ signed[at]
        total = ms.sum()
        blocks = {}
        for block, (lo, hi) in BLOCKS.items():
            blocks[block] = dict(dimensions=hi-lo, squared_distance=float(ms[lo:hi].sum()),
                mean_squared_per_feature=float(ms[lo:hi].mean()),
                total_distance_fraction=float(ms[lo:hi].sum()/total) if total else None,
                signed_mean_delta_norm=float(np.linalg.norm(md[lo:hi])))
        out[name] = dict(rows=int(at.sum()), weight=float(mass/qw.sum()), blocks=blocks,
            label_bias=(ww @ (y[at]-p[at])).tolist(),
            largest_difference_columns=np.argsort(-ms, kind='stable')[:8].tolist())
    return out
