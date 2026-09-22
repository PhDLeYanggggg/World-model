"""Descriptive frozen-head fit/support checks; never calibration or selection."""
import numpy as np


def population_summary(target, score, distance, selected, weights):
    y, p, d, s, w = map(np.asarray, (target, score, distance, selected, weights))
    n = len(d)
    if (y.shape != (n, 2) or p.shape != y.shape or s.shape != (n,) or s.dtype != bool
            or w.shape != (n,) or d.shape != (n,) or not np.isfinite(p).all()
            or not np.isfinite(d).all() or not np.isfinite(w).all()
            or (p < 0).any() or (d < 0).any() or (w < 0).any()
            or np.isinf(y).any() or not np.array_equal(np.isnan(y[:, 0]), np.isnan(y[:, 1]))):
        raise ValueError('Aligned complete-cost labels, scores, decisions and nonnegative weights required')
    known = np.isfinite(y).all(1)
    if (y[known] < 0).any():
        raise ValueError('Costs cannot be negative')
    out = {}
    for name, mask in dict(all=np.ones(n, bool), selected=s, rejected=~s).items():
        take = known & mask
        row = dict(rows=int(mask.sum()), complete=int(take.sum()), not_complete=int((mask & ~known).sum()))
        if take.any():
            h, ph = y[take, 1], p[take, 1]
            row.update(realized_harm=float(h.mean()), predicted_harm=float(ph.mean()),
                       realized_benefit=float(y[take, 0].mean()), predicted_benefit=float(p[take, 0].mean()),
                       harm_event_rate=float((h > 0).mean()), native_MAE=float(np.abs(p[take]-y[take]).mean()),
                       native_MSE=float(np.square(p[take]-y[take]).mean()),
                       harm_underpredicted=bool(ph.mean() < h.mean()),
                       harm_ratio=None if ph.sum() == 0 else float(h.sum()/ph.sum()),
                       forecast_disagreement_mean=float(d[take].mean()))
            top = max(1, int(np.ceil(.01*len(h))))
            row['top_one_percent_harm_share'] = None if h.sum() == 0 else float(np.sort(h)[-top:].sum()/h.sum())
            ww = w[take]
            row['fit_weighted_harm'] = float(np.average(h, weights=ww)) if ww.sum() else None
            row['fit_weighted_predicted_harm'] = float(np.average(ph, weights=ww)) if ww.sum() else None
        out[name] = row
    return out


def moving_zero_support(past, reference, complete, selected, draws=None):
    h, r, c, s = map(np.asarray, (past, reference, complete, selected))
    if (h.shape != (len(r), 8, 2) or r.ndim != 1 or c.shape != r.shape or s.shape != r.shape
            or c.dtype != bool or s.dtype != bool or not np.isfinite(h).all()
            or not np.isfinite(r[c]).all() or (r[c] < 0).any()):
        raise ValueError('Finite past history and complete reference-label support required')
    moving = np.any(h[:, -1] != h[:, -2], axis=1)
    exact = c & (r == 0)
    masks = dict(complete_exact_zero=exact, complete_exact_zero_stopped=exact & ~moving,
                 complete_exact_zero_moving=exact & moving)
    out = {}
    for name, mask in masks.items():
        row = dict(rows=int(mask.sum()), selected=int((mask & s).sum()))
        if draws is not None:
            a = np.asarray(draws)
            if a.shape != r.shape or a.dtype.kind not in 'iu' or (a < 0).any():
                raise ValueError('Aligned nonnegative sampler counts required')
            row.update(sampled_draws=int(a[mask].sum()), unsampled_rows=int((mask & (a == 0)).sum()))
        out[name] = row
    return out


def standardized_support(x, mean, std, selected):
    x, m, t, s = map(np.asarray, (x, mean, std, selected))
    if (x.ndim != 2 or m.shape != (x.shape[1],) or t.shape != m.shape
            or s.shape != (len(x),) or s.dtype != bool or not np.isfinite(x).all()
            or not np.isfinite(m).all() or not np.isfinite(t).all() or (t <= 0).any()):
        raise ValueError('Finite features and existing fitting-only preprocessing required')
    z = np.max(np.abs((x.astype(float)-m)/t), axis=1)
    out = {}
    for name, mask in dict(all=np.ones(len(x), bool), selected=s).items():
        v = z[mask]
        out[name] = dict(rows=len(v), max_abs_z_quantiles=None if not len(v) else
                        dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.quantile(v, [.5, .95, .99, 1])))),
                        counts_above={str(k): int((v > k).sum()) for k in (3, 5, 10, 50)})
    return out
