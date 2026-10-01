"""Source-recording empirical component margins; not conformal transfer bounds."""
import numpy as np
from src.world_model import m3w_inner_separability as core
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds

MODES = ('harm', 'reference', 'joint')


def eligible(p, moving, support):
    p, moving, support = np.asarray(p, float), np.asarray(moving), np.asarray(support)
    if (p.ndim != 2 or p.shape[1] != 5 or not np.isfinite(p).all() or (p < 0).any()
            or moving.shape != (len(p),) or support.shape != moving.shape
            or moving.dtype != bool or support.dtype != bool):
        raise ValueError('Finite causal scores and Boolean support required')
    z = core.signed(p)
    return moving & support & (z[:, 0] > 0) & (z[:, 1:] <= 0).all(1)


def record_scores(p, y, env, action, recordings):
    """Observed source labels may fit margins; unknown rows are never zeros."""
    p, y, env, action, rec = map(np.asarray, (p, y, env, action, recordings))
    if (p.shape != y.shape or p.shape != (len(env), 5) or action.shape != env.shape
            or action.dtype != bool or rec.shape != env.shape or not np.isfinite(p).all()
            or (p < 0).any() or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Aligned source arrays required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or (y[known] < 0).any():
        raise ValueError('Known nonnegative moments or wholly missing rows')
    if (y[known, 1] > env[known] + 1e-5).any():
        raise ValueError('Causal disagreement envelope violated')
    out = []
    for name in np.unique(rec):
        selected = (rec == name) & action
        ix = selected & known
        truth, pred = y[ix].sum(0), p[ix].sum(0)
        denom = np.array([env[ix].sum(), env[ix].sum(), pred[2], pred[3]], float)
        numer = np.array([truth[1]-pred[1], truth[4]-pred[4], pred[2]-truth[2], pred[3]-truth[3]])
        scores = [float(n/d) if d > 0 else None for n, d in zip(numer, denom)]
        out.append(dict(recording=str(name), known_selected=int(ix.sum()),
            unknown_selected=int((selected & ~known).sum()), scores=scores))
    return out


def fit_margin(records, quantile=.9):
    if not 0 < quantile < 1:
        raise ValueError('Fixed empirical quantile in (0,1) required')
    names = [r['recording'] for r in records]
    if len(set(names)) != len(names):
        raise ValueError('One score per whole source recording required')
    counts, margins = [], []
    for j in range(4):
        vals = [r['scores'][j] for r in records if r['scores'][j] is not None]
        if not np.isfinite(vals).all():
            raise ValueError('Finite residual scores required')
        counts.append(len(vals))
        margins.append(float(np.clip(np.quantile(vals, quantile, method='higher'), 0, 1)) if vals else 1.)
    return dict(recordings=names, component_recording_counts=counts, quantile=quantile,
        margins=margins, supported=all(n >= 1 for n in counts),
        minimum_component_recordings=min(counts), conformal_guarantee=False)


def adjust(p, env, calibration, mode):
    if mode not in MODES:
        raise ValueError('Registered component arm required')
    p, env = np.asarray(p, float), np.asarray(env, float)
    q = p.copy()
    if p.shape != (len(env), 5) or not np.isfinite(p).all() or (p < 0).any() or (env < 0).any() or not np.isfinite(env).all():
        raise ValueError('Causal nonnegative moments and envelope required')
    h, eh, r, er = calibration['margins']
    if not np.isfinite([h, eh, r, er]).all() or not (0 <= min(h, eh, r, er) <= max(h, eh, r, er) <= 1):
        raise ValueError('Margins must be in [0,1]')
    if mode in ('harm', 'joint'):
        q[:, 1] = np.minimum(env, p[:, 1] + h*env)
        q[:, 4] = np.minimum(env, p[:, 4] + eh*env)
        q[:, 1] = np.maximum(q[:, 1], q[:, 4])
    if mode in ('reference', 'joint'):
        q[:, 2] = p[:, 2]*(1-r)
        q[:, 3] = np.minimum(p[:, 3]*(1-er), q[:, 2])
    # Missing source support or a zero predicted denominator is not a safe switch.
    disabled = (q[:, 2] <= 0) | (q[:, 3] <= 0)
    if not calibration['supported']:
        disabled[:] = True
    q[disabled, 0] = 0
    return q


def calibrate(p, y, env, moving, support, recordings, quantile=.9):
    rec = np.asarray(recordings).astype(str)
    raw = eligible(p, moving, support)
    scores = record_scores(p, y, env, raw, rec)
    final = fit_margin(scores, quantile)
    oof = {m: np.zeros(len(p), bool) for m in MODES}
    folds = []
    for name in np.unique(rec):
        train = [r for r in scores if r['recording'] != name]
        fitted = fit_margin(train, quantile)
        assert name not in fitted['recordings']
        held = rec == name
        for mode in MODES:
            q = adjust(np.asarray(p)[held], np.asarray(env)[held], fitted, mode)
            oof[mode][held] = eligible(q, np.asarray(moving)[held], np.asarray(support)[held])
        folds.append(dict(held_recording=name, calibration=fitted))
    source = {}
    for mode in MODES:
        assert not (oof[mode] & ~raw).any()
        final_action = eligible(adjust(p, env, final, mode), moving, support)
        assert not (final_action & ~raw).any()
        source[mode] = dict(oof=completion_bounds(y, oof[mode], env),
                            full_fit_resubstitution=completion_bounds(y, final_action, env))
    return dict(final=final, folds=folds, record_scores=scores, source=source,
                no_per_row_future_eligibility=True, independent_calibration=False), oof


def matched(left, right, left_utility, right_utility, recordings, frames, ids):
    out = [np.zeros_like(left), np.zeros_like(right)]
    groups = {}
    for i, key in enumerate(zip(recordings, frames)):
        groups.setdefault(key, []).append(i)
    for ix in groups.values():
        at = np.array(ix); count = min(int(left[at].sum()), int(right[at].sum()))
        for j, (take, utility) in enumerate(((left, left_utility), (right, right_utility))):
            pool = at[take[at]]
            order = pool[np.lexsort((np.asarray(ids)[pool], -utility[pool]))]
            out[j][order[:count]] = True
    return tuple(out)
