"""Fitting-only temporal information probes, not trajectory deployment policies."""
import numpy as np
from src.world_model.m3w_risk_conditioned_residual import risk_features

ARMS = ('score_only', 'old_summary', 'ordered_history', 'history_neighbors')


def temporal_features(geometry, width):
    """Typed past geometry only; observed box width is a causal scale floor."""
    g, width = np.asarray(geometry, float), np.asarray(width, float)
    n = len(g)
    if (g.shape != (n, 476) or width.shape != (n,) or (width <= 0).any()
            or not np.isfinite(g).all() or not np.isfinite(width).all()):
        raise ValueError('Finite past geometry and positive observed width required')
    h = g[:, :16].reshape(n, 8, 2)
    times = g[:, 16:24]
    pos = g[:, 38:166].reshape(n, 8, 8, 2)
    nt = g[:, 166:230].reshape(n, 8, 8)
    mask = g[:, 230:294].reshape(n, 8, 8)
    if (not np.isin(mask, [0, 1]).all() or (times > 0).any()
            or (np.diff(times, axis=1) <= 0).any() or (times[:, -1] != 0).any()
            or not np.allclose(np.diff(times, axis=1), 1/12, rtol=0, atol=1e-7)
            or not np.allclose(nt[mask == 1], np.broadcast_to(times[:, None], nt.shape)[mask == 1])):
        raise ValueError('Synchronous past-only eight-step schema required')
    mask = mask.astype(bool)
    d = np.diff(h, axis=1); speed = np.linalg.norm(d, axis=2)
    moving = speed > 0; last = 6-np.argmax(moving[:, ::-1], axis=1)
    direction = d[np.arange(n), last]; norm = np.linalg.norm(direction, axis=1)
    unit = np.divide(direction, norm[:, None], out=np.zeros_like(direction), where=norm[:, None] > 0)
    unit[norm == 0] = [1, 0]
    rotation = np.stack((unit, np.column_stack((-unit[:, 1], unit[:, 0]))), axis=2)
    scale = np.maximum(speed.sum(1), width)
    def local(x):
        return (np.einsum('nvd,ndk->nvk', x.reshape(n, -1, 2), rotation)/scale[:, None, None]).reshape(x.shape)
    lh = local(h-h[:, -1, None]); ld = np.diff(lh, axis=1)
    both = moving[:, 1:] & moving[:, :-1]
    angle = np.arctan2(d[:, :-1, 0]*d[:, 1:, 1]-d[:, :-1, 1]*d[:, 1:, 0],
                      (d[:, :-1]*d[:, 1:]).sum(2))
    motion = np.column_stack((lh.reshape(n, -1), ld.reshape(n, -1),
        np.diff(ld, axis=1).reshape(n, -1), speed/scale[:, None],
        np.where(both, np.sin(angle), 0), np.where(both, np.cos(angle), 0), moving))
    rel = local(np.where(mask[..., None], pos-h[:, None], 0))
    distance = np.linalg.norm(rel, axis=3); count = mask.sum(1)
    mean = rel.sum(1)/np.maximum(count[..., None], 1)
    nearest = np.min(np.where(mask, distance, np.inf), axis=1)
    nearest[count == 0] = 0
    pair = mask[:, :, 1:] & mask[:, :, :-1]
    closing = np.where(pair, distance[:, :, :-1]-distance[:, :, 1:], 0)
    paired_count = pair.sum(1)
    neighbor = np.column_stack((count/8, nearest, mean.reshape(n, -1), count > 0,
        paired_count/8, closing.sum(1)/np.maximum(paired_count, 1),
        np.maximum(closing, 0).max(1), (pair & (closing > 0)).sum(1)/np.maximum(paired_count, 1)))
    assert motion.shape == (n, 68) and neighbor.shape == (n, 68)
    return motion, neighbor


def design_inputs(prediction, envelope, summary, history, neighbor, arm):
    if arm not in ARMS:
        raise ValueError('Unregistered context arm')
    p, e = np.asarray(prediction, float), np.asarray(envelope, float)
    risk = risk_features(p, e)
    summary, history, neighbor = map(lambda a: np.asarray(a, float), (summary, history, neighbor))
    if (summary.shape != (len(e), 7) or history.shape != (len(e), 68)
            or neighbor.shape != (len(e), 68) or np.isinf(summary).any()
            or not np.isfinite(history).all() or not np.isfinite(neighbor).all()):
        raise ValueError('Fixed past context schema required')
    parts = [np.log1p(np.column_stack((e, p))), risk]
    if arm == 'old_summary':
        parts += [np.nan_to_num(summary, nan=0), np.isfinite(summary)]
    elif arm in ('ordered_history', 'history_neighbors'):
        parts.append(history)
        if arm == 'history_neighbors': parts.append(neighbor)
    return np.column_stack(parts)


def fit(x, p, y, envelope, sites, excluded, *, ridge=.1):
    x, p, y, e, sites = map(np.asarray, (x, p, y, envelope, sites))
    if (x.ndim != 2 or p.shape != (len(x), 4) or y.shape != p.shape
            or sites.shape != (len(x),) or e.shape != sites.shape
            or set(sites) & set(excluded) or len(set(sites)) != 2 or ridge != .1
            or not np.isfinite(x).all() or not np.isfinite(p).all()
            or np.isinf(y).any() or not np.isfinite(e).all() or (e < 0).any()):
        raise ValueError('Exactly two fitting localities and excluded scoring/outer localities required')
    known = np.isfinite(y).all(1)
    if not np.array_equal(np.isnan(y).all(1), ~known):
        raise ValueError('Paired supported labels required')
    use = known & (e > 0); w = np.zeros(len(x))
    for site in sorted(set(sites)):
        ix = use & (sites == site)
        if not ix.any(): raise ValueError('Missing fitting locality support')
        w[ix] = 1/(2*ix.sum())
    xx, ww = x[use].astype(float), w[use]
    mean = ww@xx; std = np.sqrt(ww@((xx-mean)**2)).clip(1e-6)
    z = np.column_stack((np.ones(use.sum()), np.clip((xx-mean)/std, -8, 8)))
    rms = max(float(np.sqrt(ww@(y[use, 3]**2))), 1e-8)
    response = (y[use, 3]-p[use, 3])/rms
    penalty = np.eye(z.shape[1])*ridge; penalty[0, 0] = 0
    lhs = z.T@(ww[:, None]*z)+penalty; rhs = z.T@(ww*response)
    beta = np.linalg.solve(lhs, rhs)
    np.testing.assert_allclose(lhs@beta, rhs, rtol=1e-8, atol=1e-9)
    return dict(mean=mean.tolist(), std=std.tolist(), beta=beta.tolist(), rms=rms,
        ridge=ridge, training_sites=sorted(set(sites)), excluded=sorted(excluded),
        known_rows=int(known.sum()), positive_envelope_rows=int(use.sum()),
        unconstrained_fitting_residual_MSE=float(ww@((response-z@beta)**2)),
        independent_calibration=False)


def predict(model, x, prediction, envelope):
    x, p, e = np.asarray(x, float), np.asarray(prediction, float), np.asarray(envelope, float)
    risk_features(p, e)
    if x.shape != (len(p), len(model['mean'])) or not np.isfinite(x).all():
        raise ValueError('Finite aligned inference features required; no target argument')
    z = np.column_stack((np.ones(len(x)), np.clip((x-model['mean'])/model['std'], -8, 8)))
    result = p.copy()
    result[:, 3] = np.clip(p[:, 3]+z@np.asarray(model['beta'])*model['rms'], 0, p[:, 1])
    return result


def separated_queries(recordings, frames, span=228):
    """Greedy non-overlapping [query-84, query+144] intervals; not independence."""
    records = np.unique(np.column_stack((recordings, frames)), axis=0)
    total = 0
    for recording in np.unique(records[:, 0]):
        last = -np.inf
        for frame in records[records[:, 0] == recording, 1]:
            if frame-last > span:
                total += 1; last = frame
    return total


def support(y, env, sites, recordings, agents, frames):
    y, e, ss, rr, aa, ff = map(np.asarray, (y, env, sites, recordings, agents, frames))
    rows = []
    for site in sorted(set(ss)):
        known = (ss == site) & np.isfinite(y).all(1) & (e > 0)
        event = known & (y[:, 3] > 0)
        keys = np.column_stack((rr[event], aa[event]))
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        mass = np.bincount(inverse, weights=y[event, 3])
        def count(mask, columns): return len(np.unique(np.column_stack([a[mask] for a in columns]), axis=0))
        rows.append(dict(site=str(site), supported_rows=int(known.sum()), easy_harm_rows=int(event.sum()),
            recordings=int(len(np.unique(rr[known]))), event_recordings=int(len(np.unique(rr[event]))),
            tracks=count(known, [rr, aa]), event_tracks=len(mass),
            agent_queries=count(known, [rr, aa, ff]), event_agent_queries=count(event, [rr, aa, ff]),
            event_scene_queries=count(event, [rr, ff]),
            event_nonoverlap_recording_queries=separated_queries(rr[event], ff[event]),
            event_track_mass_effective_count=float(mass.sum()**2/(mass@mass)) if len(mass) else 0.,
            largest_event_track_mass_share=float(mass.max()/mass.sum()) if len(mass) else None,
            intervals_are_not_independent=True))
    return rows


def metrics(p, y, env):
    from src.evaluation.m3w_harm_tail_diagnostics import top_mass_share
    p, y, e = map(np.asarray, (p, y, env))
    known = np.isfinite(y).all(1) & (e > 0)
    if not known.any(): return dict(status='not_estimable')
    a, h, cap = p[known, 3], y[known, 3], p[known, 1]
    mean = float(h.mean()); w = np.full(len(h), 1/len(h))
    return dict(status='measured', rows=len(h), H_easy_MSE=float(((a-h)**2).mean()),
        H_all_MSE=float(((p[known, 1]-y[known, 1])**2).mean()), actual_harm_mean=mean,
        predicted_harm_mean=float(a.mean()), coverage=float(a.mean()/mean) if mean > 0 else None,
        top10_mass_share=top_mass_share(a, h, w, .1),
        frozen_cap_unavoidable_MSE=float((np.maximum(h-cap, 0)**2).mean()),
        harm_above_cap_rows=int((h > cap).sum()))
