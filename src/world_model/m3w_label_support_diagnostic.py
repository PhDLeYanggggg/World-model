"""Offline label-support diagnostics; never an inference or filtering policy."""
import numpy as np


def temporal_errors(reference, candidate, target, valid):
    r, c, t = [np.asarray(v, float) for v in (reference, candidate, target)]
    m = np.asarray(valid)
    if (r.ndim != 3 or r.shape[1:] != (12, 2) or c.shape != r.shape
            or t.shape != r.shape or m.shape != r.shape[:2] or m.dtype != bool
            or not np.isfinite(r).all() or not np.isfinite(c).all()
            or not np.isfinite(t[m]).all()):
        raise ValueError('Aligned twelve-step predictions and explicit label masks required')
    safe = np.where(m[..., None], t, 0)
    delta = np.where(m, np.linalg.norm(c-safe, axis=2)-np.linalg.norm(r-safe, axis=2), 0)
    count = m.sum(1)
    total = delta.sum(1)
    mean = np.divide(total, count, out=np.full(len(m), np.nan), where=count > 0)
    loo = np.divide(total[:, None]-delta, count[:, None]-1,
                    out=np.full(delta.shape, np.nan), where=m & (count[:, None] > 1))
    flip = ((mean[:, None] > 0) & (loo <= 0) | (mean[:, None] < 0) & (loo >= 0)) & m
    first = m[:, :6].sum(1)
    last = m[:, 6:].sum(1)
    a = np.divide(delta[:, :6].sum(1), first, out=np.full(len(m), np.nan), where=first > 0)
    b = np.divide(delta[:, 6:].sum(1), last, out=np.full(len(m), np.nan), where=last > 0)
    return dict(valid_steps=count, signed_error=mean, leave_one_out_defined=count > 1,
                leave_one_out_sign_flip=flip.any(1), early_late_defined=(first > 0) & (last > 0),
                early_late_opposite_sign=(a*b < 0))


def raw_future_quality(track, queries, xy, valid):
    """Read same tracker ID only; consistency cannot certify physical identity."""
    q, t, m = np.asarray(queries), np.asarray(xy), np.asarray(valid)
    if (np.diff(track['frame']) <= 0).any() or m.dtype != bool or t.shape != (len(q), 12, 2):
        raise ValueError('Ordered track and aligned requested labels required')
    grid = q[:, None]+12*np.arange(1, 13)
    ix = np.searchsorted(track['frame'], grid)
    exists = ix < len(track)
    exists[exists] &= track['frame'][ix[exists]] == grid[exists]
    np.testing.assert_array_equal(m, exists)
    use = np.minimum(ix, len(track)-1)
    boxes = np.stack([track[k][use] for k in ('x_min', 'y_min', 'x_max', 'y_max')], 2)
    fresh_xy = np.where(m[..., None], (boxes[..., :2]+boxes[..., 2:])/2, 0)
    np.testing.assert_array_equal(t, fresh_xy)
    current = np.searchsorted(track['frame'], q)
    np.testing.assert_array_equal(track['frame'][current], q)
    width = np.maximum(track['x_max'][current]-track['x_min'][current], 1.)
    count = m.sum(1)
    confidence = np.where(m, track['confidence'][use], 0).sum(1)
    confidence = np.divide(confidence, count, out=np.full(len(q), np.nan), where=count > 0)
    changed = m & (track['class_id'][use] != track['class_id'][current, None])
    relative_width = np.abs((boxes[..., 2]-boxes[..., 0])-
                           (track['x_max'][current]-track['x_min'][current])[:, None])/width[:, None]
    change = np.max(np.where(m, relative_width, 0), axis=1)
    change[count == 0] = np.nan
    return dict(future_mean_detector_confidence=confidence, future_class_id_changed=changed.any(1),
                future_max_width_change_over_current_width=change)


def cohort(y, selected, env, ids, recordings, agents, valid_steps, temporal):
    y, take, e = np.asarray(y), np.asarray(selected), np.asarray(env)
    n = len(y)
    if y.shape != (n, 5) or take.shape != (n,) or take.dtype != bool or e.shape != (n,):
        raise ValueError('Five moments, causal frozen actions and envelope required')
    known = np.isfinite(y).all(1)
    if not (known | np.isnan(y).all(1)).all() or not np.array_equal(known, valid_steps > 0):
        raise ValueError('Unknown outcomes cannot be zero-filled')
    yes = take & known
    harm = yes & (y[:, 1] > 0)
    safe = yes & ~harm
    strata = {'unknown': valid_steps == 0, 'one_to_three': (valid_steps > 0) & (valid_steps <= 3),
              'four_to_eleven': (valid_steps >= 4) & (valid_steps <= 11), 'all_twelve': valid_steps == 12}
    out = {}
    for name, mask in strata.items():
        chosen, observed = take & mask, yes & mask
        nh = harm & mask
        nd = nh & temporal['leave_one_out_defined']
        ed = nh & temporal['early_late_defined']
        h, r, eh, er = [float(y[observed, k].sum()) for k in (1, 2, 4, 3)]
        out[name] = dict(rows=int(mask.sum()), selected=int(chosen.sum()), known_selected=int(observed.sum()),
            harmful=int(nh.sum()), nonharmful=int((safe & mask).sum()), harm=h, reference=r,
            easy_harm=eh, easy_reference=er, easy_harm_ratio=eh/er if er > 0 else None,
            unknown_envelope=float(e[chosen & ~known].sum()),
            leave_one_out_defined_harmful=int(nd.sum()),
            leave_one_out_fragile_harmful=int((nd & temporal['leave_one_out_sign_flip']).sum()),
            early_late_defined_harmful=int(ed.sum()),
            early_late_opposite_harmful=int((ed & temporal['early_late_opposite_sign']).sum()))
    return dict(strata=out, unique_rows=int(len(np.unique(ids))),
                unique_tracks=len(set(zip(np.asarray(recordings).tolist(), np.asarray(agents).tolist()))))


def contrasts(values, harmful, safe, recordings, frames):
    """Descriptive harmful-minus-nonharmful means; paired within the same query."""
    v, h, s = np.asarray(values, float), np.asarray(harmful), np.asarray(safe)
    if v.shape != h.shape or s.shape != h.shape or h.dtype != bool or s.dtype != bool or (h & s).any():
        raise ValueError('Disjoint aligned diagnostic cohorts required')
    h = h & np.isfinite(v)
    s = s & np.isfinite(v)
    raw = float(v[h].mean()-v[s].mean()) if h.any() and s.any() else None
    groups = {}
    for i in np.flatnonzero(h | s):
        groups.setdefault((int(recordings[i]), int(frames[i])), []).append(i)
    differences = []
    for ix in groups.values():
        at = np.asarray(ix)
        if h[at].any() and s[at].any():
            differences.append(float(v[at[h[at]]].mean()-v[at[s[at]]].mean()))
    return dict(harmful_n=int(h.sum()), safe_n=int(s.sum()), raw_mean_difference=raw,
                matched_queries=len(differences),
                query_matched_mean_difference=float(np.mean(differences)) if differences else None)


def locality_interval(groups, extractor, draws=3000, seed=40091):
    site = {}
    for g in groups:
        value = extractor(g)
        if value is not None:
            site.setdefault(g['source'], []).append(float(value))
    point = {s:float(np.mean(v)) for s,v in sorted(site.items())}
    if len(point) < 2:
        return dict(localities=point, mean=None, CI95=None, status='insufficient_localities')
    a = np.asarray(list(point.values()))
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, len(a), (draws, len(a)))].mean(1)
    return dict(localities=point, mean=float(a.mean()), CI95=np.quantile(boot, [.025, .975]).tolist(),
                bootstrap_draws=draws, status='nominal_exposed_development_association_not_causal')
