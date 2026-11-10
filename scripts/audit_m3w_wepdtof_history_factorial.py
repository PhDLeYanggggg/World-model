"""Separate history anchor and velocity effects on the already scored WEPDTOF roster."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_external_transfer as old
from src.evaluation.m3w_selection_forecast_readout import errors, policy

base, np = old.base, old.np
OUT = old.OUT.parent/'wepdtof_history_factorial_v1'
ARM = ('detector', 'annotation_anchor', 'annotation_velocity', 'annotation_both')
MODE = ('cv', 'ols4')
TIMES = np.arange(1., 13.)[None, :, None]


def fit_line(h):
    """Fit the slope and last-time intercept on exactly four observed points."""
    x = h[:, -4:]
    centered = x-x[:, -1:]
    slope = np.einsum('t,ntd->nd', [-1.5, -.5, .5, 1.5], centered)/5.
    return slope, centered.mean(1)+1.5*slope


def refit(h):
    slope, offset = fit_line(h)
    return offset[:, None]+TIMES*slope[:, None]


def crossed(h, a):
    if h.shape != a.shape or h.ndim != 3 or h.shape[1:] != (8, 2) or not (
            np.isfinite(h).all() and np.isfinite(a).all()):
        raise ValueError('Expected matched finite observation histories')
    det, ann = old.method.analytic(h), old.method.analytic(a)
    shift = (a[:, -1]-h[:, -1])[:, None]
    return {m: dict(detector=d, annotation_anchor=d+shift, annotation_velocity=x,
                    annotation_both=x+shift) for m, d, x in zip(MODE, det, ann, strict=True)}


def annotation_histories(windows, q, ids, people):
    out = np.full((len(q), 8, 2), np.nan)
    for k, w in enumerate(windows):
        frames = [f['frame'] for f in w['frames']]
        if frames != [w['start']+t*base.STRIDE for t in range(8)]:
            raise ValueError('Observation clock changed')
        links = [base.associate(np.asarray(f['rows']).reshape(-1, 7),
                                {j: z[t] for j, z in people.items() if t in z})
                 for t, f in zip(frames, w['frames'], strict=True)]
        stable = dict(set.intersection(*links))
        for i in np.flatnonzero(q == k):
            j = stable.get(ids[i])
            if j is not None:
                out[i] = [people[j][t][:2] for t in frames]
    return out


def factor(ade):
    """Average both intervention orders to allocate the total ADE change."""
    d, p, v, b = (ade[k] for k in ARM)
    return dict(anchor=.5*((d-p)+(v-b)), velocity=.5*((d-v)+(p-b)), total=d-b)


def average(v):
    a = v[np.isfinite(v)]
    return float(a.mean()) if len(a) else None


def error_summary(p, y, v):
    ade, fde = errors(p, y, v)
    z = ade[np.isfinite(ade)]
    return dict(ade=average(ade), fde=average(fde), p95=float(np.quantile(z, .95)) if len(z) else None)


def analyze(windows, item, original, cut):
    h, q, ids = old.camera_arrays(windows)
    clip = item['clip']
    people = old.references(clip)
    a = annotation_histories(windows, q, ids, people)
    y, v, matched, refs = old.outcomes(windows, h, q, ids, people)
    np.testing.assert_array_equal(np.isfinite(a).all((1, 2)), matched)
    for k, val in dict(online=len(h), matched=int(matched.sum()), full=int(v.all(1).sum()),
                       missing=int((~v.any(1)).sum()), reference_histories=refs).items():
        if val != original[k]:
            raise ValueError(f'Original {k} mismatch: {clip}')
    file = old.checked(item['artifact'])
    cv, ols = old.method.analytic(h)
    with np.load(file, allow_pickle=False) as frozen:
        np.testing.assert_array_equal(q, frozen['q'])
        np.testing.assert_array_equal(ids, frozen['ids'])
        for s in old.SEEDS:
            for m, p in zip(MODE, (cv, ols), strict=True):
                np.testing.assert_array_equal(p, frozen[f'{s}_{m}'])
                if len(h):
                    values = error_summary(p, y, v)
                    r = original['seeds'][str(s)]['arms'][m]
                    for k, key in (('ade', 'ade_px'), ('fde', 'fde_px'), ('p95', 'p95_ade_px')):
                        if values[k] != r[key]:
                            raise ValueError('Original baseline metric did not replay')
    f = refit(h)
    safe = None
    all_scores = {k: error_summary(p, y, v) for k, p in dict(cv=cv, ols4=ols, ols4_refit=f).items()}
    if len(h):
        risk = policy(f, cv, cv, y, v, np.full(len(h), clip), easy_cut=cut[0], hard_cut=cut[1])
        safe = risk['recordings'][clip]
    use = matched & v.any(1)
    a0, h0 = a[use], h[use]
    scores = {k: {x: error_summary(p, y[use], v[use]) for x, p in z.items()}
              for k, z in crossed(h0, a0).items()}
    change = {k: factor({x: r['ade'] for x, r in z.items()}) for k, z in scores.items()} if use.any() else None
    return dict(source=item['source'], roster=len(h), stable_histories=int(matched.sum()), scored_histories=int(use.sum()),
                endpoint_rows=int(v[:, -1].sum()), available_steps=int(v.sum()),
                center_error=average(np.linalg.norm(a0[:, -1]-h0[:, -1], axis=-1)),
                factorial=scores, contributions=change, full_roster=all_scores, refit_risk=safe)


def contrast(rows, source_list, fun):
    """Equal clips within source, then equal sources; no window resampling."""
    vals, counts = {}, {}
    for site in source_list:
        x = [fun(r) for r in rows.values() if r['source'] == site]
        v = [a for a in x if a is not None]
        counts[site] = len(v)
        if v:
            vals[site] = float(np.mean(v))
    x = np.array(list(vals.values()))
    point, bounds = (None, None) if not len(x) else (float(x.mean()), None)
    if len(x) >= 2:
        bootstrap = np.random.default_rng(82317).choice(x, size=(3000, len(x))).mean(1)
        bounds = np.quantile(bootstrap, [.025, .975]).tolist()
    return dict(point=point, ci95=bounds, estimable_sources=len(vals), all_sources=len(source_list),
                by_source=vals, contributing_clips=counts)


def summary(rows):
    sources = sorted({r['source'] for r in rows.values()})
    def ci(f):
        return contrast(rows, sources, f)

    def gap(r, k):
        x, y = [r['factorial'][m][k]['ade'] for m in MODE]
        return x-y if x is not None and y is not None else None

    def interaction(r):
        d, a = gap(r, 'detector'), gap(r, 'annotation_both')
        return d-a if d is not None and a is not None else None

    factorial = {m: {a: {k: ci(lambda r: r['factorial'][m][a][k]) for k in ('ade', 'fde', 'p95')}
                     for a in ARM} for m in MODE}
    contributions = {m: {k: ci(lambda r: r['contributions'][m][k] if r['contributions'] else None)
                         for k in ('anchor', 'velocity', 'total')} for m in MODE}
    direct = {k: {m: ci(lambda r: r['full_roster'][k][m]) for m in ('ade', 'fde', 'p95')}
              for k in ('cv', 'ols4', 'ols4_refit')}

    def refit_gain(r, k):
        a, b = (r['full_roster'][m][k] for m in ('ols4', 'ols4_refit'))
        return a-b if a is not None and b is not None else None

    risks = [r['refit_risk'] for r in rows.values() if r['refit_risk'] is not None]
    return dict(rows={k: sum(r[k] for r in rows.values()) for k in (
        'roster', 'stable_histories', 'scored_histories', 'available_steps', 'endpoint_rows')},
        factorial=factorial, detector_minus_annotation_velocity_gap=ci(interaction),
        cv_minus_ols4_ade={k: ci(lambda r: gap(r, k)) for k in ARM},
        contributions=contributions, full_roster=direct,
        refit_gain_vs_ols4={k: ci(lambda r: refit_gain(r, k)) for k in ('ade', 'fde', 'p95')},
        refit_risk=dict(clips=len(rows), with_online_roster=len(risks), supported=sum(r['supported'] for r in risks),
                        undefined_easy=sum(r['completion_easy_risk_upper'] is None for r in risks),
                        worst_signed_easy_degradation=max((r['known_easy_net_degradation'] for r in risks
                            if r['known_easy_net_degradation'] is not None), default=None)))


def main():
    p, f = (old.OUT/k for k in ('external_readout.json', 'augmented_freeze.json'))
    r, freeze = old.read(p), old.read(f)
    if (r['forecast_freeze'] != base.sha(f) or not r['external_scoring_completed'] or r['external_tuning']
            or not r['original_2pct'] or r['dronecrowd_confirmation_access'] or len(freeze['rows']) != 15):
        raise ValueError('Require the complete original WEPDTOF readout')
    camera, windows = old.frozen_camera()
    if old.checked(freeze['camera']) != camera:
        raise ValueError('Frozen observations changed')
    original = {x['clip']: x for x in r['rows']}
    clips = {x['clip'] for x in freeze['rows']}
    if set(original) != clips or set(windows) != clips or len({x['source'] for x in original.values()}) != 13:
        raise ValueError('Cannot expand or change the previously scored sources')
    cut = old.read(old.OUT/'pilot_freeze.json')['cut_px']
    result = {}
    for item in freeze['rows']:
        c = item['clip']
        result[c] = analyze(windows[c], item, original[c], cut)
        print('verified', len(result), len(clips), flush=True)
    s = summary(result)
    if s['rows']['roster'] != 431 or s['rows']['stable_histories'] != 353:
        raise ValueError('Historical denominators changed')
    base.immutable(OUT/'readout.json', dict(binding={k: base.sha(x) for k, x in dict(
        old_readout=p, old_freeze=f, camera=camera, driver=Path(__file__), protocol=OUT/'protocol.md').items()},
        result_source='fresh_retrospective_evaluation', by_clip=result, summary=s, units='image_pixels',
        source_bootstrap_draws=3000, independent_confirmation=False, previously_opened_outcomes_reread=True,
        reserved_outcomes_read=False, original_2pct=True, training=False, deployment_changed=False))


if __name__ == '__main__':
    main()
