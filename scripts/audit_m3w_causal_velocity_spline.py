"""Decompose frozen spline performance by label support, difficulty and pairs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_causal_velocity_spline as m

old, base, prior, np = m.old, m.base, m.prior, m.np
ARMS = (*m.model.MODES, 'selected')
PAIR = ((0, 1), (1, 2), (3, 4))
KEYS = ('peers_vs_ego', 'completion_vs_peers', 'selected_vs_ols4')


def slice_metrics(cv, ols, path, d, cut):
    v = d['valid']
    y = np.where(v[..., None], d['target'], 0.)
    ref, o, p, diff = (np.linalg.norm(x, axis=-1) for x in (cv-y, ols-y, path-y, path-ols))
    w = old.weight(d['query'])
    n = v.sum(1)
    actual = np.where(v, o-p, 0.).sum(1)
    radius = np.where(v, 0., diff).sum(1)
    denom = float(w @ np.where(v, ref, 0.).sum(1))
    if denom <= 0.:
        raise ValueError('No source CV reference')
    cv_ade = np.where(v, ref, 0.).sum(1)/n.clip(1)
    easy = (n > 0) & (cv_ade > 0.) & (cv_ade <= cut[0])
    hard = (n > 0) & (cv_ade > cut[1])

    def conditional(take):
        den = w[take] @ np.where(v[take], o[take], 0.).sum(1)
        return float(100*(w[take] @ actual[take])/den) if den > 0. else None

    def pp(take, x):
        return float(100*(w[take] @ x[take])/denom)

    full, partial, none = v.all(1), (n > 0) & ~v.all(1), n == 0
    return dict(complete_known=pp(full, actual), partial_known=pp(partial, actual),
        partial_debit=pp(partial, radius), wholly_missing_debit=pp(none, radius),
        easy_known_improvement_percent_vs_ols4=conditional(easy),
        hard_known_improvement_percent_vs_ols4=conditional(hard),
        mean_peak_deviation_pixel=float(w @ diff.max(1)/w.sum()))


def main():
    f, r = (m.OUT / n for n in ('freeze.json', 'readout.json'))
    freeze, result = map(old.read, (f, r))
    if (result['freeze'] != old.sha(f) or not freeze['before_held_scoring']
            or result['a_labels_opened'] or result['independent_confirmation']):
        raise ValueError('Require frozen development forecasts')
    base.register()
    d = base.fit_population(lambda **_: None)
    slices, geometry = {}, {}
    for c in freeze['cells']:
        i = c['identity']
        _, h = base.subset(d, i['held'])
        cv, ols = prior.study.motion.bank(h['history']).swapaxes(0, 1)[:2]
        with np.load(old.checked(c['predictions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            p = {k: z[k] for k in m.model.MODES}
        p['selected'] = p[c['selected']]
        for s in sorted(set(h['sites'])):
            at = h['sites'] == s
            z = {k: h[k][at] for k in ('target', 'valid', 'query', 'history')}
            key = (str(s), i['seed'])
            slices[key] = {k: slice_metrics(cv[at], ols[at], p[k][at], z, c['cut']) for k in ARMS}
            # Selected versus OLS4 uses the frozen validation choice.
            paths = np.stack([p['ego'][at], p['peers'][at], p['completion'][at], ols[at], p['selected'][at]])
            known, lower, totals = np.zeros(len(PAIR)), np.zeros(len(PAIR)), np.zeros(2)
            for g in prior.query_groups(z['query']):
                _, effect, _, (steps, observed, _, _) = prior.geometry.query(
                    z['history'][g], paths[:, g], z['target'][g], z['valid'][g],
                    PAIR)
                if steps:
                    lower += effect[:, 1]*steps
                    totals[1] += steps
                if observed:
                    known += effect[:, 0]*observed
                    totals[0] += observed
            geometry[key] = np.stack((known, lower), axis=1)/np.where(totals > 0., totals, np.nan)
        print(dict(fold=i['fold'], seed=i['seed']), flush=True)
    sites = sorted({s for s, _ in slices})
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    cells = lambda f: [[f(s, k) for k in base.SEEDS] for s in sites]
    fields = next(iter(slices.values()))['ego']
    aggregate = {a: {f: prior.geometry.bootstrap(cells(lambda s, k: slices[s, k][a][f]), draw)
                    for f in fields} for a in ARMS}
    paired = {name: {field: prior.geometry.bootstrap(cells(lambda s, k: geometry[s, k][j, a]), draw)
                    for a, field in enumerate(('known_pair_gain_pixel', 'lower_pair_gain_pixel'))}
              for j, name in enumerate(KEYS)}
    for a in ARMS:
        gains = [aggregate[a][f]['point'] for f in ('complete_known', 'partial_known',
                                                     'partial_debit', 'wholly_missing_debit')]
        np.testing.assert_allclose(gains[0]+gains[1]-gains[2]-gains[3],
            result['summary']['effects'][f'{a}_vs_ols4']['completion_lower_pp']['point'], atol=1e-8)
    old.once(m.OUT / 'slices.json', old.stable(dict(parent=old.artifact(r), code=old.artifact(Path(__file__)),
        slices=aggregate, paired_geometry=paired, diagnostic_only=True,
        independent_confirmation=False, a_labels_opened=False)))


if __name__ == '__main__':
    main()
