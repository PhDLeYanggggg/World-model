"""Test a validation-frozen OLS4 return with an exact-displacement control."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_causal_velocity_spline as m

old, base, prior, np, torch = m.old, m.base, m.prior, m.np, m.torch
SHAPE = tuple((1.-(np.arange(1., 13.)/12.)**2).tolist())
AMPLITUDES = (.10, .25, .50, 1.)
MODES = ('uniform', 'taper')


def matched(ols, path):
    delta = path-ols
    length = np.linalg.norm(delta, axis=2)
    mass = length.sum(1)
    budget = length @ np.asarray(SHAPE)
    alpha = np.divide(budget, mass, out=np.zeros_like(mass), where=mass > 0.)
    a, b = delta*alpha[:, None, None], delta*np.asarray(SHAPE)[None, :, None]
    np.testing.assert_allclose(np.linalg.norm(a, axis=2).sum(1), np.linalg.norm(b, axis=2).sum(1),
                               atol=1e-8, rtol=1e-10)
    return dict(uniform=a, taper=b)


def calibration(d, cv, ols, path):
    roster = sorted(set(d['sites']))
    full = prior.study.motion.full_masses
    delta = matched(ols, path)
    vals = []
    for k in MODES:
        for alpha in AMPLITUDES:
            p = ols+alpha*delta[k]
            gains = []
            for site in roster:
                at = d['sites'] == site
                y, v, q = (d[f][at] for f in ('target', 'valid', 'query'))
                r = full(ols[at], p[at], cv[at], y, v, old.weight(q))
                if r['reference'] <= 0.:
                    raise ValueError('No calibration CV mass')
                gains.append(100.*(r['known_delta']-r['unknown_radius'])/r['reference'])
            vals.append(dict(mode=k, alpha=alpha, lower_pp=float(np.mean(gains))))
    best = max(vals, key=lambda x: x['lower_pp'])
    return vals, best if best['lower_pp'] > 0. else None


def main():
    torch.set_num_threads(4)
    rp, fp = (m.OUT / n for n in ('readout.json', 'freeze.json'))
    r, f = map(old.read, (rp, fp))
    if (r['freeze'] != old.sha(fp) or not f['before_held_scoring']
            or not r['original_2pct'] or r['independent_confirmation'] or r['a_labels_opened']):
        raise ValueError('Require frozen development-only forecasts')
    rp0 = m.OUT / 'tail_registration.json'
    old.once(rp0, old.stable(dict(parent=old.artifact(rp), code=old.artifact(Path(__file__)),
        exploratory_after_raw_development_readout=True, fixed_parent_arm='completion', weights=SHAPE,
        amplitudes=AMPLITUDES, matching='per-row sum of lead displacement around OLS4',
        selection='positive source-mean conservative OLS4 gain on calibration only; else OLS4',
        original_2pct=True, deployment=False, confirmation_access=False)))
    base.register()
    d = base.fit_population(lambda **_: None)
    x = m.model.causal_inputs(d['history'], prior.query_groups(d['query']))
    bank = prior.study.motion.bank(d['history'])
    frozen = []
    for c in f['cells']:
        i = c['identity']
        fit = c['fits']['completion']
        z = torch.load(old.checked(fit['artifact']), weights_only=True, map_location='cpu')
        if z['identity'] != fit['identity'] or z['step'] != m.STEPS:
            raise ValueError('Frozen parent checkpoint changed')
        net = m.model.Spline()
        net.load_state_dict(z['net'])
        ix, cal = base.subset(d, i['calibration'])
        table, best = calibration(cal, bank[ix, 0], bank[ix, 1],
            m.predict(net.eval(), [a[ix] for a in x], 'completion'))
        frozen.append(dict(identity=i, validation=table, chosen=best))
    freeze = m.OUT / 'tail_freeze.json'
    old.once(freeze, old.stable(dict(registration=old.sha(rp0), cells=frozen, before_held_scoring=True)))
    rows = []
    pairs = m.parent.vs_references((*MODES, 'selected_tail', 'completion'))
    pairs['taper_vs_uniform'] = ('uniform', 'taper')
    for c, policy in zip(f['cells'], old.read(freeze)['cells'], strict=True):
        i = c['identity']
        if i != policy['identity']:
            raise ValueError('Frozen roles changed')
        ix, h = base.subset(d, i['held'])
        cv, ols = bank[ix, 0], bank[ix, 1]
        with np.load(old.checked(c['predictions']), allow_pickle=False) as z:
            np.testing.assert_array_equal(h['ids'], z['ids'])
            parent = z['completion']
        delta = matched(ols, parent)
        spec = policy['chosen']
        p = ols if spec is None else ols+spec['alpha']*delta[spec['mode']]
        paths = {base.CV: cv, base.OLS: ols, 'completion': parent, 'selected_tail': p,
                 **{k: ols+v for k, v in delta.items()}}
        rows.extend(m.parent.rows_for(h, paths, cv, c['cut'], pairs, i))
    roster = {row['recording']: row['source'] for row in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    count = Counter(c['chosen']['mode'] if c['chosen'] else 'ols4' for c in frozen)
    old.once(m.OUT/'tail_readout.json', old.stable(dict(freeze=old.sha(freeze), summary=summary, selected=dict(count),
        exploratory=True, original_2pct=True, independent_confirmation=False, a_labels_opened=False)))
    print(dict(selected=count, effects={k: summary['effects'][k]['completion_lower_pp']
        for k in ('taper_vs_uniform', 'taper_vs_ols4', 'selected_tail_vs_ols4')}), flush=True)


if __name__ == '__main__':
    main()
