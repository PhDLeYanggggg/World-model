"""Compare risk parameterizations at the realized low-coverage query counts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_risk_component_swaps as swap

study, old, base, prior, np, torch = swap.study, swap.old, swap.base, swap.prior, swap.np, swap.torch
ARMS = ('direct', 'component', 'joint')
PAIRS = {'components_at_low_count': ('direct', 'component'), 'joint_at_low_count': ('component', 'joint')}


def main():
    torch.set_num_threads(4)
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, result = map(old.read, (f, r))
    if (result['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or not result['original_2pct']
            or result['a_labels_opened'] or result['independent_confirmation']):
        raise ValueError('Require frozen development-only low-coverage choices')
    reg = old.read(study.OUT / 'registration.json')
    parent = old.read(old.checked(reg['parents'][0]))
    base.register()
    data = base.fit_population(lambda **_: None)
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    rows, pass_rows = [], {}
    for c, source in zip(freeze['cells'], parent['cells'], strict=True):
        i = c['identity']
        if i != source['identity']:
            raise ValueError('Frozen gain roles differ')
        _, h = base.subset(data, i['held'])
        past, _ = study.study.parent.parent.context.context(h)
        shape = study.study.loader.load(source['fits'][1], past.shape[1]+37)
        nets = {k: swap.restore(x, past.shape[1]) for k, x in c['fits'].items()}
        cv, ols, b, _, pred, _, _ = study.predict(h, shape, nets)
        groups = prior.query_groups(h['query'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            anchor = z['component_free']
        a = {k: study.study.parent.model.independent(*pred[k][:3], groups, .50, match=anchor)
             for k in study.MODES}
        a['joint'] = anchor
        for q in groups:
            if len({int((a[k][q] >= 0).sum()) for k in ARMS}) != 1:
                raise ValueError('Low-coverage per-query counts changed')
        paths = {base.CV: cv, base.OLS: ols, **{k: study.study.parent.model.apply(cv, b, x) for k, x in a.items()}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            s, = np.unique(h['sites'][at])
            z = {k: x[at] for k, x in paths.items()}
            scored = scorer.score_recording(z, {}, y, v, q, str(rec),
                easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in PAIRS.items()}
            rows.append(dict(scored, source=str(s), seed=i['seed'], comparisons=terms))
            flags = np.array([scored['metrics'][k]['risk']['supported'] for k in ARMS], float)
            pass_rows.setdefault((str(s), i['seed']), []).append(flags)
        print(dict(fold=i['fold'], seed=i['seed'], views=len(rows)), flush=True)
    roster = {x['recording']: x['source'] for x in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    if summary['risk']['joint'] != result['summary']['risk']['component_free']:
        raise ValueError('Original low-count joint policy changed')
    sites = sorted(set(roster.values()))
    x = np.asarray([[np.mean(pass_rows[s, k], axis=0) for k in base.SEEDS] for s in sites])
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    pass_effects = {k: prior.geometry.bootstrap(100.*(x[:, :, ARMS.index(b)]-x[:, :, ARMS.index(a)]), draw)
                    for k, (a, b) in PAIRS.items()}
    old.once(study.OUT / 'low_count_readout.json', old.stable(dict(parent=old.artifact(r),
        code=old.artifact(Path(__file__)), summary=summary, pass_effects_pp=pass_effects,
        exact_query_match=True, diagnostic_only=True, original_2pct=True, a_labels_opened=False,
        independent_confirmation=False)))
    print(summary['risk'], pass_effects, flush=True)


if __name__ == '__main__':
    main()
