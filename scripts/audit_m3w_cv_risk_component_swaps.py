"""Hold gains and counts fixed while swapping each predicted risk block."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_risk_components as study

old, base, prior, np, torch = study.old, study.base, study.prior, study.np, study.torch
BLOCKS = {'harm_only': slice(0, 2), 'signed_only': slice(2, 3), 'reference_only': slice(3, 5)}
ARMS = ('direct_matched', *BLOCKS, 'component_matched')


def restore(fit, past):
    z = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)
    if z['identity'] != fit['identity'] or z['step'] != study.STEPS:
        raise ValueError('Risk-component checkpoint mismatch')
    net = study.model.Head(past, past+37)
    net.load_state_dict(z['net'])
    return net.eval()


def main():
    torch.set_num_threads(4)
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, result = map(old.read, (f, r))
    if (result['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or not result['original_2pct']
            or result['a_labels_opened'] or result['independent_confirmation']):
        raise ValueError('Require frozen development-only choices')
    reg = old.read(study.OUT / 'registration.json')
    parent = old.read(old.checked(reg['parents'][0]))
    base.register()
    data = base.fit_population(lambda **_: None)
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    rows, changed = [], dict.fromkeys(BLOCKS, 0)
    pairs = {f'{a}_vs_direct': ('direct_matched', a) for a in ARMS[1:]}
    for c, source in zip(freeze['cells'], parent['cells'], strict=True):
        i = c['identity']
        if i != source['identity']:
            raise ValueError('Frozen gain roles differ')
        _, h = base.subset(data, i['held'])
        p, scale = study.study.parent.parent.context.context(h)
        shape = study.study.loader.load(source['fits'][1], p.shape[1]+37)
        nets = {k: restore(x, p.shape[1]) for k, x in c['fits'].items()}
        cv, ols, bank, _, pred, m, _ = study.predict(h, shape, nets)
        groups = prior.query_groups(h['query'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            anchor = z['portfolio']
            actions = {k: z[k] for k in ('direct_matched', 'component_matched')}
        for k, part in BLOCKS.items():
            t = m['direct'].copy()
            t[..., part] = m['component'][..., part]
            g, _, legal, _ = pred['direct']
            a = study.study.parent.model.independent(g, study.model.risk(t)*scale[:, None, None],
                legal, groups, .50, match=anchor)
            actions[k] = a
            changed[k] += int((a != actions['direct_matched']).sum())
        for q in groups:
            if len({int((actions[k][q] >= 0).sum()) for k in ARMS}) != 1:
                raise ValueError('Component swap altered query counts')
        paths = {base.CV: cv, base.OLS: ols,
            **{k: study.study.parent.model.apply(cv, bank, a) for k, a in actions.items()}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            site, = np.unique(h['sites'][at])
            z = {k: x[at] for k, x in paths.items()}
            scored = scorer.score_recording(z, {}, y, v, q, str(rec),
                easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(scored, source=str(site), seed=i['seed'], comparisons=terms))
        print(dict(fold=i['fold'], seed=i['seed'], views=len(rows)), flush=True)
    roster = {x['recording']: x['source'] for x in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    for k in ('direct_matched', 'component_matched'):
        if summary['risk'][k] != result['summary']['risk'][k]:
            raise ValueError('Frozen reference arm changed')
    old.once(study.OUT / 'block_swap_readout.json', old.stable(dict(parent=old.artifact(r),
        code=old.artifact(Path(__file__)), summary=summary, changed=changed, diagnostic_only=True,
        original_2pct=True, a_labels_opened=False, independent_confirmation=False)))
    print(summary['effects'], flush=True)


if __name__ == '__main__':
    main()
