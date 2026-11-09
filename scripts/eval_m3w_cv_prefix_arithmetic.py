"""Test direct arithmetic use of predicted state at the same query counts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_prefix_state as study

old, base, prior, np, torch = study.old, study.base, study.prior, study.np, study.torch
KEYS = ('arithmetic_local', 'arithmetic_state')


def choose(packet, n):
    out = {k: np.full(n, -1, np.int16) for k in KEYS}
    prefix, rec = np.zeros(8), None
    for p in packet:
        if p['rec'] != rec:
            prefix, rec = np.zeros(8), p['rec']
        d = p['predicted']
        norm = p['scale'] * len(p['at'])
        for k in KEYS:
            risk = (study.model.marginal(prefix, d) if k == 'arithmetic_state'
                    else study.model.marginal(np.zeros(8), d))
            y = np.column_stack((d[:, :2] / p['scale'], risk / norm, np.zeros(len(d))))
            i = study.model.pick(y, p['counts'], 2., 0., p['anchor_count'])
            out[k][p['at']] = p['plans'][i]
        prefix += d[p['anchor']]
    return out


def main():
    torch.set_num_threads(4)
    f, r = (study.OUT / k for k in ('freeze.json', 'readout.json'))
    freeze, result = map(old.read, (f, r))
    if (result['freeze'] != old.sha(f) or not freeze['before_held_scoring'] or not result['original_2pct']
            or result['a_labels_opened'] or result['independent_confirmation']):
        raise ValueError('Require frozen development-only readout')
    reg = old.read(study.OUT / 'registration.json')
    shape = old.read(old.checked(reg['parents'][0]))
    base.register()
    d = base.fit_population(lambda **_: None)
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{a}_vs_{k}': (b, a) for a in KEYS for k, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs['arithmetic_at_count'] = KEYS
    rows, changed = [], 0
    for c, parent in zip(freeze['cells'], shape['cells'], strict=True):
        i = c['identity']
        if i != parent['identity']:
            raise ValueError('Changed source role')
        _, h = base.subset(d, i['held'])
        past, _ = study.study.parent.parent.context.context(h)
        net = study.study.loader.load(parent['fits'][1], past.shape[1]+37)
        cv, ols, bank, anchor, packet = study.prepare(h, net)
        a = choose(packet, len(cv))
        for p in packet:
            at = p['at']
            if any(int((a[k][at] >= 0).sum()) != int((anchor[at] >= 0).sum()) for k in KEYS):
                raise ValueError('Query count changed')
        changed += int((a[KEYS[0]] != a[KEYS[1]]).sum())
        paths = {base.CV: cv, base.OLS: ols,
            **{k: study.study.parent.model.apply(cv, bank, x) for k, x in a.items()}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q = (h[k][at] for k in ('target', 'valid', 'query'))
            site, = np.unique(h['sites'][at])
            z = {k: x[at] for k, x in paths.items()}
            scored = scorer.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(scored, source=str(site), seed=i['seed'], comparisons=terms))
        print(dict(fold=i['fold'], seed=i['seed'], views=len(rows)), flush=True)
    roster = {x['recording']: x['source'] for x in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    out = dict(parent=old.artifact(r), code=old.artifact(Path(__file__)), summary=summary, changed=changed,
        matched_exact_counts=True, original_2pct=True, independent_confirmation=False, a_labels_opened=False)
    old.once(study.OUT / 'arithmetic_readout.json', old.stable(out))
    print(summary['effects']['arithmetic_at_count'], flush=True)


if __name__ == '__main__':
    main()
