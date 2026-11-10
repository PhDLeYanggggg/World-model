"""Decompose the matched angular result into observed utility and missing debit."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_exchange_angular_head as fit


run, np, torch, matched, old, base = fit.run, fit.np, fit.torch, fit.matched, fit.old, fit.base
OUT = fit.OUT.parent/'opendd_exchange_angular_debit_v1'
CONTRASTS = (('relative', 'no_swap'), ('original', 'no_swap'), ('relative', 'original'))


def components(ref, pred, y, valid):
    known, lower = fit.direction.prior.parent.transfer.paired(ref, pred, y, valid)
    debit = np.linalg.norm(ref-pred, axis=-1)[~valid].sum()
    return dict(observed=float(known.sum()), debit=float(debit), lower=float(lower.sum()))


def evaluate(args, net):
    site, seed, motion, d, bank, menu, width, frozen, cut = args
    x, legal, rows, path, _ = menu
    cv, ols = bank[:2]
    fraction = fit.direction.model.bridge_fraction(cv, ols, path)
    x0, _ = fit.model.features(d['h'], cv, ols, fraction)
    with torch.inference_mode():
        z = net(torch.from_numpy(x0)).double()
        new = fit.model.rotate(z, torch.from_numpy(cv), torch.from_numpy(path)).numpy()
    np.testing.assert_allclose(np.linalg.norm(new-cv, axis=-1), np.linalg.norm(path-cv, axis=-1), atol=1e-11, rtol=1e-11)
    move = base.model.movement_prefix(cv, path, rows)
    total = {m: {f'{a}_minus_{b}': Counter(dict(observed=0., debit=0., lower=0.)) for a, b in CONTRASTS}
             for m in fit.MASKS}
    counts = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, q, r = old.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a, cap = np.asarray(f['score_plan']), f['cap']
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound anchor')
        pairs = old.model.exchange_menu(a, legal[q], move[q], cap)
        if not len(pairs):
            continue
        z = old.model.features(x[q], legal[q], move[q], a, cap, pairs)[0]
        with torch.inference_mode():
            scores = frozen['aligned'](matched.exp.design(torch.from_numpy(z), width, 'aligned'))
        chosen = dict(guarded=old.model.rank(scores), all_eligible=matched.matched_choice(scores))
        anchor = old.measure(d, ai, bank, path, r, a, cut)[0]
        y, v = d['y'][ai], d['valid'][ai]
        for m, j in chosen.items():
            if j is None:
                continue
            counts[m] += 1
            src, dst = pairs[j]
            p = old.measure(d, ai, bank, path, r, old.model.exchanged(a, src, dst), cut)[0]
            row = r[dst, a[dst]]
            if row < 0:
                raise ValueError('Illegal recipient')
            q = p.copy()
            q[row] = new[np.flatnonzero(ai)[row]]
            values = dict(no_swap=anchor, original=p, relative=q)
            for a0, b in CONTRASTS:
                total[m][f'{a0}_minus_{b}'].update(components(values[b], values[a0], y, v))
    mass = np.linalg.norm(cv-d['y'], axis=-1)[d['valid']].sum()
    result = {}
    for m, rows in total.items():
        r, o = (rows[k] for k in ('relative_minus_no_swap', 'original_minus_no_swap'))
        if not np.isclose(r['debit'], o['debit'], atol=1e-10, rtol=1e-11):
            raise ValueError('Direction changed the missing debit against no swap')
        result[m] = {k: {a: 100.*x/mass for a, x in v.items()} for k, v in rows.items()}
        for v in result[m].values():
            if not np.isclose(v['observed']-v['debit'], v['lower'], atol=1e-12, rtol=1e-11):
                raise ValueError('Paired bound decomposition changed')
    return dict(counts=dict(counts), pp=result)


def summarize(groups):
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(m, k, f):
        v = np.asarray([np.mean([r['pp'][m][k][f] for r in g]) for g in groups.values()])
        return dict(point_pp=float(v.mean()), ci95_pp=np.quantile(v[draws].mean(1), [.025, .975]).tolist())

    return {m: {k: {f: ci(m, k, f) for f in ('observed', 'debit', 'lower')}
                for k in (f'{a}_minus_{b}' for a, b in CONTRASTS)} for m in fit.MASKS}


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p, h, r = (fit.OUT/k for k in ('registration.json', 'head_freeze.json', 'readout.json'))
    reg = run.sha(p)
    freeze, report = run.read(h), run.read(r)
    if (report['registration'] != reg or freeze['registration'] != reg or report['checkpoint_manifest'] != run.sha(h)
            or report['external_outcome_access'] or not report['original_2pct']):
        raise ValueError('Require the completed angular direction experiment')
    out = OUT/'registration.json'
    run.once(out, run.stable(dict(inputs=[run.artifact(f) for f in (Path(__file__), OUT/'protocol.md', h, r)],
        original_2pct=True, no_training=True, external_outcome_access=False)))
    groups = {}
    for args, expected in matched.context_cells():
        s, seed = args[:2]
        meta = freeze['checkpoints'][f'{s}_{seed}_relative']
        if meta['identity'] != dict(registration=reg, held=s, train=[t for t in matched.parent.source.SITES if t != s],
                                    seed=seed, arm='relative'):
            raise ValueError('Held direction fit has an invalid role')
        net = fit.study.load(meta, lambda: fit.model.AngularHead(19), old.STEPS)
        v = evaluate(args, net)
        if v['counts'].get('guarded', 0) != expected['transitions']['aligned']['exchanged']:
            raise ValueError('Changed original guard')
        groups.setdefault(s, []).append(dict(seed=seed, **v))
    summary = summarize(groups)
    for m in fit.MASKS:
        if sum(r['counts'].get(m, 0) for g in groups.values() for r in g) != report['summary']['counts'][m]:
            raise ValueError('Changed intervention mask')
        for k in ('relative_minus_no_swap', 'relative_minus_original'):
            v = report['summary']['masks'][m]['paired'][k]['gain']
            w = summary[m][k]['lower']
            np.testing.assert_allclose([v['point_pp'], *v['ci95_pp']], [w['point_pp'], *w['ci95_pp']], atol=1e-12, rtol=1e-11)
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(out), summary=summary, by_site=groups,
        original_2pct=True, no_training=True, external_outcome_access=False, deployment_changed=False)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
