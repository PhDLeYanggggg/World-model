"""Apply learned directions to the incumbent without exchanging recipients."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_exchange_angular_head as fit


run, np, torch, matched, old, base = fit.run, fit.np, fit.torch, fit.matched, fit.old, fit.base
OUT = fit.OUT.parent/'opendd_exchange_angular_same_recipients_v1'
ARMS = ('original', 'acceleration', 'turn', 'suffix', 'observed', 'relative')


def evaluate(args, nets):
    site, seed, motion, d, bank, menu, _, _, cut = args
    _, _, rows, path, _ = menu
    cv, ols = bank[:2]
    h = d['h']
    frac = fit.direction.model.bridge_fraction(cv, ols, path)
    feature, _ = fit.model.features(h, cv, ols, frac)
    dirs = fit.direction.model.direction_bank(h, cv, path)
    pick, _ = fit.direction.model.select(fit.direction.model.suffix_scores(h, frac))
    shapes = dict(original=path, acceleration=dirs[:, 3], turn=dirs[:, 4], suffix=dirs[np.arange(len(h)), pick])
    with torch.inference_mode():
        for k, net in nets.items():
            z = net(torch.from_numpy(feature)).double()
            shapes[k] = fit.model.rotate(z, torch.from_numpy(cv), torch.from_numpy(path)).numpy()
    r0 = np.linalg.norm(path-cv, axis=-1)
    for v in shapes.values():
        np.testing.assert_allclose(np.linalg.norm(v-cv, axis=-1), r0, atol=1e-11, rtol=1e-11)
    plans = {k: cv.copy() for k in ARMS}
    count = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, _, r = old.indices(d, rec, rows)
        f = run.read(base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a = np.asarray(f['score_plan'])
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound recipient plan')
        keep = r[np.arange(2) < a[:, None]]
        if (keep < 0).any() or len(keep) != len(set(keep)):
            raise ValueError('Invalid incumbent recipients')
        dst = np.flatnonzero(ai)[keep]
        for k in ARMS:
            plans[k][dst] = shapes[k][dst]
        count.update(views=1, edits=int(a.sum()))
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    count.update(known_steps=int(valid.sum()), endpoints=int(valid[:, -1].sum()))
    ref = plans['original']
    edit = np.linalg.norm(ref-cv, axis=-1).max(1) > 0.
    arms, effect = {}, {}
    for k, p in plans.items():
        np.testing.assert_array_equal(np.linalg.norm(p-cv, axis=-1).max(1) > 0., edit)
        raw, scores = fit.direction.metric_rows(p, cv, ols, d, cut)
        if raw['edits'] != count['edits']:
            raise ValueError('Directional shape changed count')
        arms[k] = scores
        known, lower = fit.direction.prior.parent.transfer.paired(ref, p, y, valid)
        effect[k] = dict(observed_pp=float(100.*known.sum()/mass), lower_pp=float(100.*lower.sum()/mass))
    if len({z['undefined'] for z in arms.values()}) != 1:
        raise ValueError('Changing only direction changed reference support')
    return dict(counts=dict(count), arms=arms, effect=effect)


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))

    def ci(fn):
        x = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())

    arms = {k: dict(sum((Counter(r['arms'][k]) for r in rows), Counter())) for k in ARMS}
    for a in arms.values():
        a.update(available_step_mean_error=a.pop('step_error')/counts['known_steps'],
                 available_endpoint_mean_error=a.pop('endpoint_error')/counts['endpoints'])
    effects = {k: {f: ci(lambda r: r['effect'][k][f]) for f in ('observed_pp', 'lower_pp')} for k in ARMS[1:]}
    for k in effects:
        effects[k]['dual_rate_pp'] = ci(lambda r: 100.*(
            r['arms'][k]['dual']-r['arms']['original']['dual'])/r['counts']['views'])
    return dict(counts=counts, arms=arms, paired=effects)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p, h, r = (fit.OUT/k for k in ('registration.json', 'head_freeze.json', 'readout.json'))
    reg = run.sha(p)
    freeze, report = run.read(h), run.read(r)
    if (report['registration'] != reg or freeze['registration'] != reg or report['checkpoint_manifest'] != run.sha(h)
            or report['external_outcome_access'] or not report['original_2pct']):
        raise ValueError('Require the frozen angular model')
    out = OUT/'registration.json'
    files = [Path(__file__), OUT/'protocol.md', h, r]
    run.once(out, run.stable(dict(inputs=[run.artifact(f) for f in files], original_2pct=True,
        no_training=True, external_outcome_access=False, keeps_incumbent_recipients=True)))
    groups = {}
    for args, expected in matched.context_cells():
        site, seed = args[:2]
        nets = {}
        for k in ('observed', 'relative'):
            meta = freeze['checkpoints'][f'{site}_{seed}_{k}']
            if meta['identity'] != dict(registration=reg, held=site,
                    train=[s for s in matched.parent.source.SITES if s != site], seed=seed, arm=k):
                raise ValueError('Invalid held-source model')
            nets[k] = fit.study.load(meta, lambda: fit.model.AngularHead(19), old.STEPS)
        v = evaluate(args, nets)
        if any(v['arms']['original'][a] != expected['scores']['frozen'][b]
               for a, b in (('dual', 'dual_positive'), ('risk_pass', 'risk_pass'), ('undefined', 'undefined'))):
            raise ValueError('Original no-exchange decision changed')
        groups.setdefault(site, []).append(dict(seed=seed, **v))
    summary = summarize(groups)
    actual = summary['arms']['original']
    reference = report['summary']['masks']['guarded']['arms']['no_swap']
    np.testing.assert_allclose([actual[k] for k in reference], list(reference.values()), atol=1e-12, rtol=1e-12)
    if (summary['counts']['views'], summary['counts']['edits']) != (606, 780):
        raise ValueError('Changed original roster')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(out), summary=summary, by_site=groups,
        no_training=True, original_2pct=True, external_outcome_access=False, deployment_changed=False)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
