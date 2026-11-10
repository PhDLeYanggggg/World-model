"""Measure shape-only headroom on the fixed original recipient roster."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_opendd_exchange_angular_same_recipients as same
from src.evaluation import m3w_angular_incumbent_oracle as model


run, np, torch, fit, old = same.run, same.np, same.torch, same.fit, same.old
OUT = fit.OUT.parent/'opendd_angular_incumbent_oracle_v1'
ARMS = ('original', 'constant_angle', 'analytic', 'bounded_stepwise', 'unrestricted_stepwise')
ANGLES = tuple(range(-90, 91, 15))


def evaluate(args):
    site, seed, motion, d, bank, menu, _, _, cut = args
    _, _, rows, path, _ = menu
    cv, ols = bank[:2]
    p0 = cv.copy()
    count = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, _, r = old.indices(d, rec, rows)
        f = run.read(same.base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a = np.asarray(f['score_plan'])
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound incumbent')
        keep = r[np.arange(2) < a[:, None]]
        if (keep < 0).any() or len(keep) != len(set(keep)):
            raise ValueError('Invalid incumbent selection')
        g = np.flatnonzero(ai)[keep]
        p0[g] = path[g]
        count.update(views=1, edits=int(a.sum()))
    y, valid = d['y'], d['valid']
    use = np.linalg.norm(p0-cv, axis=-1).max(1) > 0.
    actual, c, t, v = p0[use], cv[use], y[use], valid[use]
    dirs = fit.direction.model.direction_bank(d['h'], cv, path)[use]
    q = dict(original=actual, constant_angle=model.choose_rowwise(actual,
        [actual, *(model.rotate(c, actual, np.deg2rad(a)) for a in ANGLES)], t, v),
        analytic=model.choose_rowwise(actual, dirs.swapaxes(0, 1), t, v),
        bounded_stepwise=model.stepwise_oracle(c, actual, t, v, np.pi/2),
        unrestricted_stepwise=model.stepwise_oracle(c, actual, t, v, np.pi))
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    out = {}
    for k, p in q.items():
        np.testing.assert_allclose(np.linalg.norm(p-c, axis=-1), np.linalg.norm(actual-c, axis=-1), atol=1e-11, rtol=1e-11)
        pred = p0.copy()
        pred[use] = p
        s = fit.direction.prior.parent.score(d, cv, ols, pred, cut)
        g, low = fit.direction.prior.parent.transfer.paired(p0, pred, y, valid)
        if s['edits'] != count['edits'] or (low.sum(1) < -1e-10).any():
            raise ValueError('An oracle performed worse than keeping the incumbent')
        by_rec = []
        for rec in sorted(set(d['rec'])):
            at = d['rec'] == rec
            denom = np.linalg.norm(cv[at]-y[at], axis=-1)[valid[at]].sum()
            if denom > 0.:
                by_rec.append(float(100.*low[at].sum()/denom))
        out[k] = dict(dual=s['dual_positive'], risk_pass=s['risk_pass'], undefined=s['undefined'],
            observed_pp=float(100.*g.sum()/mass), lower_pp=float(100.*low.sum()/mass),
            positive_views=sum(z > 1e-6 for z in by_rec), above_0_10pp=sum(z > .10 for z in by_rec))
    bounded = out['bounded_stepwise']['lower_pp']
    unrestricted = out['unrestricted_stepwise']['lower_pp']
    if (bounded+1e-10 < out['constant_angle']['lower_pp'] or
        unrestricted+1e-10 < max(bounded, out['analytic']['lower_pp'])):
        raise ValueError('Stepwise analytic upper bound contradicted a feasible direction')
    return dict(counts=dict(count), arms=out)


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    counts = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    result = {}
    for k in ARMS:
        a = {f: sum(r['arms'][k][f] for r in rows) for f in (
            'dual', 'risk_pass', 'undefined', 'positive_views', 'above_0_10pp')}
        x = np.asarray([np.mean([r['arms'][k]['lower_pp'] for r in g]) for g in groups.values()])
        a.update(lower_pp=float(x.mean()), worst_location_pp=float(x.min()),
                 ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
        result[k] = a
    return dict(counts=counts, arms=result)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = OUT/'registration.json'
    inputs = [Path(__file__), Path(model.__file__), OUT/'protocol.md', same.OUT/'readout.json',
              same.OUT/'registration.json']
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in inputs], finite_angles=ANGLES,
        original_2pct=True, no_training=True, external_outcome_access=False, label_aware_only=True)))
    groups = {}
    for args, ref in fit.matched.context_cells():
        s, seed = args[:2]
        r = evaluate(args)
        if any(r['arms']['original'][a] != ref['scores']['frozen'][b]
               for a, b in (('dual', 'dual_positive'), ('risk_pass', 'risk_pass'), ('undefined', 'undefined'))):
            raise ValueError('Frozen recipients did not reproduce')
        groups.setdefault(s, []).append(dict(seed=seed, **r))
    summary = summarize(groups)
    if summary['counts'] != dict(views=606, edits=780) or summary['arms']['original']['dual'] != 60:
        raise ValueError('Wrong fixed-recipient roster')
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=summary, by_site=groups,
        original_2pct=True, label_aware_only=True, external_outcome_access=False, no_training=True)))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
