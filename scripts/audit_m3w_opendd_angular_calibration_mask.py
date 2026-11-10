"""Separate the learned rotation mask from the direction used on that mask."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_angular_calibration as cal


fit, run, torch, np, old = cal.fit, cal.run, cal.torch, cal.np, cal.old
OUT = cal.OUT.parent/'opendd_angular_calibration_mask_v1'
CONTRASTS = ('constant_minus_model', 'constant_vs_incumbent', 'model_vs_incumbent',
             'random_mask_vs_incumbent', 'constant_minus_random_mask', 'uniform_angle_vs_incumbent',
             'constant_minus_uniform_angle')


def selection_weights(recordings, eligible, rotates):
    if np.any(rotates & ~eligible):
        raise ValueError('Rotations must be incumbent edits')
    weights = np.zeros(len(eligible))
    flexible = 0
    for rec in set(recordings[eligible]):
        r = (recordings == rec) & eligible
        n, k = int(r.sum()), int((r & rotates).sum())
        weights[r] = k/n
        flexible += int(0 < k < n)
    if not np.isclose(weights.sum(), rotates.sum(), atol=1e-10, rtol=0.):
        raise ValueError('Random mask changed expected spend')
    return weights, flexible


def evaluate(args, net, spec):
    s, seed, motion, d, bank, menu, _, _, _ = args
    _, _, rows, path, _ = menu
    cv = bank[0]
    use = np.zeros(len(cv), dtype=bool)
    count = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, _, r = old.indices(d, rec, rows)
        f = run.read(cal.prev.prior.same.base.PRIVATE/f'{s}_{seed}_{i:04d}.json')
        a = np.asarray(f['score_plan'])
        if f['identity'] != dict(registration=motion, site=s, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound incumbent')
        selected = r[np.arange(2) < a[:, None]]
        if (selected < 0).any() or len(selected) != len(set(selected)):
            raise ValueError('Illegal incumbent recipients')
        use[np.flatnonzero(ai)[selected]] = True
        count.update(views=1, edits=int(a.sum()))
    anchor = np.where(use[:, None, None], path, cv)
    idx, score = cal.predict(d, bank, path, net)
    a = np.where(use, cal.rule.indices(idx, score, spec['threshold']['mean']), 0)
    rotate = a != 0
    theta = cal.prev.ANGLES[spec['matched_constant']]
    g = cal.prev.geometry.rotate
    p = np.where(rotate[:, None, None], g(cv, anchor, cal.prev.ANGLES[a][:, None]), anchor)
    all_constant = np.where(use[:, None, None], g(cv, anchor, theta), anchor)
    constant = np.where(rotate[:, None, None], all_constant, anchor)
    y, valid = d['y'], d['valid']
    debit = fit.direction.prior.parent.transfer.paired
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()

    def lower(ref, pred):
        return 100.*debit(ref, pred, y, valid)[1].sum(1)/mass

    weights, flexible = selection_weights(d['rec'], use, rotate)
    v = float(lower(anchor, constant).sum())
    random = float(np.dot(lower(anchor, all_constant), weights))
    uniform = float(np.mean([lower(anchor, np.where(rotate[:, None, None], g(cv, anchor, t), anchor)).sum()
                             for t in cal.prev.ANGLES[1:]]))
    count.update(rotations=int(rotate.sum()), rotation_views=len(set(d['rec'][rotate])), flexible_views=flexible)
    return dict(counts=dict(count), lower=dict(constant_minus_model=float(lower(p, constant).sum()),
        constant_vs_incumbent=v, model_vs_incumbent=float(lower(anchor, p).sum()),
        random_mask_vs_incumbent=random, constant_minus_random_mask=v-random,
        uniform_angle_vs_incumbent=uniform, constant_minus_uniform_angle=v-uniform))


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))
    result = {}
    for k in CONTRASTS:
        x = np.array([np.mean([r['lower'][k] for r in g]) for g in groups.values()])
        result[k] = dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                         positive_locations=int((x > 0.).sum()))
    return dict(counts=dict(sum((Counter(r['counts']) for r in rows), Counter())), paired=result)


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    freeze, report = (cal.OUT/k for k in ('calibration_freeze.json', 'readout.json'))
    f, r = run.read(freeze), run.read(report)
    if (r['calibration'] != run.sha(freeze) or r['registration'] != f['registration'] or
            not f['before_outer_score'] or r['external_outcome_access']):
        raise ValueError('Require completed, frozen calibration')
    angular = cal.prev.OUT/'head_freeze.json'
    fit0 = run.read(angular)
    reg = run.sha(cal.prev.OUT/'registration.json')
    if fit0['registration'] != reg:
        raise ValueError('Unbound outer fit')
    p = OUT/'registration.json'
    files = [Path(__file__), OUT/'protocol.md', freeze, report, angular]
    run.once(p, run.stable(dict(inputs=[run.artifact(q) for q in files], no_training=True,
        original_2pct=True, external_outcome_access=False, no_held_tuning=True)))
    groups = {}
    for args, _ in fit.matched.context_cells():
        s, seed = args[:2]
        spec = f['cells'][f'{s}_{seed}']
        train = [t for t in fit.matched.parent.source.SITES if t != s]
        meta = fit0['checkpoints'][f'{s}_{seed}_factor']
        if meta['identity'] != dict(registration=reg, held=s, train=train, seed=seed, arm='factor'):
            raise ValueError('Invalid held-source role')
        net = cal.study.load(meta, lambda: cal.prev.model.ChoiceHead(19), old.STEPS)
        row = evaluate(args, net, spec)
        ref = next(v for v in r['by_site'][s] if v['seed'] == seed)
        for a, b in (('constant_vs_incumbent', 'constant_at_mean'), ('model_vs_incumbent', 'mean')):
            if not np.isclose(row['lower'][a], ref['arms'][b]['gain_pp'], atol=1e-12, rtol=1e-11):
                raise ValueError('Matched mask did not reproduce the calibrated comparison')
        if row['counts']['rotations'] != ref['arms']['mean']['rotated_edits']:
            raise ValueError('Matched mask changed rotation count')
        groups.setdefault(s, []).append(dict(seed=seed, **row))
    s = summarize(groups)
    run.once(OUT/'readout.json', run.stable(dict(registration=run.sha(p), summary=s, by_site=groups,
        no_training=True, original_2pct=True, external_outcome_access=False, deployment_changed=False)))
    print(s, flush=True)


if __name__ == '__main__':
    main()
