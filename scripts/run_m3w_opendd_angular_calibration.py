"""Calibrate angular acceptance on cross-fitted donor locations."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if __name__ == '__main__' and (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_angular_choice as prev
from src.evaluation import m3w_angular_score_calibration as rule


fit, run, np, torch, old = prev.fit, prev.run, prev.np, prev.torch, prev.old
exp, study = fit.matched.exp, fit.study
OUT = fit.OUT.parent/'opendd_angular_calibration_v1'
PRIVATE = fit.PRIVATE.parent/OUT.name
ARMS = ('original', 'zero', 'mean', 'stable', 'constant_at_mean')


def setup():
    context = exp.register()
    loader = fit.matched.parent.prior.prior.old
    sites = {s: loader.load_site(s) for s in fit.matched.parent.source.SITES}
    files = [Path(__file__), Path(rule.__file__), OUT/'protocol.md', prev.OUT/'head_freeze.json',
             prev.OUT/'readout.json', prev.OUT/'registration.json', exp.OUT/'registration.json']
    for s, d in sites.items():
        p = loader.source.prior.PRIVATE/f'{s}_cohort.npz'
        files.append(p)
        with np.load(p, allow_pickle=False) as z:
            d.update(recordings=z['recording'], starts=z['start'])
    p = OUT/'registration.json'
    run.once(p, run.stable(dict(inputs=[run.artifact(f) for f in files], margins=rule.MARGINS,
        seeds=study.prior.SEEDS, steps=old.STEPS, workers=0, threads=4, original_2pct=True,
        no_held_tuning=True, external_outcome_access=False)))
    return run.sha(p), context[6], sites


@torch.inference_mode()
def predict(d, bank, path, net):
    cv, ols = bank[:2]
    a = fit.direction.model.bridge_fraction(cv, ols, path)
    x, _ = fit.model.features(d['h'], cv, ols, a)
    z = net(torch.from_numpy(x))
    score, index = (z[:, 0]-z[:, 1].clamp_min(0.)).max(1)
    return index.numpy(), score.numpy()


def active_from_plan(d, rows, counts):
    take = rows[np.arange(2) < counts[:, None]]
    if (take < 0).any() or len(take) != len(set(take)):
        raise ValueError('Illegal selected rows')
    use = np.zeros(len(d['h']), dtype=bool)
    use[take] = True
    return use


def metrics(d, bank, anchor, use, indices, cut):
    cv, ols = bank[:2]
    idx = np.where(use, indices, 0)
    candidate = np.where((idx != 0)[:, None, None], prev.geometry.rotate(cv, anchor, prev.ANGLES[idx][:, None]), anchor)
    np.testing.assert_allclose(np.linalg.norm(candidate-cv, axis=-1), np.linalg.norm(anchor-cv, axis=-1), atol=1e-11, rtol=1e-11)
    raw, vals = fit.direction.metric_rows(candidate, cv, ols, d, cut)
    if raw['edits'] != int(use.sum()):
        raise ValueError('Rotation changed the actual edit count')
    y, valid = d['y'], d['valid']
    mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    if mass <= 0.:
        raise ValueError('No available reference mass')
    effect = fit.direction.prior.parent.transfer.paired(anchor, candidate, y, valid)[1].sum()
    return dict(**vals, gain_pp=float(100.*effect/mass), rotated_edits=int((idx != 0).sum()),
                rotated_views=len(set(d['rec'][idx != 0])), views=raw['views'])


def inner_menu(outer, cal, seed, sites, heads):
    omitted = sorted((outer, cal))
    train = [s for s in sites if s not in omitted]
    meta = heads['rank_heads']['rank_'+'_'.join(omitted)+f'_{seed}']
    if meta['identity'] != dict(registration=heads['registration'], omitted=omitted, fit=train, seed=seed):
        raise ValueError('Calibration nomination used a held label')
    rank = study.load(meta, lambda: study.prior.model.SwapValue(False), study.RANK_STEPS)
    d = sites[cal]
    bank = fit.matched.parent.prior.analytic(d)
    _, legal, rows, path, _ = exp.prefix.menu_for(d, bank, rank)
    scores = np.tile([0., 1., -1.], (len(rows), 1))
    counts = study.model.streaming_count(scores, legal, d['recordings'], d['starts'], study.RATE)
    use = active_from_plan(d, rows, counts)
    cut = np.quantile(np.concatenate([sites[s]['cut_samples'] for s in train]), [.25, .75])
    return bank, path, use, cut


def donor_constant(table):
    value = []
    for _, cv, _, path, y, scale, valid in table:
        g, debit = prev.model.targets(cv, path, y, valid, scale)
        value.append((g-debit).mean(0))
    return 1+int(torch.stack(value).mean(0).argmax())


def calibrate(reg, heads, sites, replay, beat):
    samples = {s: fit.donor(d) for s, d in sites.items()}
    fitted, nets, cells, updates = {}, {}, {}, 0
    for s in sites:
        for seed in study.prior.SEEDS:
            thresholds = {}
            for cal in sites:
                if s == cal:
                    continue
                omitted = sorted((s, cal))
                train = [t for t in sites if t not in omitted]
                name = '_'.join(omitted)+f'_{seed}'
                identity = dict(registration=reg, omitted=omitted, train=train, seed=seed, arm='factor')
                if name not in nets:
                    table = [samples[t] for t in train]

                    def loss(net, x, cv, _, anchor, y, scale, valid):
                        g, debit = prev.model.targets(cv, anchor, y, valid, scale)
                        return prev.model.loss(net(x), g, debit, True)

                    nets[name], fitted[name], n = study.fit(identity, str(PRIVATE/name), seed,
                        lambda: prev.model.ChoiceHead(table[0][0].shape[1]), old.STEPS, prev.prior.batches(table),
                        loss, replay, beat)
                    updates += n
                if fitted[name]['identity'] != identity:
                    raise ValueError('Cross-fitting reused the wrong source roster')
                bank, path, use, cut = inner_menu(s, cal, seed, sites, heads)
                d = sites[cal]
                idx, score = predict(d, bank, path, nets[name])
                anchor = np.where(use[:, None, None], path, bank[0])
                thresholds[cal] = dict(edits=int(use.sum()), predictor=name, results=[
                    metrics(d, bank, anchor, use, rule.indices(idx, score, m), cut) for m in rule.MARGINS])
            rows = [v['results'] for v in thresholds.values()]
            choice = dict(mean=rule.select_threshold(rows), stable=rule.select_threshold(rows, stable=True))
            cells[f'{s}_{seed}'] = dict(held=s, seed=seed, threshold=choice, calibration=thresholds,
                matched_constant=donor_constant([samples[t] for t in sites if t != s]))
            beat(phase='calibration_cell', held=s, seed=seed, thresholds=choice, updates=updates)
    if len(fitted) != 63 or len(cells) != 21:
        raise ValueError('Incomplete nested calibration')
    p = OUT/'calibration_freeze.json'
    run.once(p, run.stable(dict(registration=reg, checkpoints=fitted, cells=cells, masks_frozen=True,
        before_outer_score=True, original_2pct=True, external_outcome_access=False)))
    beat(phase='calibrated', updates=updates, replay=replay, freeze=run.sha(p))


def outer_row(args, net, spec):
    site, seed, motion, d, bank, menu, _, _, cut = args
    _, _, rows, path, _ = menu
    anchor = bank[0].copy()
    use = np.zeros(len(anchor), dtype=bool)
    count = Counter()
    for i, rec in enumerate(sorted(set(d['rec']))):
        ai, _, r = old.indices(d, rec, rows)
        f = run.read(prev.prior.same.base.PRIVATE/f'{site}_{seed}_{i:04d}.json')
        a = np.asarray(f['score_plan'])
        if f['identity'] != dict(registration=motion, site=site, seed=seed, ordinal=i) or a.sum() != f['edits']:
            raise ValueError('Unbound incumbent')
        selected = r[np.arange(2) < a[:, None]]
        if (selected < 0).any() or len(selected) != len(set(selected)):
            raise ValueError('Duplicate incumbent recipient')
        take = np.flatnonzero(ai)[selected]
        anchor[take], use[take] = path[take], True
        count.update(views=1, edits=int(a.sum()))
    idx, score = predict(d, bank, path, net)
    choice = {k: rule.indices(idx, score, m) for k, m in spec['threshold'].items()}
    choice.update(original=np.zeros_like(idx), zero=rule.indices(idx, score, 0.),
                  constant_at_mean=np.where(choice['mean'] != 0, spec['matched_constant'], 0))
    arms = {k: metrics(d, bank, anchor, use, a, cut) for k, a in choice.items()}
    if (arms['mean']['rotated_edits'] != arms['constant_at_mean']['rotated_edits']
            or len({a['undefined'] for a in arms.values()}) != 1):
        raise ValueError('Rotation mask or original reference support changed')
    return dict(counts=dict(count), arms=arms, thresholds=spec['threshold'])


def summarize(groups):
    rows = [r for g in groups.values() for r in g]
    draws = np.random.default_rng(817).integers(len(groups), size=(3000, len(groups)))

    def ci(fn):
        x = np.asarray([np.mean([fn(r) for r in g]) for g in groups.values()])
        return dict(point_pp=float(x.mean()), ci95_pp=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                    positive_locations=int((x > 0.).sum()), worst_location_pp=float(x.min()))

    count = dict(sum((Counter(r['counts']) for r in rows), Counter()))
    arms = {k: {f: sum(r['arms'][k][f] for r in rows)
                for f in ('dual', 'risk_pass', 'undefined', 'rotated_edits', 'rotated_views')} for k in ARMS}
    for k in ARMS:
        arms[k]['vs_incumbent'] = ci(lambda r: r['arms'][k]['gain_pp'])
    return dict(counts=count, arms=arms, active_thresholds={k: sum(r['thresholds'][k] is not None for r in rows)
        for k in ('mean', 'stable')})


def score_outer(reg):
    p = OUT/'calibration_freeze.json'
    f = run.read(p)
    if f['registration'] != reg or not f['before_outer_score'] or f['external_outcome_access']:
        raise ValueError('Require frozen calibration before the held readout')
    freeze_path = prev.OUT/'head_freeze.json'
    old_readout = run.read(prev.OUT/'readout.json')
    old_fit = run.read(freeze_path)
    key = run.sha(prev.OUT/'registration.json')
    if (old_readout['registration'] != key or old_fit['registration'] != key or
            old_readout['checkpoint_manifest'] != run.sha(freeze_path) or old_readout['external_outcome_access']):
        raise ValueError('Unbound outer model')
    groups = {}
    for args, _ in fit.matched.context_cells():
        s, seed = args[:2]
        spec = f['cells'][f'{s}_{seed}']
        donors = [t for t in fit.matched.parent.source.SITES if t != s]
        if spec['held'] != s or spec['seed'] != seed or set(spec['calibration']) != set(donors):
            raise ValueError('A held location entered its own calibration')
        meta = old_fit['checkpoints'][f'{s}_{seed}_factor']
        if meta['identity'] != dict(registration=key, held=s, train=donors, seed=seed, arm='factor'):
            raise ValueError('Invalid final-fit training roles')
        net = study.load(meta, lambda: prev.model.ChoiceHead(19), old.STEPS)
        r = outer_row(args, net, spec)
        ref = next(v for v in old_readout['by_site'][s] if v['seed'] == seed)
        if any(r['arms'][a][k] != ref['arms'][b][k] for a, b in (('original','original'), ('zero','factor'))
               for k in ('dual','risk_pass','undefined','rotated_edits','rotated_views')):
            raise ValueError('Zero-threshold control did not reproduce')
        groups.setdefault(s, []).append(dict(seed=seed, **r))
    summary = summarize(groups)
    if summary['counts'] != dict(views=606, edits=780):
        raise ValueError('Wrong held evaluation cohort')
    run.once(OUT/'readout.json', run.stable(dict(registration=reg, calibration=run.sha(p),
        summary=summary, by_site=groups, original_2pct=True, no_training_at_score=True,
        external_outcome_access=False, deployment_changed=False)))
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('calibrate', 'score'))
    parser.add_argument('--replay', action='store_true')
    a = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**e):
            row = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **e)
            run.kin.prior.source.atomic(PRIVATE/'heartbeat.json', row)
            if e['phase'] in ('calibration_cell', 'calibrated'):
                print(row, flush=True)

        reg, heads, sites = setup()
        if a.phase == 'calibrate':
            calibrate(reg, heads, sites, a.replay, beat)
        else:
            score_outer(reg)
            beat(phase='scored', updates=0)


if __name__ == '__main__':
    main()
