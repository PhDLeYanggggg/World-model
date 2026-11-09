"""Compare direct and factorized risk with frozen gains and exact matched counts."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_prefix_state as state
from src.world_model import m3w_cv_risk_components as model

study, old, base, prior, np, torch = state.study, state.old, state.base, state.prior, state.np, state.torch
OUT = state.OUT.parent / 'moving_cv_risk_components_v1'
PRIVATE = state.PRIVATE.parent / OUT.name
STEPS = 1400
MODES = ('direct', 'component')
ARMS = ('portfolio', 'shape_matched', 'direct_matched', 'component_matched',
        'direct_free', 'component_free', 'calibrated')
PRIMARY = dict(component_at_count=('direct_matched', 'component_matched'),
    component_vs_shape=('shape_matched', 'component_matched'),
    component_free=('direct_free', 'component_free'))
GRID = tuple((k, f, s) for k in MODES for f in (.25, .50, 1.) for s in (0., .02))
FIELDS = ('all_harm', 'easy_harm', 'signed_easy', 'all_reference', 'easy_reference',
          'wholly_unknown_brier', 'easy_brier', 'missing_debit')


def sampler(d, cut):
    cv, ols, b = study.parent.analytic(d['history'])
    past, scale = study.parent.parent.context.context(d)
    legal = (b != cv[:, None]).any((2, 3))
    groups = [np.flatnonzero((d['sites'] == s) & legal.any(1)) for s in sorted(set(d['sites']))]
    if any(not len(g) for g in groups):
        raise ValueError('Every fitting source needs legal paths')
    fractions = np.asarray(study.parent.model.AMPLITUDES)

    def draw(rng):
        i = prior.draw(groups, rng)
        j = np.where(legal[i], rng.random(legal[i].shape), -1.).argmax(1)
        a = fractions[rng.integers(len(fractions), size=len(i))]
        p = cv[i] + a[:, None, None]*(b[i, j]-cv[i])
        x = study.parent.parent.context.model.action_features(past[i], cv[i], ols[i], p, scale[i])
        t = model.targets(cv[i], p, d['target'][i], d['valid'][i], scale[i], cut)
        return tuple(torch.from_numpy(z) for z in (x, *t))

    return draw, past.shape[1]


@torch.inference_mode()
def predict(d, shape, nets):
    cv, ols, b = study.parent.analytic(d['history'])
    frozen = study.parent.predictions({'shape': shape}, d, cv, ols, b)['shape']
    g, _, legal, _ = frozen
    past, scale = study.parent.parent.context.context(d)
    moments = {k: np.empty((*legal.shape, 5), np.float32) for k in MODES}
    probability = {k: np.empty((len(cv), 3), np.float32) for k in MODES}
    for a in range(legal.shape[1]):
        j, f = divmod(a, study.parent.model.M)
        p = cv + study.parent.model.AMPLITUDES[f]*(b[:, j]-cv)
        radius = (np.linalg.norm(p-cv, axis=-1).max(1) / scale).astype(np.float32)
        for i in range(0, len(cv), 4096):
            q = slice(i, i+4096)
            x = torch.from_numpy(study.parent.parent.context.model.action_features(past[q], cv[q], ols[q], p[q], scale[q]))
            for k, net in nets.items():
                m, logits = model.decode(*net(x), torch.from_numpy(radius[q]), k == 'component')
                moments[k][q, a] = m.numpy()
                if a == 0:
                    probability[k][q] = logits.softmax(1).numpy()
    pred = {k: (g, model.risk(m)*scale[:, None, None], legal, m[:, 0, 3:]*scale[:, None])
            for k, m in moments.items()}
    if any(not np.isfinite(x).all() for p in pred.values() for x in p):
        raise ValueError('Nonfinite moments')
    return cv, ols, b, frozen, pred, moments, probability


def calibrate(d, cv, ols, b, pred, groups, cut):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    recs = [np.flatnonzero(d['recording'] == r) for r in np.unique(d['recording'])]
    table = []
    for k, f, s in GRID:
        a = study.parent.model.portfolio(*pred[k], groups, f, s)
        path = study.parent.model.apply(cv, b, a)
        rows = []
        for at in recs:
            y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
            rec, = np.unique(d['recording'][at])
            site, = np.unique(d['sites'][at])
            r = scorer.score_recording({base.CV: cv[at], 'p': path[at]}, {}, y, v, q, str(rec),
                easy_cut=cut[0], hard_cut=cut[1])
            g = [full(p[at], path[at], cv[at], y, v, old.weight(q)) for p in (cv, ols)]
            rows.append((site, r['metrics']['p']['risk']['supported'], g))
        gain = [[scorer.merge_comparisons([x[2][j] for x in rows if x[0] == t])['completion_lower_pp']
                 for j in range(2)] for t in sorted(set(d['sites']))]
        table.append(dict(spec=(k, f, s), edits=int((a >= 0).sum()), passes=sum(x[1] for x in rows),
            lower=np.mean(gain, axis=0).tolist()))
    good = [x for x in table if x['edits'] and x['passes'] == len(recs) and min(x['lower']) > 0.]
    return table, max(good, key=lambda x: (min(x['lower']), -x['edits'])) if good else None


def cell(d, c, reg, beat, replay):
    i, cut = c['identity'], c['cut']
    tr = old.teacher.fit_role(d, i)
    draw, past = sampler(tr, cut[0])
    nets, fits, updates, rng0 = {}, {}, 0, None
    for k in MODES:
        def objective(net, x, y, cat, radius):
            return model.objective(net, x, y, cat, radius, k == 'component')

        name = f'{k}_{i["fold"]}_{i["seed"]}'
        net, fit, n = prior.prefix.trainer.fit(dict(i, arm=k, registration=reg), name, i['seed'],
            lambda: model.Head(past, past+37), STEPS, draw, objective, replay, beat)
        rng = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)['rng']
        if rng0 is not None and rng0 != rng:
            raise ValueError('Risk-head samples changed')
        nets[k], fits[k], updates, rng0 = net, fit, updates + n, rng
    shape = study.loader.load(c['fits'][1], past+37)
    _, cal = base.subset(d, i['calibration'])
    cv, ols, b, _, pred, _, _ = predict(cal, shape, nets)
    table, sel = calibrate(cal, cv, ols, b, pred, prior.query_groups(cal['query']), cut)
    _, h = base.subset(d, i['held'])
    cv, _, _, shape_pred, pred, moment, probability = predict(h, shape, nets)
    groups = prior.query_groups(h['query'])
    a = study.parent.model.portfolio(*shape_pred, groups, .50, 0.)
    out = dict(portfolio=a, shape_matched=study.parent.model.independent(*shape_pred[:3], groups, .50, match=a))
    for k in MODES:
        out[f'{k}_matched'] = study.parent.model.independent(*pred[k][:3], groups, .50, match=a)
        out[f'{k}_free'] = study.parent.model.portfolio(*pred[k], groups, .50, 0.)
    out['calibrated'] = (np.full_like(a, -1) if sel is None else
        study.parent.model.portfolio(*pred[sel['spec'][0]], groups, *sel['spec'][1:]))
    at = np.flatnonzero(a >= 0)
    path = prior.study.parent.parent.save_actions(PRIVATE / f'choices_{i["fold"]}_{i["seed"]}.npz',
        dict(ids=h['ids'], **out, **{f'{k}_moment': moment[k][at, a[at]] for k in MODES},
             **{f'{k}_probability': probability[k][at] for k in MODES}))
    return dict(identity=i, cut=cut, fits=fits, ladder=table, selected=sel, choices=path), updates


def evaluate(cells, data, beat):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{k}_vs_{t}': (b, k) for k in ARMS for t, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs.update(PRIMARY)
    rows, stats, geo, changed = [], {}, {}, dict.fromkeys(PRIMARY, 0)
    for c in cells:
        i = c['identity']
        _, h = base.subset(data, i['held'])
        cv, ols, bank = study.parent.analytic(h['history'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            a = {k: z[k] for k in ARMS}
            p = {base.CV: cv, base.OLS: ols, **{k: study.parent.model.apply(cv, bank, a[k]) for k in ARMS}}
            at = np.flatnonzero(a['portfolio'] >= 0)
            y, v = h['target'][at], h['valid'][at]
            scale = prior.model.scale(h['history'][at])
            target, cat, radius = model.targets(cv[at], p['portfolio'][at], y, v, scale, c['cut'][0])
            err = {}
            for k in MODES:
                q = z[f'{k}_probability']
                err[k] = np.column_stack((np.abs(z[f'{k}_moment']-target),
                    (q[:, 0]-(cat == 0))**2, (q[:, 1]-(cat == 1))**2,
                    np.abs(q[:, 0]-(cat == 0))*radius))
            for site in np.unique(h['sites']):
                use = h['sites'][at] == site
                if not use.any():
                    raise ValueError('No common diagnostic exposures in a source')
                stats[str(site), i['seed']] = np.stack([err[k][use].mean(0, dtype=np.float64) for k in MODES])
        for g in prior.query_groups(h['query']):
            if len({int((a[k][g] >= 0).sum()) for k in ARMS[:4]}) != 1:
                raise ValueError('Matched count changed')
        for k, (l, r) in PRIMARY.items():
            changed[k] += int((a[l] != a[r]).sum())
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q, history = (h[k][at] for k in ('target', 'valid', 'query', 'history'))
            z = {k: x[at] for k, x in p.items()}
            site, = np.unique(h['sites'][at])
            r = scorer.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(site), seed=i['seed'], comparisons=terms))
            b = geo.setdefault((str(site), i['seed']), np.zeros((len(PRIMARY), 2)))
            for g in prior.query_groups(q):
                arr = np.stack([z[k][g] for pair in PRIMARY.values() for k in pair])
                _, u, _, (steps, _, _, _) = prior.geometry.query(history[g], arr, y[g], v[g],
                    tuple((2*j, 2*j+1) for j in range(len(PRIMARY))))
                if steps:
                    b[:, 0] += u[:, 1]
                    b[:, 1] += 1
        beat(phase='held_scored', fold=i['fold'], seed=i['seed'])
    roster = {r['recording']: r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    x = np.asarray([[stats[s, k] for k in base.SEEDS] for s in sites])
    acc = {k: dict(direct=float(x[:, :, 0, j].mean()), component=float(x[:, :, 1, j].mean()),
        direct_minus_component=prior.geometry.bootstrap(x[:, :, 0, j]-x[:, :, 1, j], draw))
        for j, k in enumerate(FIELDS)}
    x = np.asarray([[geo[s, k] for k in base.SEEDS] for s in sites])
    p = np.divide(x[..., 0], x[..., 1], out=np.full_like(x[..., 0], np.nan), where=x[..., 1] > 0.)
    geometry = {k: prior.geometry.bootstrap(p[:, :, j], draw) for j, k in enumerate(PRIMARY)}
    return dict(summary=summary, changed=changed, geometry=geometry, common_exposure=acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', action='store_true')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    with (PRIVATE / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-start, 3), **v)
            prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json', z)
            if v.get('phase') != 'fit' or v.get('step') in (100, STEPS):
                print(z, flush=True)

        refs = [study.parent.OUT / 'freeze.json', study.OUT / 'readout.json', state.OUT / 'readout.json']
        f, r, p = map(old.read, refs)
        if (not f['before_held_scoring'] or any(x['a_labels_opened'] or x['independent_confirmation']
                or not x['original_2pct'] for x in (r, p))):
            raise ValueError('Require frozen causal development parents')
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(x) for x in refs], original_2pct=True,
            files=[old.artifact(x) for x in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            grid=GRID, steps=STEPS, seeds=base.SEEDS, frozen_gain=True)))
        base.register()
        d = base.fit_population(beat)
        cells, updates = [], 0
        for c in f['cells']:
            x, n = cell(d, c, old.sha(rp), beat, args.replay)
            cells.append(x)
            updates += n
            beat(phase='cell_frozen', fold=c['identity']['fold'], seed=c['identity']['seed'], updates=updates)
        fp = OUT / 'freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(rp), cells=cells,
            optimizer_steps=18*STEPS, before_held_scoring=True)))
        result = evaluate(old.read(fp)['cells'], d, beat)
        if result['summary']['risk']['portfolio'] != r['summary']['risk']['portfolio']:
            raise ValueError('Frozen portfolio changed')
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fp), **result,
            qualified=sum(c['selected'] is not None for c in cells), original_2pct=True,
            independent_confirmation=False, a_labels_opened=False)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
