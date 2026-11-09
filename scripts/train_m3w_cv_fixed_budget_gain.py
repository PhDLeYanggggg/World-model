"""Train utility heads under an unchanged count and frozen predicted-risk budget."""
import argparse
import fcntl
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch Python')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_risk_component_swaps as parent
from src.world_model import m3w_cv_fixed_budget_gain as model

risk = parent.study
study, old, base, prior, np, torch = risk.study, parent.old, parent.base, parent.prior, parent.np, parent.torch
shape_api = study.parent
context = shape_api.parent.context
amplitude = shape_api.model
OUT = risk.OUT.parent / 'moving_cv_fixed_budget_gain_v1'
PRIVATE = risk.PRIVATE.parent / OUT.name
STEPS = 1400
MODES = ('direct', 'factor')
ARMS = ('anchor', 'geometry', *MODES, 'calibrated')
COMPARE = dict(factor_vs_direct=('direct', 'factor'), direct_vs_geometry=('geometry', 'direct'),
               factor_vs_geometry=('geometry', 'factor'))


def rollout(cv, bank, a):
    j, f = np.divmod(a, amplitude.M)
    if np.ndim(a):
        return cv + np.asarray(amplitude.AMPLITUDES)[f, None, None]*(bank[np.arange(len(cv)), j]-cv)
    return cv + amplitude.AMPLITUDES[f]*(bank[:, j]-cv)


def sampler(d, cv, ols, bank, prediction):
    _, r, legal, _ = prediction
    past, scale = context.context(d)
    groups = [np.flatnonzero((d['sites'] == s) & legal.any(1)) for s in sorted(set(d['sites']))]
    if any(not len(g) for g in groups):
        raise ValueError('Every fitting source needs legal actions')
    low = np.argsort(np.where(legal, r.clip(0.).sum(-1), np.inf), axis=1, kind='stable')
    limit = np.minimum(legal.sum(1), 5)

    def draw(rng):
        i = prior.draw(groups, rng)
        j = np.where(legal[i], rng.random(legal[i].shape), -1.).argmax(1)
        favor = rng.random(len(i)) < .50
        j[favor] = low[i[favor], (rng.random(favor.sum())*limit[i[favor]]).astype(int)]
        p = rollout(cv[i], bank[i], j)
        x = context.model.action_features(past[i], cv[i], ols[i], p, scale[i])
        t = model.targets(cv[i], ols[i], p, d['target'][i], d['valid'][i], scale[i])
        return tuple(torch.from_numpy(z) for z in (x, *t))

    return draw, past.shape[1]


def packet(d, shape, risk_nets):
    cv, ols, b, _, pred, _, _ = risk.predict(d, shape, risk_nets)
    groups = prior.query_groups(d['query'])
    pred = pred['component']
    anchor = amplitude.portfolio(*pred, groups, .50, 0.)
    return cv, ols, b, pred, groups, anchor


@torch.inference_mode()
def scores(d, cv, ols, b, legal, nets, *, hindsight=False):
    past, scale = context.context(d)
    arms = ('hindsight',) if hindsight else ('geometry', *nets)
    out = {k: np.zeros(legal.shape) for k in arms}
    for a in range(legal.shape[1]):
        use = np.flatnonzero(legal[:, a])
        for i in range(0, len(use), 4096):
            q = use[i:i+4096]
            p = rollout(cv[q], b[q], a)
            if hindsight:
                *_, g = model.targets(cv[q], ols[q], p, d['target'][q], d['valid'][q], scale[q])
                out['hindsight'][q, a] = g[:, 1]*scale[q]
                continue
            t = model.missing_terms(cv[q], ols[q], p, scale[q])
            out['geometry'][q, a] = t[..., 1].mean(1)*scale[q]
            x = torch.from_numpy(context.model.action_features(past[q], cv[q], ols[q], p, scale[q]))
            geom = torch.from_numpy(t.astype(np.float32))
            for k, net in nets.items():
                out[k][q, a] = model.decode(*net(x), geom, k == 'factor').numpy()[:, 1]*scale[q]
    if any(not np.isfinite(p).all() for p in out.values()):
        raise ValueError('Nonfinite utility')
    return out


def optimize(pred, groups, anchor, utility):
    gain, r, legal, mass = pred
    out = {k: anchor.copy() for k in utility}
    for q in groups:
        if not (anchor[q] >= 0).any():
            continue
        budget = model.Budget(legal[q], r[q], mass[q], gain[q, :, 0], anchor[q])
        for k, value in utility.items():
            out[k][q] = budget.solve(value[q])
            if (out[k][q] >= 0).sum() != budget.count:
                raise ValueError('Count changed')
    return out


def rows_for(d, paths, cv, cut, pairs, ident):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    rows = []
    for rec in np.unique(d['recording']):
        at = d['recording'] == rec
        y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
        site, = np.unique(d['sites'][at])
        z = {k: x[at] for k, x in paths.items()}
        scored = scorer.score_recording(z, {}, y, v, q, str(rec), easy_cut=cut[0], hard_cut=cut[1])
        w = old.weight(q)
        terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
        rows.append(dict(scored, source=str(site), seed=ident['seed'], comparisons=terms))
    return rows


def paths_for(cv, ols, b, actions):
    return {base.CV: cv, base.OLS: ols, **{k: amplitude.apply(cv, b, a) for k, a in actions.items()}}


def vs_references(arms):
    return {f'{a}_vs_{k}': (ref, a) for a in arms for k, ref in (('cv', base.CV), ('ols4', base.OLS))}


def calibrate(d, cv, ols, b, a, cut, ident):
    scorer = prior.prior.parent.readout
    rows = rows_for(d, paths_for(cv, ols, b, a), cv, cut, vs_references(a), ident)
    sources = sorted(set(x['source'] for x in rows))
    table = {}
    for k, x in a.items():
        lower = [[scorer.merge_comparisons([r['comparisons'][f'{k}_vs_{ref}']
                  for r in rows if r['source'] == s])['completion_lower_pp']
                  for ref in ('cv', 'ols4')] for s in sources]
        table[k] = dict(edits=int((x >= 0).sum()), lower=np.mean(lower, axis=0).tolist(),
                        passes=sum(r['metrics'][k]['risk']['supported'] for r in rows))
    good = [k for k, v in table.items() if v['edits'] and v['passes'] == len(rows) and min(v['lower']) > 0.]
    return table, max(good, key=lambda k: min(table[k]['lower'])) if good else None


def gain_actions(d, pack, nets):
    cv, ols, b, pred, groups, anchor = pack
    return dict(anchor=anchor, **optimize(pred, groups, anchor, scores(d, cv, ols, b, pred[2], nets)))


def parents(c, sc, d):
    if c['identity'] != sc['identity'] or c['cut'] != sc['cut']:
        raise ValueError('Frozen roles or cuts differ')
    past, _ = context.context(d)
    return (study.loader.load(sc['fits'][1], past.shape[1]+37),
            {k: parent.restore(x, past.shape[1]) for k, x in c['fits'].items()})


def cell(d, c, sc, reg, beat, replay):
    i, cut = c['identity'], c['cut']
    tr = old.teacher.fit_role(d, i)
    shape, risk_nets = parents(c, sc, tr)
    cv, ols, b, pred, _, _ = packet(tr, shape, risk_nets)
    draw, past = sampler(tr, cv, ols, b, pred)
    nets, fits, updates, rng0 = {}, {}, 0, None
    for k in MODES:
        def objective(net, *x):
            return model.objective(net, *x, k == 'factor')

        name = f'{k}_{i["fold"]}_{i["seed"]}'
        net, fit, n = prior.prefix.trainer.fit(dict(i, arm=k, registration=reg), name, i['seed'],
            lambda: model.Head(past, past+37), STEPS, draw, objective, replay, beat)
        rng = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)['rng']
        if rng0 is not None and rng0 != rng:
            raise ValueError('Gain-head samples changed')
        nets[k], fits[k], updates, rng0 = net, fit, updates+n, rng
    _, cal = base.subset(d, i['calibration'])
    cp = packet(cal, shape, risk_nets)
    table, selected = calibrate(cal, *cp[:3], gain_actions(cal, cp, nets), cut, i)
    _, held = base.subset(d, i['held'])
    hp = packet(held, shape, risk_nets)
    with np.load(old.checked(c['choices']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], held['ids'])
        np.testing.assert_array_equal(z['component_free'], hp[-1])
    actions = gain_actions(held, hp, nets)
    actions['calibrated'] = (np.full_like(hp[-1], -1) if selected is None else actions[selected])
    path = prior.study.parent.parent.save_actions(PRIVATE / f'choices_{i["fold"]}_{i["seed"]}.npz',
        dict(ids=held['ids'], **actions))
    return dict(identity=i, cut=cut, fits=fits, choices=path, calibration=table, selected=selected), updates


def evaluate(d, cells, risk_cells, shape_cells, beat):
    arms = (*ARMS, 'hindsight')
    pairs = dict(vs_references(arms), **COMPARE,
                 **{f'{k}_vs_anchor': ('anchor', k) for k in arms if k != 'anchor'})
    rows, changes, counts = [], {}, {}
    for c, rc, sc in zip(cells, risk_cells, shape_cells, strict=True):
        i = c['identity']
        _, h = base.subset(d, i['held'])
        shape, risk_nets = parents(rc, sc, h)
        cv, ols, b, pred, groups, anchor = packet(h, shape, risk_nets)
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            np.testing.assert_array_equal(z['anchor'], anchor)
            a = {k: z[k] for k in ARMS}
        # Label-aware diagnostic only after the causal choices are frozen.
        a.update(optimize(pred, groups, anchor, scores(h, cv, ols, b, pred[2], {}, hindsight=True)))
        for k, v in a.items():
            counts[k] = counts.get(k, 0)+int((v >= 0).sum())
            changes[k] = changes.get(k, 0)+sum(int((v[q] != anchor[q]).any()) for q in groups)
        rows.extend(rows_for(h, paths_for(cv, ols, b, a), cv, c['cut'], pairs, i))
        beat(phase='held_scored', fold=i['fold'], seed=i['seed'])
    roster = {r['recording']: r['source'] for r in rows}
    result = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    return dict(summary=result, changed_queries=changes, edits=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Require and verify completed checkpoints')
    args = parser.parse_args()
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

        refs = [risk.OUT / k for k in ('freeze.json', 'readout.json', 'registration.json', 'low_count_readout.json')]
        f, r, reg, low = map(old.read, refs)
        if (r['freeze'] != old.sha(refs[0]) or f['registration'] != old.sha(refs[2])
                or not f['before_held_scoring'] or not r['original_2pct']
                or r['a_labels_opened'] or r['independent_confirmation']):
            raise ValueError('Require frozen development parents')
        sp = old.checked(reg['parents'][0])
        shape = old.read(sp)
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(x) for x in (*refs, sp)],
            original_2pct=True, files=[old.artifact(x) for x in
                (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            steps=STEPS, seeds=base.SEEDS, exact_count=True, no_confirmation_labels=True)))
        base.register()
        d = base.fit_population(beat)
        cells, updates = [], 0
        for c, sc in zip(f['cells'], shape['cells'], strict=True):
            x, n = cell(d, c, sc, old.sha(rp), beat, args.replay)
            cells.append(x)
            updates += n
            beat(phase='cell_frozen', fold=c['identity']['fold'], seed=c['identity']['seed'], updates=updates)
        fp = OUT / 'freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(rp), cells=cells, before_held_scoring=True,
            optimizer_steps=18*STEPS)))
        result = evaluate(d, old.read(fp)['cells'], f['cells'], shape['cells'], beat)
        if result['edits']['anchor'] != 21242 or any(result['edits'][k] != result['edits']['anchor']
                for k in ('geometry', *MODES, 'hindsight')):
            raise ValueError('Matched edit total changed')
        if result['summary']['risk']['anchor'] != r['summary']['risk']['component_free']:
            raise ValueError('Anchor readout changed')
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fp), **result,
            qualified=sum(c['selected'] is not None for c in cells), original_2pct=True,
            independent_confirmation=False, a_labels_opened=False, hindsight_diagnostic_only=True)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
