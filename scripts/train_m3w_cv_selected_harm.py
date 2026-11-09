"""Test upper-quantile harm prediction on the actions a frozen policy nominates."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_amplitude_response as audit
from scripts import train_m3w_cv_amplitude_response as parent
from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model import m3w_cv_selected_harm as model

old, base, prior, np, torch = parent.old, parent.base, parent.prior, parent.np, parent.torch
OUT = parent.OUT.parent / 'moving_cv_selected_harm_v1'
PRIVATE = parent.PRIVATE.parent / OUT.name
STEPS = 1200
MODES = {'median': .50, 'tail': .85}
GRID = tuple((m, p, t) for m in MODES for p in (1., 2., 4.) for t in (0., .02))
ARMS = ('parent', *MODES, 'calibrated')


def packet(d, cv, ols, bank, shape, anchor=None):
    g, r, legal, ref = parent.predictions({'shape': shape}, d, cv, ols, bank)['shape']
    q = prior.query_groups(d['query'])
    if anchor is None:
        anchor = parent.model.portfolio(g, r, legal, ref, q, .50, 0.)
    choice = model.proposals(g, r, legal, anchor)
    path = parent.model.apply(cv, bank, choice)
    past, scale = parent.parent.context.context(d)
    x = parent.parent.context.model.action_features(past, cv, ols, path, scale)
    at = np.arange(len(cv))
    gain, risk = (np.where(choice[:, None] >= 0, a[at, choice], 0.) for a in (g, r))
    size = np.empty(len(cv))
    fraction = np.empty(len(cv))
    for ix in q:
        size[ix], fraction[ix] = len(ix), (anchor[ix] >= 0).mean()
    extra = np.column_stack((gain / scale[:, None], risk / scale[:, None],
        ref / scale[:, None], anchor >= 0, np.log1p(size), fraction))
    x = np.concatenate((x, extra.astype(np.float32)), axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Invalid causal nomination features')
    return dict(x=x, scale=scale, gain=gain, ref=ref, path=path, anchor=anchor, choice=choice, groups=q)


def sampler(d, cv, p, cut):
    y = model.targets(cv, p['path'], d['target'], d['valid'], p['scale'], cut)
    groups = [np.flatnonzero((d['sites'] == site) & (p['choice'] >= 0) & ((p['anchor'] >= 0) == nominated))
              for site in sorted(set(d['sites'])) for nominated in (True, False)]
    groups = [g for g in groups if len(g)]
    if not groups:
        raise ValueError('No fit-role proposals')

    def draw(rng):
        at = prior.draw(groups, rng)
        return torch.from_numpy(p['x'][at]), torch.from_numpy(y[at])

    return draw, [len(g) for g in groups]


@torch.inference_mode()
def estimates(nets, x):
    result = {k: np.concatenate([model.decode(net(torch.from_numpy(x[i:i+4096]))).numpy()
        for i in range(0, len(x), 4096)]) for k, net in nets.items()}
    if any(not np.isfinite(p).all() for p in result.values()):
        raise ValueError('Invalid causal harm predictions')
    return result


def choose(p, prediction, penalty, margin=None):
    s = model.rank(p['gain'], prediction, p['ref'], p['scale'], penalty)
    return model.select(p['choice'], p['anchor'], s, p['groups'], margin=margin)


def calibrate(d, c, o, b, p, est, cut):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    recs = [np.flatnonzero(d['recording'] == r) for r in np.unique(d['recording'])]
    table = []
    for k, penalty, margin in GRID:
        a = choose(p, est[k], penalty, margin)
        z = parent.model.apply(c, b, a)
        rows = []
        for ix in recs:
            y, v, q = (d[k][ix] for k in ('target', 'valid', 'query'))
            rec, = np.unique(d['recording'][ix])
            site, = np.unique(d['sites'][ix])
            r = scorer.score_recording({base.CV: c[ix], 'p': z[ix]}, {}, y, v, q, str(rec),
                easy_cut=cut[0], hard_cut=cut[1])
            g = [full(left[ix], z[ix], c[ix], y, v, old.weight(q)) for left in (c, o)]
            rows.append((site, r['metrics']['p']['risk']['supported'], g))
        lower = [[scorer.merge_comparisons([x[2][j] for x in rows if x[0] == site])['completion_lower_pp']
                  for j in range(2)] for site in sorted(set(d['sites']))]
        table.append(dict(mode=k, penalty=penalty, margin=margin, edits=int((a >= 0).sum()),
            passes=sum(x[1] for x in rows), lower_pp=np.mean(lower, axis=0).tolist()))
    good = [r for r in table if r['edits'] and r['passes'] == len(recs) and min(r['lower_pp']) > 0.]
    return table, max(good, key=lambda r: (min(r['lower_pp']), -r['edits'])) if good else None


def cell(d, donor, frozen, registration, beat, replay):
    ident, cut = frozen['identity'], frozen['cut']
    tr = old.teacher.fit_role(d, ident)
    cv, ols, bank = parent.analytic(tr['history'])
    past, _ = parent.parent.context.context(tr)
    shape = audit.load(frozen['fits'][1], past.shape[1] + 37)
    p = packet(tr, cv, ols, bank, shape)
    draw, groups = sampler(tr, cv, p, cut[0])
    width = p['x'].shape[1]
    fits, nets, updates, rng0 = {}, {}, 0, None
    for k, tau in MODES.items():
        def loss(net, x, y):
            return model.loss(net, x, y, tau)

        name = f'{k}_{ident["fold"]}_{ident["seed"]}'
        net, fit, n = prior.prefix.trainer.fit(dict(ident, arm=k, registration=registration), name,
            ident['seed'], lambda: prior.model.head(width, 3), STEPS, draw, loss, replay, beat)
        rng = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)['rng']
        if rng0 is not None and rng != rng0:
            raise ValueError('Quantile control changed training draws')
        rng0 = rng
        fits[k], nets[k], updates = fit, net, updates + n
    _, cal = base.subset(d, ident['calibration'])
    cv, ols, bank = parent.analytic(cal['history'])
    p = packet(cal, cv, ols, bank, shape)
    table, sel = calibrate(cal, cv, ols, bank, p, estimates(nets, p['x']), cut)
    hi, h = base.subset(d, ident['held'])
    cv, ols, bank = prior.bank(h['history'],
        {k: v[hi] for k, v in prior.read_donor(donor, ident['seed'], d).items()})
    with np.load(old.checked(frozen['choices']), allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], h['ids'])
        anchor = z['shape_joint']
    p = packet(h, cv, ols, bank, shape, anchor)
    est = estimates(nets, p['x'])
    choices = {k: choose(p, v, 2.) for k, v in est.items()}
    choices['calibrated'] = (np.full_like(anchor, -1) if sel is None else
        choose(p, est[sel['mode']], sel['penalty'], sel['margin']))
    path = prior.study.parent.parent.save_actions(PRIVATE / f'choices_{ident["fold"]}_{ident["seed"]}.npz',
        dict(ids=h['ids'], anchor=anchor, proposal=p['choice'], **choices,
             **{f'est_{k}': v for k, v in est.items()}))
    return dict(identity=ident, cut=cut, groups=groups, fits=fits, ladder=table, selected=sel, choices=path), updates


def evaluate(cells, d, donor, beat):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{a}_vs_{k}': (b, a) for a in ARMS for k, b in (('cv', base.CV), ('ols4', base.OLS))}
    primary = dict(median_vs_parent=('parent', 'median'), tail_vs_parent=('parent', 'tail'),
        tail_vs_median=('median', 'tail'))
    pairs.update(primary)
    rows, geom, exceed = [], {}, {}
    swaps, peak, oracle = dict.fromkeys(MODES, 0), dict.fromkeys(ARMS, 0.), []
    for cell in cells:
        ident = cell['identity']
        hi, h = base.subset(d, ident['held'])
        seed = ident['seed']
        c, o, b = prior.bank(h['history'], {k: v[hi] for k, v in prior.read_donor(donor, seed, d).items()})
        _, scale = parent.parent.context.context(h)
        with np.load(old.checked(cell['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            a, proposal = z['anchor'], z['proposal']
            paths = {base.CV: c, base.OLS: o, 'parent': parent.model.apply(c, b, a)}
            paths.update({k: parent.model.apply(c, b, z[k]) for k in (*MODES, 'calibrated')})
            estimate = {k: z[f'est_{k}'] for k in MODES}
            for k in MODES:
                np.testing.assert_array_equal(z[k][z[k] >= 0], proposal[z[k] >= 0])
                swaps[k] += int(((a < 0) & (z[k] >= 0)).sum())
        for k in ARMS:
            peak[k] += float(np.linalg.norm(paths[k] - c, axis=-1).max(1).sum())
        p = parent.model.apply(c, b, proposal)
        for rec in np.unique(h['recording']):
            ix = h['recording'] == rec
            y, v, q, history = (h[k][ix] for k in ('target', 'valid', 'query', 'history'))
            local = {k: z[ix] for k, z in paths.items()}
            s, = np.unique(h['sites'][ix])
            w, groups = old.weight(q), prior.query_groups(q)
            r = scorer.score_recording(local, {}, y, v, q, str(rec),
                easy_cut=cell['cut'][0], hard_cut=cell['cut'][1])
            terms = {k: full(local[l], local[r], c[ix], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(s), seed=seed, comparisons=terms))
            mass, offset, gain, _ = candidate_terms(c[ix], o[ix], p[ix, None], y, v)
            upper = model.upper_at_count(gain[:, 0], proposal[ix], a[ix], groups, w)
            oracle.append([float(x) for x in upper + [0., w @ offset]] if w @ mass > 0. else [None, None])
            t = model.targets(c[ix], p[ix], y, v, scale[ix], cell['cut'][0])
            selected = a[ix] >= 0
            stat = exceed.setdefault((str(s), seed), np.zeros((len(MODES), 2, 3)))
            for j, k in enumerate(MODES):
                err = t[selected] - estimate[k][ix][selected]
                stat[j, 0] += (err > 0.).sum(0)
                stat[j, 1] += selected.sum()
            z = geom.setdefault((str(s), seed), np.zeros((len(primary), 4)))
            for g in groups:
                n = (local['parent'][g] != c[ix][g]).any((1, 2)).sum()
                if any((local[k][g] != c[ix][g]).any((1, 2)).sum() != n for k in MODES):
                    raise ValueError('Unmatched intervention count')
                arr = np.stack([local[k][g] for pair in primary.values() for k in pair])
                _, u, _, (steps, known, _, _) = prior.geometry.query(history[g], arr, y[g], v[g],
                    tuple((2*i, 2*i+1) for i in range(len(primary))))
                if steps:
                    z[:, 1] += u[:, 1]
                    z[:, 3] += 1
                if known:
                    z[:, 0] += u[:, 0]
                    z[:, 2] += 1
        beat(phase='held_scored', fold=ident['fold'], seed=seed)
    roster = {r['recording']: r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    draws = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    x = np.asarray([[geom[s, k] for k in base.SEEDS] for s in sites])
    x = np.divide(x[..., :2], x[..., 2:], out=np.full_like(x[..., :2], np.nan), where=x[..., 2:] > 0.)
    geometry = {k: dict(observed_pixel=prior.geometry.bootstrap(x[:, :, i, 0], draws),
        completion_lower_pixel=prior.geometry.bootstrap(x[:, :, i, 1], draws)) for i, k in enumerate(primary)}
    exceedance = {}
    for i, k in enumerate(MODES):
        stat = np.asarray([[exceed[s, seed][i] for seed in base.SEEDS] for s in sites])
        rate = np.divide(stat[:, :, 0], stat[:, :, 1], out=np.full_like(stat[:, :, 0], np.nan), where=stat[:, :, 1] > 0.)
        exceedance[k] = [prior.geometry.bootstrap(rate[:, :, j], draws) for j in range(rate.shape[-1])]
    upper = {k: sum(x[j] is not None and x[j] <= 0. for x in oracle) for j, k in enumerate(('cv', 'ols4'))}
    known_only = {}
    for k in ARMS:
        risk = [r['metrics'][k]['risk'] for r in rows]
        known_only[k] = sum(all(x[key] is not None and x[key] <= .02
            for key in ('known_all_risk', 'known_easy_risk', 'known_easy_net_degradation')) for x in risk)
    return dict(summary=summary, peak=peak, swaps=swaps, geometry=geometry, known_only_passes=known_only,
        matched_count_one_path_excluded=upper, upper_undefined=sum(x[0] is None for x in oracle),
        parent_nomination_exceedance=exceedance)


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

        refs = [parent.OUT / k for k in ('registration.json', 'freeze.json', 'readout.json')]
        reg, frozen, report = map(old.read, refs)
        if (frozen['registration'] != old.sha(refs[0]) or report['freeze'] != old.sha(refs[1])
                or not frozen['before_held_scoring'] or not reg['original_2pct'] or not report['original_2pct']
                or report['a_labels_opened'] or report['independent_confirmation']):
            raise ValueError('Require frozen development-only parent')
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(p) for p in refs],
            files=[old.artifact(p) for p in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            quantiles=MODES, grid=GRID, steps=STEPS, seeds=base.SEEDS, original_2pct=True)))
        _, donor, _ = base.register()
        d = base.fit_population(beat)
        cells, updates = [], 0
        for f in frozen['cells']:
            c, n = cell(d, donor, f, old.sha(rp), beat, args.replay)
            cells.append(c)
            updates += n
            beat(phase='cell_frozen', fold=c['identity']['fold'], seed=c['identity']['seed'], updates=updates)
        fp = OUT / 'freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(rp), cells=cells,
            optimizer_steps=18*STEPS, before_held_scoring=True)))
        result = evaluate(old.read(fp)['cells'], d, donor, beat)
        if result['summary']['risk']['parent'] != report['summary']['risk']['shape_joint']:
            raise ValueError('Parent risk table changed')
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fp), **result,
            qualified=sum(c['selected'] is not None for c in cells), a_labels_opened=False,
            independent_confirmation=False, original_2pct=True)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
