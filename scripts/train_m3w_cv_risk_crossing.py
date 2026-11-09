"""Train matched risk-boundary and utility-only action rankers."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_risk_frontier as frontier
from scripts import run_m3w_cv_minimum_budget as minimum
from scripts import run_m3w_cv_slate_supervision as slate
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model import m3w_cv_risk_crossing as model

study, old, base, prior, np, torch = frontier.study, frontier.old, frontier.base, frontier.prior, frontier.np, frontier.torch
OUT = frontier.OUT.parent / 'moving_cv_risk_crossing_v1'
PRIVATE = frontier.PRIVATE.parent / OUT.name
STEPS = 1400
MODES = ('utility', 'risk')
ARMS = ('portfolio', 'independent', 'utility_matched', 'risk_matched', 'utility_free', 'risk_free', 'calibrated')
PRIMARY = dict(risk_vs_utility_at_count=('utility_matched', 'risk_matched'),
    risk_vs_independent_at_count=('independent', 'risk_matched'),
    risk_vs_portfolio_at_count=('portfolio', 'risk_matched'), risk_vs_utility_free=('utility_free', 'risk_free'))
GRID = tuple((a, strength, f, slack) for a in ('base', *MODES)
             for strength in ((0.,) if a == 'base' else (.25, .5, 1.))
             for f in (.25, .50, 1.) for slack in (0., .02))


def bank_paths(cv, bank):
    ray, amp = np.divmod(np.arange(bank.shape[1] * study.parent.model.M), study.parent.model.M)
    return cv[:, None] + np.asarray(study.parent.model.AMPLITUDES)[amp][None, :, None, None] * (bank[:, ray] - cv[:, None])


def quantities(cv, ols, p, y, valid, query, cut):
    mass, offset, gain, _ = study.candidate_terms(cv, ols, p, y, valid)
    w = old.weight(query)
    ref = w @ mass
    base_error, _ = errors(cv, y, valid)
    easy = valid.any(1) & (base_error > 0.) & (base_error <= cut)
    risk = study.risk_excess(cv, p, y, valid, cut)
    safe = np.where(valid[..., None], y, 0.)
    e = (np.linalg.norm(p - safe[:, None], axis=-1) * valid[:, None]).sum(2) / valid.sum(1).clip(1)[:, None]
    signed = np.where(easy[:, None], e - base_error[:, None], 0.)
    support = np.broadcast_to(easy[:, None], signed.shape)
    z = np.concatenate((gain * (100. * w / ref)[:, None, None], -risk, -signed[..., None], support[..., None]), axis=-1)
    required = np.r_[model.GAIN_MARGIN, model.GAIN_MARGIN - 100. * (w @ offset) / ref, 0., 0.,
                     -.02 * base_error[easy].sum(), 1.]
    return z, required


def candidates(anchor, legal, h, groups):
    # Use every alternative path on the same recipient. Also test paths
    # on the two closest currently unedited rows of the same query.
    left = np.flatnonzero(anchor >= 0)
    n = legal.shape[1]
    i = np.repeat(left, n)
    j = i.copy()
    arm = np.tile(np.arange(n), len(left))
    cross = []
    for q in groups:
        a = q[anchor[q] >= 0]
        b = q[(anchor[q] < 0) & legal[q].any(1)]
        if len(a) and len(b):
            dist = np.square(h[a, -1, None, :] - h[b, -1, :]).sum(-1)
            nearest = b[np.argsort(dist, axis=1, kind='stable')[:, :2]]
            cross.append((np.repeat(a, nearest.shape[1] * n), np.repeat(nearest.ravel(), n),
                          np.tile(np.arange(n), nearest.size)))
    if cross:
        i, j, arm = (np.concatenate((x, *col)) for x, col in zip((i, j, arm), zip(*cross), strict=True))
    mask = legal[j, arm] & ((i != j) | (arm != anchor[i]))
    return i[mask], anchor[i[mask]], j[mask], arm[mask]


def training(d, ident, cut, parents):
    tr = old.teacher.fit_role(d, ident)
    cv, ols, bank = study.parent.analytic(tr['history'])
    f = ident['fold']
    actions, masks = [], []
    for receipt in parents:
        cell = receipt['cells'][f]
        if cell['train'] != ident['train']:
            raise ValueError('Training witness escaped its source role')
        with np.load(old.checked(cell['teacher']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], tr['ids'])
            actions.append(model.to_current(z['action'], base.FRACTIONS,
                study.parent.model.AMPLITUDES, bank.shape[1]))
            masks.append(z['verified'])
    np.testing.assert_array_equal(masks[0], masks[1])
    past, scale = study.parent.parent.context.context(tr)
    packed, by_record, risk_axes, reverified = [], [], np.zeros(3, int), 0
    score = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    for rec in np.unique(tr['recording']):
        at = np.flatnonzero(tr['recording'] == rec)
        if not masks[0][at].all():
            continue
        y, v, q = (tr[k][at] for k in ('target', 'valid', 'query'))
        h, c, o, b = tr['history'][at], cv[at], ols[at], bank[at]
        p = bank_paths(c, b)
        legal = (p != c[:, None]).any((2, 3))
        z, required = quantities(c, o, p, y, v, q, cut[0])
        pairs = []
        for a in {t[at].tobytes(): t[at] for t in actions}.values():
            forecast = study.parent.model.apply(c, b, a)
            proof = score.score_recording({base.CV: c, 'p': forecast}, {}, y, v, q, str(rec),
                easy_cut=cut[0], hard_cut=cut[1])
            gain = [score.merge_comparisons([full(ref, forecast, c, y, v, old.weight(q))])['completion_lower_pp']
                    for ref in (c, o)]
            if not proof['metrics']['p']['risk']['supported'] or min(gain) <= 0.:
                raise ValueError('Mapped training witness failed original scorer')
            reverified += 1
            i, ai, j, aj = candidates(a, legal, h, prior.query_groups(q))
            keep, utility, risk = model.crossings(z, required, a, i, ai, j, aj)
            if keep.any():
                pairs.append(np.column_stack((at[i[keep]], ai[keep], at[j[keep]], aj[keep], utility[keep])))
                risk_axes += risk[keep].sum(0)
        if pairs:
            block = np.unique(np.concatenate(pairs), axis=0)
            by_record.append(len(block))
            packed.append(block)
    if not packed:
        raise ValueError('No risk-boundary pairs in fitting sources')
    p = np.concatenate(packed)
    if not (p[:, -1] != 0).any():
        raise ValueError('No utility labels for the matched comparison')
    sizes = np.asarray(by_record)
    offsets = np.r_[0, sizes.cumsum()]
    # Square-root allocation keeps large recordings visible without
    # repeatedly drawing a single pair from a tiny recording.
    prob = np.sqrt(sizes) / np.sqrt(sizes).sum()

    def draw(rng):
        group = rng.choice(len(sizes), size=256, p=prob)
        z = p[offsets[group] + rng.integers(sizes[group])]
        ij = z[:, [0, 2]].ravel()
        ray, amp = np.divmod(z[:, [1, 3]].ravel(), study.parent.model.M)
        a = cv[ij] + np.asarray(study.parent.model.AMPLITUDES)[amp, None, None] * (bank[ij, ray] - cv[ij])
        x = study.parent.parent.context.model.action_features(past[ij], cv[ij], ols[ij], a, scale[ij])
        return torch.from_numpy(x.reshape(len(z), 2, -1)), torch.from_numpy(z[:, -1].copy())

    stats = dict(verified_witnesses=reverified, pair_recordings=len(by_record), pairs=len(p),
        recipient_exchanges=int((p[:, 0] != p[:, 2]).sum()), utility_prefers_unsafe=int((p[:, -1] < 0).sum()),
        utility_ties=int((p[:, -1] == 0).sum()), risk_crossings_before_dedup=risk_axes.tolist())
    return draw, past.shape[1]+37, stats


@torch.inference_mode()
def predictions(d, shape, nets):
    cv, ols, b = study.parent.analytic(d['history'])
    g, r, legal, ref = study.parent.predictions({'shape': shape}, d, cv, ols, b)['shape']
    past, scale = study.parent.parent.context.context(d)
    result = {k: np.empty(legal.shape, np.float32) for k in nets}
    for a in range(legal.shape[1]):
        ray, amp = divmod(a, study.parent.model.M)
        p = cv + study.parent.model.AMPLITUDES[amp] * (b[:, ray] - cv)
        for i in range(0, len(cv), 4096):
            q = slice(i, i+4096)
            x = study.parent.parent.context.model.action_features(past[q], cv[q], ols[q], p[q], scale[q])
            for k, net in nets.items():
                result[k][q, a] = net(torch.from_numpy(x)).flatten().numpy()
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError('Nonfinite action scores')
    return cv, ols, b, g, r, legal, ref, scale, result


def decode(z, groups, k, strength, fraction, slack):
    _, _, _, g, r, l, ref, scale, prediction = z
    if k != 'base':
        g = model.adjusted_gain(g, prediction[k], scale, strength)
    return study.parent.model.portfolio(g, r, l, ref, groups, fraction, slack)


def calibrate(d, z, groups, cut):
    cv, ols, bank = z[:3]
    score = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    recs = [np.flatnonzero(d['recording'] == r) for r in np.unique(d['recording'])]
    table = []
    for spec in GRID:
        a = decode(z, groups, *spec)
        p = study.parent.model.apply(cv, bank, a)
        rows = []
        for at in recs:
            y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
            rec, = np.unique(d['recording'][at])
            s, = np.unique(d['sites'][at])
            r = score.score_recording({base.CV: cv[at], 'p': p[at]}, {}, y, v, q, str(rec),
                easy_cut=cut[0], hard_cut=cut[1])
            g = [full(x[at], p[at], cv[at], y, v, old.weight(q)) for x in (cv, ols)]
            rows.append((s, r['metrics']['p']['risk']['supported'], g))
        lo = [[score.merge_comparisons([x[2][j] for x in rows if x[0] == s])['completion_lower_pp']
               for j in range(2)] for s in sorted(set(d['sites']))]
        table.append(dict(spec=spec, edits=int((a >= 0).sum()), passes=sum(x[1] for x in rows),
            lower=np.mean(lo, axis=0).tolist()))
    good = [r for r in table if r['edits'] and r['passes'] == len(recs) and min(r['lower']) > 0.]
    return table, max(good, key=lambda x: (min(x['lower']), -x['edits'])) if good else None


def cell(d, c, draw, width, reg, beat, replay):
    ident, cut = c['identity'], c['cut']
    nets, fits, updates, rng = {}, {}, 0, None
    for a in MODES:
        def objective(net, x, y):
            return model.objective(net, x, y, a == 'risk')

        name = f'{a}_{ident["fold"]}_{ident["seed"]}'
        net, fit, n = prior.prefix.trainer.fit(dict(ident, arm=a, registration=reg), name, ident['seed'],
            lambda: prior.model.head(width, 1), STEPS, draw, objective, replay, beat)
        r = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)['rng']
        if rng is not None and rng != r:
            raise ValueError('Training streams diverged')
        nets[a], fits[a], updates, rng = net, fit, updates+n, r
    shape = study.loader.load(c['fits'][1], width)
    _, cal = base.subset(d, ident['calibration'])
    z = predictions(cal, shape, nets)
    ladder, selected = calibrate(cal, z, prior.query_groups(cal['query']), cut)
    _, h = base.subset(d, ident['held'])
    q = prior.query_groups(h['query'])
    z = predictions(h, shape, nets)
    _, _, _, g, r, l, _, scale, p = z
    a = decode(z, q, 'base', 0., .50, 0.)
    out = dict(portfolio=a, independent=study.parent.model.independent(g, r, l, q, .50, match=a))
    for k in MODES:
        gain = model.adjusted_gain(g, p[k], scale, .5)
        out[f'{k}_matched'] = study.parent.model.independent(gain, r, l, q, .50, match=a)
        out[f'{k}_free'] = decode(z, q, k, .5, .50, 0.)
    out['calibrated'] = (np.full_like(a, -1) if selected is None else decode(z, q, *selected['spec']))
    path = prior.study.parent.parent.save_actions(PRIVATE / f'choices_{ident["fold"]}_{ident["seed"]}.npz',
        dict(ids=h['ids'], **out))
    return dict(identity=ident, cut=cut, fits=fits, ladder=ladder, selected=selected, choices=path), updates


def evaluate(cells, d, beat):
    score = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{a}_vs_{k}': (b, a) for a in ARMS for k, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs.update(PRIMARY)
    rows, geometry, changed = [], {}, dict.fromkeys(PRIMARY, 0)
    for c in cells:
        i = c['identity']
        _, h = base.subset(d, i['held'])
        cv, ols, bank = study.parent.analytic(h['history'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            a = {k: z[k] for k in ARMS}
            paths = {base.CV: cv, base.OLS: ols, **{k: study.parent.model.apply(cv, bank, a[k]) for k in ARMS}}
        for q in prior.query_groups(h['query']):
            if len({int((a[k][q] >= 0).sum()) for k in ARMS[:4]}) != 1:
                raise ValueError('Matched query counts changed')
        for k, (l, r) in PRIMARY.items():
            changed[k] += int((a[l] != a[r]).sum())
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q, history = (h[k][at] for k in ('target', 'valid', 'query', 'history'))
            s, = np.unique(h['sites'][at])
            z = {k: p[at] for k, p in paths.items()}
            r = score.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(s), seed=i['seed'], comparisons=terms))
            out = geometry.setdefault((str(s), i['seed']), np.zeros((len(PRIMARY), 2)))
            for g in prior.query_groups(q):
                p = np.stack([z[k][g] for pair in PRIMARY.values() for k in pair])
                _, u, _, (steps, _, _, _) = prior.geometry.query(history[g], p, y[g], v[g],
                    tuple((2*j, 2*j+1) for j in range(len(PRIMARY))))
                if steps:
                    out[:, 0] += u[:, 1]
                    out[:, 1] += 1
        beat(phase='held_scored', fold=i['fold'], seed=i['seed'])
    roster = {r['recording']: r['source'] for r in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    a = np.asarray([[geometry[s, k] for k in base.SEEDS] for s in sites])
    b = np.divide(a[..., 0], a[..., 1], out=np.full_like(a[..., 0], np.nan), where=a[..., 1] > 0.)
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    return summary, {k: prior.geometry.bootstrap(b[:, :, i], draw) for i, k in enumerate(PRIMARY)}, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay', action='store_true')
    args = ap.parse_args()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    prior.prefix.trainer.PRIVATE = PRIVATE
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t = time.monotonic()
    with (PRIVATE / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def beat(**v):
            z = dict(pid=os.getpid(), seconds=round(time.monotonic()-t, 3), **v)
            prior.donor.prior.prior.source.atomic(PRIVATE / 'heartbeat.json', z)
            if v.get('phase') != 'fit' or v.get('step') in (100, STEPS):
                print(z, flush=True)

        refs = [study.parent.OUT / 'freeze.json', slate.OUT / 'teacher_freeze.json',
                minimum.OUT / 'teacher_freeze.json', frontier.OUT / 'integer_readout.json',
                study.OUT / 'readout.json']
        shape, witnesses, reduced, frontier_report, previous = map(old.read, refs)
        if (not shape['before_held_scoring'] or not witnesses['analytic_only'] or witnesses['independent_confirmation']
                or not reduced['train_only'] or any(x['a_labels_opened'] or x['independent_confirmation']
                    or not x['original_2pct'] for x in (frontier_report, previous))):
            raise ValueError('Require source-disjoint training witnesses')
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(p) for p in refs], original_2pct=True,
            files=[old.artifact(p) for p in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            grid=GRID, steps=STEPS, seeds=base.SEEDS, train_only_witnesses=True)))
        base.register()
        d = base.fit_population(beat)
        cells, stats, updates = [], [], 0
        for f in range(3):
            group = [x for x in shape['cells'] if x['identity']['fold'] == f]
            c0 = group[0]
            if any(x['identity']['train'] != c0['identity']['train'] or x['cut'] != c0['cut'] for x in group):
                raise ValueError('Seeds disagree on fitting roles')
            draw, width, labels = training(d, c0['identity'], c0['cut'], (witnesses, reduced))
            stats.append(dict(fold=f, **labels))
            beat(phase='training_pairs', **stats[-1])
            for c in group:
                result, n = cell(d, c, draw, width, old.sha(rp), beat, args.replay)
                cells.append(result)
                updates += n
                beat(phase='cell_frozen', fold=f, seed=c['identity']['seed'], updates=updates)
        fp = OUT / 'freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(rp), cells=cells, teacher_stats=stats,
            optimizer_steps=18*STEPS, before_held_scoring=True)))
        summary, geometry, changed = evaluate(old.read(fp)['cells'], d, beat)
        if summary['risk']['portfolio'] != previous['summary']['risk']['portfolio']:
            raise ValueError('Prior analytic portfolio changed')
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fp), summary=summary, geometry=geometry,
            changed=changed, qualified=sum(c['selected'] is not None for c in cells),
            a_labels_opened=False, independent_confirmation=False, original_2pct=True)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
