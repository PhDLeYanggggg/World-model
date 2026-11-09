"""Fit and compare variable-count whole-query intervention scorers."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_cv_amplitude_response as loader
from scripts import train_m3w_cv_selected_harm as harm
from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.world_model import m3w_cv_joint_count as model

parent, old, base, prior, np, torch = harm.parent, harm.old, harm.base, harm.prior, harm.np, harm.torch
OUT = harm.OUT.parent / 'moving_cv_joint_count_v1'
PRIVATE = harm.PRIVATE.parent / OUT.name
STEPS = 1200
MODES = ('no_pair', 'joint')
ARMS = ('portfolio', 'independent_quarter', 'no_pair', 'joint', 'independent_matched',
        'joint_fixed', 'no_pair_matched', 'calibrated')
GRID = tuple((a, p, w, t) for a in MODES for p in (1., 2., 4.) for w in (0., .25) for t in (0., .02))
PRIMARY = dict(joint_at_count=('independent_matched', 'joint'), count_choice=('joint_fixed', 'joint'),
    pair_features_at_count=('no_pair_matched', 'joint'))


def paths(cv, b, actions):
    j, k = np.divmod(actions.clip(0), parent.model.M)
    alpha = np.asarray(parent.model.AMPLITUDES)[k, None, None]
    candidate = cv[None] + alpha * (b[np.arange(len(cv)), j] - cv[None])
    return np.where((actions >= 0)[..., None, None], candidate, cv[None])


def pair_inputs(h, cv, forecast, mask, scale, edges):
    i, j = edges
    if not len(i):
        return np.zeros((len(forecast), model.PAIR_FEATURES))
    relative = (h[i, -1] - h[j, -1])[:, None]
    ref = cv[i] - cv[j]
    new = forecast[:, i] - forecast[:, j]
    d = np.linalg.norm(new - ref[None], axis=-1) / scale
    change = (np.linalg.norm(ref + relative, axis=-1) -
              np.linalg.norm(new + relative[None], axis=-1)) / scale
    return np.column_stack((d.mean((1, 2)), d.max((1, 2)), change.clip(0.).mean((1, 2)),
        change.clip(0.).max((1, 2)), (mask[:, i] & mask[:, j]).mean(1),
        np.full(len(forecast), np.linalg.norm(relative, axis=-1).mean() / scale)))


def truth(cv, ols, forecast, y, v, scale, cut, edges):
    p = forecast.swapaxes(0, 1)
    _, _, gain, _ = candidate_terms(cv, ols, p, y, v)
    risk = risk_excess(cv, p, y, v, cut)
    safe = np.where(v[..., None], y, 0.)
    known = v.sum(1)
    base_error = (np.linalg.norm(cv - safe, axis=-1) * v).sum(1) / known.clip(1)
    e = (np.linalg.norm(p - safe[:, None], axis=-1) * v[:, None]).sum(2) / known.clip(1)[:, None]
    easy = (base_error > 0.) & (base_error <= cut)
    signed = np.where(easy[:, None], e - base_error[:, None], 0.).mean(0)
    i, j = edges
    pair = np.zeros(len(forecast))
    if len(i):
        rel = safe[i] - safe[j]
        a = cv[i] - cv[j]
        b = forecast[:, i] - forecast[:, j]
        observed = v[i] & v[j]
        diff = np.linalg.norm(a - rel, axis=-1) - np.linalg.norm(b - rel[None], axis=-1)
        pair = np.where(observed[None], diff, -np.linalg.norm(a - b, axis=-1)).mean((1, 2))
    return model.terms(gain.mean(0) / 12., risk.mean(0), signed, pair, scale).astype(np.float32)


def packet(d, shape, cut=None):
    h = d['history']
    cv, ols, bank = parent.analytic(h)
    gain, risk, legal, ref = parent.predictions({'shape': shape}, d, cv, ols, bank)['shape']
    groups = prior.query_groups(d['query'])
    anchor = parent.model.portfolio(gain, risk, legal, ref, groups, .50, 0.)
    scale = prior.model.scale(h)
    out = []
    for at in groups:
        c, o, b = cv[at], ols[at], bank[at]
        l, g, r = legal[at], gain[at], risk[at]
        s = float(scale[at].mean())
        a = np.arange(l.shape[1], dtype=np.int16)
        ray, amp = np.divmod(a, parent.model.M)
        all_path = c[:, None] + np.asarray(parent.model.AMPLITUDES)[amp][None, :, None, None] * (b[:, ray] - c[:, None])
        agreement = np.linalg.norm(c - o, axis=-1).mean(1)[:, None] - np.linalg.norm(all_path - o[:, None], axis=-1).mean(2)
        plans, independent = model.menu(g, r, l, agreement, anchor[at])
        active = plans >= 0
        forecast = paths(c, b, plans)
        projected = np.concatenate((g, r, agreement[..., None]), axis=-1)[np.arange(len(at)), plans]
        z = np.where(active[..., None], projected, 0.) / s
        disp = np.linalg.norm(forecast - c, axis=-1) / s
        edges = prior.geometry.local_edges(h[at])
        info = np.column_stack((z.mean(1), z.std(1), z.min(1), z.max(1),
            disp.mean((1, 2)), disp.max((1, 2)), active.mean(1),
            np.tile([np.log1p(len(at)), np.log1p(s), l.any(1).mean(), ref[at, 1].mean() / s], (len(plans), 1)),
            pair_inputs(h[at], c, forecast, active, s, edges)))
        features = np.arcsinh(info).astype(np.float32)
        target = None if cut is None else truth(c, o, forecast, d['target'][at], d['valid'][at], s, cut, edges)
        if not np.isfinite(features).all() or (target is not None and not np.isfinite(target).all()):
            raise ValueError('Nonfinite slate representation')
        out.append(dict(at=at, x=features, y=target, plans=plans, counts=active.sum(1),
            independent=independent, anchor_count=int((anchor[at] >= 0).sum())))
    return cv, ols, bank, anchor, out


def sample_query(d, packet):
    usable = [p for p in packet if len(p['x']) > 1]
    sites = np.asarray([d['sites'][p['at'][0]] for p in usable])
    groups = [np.flatnonzero(sites == k) for k in sorted(set(d['sites']))]
    if any(not len(g) for g in groups):
        raise ValueError('Every fit source needs a nontrivial slate')

    def draw(rng):
        at = prior.draw(groups, rng)
        x, y = [], []
        for k in at:
            p = usable[k]
            i = rng.choice(len(p['x']), size=2, replace=False)
            x.append(p['x'][i])
            y.append(p['y'][i])
        return torch.from_numpy(np.stack(x)), torch.from_numpy(np.stack(y))

    return draw, [len(g) for g in groups]


@torch.inference_mode()
def predict(nets, packet):
    x = torch.from_numpy(np.concatenate([p['x'] for p in packet]))
    cuts = np.cumsum([len(p['x']) for p in packet])[:-1]
    out = {}
    for a, net in nets.items():
        if a == 'no_pair':
            xx = torch.cat((x[:, :-model.PAIR_FEATURES], torch.zeros_like(x[:, -model.PAIR_FEATURES:])), 1)
        else:
            xx = x
        y = torch.cat([net(chunk) for chunk in xx.split(4096)]).numpy()
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite query scorer predictions')
        out[a] = np.split(y, cuts)
    return out


def choose(packet, estimates, n, *, matched_controls=False, setting=('joint', 2., .25, 0.)):
    names = ARMS[1:-1] if matched_controls else ('selected',)
    out = {a: np.full(n, -1, np.int16) for a in names}
    a, price, weight, margin = setting
    for j, p in enumerate(packet):
        counts = p['counts']
        at = p['at']
        i = model.select(estimates[a][j], counts, price, weight, margin)
        if not matched_controls:
            out['selected'][at] = p['plans'][i]
            continue
        k = int(counts[i])
        independent = p['independent']
        quarter = int(np.ceil(.25 * int(counts.max())))
        choices = dict(independent_quarter=independent[quarter], joint=i,
            no_pair=model.select(estimates['no_pair'][j], counts, price, weight, margin),
            independent_matched=independent[k],
            joint_fixed=model.select(estimates['joint'][j], counts, price, weight, margin, p['anchor_count']),
            no_pair_matched=model.select(estimates['no_pair'][j], counts, price, weight, margin, k))
        for key, s in choices.items():
            out[key][at] = p['plans'][s]
    return out


def calibrate(d, c, o, b, packet, estimates, cut):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    recs = [np.flatnonzero(d['recording'] == r) for r in np.unique(d['recording'])]
    table = []
    for setting in GRID:
        a = choose(packet, estimates, len(c), setting=setting)['selected']
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
        gains = [[scorer.merge_comparisons([x[2][j] for x in rows if x[0] == site])['completion_lower_pp']
                  for j in range(2)] for site in sorted(set(d['sites']))]
        table.append(dict(setting=setting, edits=int((a >= 0).sum()),
            passes=sum(x[1] for x in rows), lower_pp=np.mean(gains, axis=0).tolist()))
    good = [x for x in table if x['edits'] and x['passes'] == len(recs) and min(x['lower_pp']) > 0.]
    return table, max(good, key=lambda x: (min(x['lower_pp']), -x['edits'])) if good else None


def cell(d, frozen, reg, beat, replay):
    ident, cut = frozen['identity'], frozen['cut']
    tr = old.teacher.fit_role(d, ident)
    x, _ = parent.parent.context.context(tr)
    shape = loader.load(frozen['fits'][1], x.shape[1] + 37)
    *_, train = packet(tr, shape, cut[0])
    draw, groups = sample_query(tr, train)
    fits, nets, updates, rng0 = {}, {}, 0, None
    for a in MODES:
        def objective(net, x, y):
            return model.loss(net, x, y, a == 'joint')

        name = f'{a}_{ident["fold"]}_{ident["seed"]}'
        net, fit, n = prior.prefix.trainer.fit(dict(ident, arm=a, registration=reg), name,
            ident['seed'], lambda: prior.model.head(train[0]['x'].shape[1], 6), STEPS, draw, objective, replay, beat)
        rng = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)['rng']
        if rng0 is not None and rng != rng0:
            raise ValueError('Geometry ablation changed training draws')
        fits[a], nets[a], updates, rng0 = fit, net, updates + n, rng
    _, cal = base.subset(d, ident['calibration'])
    cv, ols, b, _, p = packet(cal, shape)
    table, sel = calibrate(cal, cv, ols, b, p, predict(nets, p), cut)
    _, h = base.subset(d, ident['held'])
    cv, _, _, a, p = packet(h, shape)
    est = predict(nets, p)
    out = choose(p, est, len(cv), matched_controls=True)
    out['portfolio'] = a
    out['calibrated'] = (np.full_like(a, -1) if sel is None else
        choose(p, est, len(cv), setting=sel['setting'])['selected'])
    path = prior.study.parent.parent.save_actions(PRIVATE / f'choices_{ident["fold"]}_{ident["seed"]}.npz',
        dict(ids=h['ids'], **out))
    return dict(identity=ident, cut=cut, fits=fits, ladder=table, selected=sel, choices=path,
        fit_queries=groups, held_options=sum(len(q['plans']) for q in p)), updates


def evaluate(cells, d, beat):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{a}_vs_{k}': (b, a) for a in ARMS for k, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs.update(PRIMARY)
    rows, geo, count = [], {}, dict(total=0, higher=0, lower=0, same=0, all=0, zero=0)
    changed = dict.fromkeys(PRIMARY, 0)
    for c in cells:
        ident = c['identity']
        _, h = base.subset(d, ident['held'])
        cv, ols, bank = parent.analytic(h['history'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            actions = {a: z[a] for a in ARMS}
        paths = {base.CV: cv, base.OLS: ols, **{a: parent.model.apply(cv, bank, z) for a, z in actions.items()}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q, hist = (h[k][at] for k in ('target', 'valid', 'query', 'history'))
            z = {k: p[at] for k, p in paths.items()}
            a = {k: p[at] for k, p in actions.items()}
            s, = np.unique(h['sites'][at])
            seed = ident['seed']
            r = scorer.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(s), seed=seed, comparisons=terms))
            block = geo.setdefault((str(s), seed), np.zeros((len(PRIMARY), 4)))
            for g in prior.query_groups(q):
                n = {k: int((p[g] >= 0).sum()) for k, p in a.items()}
                k = n['joint']
                if k != n['independent_matched'] or k != n['no_pair_matched'] or n['joint_fixed'] != n['portfolio']:
                    raise ValueError('Query count control changed')
                count['total'] += 1
                count['higher' if k > n['portfolio'] else 'lower' if k < n['portfolio'] else 'same'] += 1
                count['all'] += k == len(g)
                count['zero'] += not k
                for name, (l, r) in PRIMARY.items():
                    changed[name] += int((a[l][g] != a[r][g]).sum())
                arr = np.stack([z[k][g] for pair in PRIMARY.values() for k in pair])
                _, u, _, (steps, known, _, _) = prior.geometry.query(hist[g], arr, y[g], v[g],
                    tuple((2*j, 2*j+1) for j in range(len(PRIMARY))))
                if steps:
                    block[:, 1] += u[:, 1]
                    block[:, 3] += 1
                if known:
                    block[:, 0] += u[:, 0]
                    block[:, 2] += 1
        beat(phase='held_scored', fold=ident['fold'], seed=ident['seed'])
    roster = {x['recording']: x['source'] for x in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    x = np.asarray([[geo[s, k] for k in base.SEEDS] for s in sites])
    p = np.divide(x[..., :2], x[..., 2:], out=np.full_like(x[..., :2], np.nan), where=x[..., 2:] > 0.)
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    geometry = {k: dict(observed_pixel=prior.geometry.bootstrap(p[:, :, i, 0], draw),
        completion_lower_pixel=prior.geometry.bootstrap(p[:, :, i, 1], draw)) for i, k in enumerate(PRIMARY)}
    return dict(summary=summary, geometry=geometry, query_counts=count, changed_row_actions=changed)


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
        refs += [harm.OUT / 'capacity.json']
        reg, frozen, report, capacity = map(old.read, refs)
        if (frozen['registration'] != old.sha(refs[0]) or report['freeze'] != old.sha(refs[1])
                or not frozen['before_held_scoring'] or not reg['original_2pct']
                or any(z['a_labels_opened'] or z['independent_confirmation'] or not z['original_2pct']
                       for z in (report, capacity))):
            raise ValueError('Require frozen development-only parent')
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(p) for p in refs],
            files=[old.artifact(p) for p in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            analytic_only=True, grid=GRID, steps=STEPS, seeds=base.SEEDS, original_2pct=True)))
        base.register()
        d = base.fit_population(beat)
        cells, updates = [], 0
        for f in frozen['cells']:
            c, n = cell(d, f, old.sha(rp), beat, args.replay)
            cells.append(c)
            updates += n
            beat(phase='cell_frozen', fold=c['identity']['fold'], seed=c['identity']['seed'], updates=updates)
        fp = OUT / 'freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(rp), cells=cells, optimizer_steps=18*STEPS,
            before_held_scoring=True)))
        result = evaluate(old.read(fp)['cells'], d, beat)
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fp), **result,
            qualified=sum(c['selected'] is not None for c in cells), a_labels_opened=False,
            independent_confirmation=False, original_2pct=True)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
