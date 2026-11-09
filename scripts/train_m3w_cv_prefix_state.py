"""Fit state-aware query scores using causal recording-prefix accounting."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import train_m3w_cv_joint_count as study
from scripts import train_m3w_cv_risk_crossing as crossing
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model import m3w_cv_prefix_state as model

old, base, prior, np, torch = study.old, study.base, study.prior, study.np, study.torch
OUT = crossing.OUT.parent / 'moving_cv_prefix_state_v1'
PRIVATE = crossing.PRIVATE.parent / OUT.name
STEPS = 1400
MODES = ('masked', 'state')
ARMS = ('portfolio', 'independent', 'masked_matched', 'state_matched', 'loop_matched',
        'masked_free', 'state_free', 'loop_free', 'calibrated')
PRIMARY = dict(state_at_count=('masked_matched', 'state_matched'),
    state_vs_independent=('independent', 'state_matched'),
    loop_at_count=('state_matched', 'loop_matched'), state_free=('masked_free', 'state_free'),
    loop_free=('state_free', 'loop_free'))
GRID = tuple((k, p, m) for k in MODES for p in (.5, 1., 2., 4.) for m in (0., .02))


def summaries(gain, risk, ref, active):
    """Inputs carry [option, agent, field]; no edit has zero incremental gain."""
    g = np.where(active[..., None], gain, 0.).mean(1)
    r = np.where(active[..., None], risk, 0.).sum(1)
    total = ref.sum(0)
    r[:, 2] -= .02 * total[1]
    support = (active * ref[:, 1]).sum(1)
    return np.column_stack((g, r, support, np.broadcast_to(total, (len(active), 2))))


def prepare(d, shape, cut=None):
    # The entire roster is processed in batches, but each option and feature
    # here is a function of one query, with no cross-query feature yet.
    cv, ols, b, anchor, packet = study.packet(d, shape, cut)
    g, r, _, ref = study.parent.predictions({'shape': shape}, d, cv, ols, b)['shape']
    scale = prior.model.scale(d['history'])
    last_query = -1
    last_rec = -1
    for p in packet:
        at, plans = p['at'], p['plans']
        rec, = np.unique(d['recording'][at])
        q, = np.unique(d['query'][at])
        if q <= last_query or rec < last_rec:
            raise ValueError('Queries not in causal recording order')
        last_query, last_rec = q, rec
        a, = np.flatnonzero((plans == anchor[at]).all(1))
        s = float(scale[at].mean())
        mask = plans >= 0
        rows = at[None]
        v = summaries(g[rows, plans], r[rows, plans], ref[at], mask)
        p.update(rec=int(rec), query=int(q), anchor=int(a), scale=s, predicted=v)
        if cut is not None:
            y = p['y'] * s
            e, _ = errors(cv[at], d['target'][at], d['valid'][at])
            easy = d['valid'][at].any(1) & (e > 0.) & (e <= cut)
            ref_actual = np.c_[e, np.where(easy, e, 0.)]
            actual = summaries(np.broadcast_to(y[:, None, :2], (*mask.shape, 2)),
                np.broadcast_to(y[:, None, 2:5], (*mask.shape, 3)), ref_actual, np.ones_like(mask))
            actual[:, 5] = (mask * ref_actual[:, 1]).sum(1)
            p['actual'] = actual
    return cv, ols, b, anchor, packet


def shadow_packet(packet, train=False):
    state, actual, rec, elapsed = np.zeros(8), np.zeros(8), None, 0
    changed = 0
    for p in packet:
        if p['rec'] != rec:
            state, actual, rec, elapsed = np.zeros(8), np.zeros(8), p['rec'], 0
        d = p['predicted']
        n, s = len(p['at']), p['scale']
        p['state_x'] = model.features(p['x'], state, d, elapsed, n, s, True)
        if train:
            a = p['actual']
            y = model.target(p['y'], actual, a, n, s)
            zero = model.target(p['y'], np.zeros_like(actual), a, n, s)
            changed += int((np.abs(y - zero).max(1) > 1e-6).sum())
            p['y'] = y
            actual += a[p['anchor']]
        state += d[p['anchor']]
        elapsed += 1
    return changed


def sampler(d, packet):
    return study.sample_query(d, [dict(p, x=p['state_x']) for p in packet])


@torch.inference_mode()
def predictions(nets, packet):
    x = torch.from_numpy(np.concatenate([p['state_x'] for p in packet]))
    indices = np.cumsum([len(p['x']) for p in packet])[:-1]
    out = {}
    for a, net in nets.items():
        z = x if a == 'state' else torch.cat((x[:, :-model.WIDTH], torch.zeros_like(x[:, -model.WIDTH:])), 1)
        y = torch.cat([net(t) for t in z.split(4096)]).numpy()
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite state predictions')
        out[a] = np.split(y, indices)
    return out


def choose(packet, scores, n, price, margin, match=False):
    out = np.full(n, -1, np.int16)
    for p, y in zip(packet, scores, strict=True):
        i = model.pick(y, p['counts'], price, margin, p['anchor_count'] if match else None)
        out[p['at']] = p['plans'][i]
    return out


@torch.inference_mode()
def closed_loop(packet, net, n, match):
    out = np.full(n, -1, np.int16)
    rec, prefix, elapsed = None, np.zeros(8), 0
    for p in packet:
        if p['rec'] != rec:
            rec, prefix, elapsed = p['rec'], np.zeros(8), 0
        d = p['predicted']
        x = model.features(p['x'], prefix, d, elapsed, len(p['at']), p['scale'], True)
        y = net(torch.from_numpy(x)).numpy()
        i = model.pick(y, p['counts'], 2., 0., p['anchor_count'] if match else None)
        out[p['at']] = p['plans'][i]
        prefix += d[i]
        elapsed += 1
    return out


def calibrate(d, cv, ols, b, p, pred, cut):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    recs = [np.flatnonzero(d['recording'] == r) for r in np.unique(d['recording'])]
    table = []
    for k, price, margin in GRID:
        a = choose(p, pred[k], len(cv), price, margin)
        forecast = study.parent.model.apply(cv, b, a)
        rows = []
        for at in recs:
            y, v, q = (d[k][at] for k in ('target', 'valid', 'query'))
            rec, = np.unique(d['recording'][at])
            site, = np.unique(d['sites'][at])
            r = scorer.score_recording({base.CV: cv[at], 'p': forecast[at]}, {}, y, v, q, str(rec),
                easy_cut=cut[0], hard_cut=cut[1])
            g = [full(x[at], forecast[at], cv[at], y, v, old.weight(q)) for x in (cv, ols)]
            rows.append((site, r['metrics']['p']['risk']['supported'], g))
        lo = [[scorer.merge_comparisons([x[2][j] for x in rows if x[0] == s])['completion_lower_pp']
               for j in range(2)] for s in sorted(set(d['sites']))]
        table.append(dict(spec=(k, price, margin), edits=int((a >= 0).sum()), passes=sum(x[1] for x in rows),
            lower=np.mean(lo, axis=0).tolist()))
    good = [r for r in table if r['edits'] and r['passes'] == len(recs) and min(r['lower']) > 0.]
    return table, max(good, key=lambda x: (min(x['lower']), -x['edits'])) if good else None


def cell(d, c, reg, beat, replay):
    ident, cut = c['identity'], c['cut']
    tr = old.teacher.fit_role(d, ident)
    past, _ = study.parent.parent.context.context(tr)
    shape = study.loader.load(c['fits'][1], past.shape[1] + 37)
    *_, p = prepare(tr, shape, cut[0])
    n_changed = shadow_packet(p, train=True)
    sample, counts = sampler(tr, p)
    width = p[0]['state_x'].shape[1]
    nets, fits, updates, rng0 = {}, {}, 0, None
    beat(phase='prefix_targets', fold=ident['fold'], seed=ident['seed'], changed=n_changed,
        options=sum(len(q['x']) for q in p), queries=counts)
    for k in MODES:
        def loss(net, x, y):
            return model.objective(net, x, y, k == 'state')

        name = f'{k}_{ident["fold"]}_{ident["seed"]}'
        net, fit, n = prior.prefix.trainer.fit(dict(ident, arm=k, registration=reg), name, ident['seed'],
            lambda: prior.model.head(width, 6), STEPS, sample, loss, replay, beat)
        rng = torch.load(old.checked(fit['artifact']), map_location='cpu', weights_only=True)['rng']
        if rng0 is not None and rng0 != rng:
            raise ValueError('State-mask ablation changed training samples')
        nets[k], fits[k], updates, rng0 = net, fit, updates + n, rng
    _, cal = base.subset(d, ident['calibration'])
    cv, ols, b, _, p = prepare(cal, shape)
    shadow_packet(p)
    ladder, selected = calibrate(cal, cv, ols, b, p, predictions(nets, p), cut)
    _, held = base.subset(d, ident['held'])
    cv, _, _, a, p = prepare(held, shape)
    shadow_packet(p)
    pred = predictions(nets, p)
    out = dict(portfolio=a, independent=np.full_like(a, -1))
    for q in p:
        out['independent'][q['at']] = q['plans'][q['independent'][q['anchor_count']]]
    for k in MODES:
        out[f'{k}_matched'] = choose(p, pred[k], len(cv), 2., 0., match=True)
        out[f'{k}_free'] = choose(p, pred[k], len(cv), 2., 0.)
    out['loop_matched'] = closed_loop(p, nets['state'], len(cv), True)
    out['loop_free'] = closed_loop(p, nets['state'], len(cv), False)
    out['calibrated'] = np.full_like(a, -1) if selected is None else choose(p, pred[selected['spec'][0]],
        len(cv), *selected['spec'][1:])
    path = prior.study.parent.parent.save_actions(PRIVATE / f'choices_{ident["fold"]}_{ident["seed"]}.npz',
        dict(ids=held['ids'], **out))
    return dict(identity=ident, cut=cut, fits=fits, ladder=ladder, selected=selected, choices=path,
        training_queries=counts, changed_training_targets=n_changed), updates


def evaluate(cells, d, beat, *, export=False):
    scorer = prior.prior.parent.readout
    full = prior.study.motion.full_masses
    pairs = {f'{k}_vs_{tag}': (b, k) for k in ARMS for tag, b in (('cv', base.CV), ('ols4', base.OLS))}
    pairs.update(PRIMARY)
    rows, geom, changed = [], {}, dict.fromkeys(PRIMARY, 0)
    for c in cells:
        i = c['identity']
        _, h = base.subset(d, i['held'])
        cv, ols, b = study.parent.analytic(h['history'])
        with np.load(old.checked(c['choices']), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], h['ids'])
            a = {k: z[k] for k in ARMS}
        for q in prior.query_groups(h['query']):
            if len({int((a[k][q] >= 0).sum()) for k in ARMS[:5]}) != 1:
                raise ValueError('Matched query counts changed')
        for k, (l, r) in PRIMARY.items():
            changed[k] += int((a[l] != a[r]).sum())
        p = {base.CV: cv, base.OLS: ols, **{k: study.parent.model.apply(cv, b, a[k]) for k in ARMS}}
        for rec in np.unique(h['recording']):
            at = h['recording'] == rec
            y, v, q, history = (h[k][at] for k in ('target', 'valid', 'query', 'history'))
            site, = np.unique(h['sites'][at])
            z = {k: x[at] for k, x in p.items()}
            r = scorer.score_recording(z, {}, y, v, q, str(rec), easy_cut=c['cut'][0], hard_cut=c['cut'][1])
            w = old.weight(q)
            terms = {k: full(z[l], z[r], cv[at], y, v, w) for k, (l, r) in pairs.items()}
            rows.append(dict(r, source=str(site), seed=i['seed'], comparisons=terms))
            if export:
                continue
            out = geom.setdefault((str(site), i['seed']), np.zeros((len(PRIMARY), 2)))
            for g in prior.query_groups(q):
                arr = np.stack([z[k][g] for pair in PRIMARY.values() for k in pair])
                _, u, _, (steps, _, _, _) = prior.geometry.query(history[g], arr, y[g], v[g],
                    tuple((2*j, 2*j+1) for j in range(len(PRIMARY))))
                if steps:
                    out[:, 0] += u[:, 1]
                    out[:, 1] += 1
        beat(phase='held_scored', fold=i['fold'], seed=i['seed'])
    if export:
        return rows
    roster = {x['recording']: x['source'] for x in rows}
    summary = prior.prefix.grouped.grouped_readout(rows, roster, base.SEEDS)
    sites = sorted(set(roster.values()))
    a = np.asarray([[geom[s, k] for k in base.SEEDS] for s in sites])
    z = np.divide(a[..., 0], a[..., 1], out=np.full_like(a[..., 0], np.nan), where=a[..., 1] > 0.)
    draw = np.random.default_rng(73107).integers(len(sites), size=(3000, len(sites)))
    return summary, {k: prior.geometry.bootstrap(z[:, :, j], draw) for j, k in enumerate(PRIMARY)}, changed


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

        refs = [study.parent.OUT / 'freeze.json', study.OUT / 'freeze.json', study.OUT / 'readout.json',
                crossing.OUT / 'readout.json']
        shape, counts, baseline, reason = map(old.read, refs)
        if (not shape['before_held_scoring'] or not counts['before_held_scoring'] or baseline['freeze'] != old.sha(refs[1])
                or any(x['a_labels_opened'] or x['independent_confirmation'] or not x['original_2pct']
                       for x in (baseline, reason))):
            raise ValueError('Require original exposed development parents')
        rp = OUT / 'registration.json'
        old.once(rp, old.stable(dict(parents=[old.artifact(p) for p in refs],
            files=[old.artifact(p) for p in (Path(__file__), Path(model.__file__), OUT / 'protocol.md')],
            grid=GRID, steps=STEPS, seeds=base.SEEDS, original_2pct=True, causal_prefix=True)))
        base.register()
        d = base.fit_population(beat)
        cells, updates = [], 0
        for f, c in zip(shape['cells'], counts['cells'], strict=True):
            if f['identity'] != c['identity'] or f['cut'] != c['cut']:
                raise ValueError('Source roles or fitting cuts changed')
            x, n = cell(d, f, old.sha(rp), beat, args.replay)
            cells.append(x)
            updates += n
            beat(phase='cell_frozen', fold=f['identity']['fold'], seed=f['identity']['seed'], updates=updates)
        fp = OUT / 'freeze.json'
        old.once(fp, old.stable(dict(registration=old.sha(rp), cells=cells,
            optimizer_steps=18*STEPS, before_held_scoring=True)))
        s, g, changed = evaluate(old.read(fp)['cells'], d, beat)
        if s['risk']['portfolio'] != baseline['summary']['risk']['portfolio']:
            raise ValueError('Original analytic portfolio changed')
        old.once(OUT / 'readout.json', old.stable(dict(freeze=old.sha(fp), summary=s, geometry=g,
            changed=changed, qualified=sum(x['selected'] is not None for x in cells),
            original_2pct=True, independent_confirmation=False, a_labels_opened=False)))
        beat(phase='complete', updates=updates, replay=args.replay)


if __name__ == '__main__':
    main()
