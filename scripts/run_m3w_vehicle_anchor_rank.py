"""Train and audit a fixed-menu, additive-anchored OpenDD comparison."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_m3w_vehicle_relative_ceiling as audit
from src.world_model import m3w_vehicle_anchor_rank as model

study, base, pair, old, np, torch = audit.study, audit.base, audit.pair, audit.old, audit.np, audit.torch
OUT = audit.OUT.parent / 'opendd_vehicle_anchor_rank_v1'
LOCAL = audit.LOCAL.parent / OUT.name
FIT = ('dual_only', 'relative')
ARMS = ('additive', 'joint', *FIT)


def register():
    source, forecasts, frozen, _, binding = audit.settings()
    _, _, _, cost, _ = study.register()
    refs = (Path(__file__), Path(model.__file__), OUT/'protocol.md', audit.OUT/'readout.json')
    p = OUT/'registration.json'
    old.write(p, dict(files=binding | {str(x.relative_to(base.prior.ROOT)): audit.sha(x) for x in refs},
        arms=ARMS, steps=study.STEPS, seeds=base.SEEDS, cap=study.CAP, original_2pct=True,
        workers=0, threads=4, external_outcomes_read=False))
    return audit.sha(p), source, frozen, cost, forecasts


def beat(**msg):
    audit.write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), **msg))
    print(msg, flush=True)


def features(m, rows, family, i, j, anchor):
    a, b, r = study.input_rows(m, rows, family, i, j)
    ref = study.input_rows(m, rows, family, anchor[:1], anchor[1:])
    keep = (np.isin(rows[i], rows[anchor]).astype(np.float32)+np.isin(rows[j], rows[anchor]))[:, None]/2
    return (a, b, r, *(np.repeat(v, len(i), axis=0) for v in ref), keep)


def anchor_actions(packet, k, rows, paths):
    idx = []
    for actor, path in zip(packet['additive_rows'][k], packet['additive_paths'][k], strict=True):
        match = np.flatnonzero((rows == actor) & (paths == path).all(axis=(1, 2)))
        if not len(match):
            raise ValueError('Frozen additive action is not in the causal bank')
        idx.append(int(match[0]))
    return np.array(sorted(idx))


def trajectory_lower(cv, p, rows, anchor, i, j, target, valid):
    ref = cv.copy()
    ref[rows[anchor]] = p[anchor]
    r, y, v = ref[rows], target[rows], valid[rows]
    baseline = base.parent.paired(r, cv[rows], y, v)[1].sum(1)
    delta = base.parent.paired(r, p, y, v)[1].sum(1)-baseline
    return ref, baseline[anchor].sum()+delta[i]+delta[j]


def donor_batch(d, m, cut, rng):
    cv, ols, y, valid, w = m['cv'], m['ols4'], d['future'], d['valid'], d['weight']
    names, rec = np.unique(d['recording'], return_inverse=True)
    mass = np.bincount(rec, weights=audit.moments(cv, cv, y, valid, cut[0])['ref_easy'])
    ols_base = base.parent.paired(ols, cv, y, valid)[1].sum(1)
    offset = np.bincount(rec, weights=w*ols_base)
    batches = []
    for name, (lo, hi) in m['slots'].items():
        q = slice(lo, hi)
        rows, family, i, j = study.choices(m['peak'], lo, hi)
        # The anchor is drawn before constructing or inspecting any labels.
        a = rng.integers(len(i))
        anchor = np.array([i[a], j[a]])
        eligible = np.flatnonzero(model.alternatives(rows, i, j, anchor))
        if not len(eligible):
            continue
        if len(eligible) > study.CAP:
            eligible = rng.choice(eligible, study.CAP, replace=False)
        i, j = i[eligible], j[eligible]
        p = m['paths'][rows, family]
        g = lambda reference, pred: base.parent.paired(reference, pred, y[rows], valid[rows])[1].sum(1)
        k = np.searchsorted(names, name)
        if names[k] != name or not np.all(rec[q] == k) or not np.all(w[q] == w[lo]):
            raise ValueError('Misaligned donor recording or unequal weights')
        gains = np.column_stack((g(cv[rows], p), g(ols[rows], p)-ols_base[rows]))*w[rows, None]
        v = audit.moments(cv[rows], p, y[rows], valid[rows], cut[0])
        risk = np.column_stack([v[key] for key in ('all_excess', 'easy_excess', 'easy_signed')])
        radius = m['peak'][rows, family]
        scale = radius[i]+radius[j]
        labels = study.model.labels(i, j, risk, v['ref_easy'], gains, .02*mass[k], offset[k], 12.*w[lo]*scale[:, None])
        ref, t = trajectory_lower(cv[q], p, rows-lo, anchor, i, j, y[q], valid[q])
        rel = audit.relative_lower(cv[q], ref, y[q], valid[q], rows-lo, p, i, j)
        target = np.column_stack((labels, t/(12.*scale),
                                  rel*len(ref)/scale)).astype(np.float32)
        batches.append((*features(m, rows, family, i, j, anchor), target))
    return tuple(torch.from_numpy(np.concatenate(x)) for x in zip(*batches, strict=True))


@torch.inference_mode()
def choose(m, nets, packet):
    for k in ('cv', 'ols4'):
        np.testing.assert_array_equal(m[k], packet[k])
    np.testing.assert_array_equal(list(m['slots']), packet['records'])
    np.testing.assert_array_equal(list(m['slots'].values()), packet['bounds'])
    output = {k: packet[k] for k in ('cv', 'ols4', 'records', 'bounds',
              *(f'{a}_{k}' for a in ARMS[:2] for k in ('rows', 'paths')))}
    result = {a: [] for a in FIT}
    for k, (lo, hi) in enumerate(packet['bounds']):
        rows, family, i, j = study.choices(m['peak'], lo, hi)
        p = m['paths'][rows, family]
        anchor = anchor_actions(packet, k, rows, p)
        same = np.all(p[i] == p[anchor[0]], axis=(1, 2)) & np.all(p[j] == p[anchor[1]], axis=(1, 2))
        same &= (rows[i] == rows[anchor[0]]) & (rows[j] == rows[anchor[1]])
        ok = model.alternatives(rows, i, j, anchor) & ~same
        i, j = i[ok], j[ok]
        if not len(i):
            for a in FIT:
                result[a].append((rows[anchor], p[anchor], -np.inf))
            continue
        inputs = tuple(torch.from_numpy(v).split(4096) for v in features(m, rows, family, i, j, anchor))
        for a in FIT:
            v = torch.cat([nets[a](*x) for x in zip(*inputs, strict=True)]).numpy()
            s = model.merit(v, a == 'relative')
            if not np.isfinite(s).all():
                raise ValueError('Nonfinite prediction')
            pick = s.argmax()
            ij = [i[pick], j[pick]]
            result[a].append((rows[ij], p[ij], s[pick]))
    for a, entries in result.items():
        for k, v in zip(('rows', 'paths', 'scores'), zip(*entries, strict=True), strict=True):
            output[f'{a}_{k}'] = np.asarray(v)
        r = lambda arm: np.linalg.norm(output[f'{arm}_paths']-m['cv'][output[f'{arm}_rows']], axis=-1).max(-1)
        np.testing.assert_allclose(r(a), r('additive'), atol=1e-9, rtol=1e-9)
    return output


def compare(p, ref, z, d, cut):
    _, lower = base.parent.paired(ref, p, d['future'], d['valid'])
    t = np.array([d['weight'][lo:hi] @ lower[lo:hi].sum(1) for lo, hi in z['bounds']])
    r = np.array([audit.pair_reduction(p[None, lo:hi], ref[lo:hi], d['future'][lo:hi],
                                     d['valid'][lo:hi])[0, 1] for lo, hi in z['bounds']])
    status = pair.evaluate(p, z, d, cut)
    return status, status['dual'] & (t >= -1e-10) & (r > audit.INTERACTION_TOL), t, r


def fit(key, name, seed, sites, donors, batches):
    result, weights = {}, {}
    original = old.LOCAL
    try:
        old.LOCAL = LOCAL
        for a in FIT:
            torch.manual_seed(seed)
            ident = dict(name=f'{name}_{a}', registration=key, seed=seed, excluded=sites, fit_sites=donors)
            loss = lambda net, batch: model.loss(net, batch, a == 'relative')
            result[a], weights[a] = old.fit(ident, batches, loss, model.AnchorValue(), study.STEPS, old.BATCH)
    finally:
        old.LOCAL = original
    return result, weights


def freeze(key, source, parent, cost, forecasts):
    data, cells = base.parent.load(source), {}
    for name, c in parent['cells'].items():
        sites, cut, seed, donors = list(c['packets']), c['cut'], c['seed'], c['fit_sites']
        if set(donors) != set(base.SITES)-set(sites):
            raise ValueError('Training location overlaps calibration or held location')
        receipt = LOCAL/f'{name}.json'
        if receipt.exists():
            x = old.read(receipt)
            if (x['registration'] != key
                    or any(audit.sha(LOCAL/r['file']) != r['sha256'] for r in x['packets'].values())
                    or any(audit.sha(LOCAL/f"{r['identity']['name']}.pt") != r['sha256'] for r in x['weights'].values())):
                raise ValueError('Resume receipt changed')
            cells[name] = x
            continue
        batches = []
        for donor in donors:
            excluded = (*sites, donor)
            tag = base.triple_id(excluded, seed)
            net = base.load_net(base.LOCAL/f'{tag}.pt', cost['triple_checkpoints'][tag],
                                old.motion.motion.network(17), excluded)
            d = data[donor]
            rng = np.random.default_rng([2917, seed, base.SITES.index(donor), *map(base.SITES.index, sites)])
            batches.append(donor_batch(d, study.menu(d, net), cut, rng))
        nets, weights = fit(key, name, seed, sites, donors, batches)
        tag = old.double_name(sites, seed)
        net = base.load_net(old.LOCAL/f'{tag}.pt', forecasts['double_excluded'][tag],
                            old.motion.motion.network(17), sites)
        packets, rules = {}, {}
        for site, r in c['packets'].items():
            with audit.prior.packet(study.LOCAL, r) as z:
                p = LOCAL/f'{name}_{site}.npz'
                pkt = choose(study.menu(data[site], net), nets, z)
                packets[site] = dict(file=p.name, sha256=base.parent.indices.save_npz(p, pkt))
        for held, cal in (sites, sites[::-1]):
            with audit.prior.packet(LOCAL, packets[cal]) as z:
                ref = pair.prediction(z, 'additive')
                arms = {}
                for a in FIT:
                    _, passed, _, _ = compare(pair.prediction(z, a), ref, z, data[cal], cut)
                    s = z[f'{a}_scores']
                    eligible = np.isfinite(s)
                    t, n = base.calibration(s[eligible], passed[eligible])
                    arms[a] = dict(threshold=t, calibration_recordings=n)
                rules[held] = dict(calibration_site=cal, arms=arms)
        cells[name] = dict(registration=key, packets=packets, rules=rules, cut=cut, seed=seed,
                           weights=weights, donor_pairs=[len(b[0]) for b in batches])
        old.write(receipt, cells[name])
        beat(phase='freeze', cell=name, complete=len(cells))
    old.write(OUT/'freeze.json', dict(registration=key, cells=cells, held_choices_frozen=True))


def score(key, source, parent):
    fp = OUT/'freeze.json'
    f = old.read(fp)
    if f['registration'] != key or set(f['cells']) != set(parent['cells']) or not f['held_choices_frozen']:
        raise ValueError('Require all predictions frozen')
    data, views, distinct = base.parent.load(source), [], set()
    for name, c in f['cells'].items():
        for site, packet in c['packets'].items():
            rule = c['rules'][site]
            if site == rule['calibration_site'] or site in parent['cells'][name]['fit_sites']:
                raise ValueError('Scoring is not location-excluded')
            d, arms = data[site], {}
            with audit.prior.packet(LOCAL, packet) as z:
                cv = z['cv']
                distinct.update(z['records'].tolist())
                mass = d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1)
                ref = pair.prediction(z, 'additive')
                for a in ARMS:
                    status, win, t, r = compare(pair.prediction(z, a), ref, z, d, c['cut'])
                    info = dict(count=len(win), dual=int(status['dual'].sum()), full_win=int(win.sum()),
                                ade_lower_pp=float(100*t.sum()/mass), relative_lower_native=float(r.mean()))
                    if a in FIT:
                        cut = rule['arms'][a]['threshold']
                        s = z[f'{a}_scores']
                        use = s >= cut if cut is not None else np.zeros_like(win)
                        info.update(eligible=int(np.isfinite(s).sum()), admitted=int(use.sum()),
                            admitted_dual=int((use & status['dual']).sum()), admitted_full_win=int((use & win).sum()),
                            admitted_missing=int((use & status['missing']).sum()),
                            gated_ade_lower_pp=float(100*t[use].sum()/mass),
                            gated_relative_lower_native=float(np.where(use, r, 0.).mean()))
                    arms[a] = info
            views.append(dict(site=site, seed=c['seed'], calibration_site=rule['calibration_site'], arms=arms))
        beat(phase='score', cell=name, complete=len(views))
    if len(views) != 126 or len(distinct) != 493:
        raise ValueError('Evaluation roster changed')
    totals, intervals, by_seed = {}, {}, {}
    draws = np.random.default_rng(2917).integers(len(base.SITES), size=(3000, len(base.SITES)))
    for a in ARMS:
        totals[a] = {k: sum(v['arms'][a][k] for v in views) for k in views[0]['arms'][a]
                     if isinstance(views[0]['arms'][a][k], int)}
        by_seed[a] = {s: {k: sum(v['arms'][a][k] for v in views if v['seed'] == s) for k in totals[a]}
                      for s in base.SEEDS}
        intervals[a] = {}
        for field in ('ade_lower_pp', 'relative_lower_native', 'gated_ade_lower_pp', 'gated_relative_lower_native'):
            if field not in views[0]['arms'][a]:
                continue
            x = np.array([np.mean([v['arms'][a][field] for v in views if v['site'] == s]) for s in base.SITES])
            intervals[a][field] = dict(site_means=x.tolist(), mean=float(x.mean()),
                                       ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    if (totals['joint']['full_win'], totals['additive']['dual'], totals['joint']['dual']) != (242, 2975, 2952):
        raise ValueError('Original controls did not replay')
    old.write(OUT/'readout.json', dict(result_source='fresh_run', freeze=audit.sha(fp), registration=key,
        totals=totals, intervals=intervals, by_seed=by_seed, sites=base.SITES, directions=len(views),
        distinct_recordings=len(distinct), original_2pct_unchanged=True, external_outcomes_read=False,
        independent_confirmation=False, deployment_changed=False))
    print(totals, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('freeze', 'score'))
    args = cli.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, parent, cost, forecasts = register()
        if args.phase == 'freeze':
            freeze(key, source, parent, cost, forecasts)
        else:
            score(key, source, parent)
