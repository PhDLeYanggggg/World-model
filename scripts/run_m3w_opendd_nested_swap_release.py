"""Nested OpenDD support-aware release: fit, calibrate, score."""

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use native arm64 .venv-pytorch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_swap_support as support
from scripts import run_m3w_vehicle_pair_learning as original

run, model = support.old, support.model
old, np, torch, base = original.old, original.np, original.torch, original.base
read, write, sha = support.read, support.write, support.sha
OUT = support.OUT.parent/'opendd_nested_swap_release_v1'
LOCAL = support.LOCAL.parent/OUT.name
ARMS = ('additive', 'sign', 'magnitude')
STEPS, BATCH = 300, 160


def beat(phase, **info):
    write(LOCAL/'heartbeat.json', dict(pid=os.getpid(), phase=phase, **info))
    print(phase, info, flush=True)


def register():
    source, cells, cost = original_inputs()
    f = original.OUT/'freeze.json'
    r = support.OUT/'readout.json'
    allocation = support.OUT.parent/'opendd_swap_support_allocation_v1/readout.json'
    if not read(r)['summary']['allocation_screen'] or read(allocation)['deployment_changed']:
        raise ValueError('Require prior verified ranking and unchanged deployment')
    refs = (Path(__file__), OUT/'protocol.md', f, r, allocation, base.OUT/'freeze.json',
            Path(model.__file__), Path(original.__file__))
    p = OUT/'registration.json'
    write(p, dict(files={str(v.relative_to(base.prior.ROOT)): sha(v) for v in refs},
        sites=base.SITES, seeds=base.SEEDS, workers=0, threads=4, steps=STEPS, batch=BATCH,
        original_2pct=True, no_external_outcomes=True), immutable=True)
    return sha(p), source, cells, cost


def original_inputs():
    key, source, _, _, _ = original.register()
    f = read(original.OUT/'freeze.json')
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require archived double-excluded incumbents')
    cost = read(base.OUT/'freeze.json')
    if len(cost['triple_checkpoints']) != 105:
        raise ValueError('Require the full triple-excluded checkpoint bank')
    return source, f['cells'], cost


def donor_packet(d, net):
    m = original.menu(d, net)
    cv, p = m['cv'], m['paths'][:, 2]
    peak = np.linalg.norm(p-cv, axis=-1).max(axis=1)
    rows, bounds, names = [], [], []
    for name, (lo, hi) in m['slots'].items():
        idx = np.argsort(-peak[lo:hi], kind='stable')[:2]+lo
        if (peak[idx] > 1e-12).all():
            rows.append(idx)
            bounds.append((lo, hi))
            names.append(name)
    x = np.stack(rows)
    return dict(cv=cv, ols4=m['ols4'], bounds=np.asarray(bounds), records=np.asarray(names),
                additive_rows=x, additive_paths=p[x])


def donor_batch(key, seed, pair, site, d, feature, triples):
    excluded = tuple(sorted((*pair, site)))
    tag = base.triple_id(excluded, seed)
    ck = triples[tag]
    identity = dict(key=key, donor=site, excludes=list(excluded), ckpt=ck['sha256'], cohort=sha(feature['file']))
    dest = LOCAL/f'{tag}_{site}_donor.npz'
    receipt = dest.with_suffix('.json')
    if not receipt.exists():
        net = base.load_net(base.LOCAL/f'{tag}.pt', ck, old.motion.motion.network(17), excluded)
        p = donor_packet(d, net)
        cv, anchor, allowed = run.old.geometry(p, feature)
        y, v = d['future'], d['valid']
        pairs = []
        for q, a, b, new, x in run.menus(p, feature, cv, anchor, allowed):
            if len(x):
                inp, _, g = support.candidate(feature, cv, anchor, q, a, b, new, x)
                pairs.append((inp, model.targets(y[q], v[q], a, b, g)))
        x, y = (np.concatenate(z) for z in zip(*pairs, strict=True))
        digest = base.parent.indices.save_npz(dest, dict(x=x, y=y))
        write(receipt, dict(identity=identity, sha256=digest, candidates=len(x)), immutable=True)
    r = read(receipt)
    if r['identity'] != identity or sha(dest) != r['sha256']:
        raise ValueError('Changed nested donor batch')
    with np.load(dest, allow_pickle=False) as z:
        return tuple(torch.from_numpy(z[k]) for k in ('x', 'y')), r


def fit(identity, data):
    torch.manual_seed(identity['seed'])
    network = model.network(data[0][0].shape[1])
    # Reuse the atomic optimizer and sampler checkpoint implementation.
    directory = old.LOCAL
    try:
        old.LOCAL = LOCAL
        return old.fit(identity, data, lambda m, z: model.loss(m(z[0]), z[1]), network, STEPS, BATCH)
    finally:
        old.LOCAL = directory


@torch.inference_mode()
def propose(p, f, net):
    cv, anchor, allowed = run.old.geometry(p, f)
    rows = len(p['bounds'])
    out = {k: np.full((rows, 2), [-1., 0.]) for k in ARMS[1:]}
    for i, (q, a, b, new, x) in enumerate(run.menus(p, f, cv, anchor, allowed)):
        if not len(x):
            continue
        features, scale, _ = support.candidate(f, cv, anchor, q, a, b, new, x)
        prob = net(torch.from_numpy(features)).sigmoid().numpy()
        values = model.lower(prob[..., :3])*scale
        for arm, val in (('sign', prob[..., 3]), ('magnitude', values)):
            merit = val[:, 0]+.5*val[:, 1]
            j = int(merit.argmax())
            if arm == 'sign' or merit[j] > 0.:
                out[arm][i] = [j, merit[j]]
    return out


def freeze(key, source, cells, costs):
    data, features = base.parent.load(source), run.old.prior.load_features(source)
    output = {}
    for name, c in cells.items():
        run.old.check_cell(c)
        excluded = sorted(c['packets'])
        seed = c['seed']
        f = LOCAL/f'{name}_predictions.json'
        if not f.exists():
            ds, proofs = [], {}
            for s in c['fit_sites']:
                d, proofs[s] = donor_batch(key, seed, excluded, s, data[s], features[s], costs['triple_checkpoints'])
                ds.append(d)
            net, ck = fit(dict(name=name, key=key, seed=seed, excluded=excluded, fit_sites=c['fit_sites'],
                               donors=proofs), ds)
            packets = {}
            for s, r in c['packets'].items():
                with support.packet(r) as p:
                    z = propose(p, features[s], net)
                path = LOCAL/f'{name}_{s}.npz'
                packets[s] = dict(file=path.name, sha256=base.parent.indices.save_npz(path, z), original=r['sha256'])
            write(f, dict(key=key, fit=proofs, checkpoint=ck, packets=packets), immutable=True)
        v = read(f)
        if (v['key'] != key or set(v['fit']) != set(c['fit_sites']) or set(v['packets']) != set(excluded)
                or sha(LOCAL/f'{name}.pt') != v['checkpoint']['sha256']):
            raise ValueError('Changed nested fit lineage')
        output[name] = v
        beat('freeze', completed=len(output), total=63)
    write(OUT/'freeze.json', dict(key=key, cells=output, no_external_outcomes=True), immutable=True)


def candidates(p, feature, actions):
    cv, anchor, allowed = run.old.geometry(p, feature)
    forecasts = dict(additive=anchor)
    forecasts.update({k: anchor.copy() for k in ARMS[1:]})
    for i, (q, a, b, new, _) in enumerate(run.menus(p, feature, cv, anchor, allowed)):
        for k in ARMS[1:]:
            j = int(actions[k][i, 0])
            if j >= 0:
                forecasts[k][q.start+a[j]], forecasts[k][q.start+b[j]] = cv[q.start+a[j]], new[j]
    return cv, forecasts


def per_record_gain(ref, pred, p, d):
    names, rec = np.unique(d['recording'], return_inverse=True)
    idx = np.searchsorted(names, p['records'])
    np.testing.assert_array_equal(names[idx], p['records'])
    _, gain = base.parent.paired(ref, pred, d['future'], d['valid'])
    return np.bincount(rec, weights=d['weight']*gain.sum(1))[idx]


def inspect(p, f, d, cut, z):
    cv, forecasts = candidates(p, f, z)
    results = {}
    for a, pred in forecasts.items():
        score = p['additive_scores'] if a == 'additive' else z[a][:, 1]
        allowed = np.ones(len(score), dtype=bool) if a == 'additive' else z[a][:, 0] >= 0.
        r = original.pair.evaluate(pred, p, d, cut)
        gain = per_record_gain(forecasts['additive'], pred, p, d) if a != 'additive' else np.zeros(len(score))
        strict = r['dual'] if a == 'additive' else r['dual'] & (gain > 0.)
        results[a] = dict(score=score, enabled=allowed, strict=strict, **r)
    return cv, forecasts, results


def phase_input(key):
    p = OUT/'freeze.json'
    run.old.require_published(p)
    v = read(p)
    if v['key'] != key or not v['no_external_outcomes']:
        raise ValueError('Require the published pre-outcome freeze')
    return v


def opened(site, name, c, frozen):
    r, p = frozen['cells'][name]['packets'][site], c['packets'][site]
    if site in frozen['cells'][name]['fit']:
        raise ValueError('Calibration or held site was used to fit the selector')
    return support.packet(p), np.load(support.checked_local(r, LOCAL, p['sha256']), allow_pickle=False)


def calibrate(key, source, cells):
    frozen = phase_input(key)
    data, features = base.parent.load(source), run.old.prior.load_features(source)
    out = {}
    for name, c in cells.items():
        for held, rule in c['rules'].items():
            cal = rule['calibration_site']
            if cal == held or set((cal, held)) != set(c['packets']):
                raise ValueError('Invalid calibration assignment')
            p, z = opened(cal, name, c, frozen)
            with p as packet, z as actions:
                _, _, scores = inspect(packet, features[cal], data[cal], c['cut'], actions)
            release = {}
            for a, x in scores.items():
                use = x['enabled']
                threshold, count = base.calibration(x['score'][use], x['strict'][use])
                release[a] = dict(threshold=threshold, calibration_recordings=count,
                                  eligible=int(use.sum()), hindsight_strict=int((use & x['strict']).sum()))
            if any(release['additive'][k] != v for k, v in rule['arms']['additive'].items()):
                raise ValueError('The original additive threshold did not reproduce')
            out[f'{name}/{held}'] = dict(calibration=cal, held=held, seed=c['seed'], arms=release)
        beat('calibrate', completed=len(out), total=126)
    write(OUT/'calibration_freeze.json', dict(key=key, freeze=sha(OUT/'freeze.json'),
        rules=out, original_2pct=True, external_outcomes_read=False), immutable=True)


def held_metrics(p, d, cv, forecasts, scores, thresholds):
    arms = {}
    for a, x in scores.items():
        t = thresholds[a]['threshold']
        use = x['enabled'] & (x['score'] >= t) if t is not None else np.zeros(len(x['score']), dtype=bool)
        released = np.isin(d['recording'], p['records'][use])
        pred = np.where(released[:, None, None], forecasts[a], cv)
        y, valid, w = d['future'], d['valid'], d['weight']
        roster = np.isin(d['recording'], p['records'])
        mass = float(w[roster] @ np.where(valid[roster], np.linalg.norm(cv[roster]-y[roster], axis=-1), 0.).sum(1))
        observed, lower = base.parent.paired(cv[roster], pred[roster], y[roster], valid[roster])
        cv_pp = float(100.*(w[roster] @ lower.sum(1))/mass)
        obs_pp = float(100.*(w[roster] @ observed.sum(1))/mass)
        _, ols = base.parent.paired(p['ols4'][roster], pred[roster], y[roster], valid[roster])
        rel = sum((run.relative_contrast(pred[lo:hi], cv[lo:hi], y[lo:hi], valid[lo:hi]) for (lo, hi), take
                   in zip(p['bounds'], use, strict=True) if take), start=np.zeros(3))/len(use)
        arms[a] = dict(released=int(use.sum()), risk=int((use & x['risk']).sum()), dual=int((use & x['dual']).sum()),
            strict=int((use & x['strict']).sum()), missing=int((use & x['missing']).sum()),
            observed_cv_pp=obs_pp, lower_cv_pp=cv_pp, lower_ols_pp=float(100.*(w[roster] @ ols.sum(1))/mass),
            relative_observed=float(rel[0]), relative_lower=float(rel[1]), eligible=int(x['enabled'].sum()))
    return arms


def summarize(rows, thresholds):
    count_keys = ('released', 'risk', 'dual', 'strict', 'missing', 'eligible')
    metrics = ('observed_cv_pp', 'lower_cv_pp', 'lower_ols_pp', 'relative_observed', 'relative_lower')
    draw = np.random.default_rng(211).integers(7, size=(3000, 7))
    groups = {s: [r for r in rows if r['held'] == s] for s in base.SITES}
    out = {}
    for a in ARMS:
        c = {k: sum(v['arms'][a][k] for v in rows) for k in count_keys}
        c['calibration_active_directions'] = sum(v['arms'][a]['threshold'] is not None for v in thresholds.values())
        c['held_active_directions'] = sum(v['arms'][a]['released'] > 0 for v in rows)
        c['active_directions_all_strict'] = sum(0 < v['arms'][a]['released'] == v['arms'][a]['strict'] for v in rows)
        c['effects'] = {}
        for k in metrics:
            v = np.array([np.mean([np.mean([r['arms'][a][k] for r in g if r['seed'] == seed])
                                   for seed in base.SEEDS]) for g in groups.values()])
            c['effects'][k] = dict(point=float(v.mean()), ci95=np.quantile(v[draw].mean(1), [.025, .975]).tolist(),
                                  by_site=dict(zip(groups, v.tolist(), strict=True)))
        out[a] = c
    return out


def score(key, source, cells):
    frozen = phase_input(key)
    p = OUT/'calibration_freeze.json'
    run.old.require_published(p)
    rules = read(p)
    if rules['key'] != key or rules['freeze'] != sha(OUT/'freeze.json') or not rules['original_2pct']:
        raise ValueError('Require the published thresholds')
    data, features = base.parent.load(source), run.old.prior.load_features(source)
    rows = []
    for name, c in cells.items():
        for held in c['packets']:
            r = rules['rules'][f'{name}/{held}']
            ident = dict(key=key, calibration=sha(p), held=held, seed=c['seed'], donor_cal=r['calibration'])
            if r['held'] != held or r['calibration'] == held or held in c['fit_sites']:
                raise ValueError('Held location used to fit or calibrate')
            dest = LOCAL/f'{name}_{held}_score.json'
            if not dest.exists():
                pck, a = opened(held, name, c, frozen)
                with pck as packet, a as actions:
                    d = data[held]
                    cv, pred, v = inspect(packet, features[held], d, c['cut'], actions)
                    arm = held_metrics(packet, d, cv, pred, v, r['arms'])
                write(dest, dict(identity=ident, arms=arm), immutable=True)
            row = read(dest)
            if row['identity'] != ident:
                raise ValueError('Changed held-score receipt')
            rows.append(row)
        beat('score', completed=len(rows), total=126)
    s = summarize([dict(**r['identity'], arms=r['arms']) for r in rows], rules['rules'])
    if len(rows) != 126 or (s['additive']['released'], s['additive']['risk'], s['additive']['dual']) != (65, 32, 19):
        raise ValueError('Archived release count, risk, or dual results did not reproduce')
    write(OUT/'readout.json', dict(key=key, calibration=sha(p), summary=s, by_direction=rows,
        original_2pct=True, independent_confirmation=False, external_outcomes_read=False,
        deployment_changed=False, submission_ready=False), immutable=True)
    print({a: {k: v for k, v in row.items() if k != 'effects'} for a, row in s.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze', 'calibrate', 'score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        k, source, cells, cost = register()
        if args.phase == 'freeze':
            freeze(k, source, cells, cost)
        elif args.phase == 'calibrate':
            calibrate(k, source, cells)
        else:
            score(k, source, cells)
