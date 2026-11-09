"""Fit matched direct pair/value heads with source-excluded forecast labels."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_support as pair
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_vehicle_pair_learning as model

base, old, np, torch = pair.base, pair.old, pair.np, pair.torch
OUT = pair.OUT.parent / 'opendd_vehicle_pair_learning_v1'
LOCAL = pair.LOCAL.parent / OUT.name
ARMS = ('archived_joint', 'additive', 'same_family', 'joint')
STEPS, CAP = 200, 256


def register():
    key, source, _, cost, forecasts = pair.register()
    f = old.read(pair.OUT / 'freeze.json')
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require archived choices and verified models')
    refs = (Path(__file__), Path(model.__file__), OUT / 'protocol.md', pair.OUT / 'freeze.json',
            pair.OUT / 'registration.json', Path(moments.__code__.co_filename))
    p = OUT / 'registration.json'
    old.write(p, dict(files={str(k.resolve().relative_to(base.prior.ROOT)): base.prior.sha(k) for k in refs},
        arms=ARMS, seeds=base.SEEDS, steps=STEPS, max_train_pairs_per_record=CAP, original_2pct=True,
        workers=0, threads=4, external_outcomes_read=False))
    return base.prior.sha(p), source, f, cost, forecasts


def menu(d, net):
    cv, ols, offsets, paths, x, units, _ = old.menu(d, old.forecast(net, d))
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    slots = pair.fixed_pairs(offsets, d['recording'], peak)
    speed = np.diff(d['history'][:, -4:], axis=1).mean(axis=1)
    return dict(cv=cv, ols4=ols, paths=paths, x=x, units=units, peak=peak, slots=slots, speed=speed)


def choices(peak, lo, hi, rng=None):
    row, family = np.nonzero(peak[lo:hi] > 0.)
    row += lo
    i, j = np.triu_indices(len(row), 1)
    keep = row[i] != row[j]
    i, j = i[keep], j[keep]
    if rng is not None and len(i) > CAP:
        take = rng.choice(len(i), CAP, replace=False)
        i, j = i[take], j[take]
    return row, family, i, j


def relative(m, a, b, ka, kb):
    def norm(v):
        return np.linalg.norm(v, axis=1)

    def cosine(u, v):
        return (u*v).sum(1)/np.maximum(norm(u)*norm(v), 1e-12)

    u, v = m['speed'][a], m['speed'][b]
    ca, cb = m['cv'][a, -1], m['cv'][b, -1]
    x = (m['paths'][a, ka]-m['cv'][a]).reshape(len(a), 24)
    y = (m['paths'][b, kb]-m['cv'][b]).reshape(len(a), 24)
    unit = np.maximum(.5*(m['units'][a]+m['units'][b]), 1e-12)
    radius = np.maximum(.5*(m['peak'][a, ka]+m['peak'][b, kb]), 1e-12)
    return np.column_stack((np.log1p(12.*norm(u-v)/unit), cosine(u, v),
        np.log1p(norm(ca-cb)/unit), np.log1p(np.abs(norm(ca)-norm(cb))/unit),
        np.log1p(norm(x-y)/(np.sqrt(12.)*radius)), cosine(x, y))).astype(np.float32)


def input_rows(m, row, family, i, j):
    a, b, ka, kb = row[i], row[j], family[i], family[j]
    return m['x'][a, ka], m['x'][b, kb], relative(m, a, b, ka, kb)


def donor_batch(d, m, cut, rng):
    cv, ols = m['cv'], m['ols4']
    y, valid, w = d['future'], d['valid'], d['weight']
    rec, ix = np.unique(d['recording'], return_inverse=True)
    mass = np.bincount(ix, weights=moments(cv, cv, y, valid, cut[0])['ref_easy'])
    ols_base = base.parent.paired(ols, cv, y, valid)[1].sum(axis=1)
    offset = np.bincount(ix, weights=w*ols_base)
    batches = []
    for name, (lo, hi) in m['slots'].items():
        row, family, i, j = choices(m['peak'], lo, hi, rng)
        p = m['paths'][row, family]
        m0 = moments(cv[row], p, y[row], valid[row], cut[0])
        risk = np.column_stack([m0[k] for k in ('all_excess', 'easy_excess', 'easy_signed')])
        gain = np.column_stack((
            base.parent.paired(cv[row], p, y[row], valid[row])[1].sum(1),
            base.parent.paired(ols[row], p, y[row], valid[row])[1].sum(1)-ols_base[row]))
        k = np.searchsorted(rec, name)
        if rec[k] != name or not np.all(w[lo:hi] == w[lo]):
            raise ValueError('Invalid donor recording or unequal query weights')
        radii = m['peak'][row, family]
        scale = 12.*w[lo]*(radii[i]+radii[j])[:, None]
        label = model.labels(i, j, risk, m0['ref_easy'], gain*w[row, None], .02*mass[k], offset[k], scale)
        batches.append((*input_rows(m, row, family, i, j), label))
    if not batches:
        raise ValueError('No eligible training pairs for this donor')
    return tuple(torch.from_numpy(np.concatenate(a, axis=0)) for a in zip(*batches, strict=True))


def fit(identity, batches, joint):
    original = old.LOCAL
    try:
        old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return old.fit(identity, batches, model.loss, model.PairValue(joint), STEPS, old.BATCH)
    finally:
        old.LOCAL = original


@torch.inference_mode()
def choose(m, models, archived):
    np.testing.assert_array_equal(m['cv'], archived['cv'])
    np.testing.assert_array_equal(m['ols4'], archived['ols4'])
    np.testing.assert_array_equal(list(m['slots']), archived['records'])
    np.testing.assert_array_equal(list(m['slots'].values()), archived['bounds'])
    packet = {k: archived[k] for k in ('cv', 'ols4', 'records', 'bounds')}
    for k in ('paths', 'rows', 'scores'):
        packet[f'archived_joint_{k}'] = archived[f'joint_{k}']
    results = {a: [] for a in ARMS[1:]}
    for lo, hi in m['slots'].values():
        row, family, i, j = choices(m['peak'], lo, hi)
        features = input_rows(m, row, family, i, j)
        x = [torch.from_numpy(a) for a in features]
        scores = {}
        for arm, net in models.items():
            scores[arm] = model.merit(torch.cat([net(*a) for a in zip(*(v.split(4096) for v in x), strict=True)]).numpy())
        scores['same_family'] = np.where(family[i] == family[j], scores['joint'], -np.inf)
        for arm, s in scores.items():
            k = int(s.argmax())
            q = [i[k], j[k]]
            if not np.isfinite(s[k]):
                raise ValueError('Nonfinite chosen score')
            results[arm].append((row[q], m['paths'][row[q], family[q]], s[k]))
    radius = np.linalg.norm(packet['archived_joint_paths']-packet['cv'][packet['archived_joint_rows']], axis=-1).max(axis=-1)
    for arm, entries in results.items():
        for k, values in zip(('rows', 'paths', 'scores'), zip(*entries, strict=True), strict=True):
            packet[f'{arm}_{k}'] = np.asarray(values)
        a = np.linalg.norm(packet[f'{arm}_paths']-packet['cv'][packet[f'{arm}_rows']], axis=-1).max(axis=-1)
        np.testing.assert_allclose(a, radius, atol=1e-9, rtol=1e-9)
    return packet


def freeze(key, source, f, cost, forecasts):
    data = base.parent.load(source)
    cells = {}
    for name, c in f['cells'].items():
        excluded, cut, seed = list(c['packets']), c['cut'], c['seed']
        receipt = LOCAL / f'{name}.json'
        if receipt.exists():
            row = old.read(receipt)
            if (row['registration'] != key
                    or any(base.prior.sha(LOCAL / p['file']) != p['sha256'] for p in row['packets'].values())
                    or any(base.prior.sha(LOCAL / f"{p['identity']['name']}.pt") != p['sha256'] for p in row['weights'].values())):
                raise ValueError('Frozen cell changed')
            cells[name] = row
            continue
        batches = []
        for donor in c['fit_sites']:
            if donor in excluded:
                raise ValueError('Training source overlaps calibration or held source')
            sites = (*excluded, donor)
            tag = base.triple_id(sites, seed)
            net = base.load_net(base.LOCAL / f'{tag}.pt', cost['triple_checkpoints'][tag],
                                old.motion.motion.network(17), sites)
            d = data[donor]
            rng = np.random.default_rng([seed, base.SITES.index(donor), *[base.SITES.index(s) for s in excluded]])
            batches.append(donor_batch(d, menu(d, net), cut, rng))
        models, weights = {}, {}
        for arm in ('additive', 'joint'):
            identity = dict(name=f'{name}_{arm}', registration=key, seed=seed,
                            fit_sites=c['fit_sites'], excluded=excluded, role=arm)
            models[arm], weights[arm] = fit(identity, batches, joint=(arm == 'joint'))
        tag = old.double_name(excluded, seed)
        net = base.load_net(old.LOCAL / f'{tag}.pt', forecasts['double_excluded'][tag],
                            old.motion.motion.network(17), excluded)
        packets = {}
        for site, r in c['packets'].items():
            archived = pair.LOCAL / r['file']
            if base.prior.sha(archived) != r['sha256']:
                raise ValueError('Archived comparison packet changed')
            with np.load(archived, allow_pickle=False) as z:
                pkt = choose(menu(data[site], net), models, z)
            p = LOCAL / f'{name}_{site}.npz'
            packets[site] = dict(file=p.name, sha256=base.parent.indices.save_npz(p, pkt))
        rules = {}
        for held, cal in (excluded, excluded[::-1]):
            arms = {'archived_joint': c['rules'][held]['arms']['joint']}
            with np.load(LOCAL / packets[cal]['file'], allow_pickle=False) as z:
                for arm in ARMS[1:]:
                    v = pair.evaluate(pair.prediction(z, arm), z, data[cal], cut)
                    t, n = base.calibration(z[f'{arm}_scores'], v['dual'])
                    arms[arm] = dict(threshold=t, calibration_recordings=n)
            rules[held] = dict(calibration_site=cal, arms=arms)
        cells[name] = dict(registration=key, seed=seed, fit_sites=c['fit_sites'], cut=cut,
            donor_pairs=[len(b[0]) for b in batches], weights=weights, packets=packets, rules=rules)
        old.write(receipt, cells[name])
        beat(phase='freeze', name=name, count=len(cells))
    old.write(OUT / 'freeze.json', dict(registration=key, cells=cells, held_choices_frozen=True))


def beat(**kw):
    import os
    base.prior.write_json(LOCAL / 'heartbeat.json', dict(pid=os.getpid(), **kw))
    print(kw, flush=True)


def score(key, source):
    fp = OUT / 'freeze.json'
    f = old.read(fp)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require all frozen choices')
    data, views = base.parent.load(source), []
    for name, c in f['cells'].items():
        for site, packet in c['packets'].items():
            p = LOCAL / packet['file']
            if base.prior.sha(p) != packet['sha256'] or site in c['fit_sites']:
                raise ValueError('Changed choices or source role')
            rule = c['rules'][site]
            d, arms = data[site], {}
            with np.load(p, allow_pickle=False) as z:
                cv = z['cv']
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
                pred = {a: pair.prediction(z, a) for a in ARMS}
                for arm in ARMS:
                    s = pair.evaluate(pred[arm], z, d, c['cut'])
                    t = rule['arms'][arm]['threshold']
                    use = z[f'{arm}_scores'] >= t if t is not None else np.zeros(len(s['dual']), bool)
                    arms[arm] = dict(matched=len(use), matched_risk=int(s['risk'].sum()), matched_dual=int(s['dual'].sum()),
                        matched_missing=int(s['missing'].sum()), calibration=rule['arms'][arm]['calibration_recordings'],
                        released=int(use.sum()), released_risk=int((use & s['risk']).sum()),
                        released_dual=int((use & s['dual']).sum()), released_missing=int((use & s['missing']).sum()))
                contrasts = {a: pair.paired(pred['joint'], pred[a], d, z['bounds'], mass) for a in ARMS[:-1]}
            views.append(dict(site=site, seed=c['seed'], calibration_site=rule['calibration_site'],
                              arms=arms, contrasts=contrasts))
        beat(phase='score', name=name, count=len(views))
    summary = {a: {k: sum(v['arms'][a][k] for v in views) for k in views[0]['arms'][a]} for a in ARMS}
    for a in ARMS:
        active = [v['arms'][a] for v in views if v['arms'][a]['released']]
        summary[a].update(active=len(active), all_dual=sum(v['released']==v['released_dual'] for v in active))
    parent = old.read(pair.OUT / 'readout.json')['summary']['joint']
    for k, v in (('matched', 'matched_recordings'), ('matched_dual', 'matched_dual'), ('released', 'released_recordings'),
                 ('released_dual', 'released_dual')):
        if summary['archived_joint'][k] != parent[v]:
            raise ValueError('Archived comparator does not replay')
    draws = np.random.default_rng(2917).integers(len(base.SITES), size=(3000, len(base.SITES)))
    intervals = {}
    for a in ARMS[:-1]:
        intervals[a] = {}
        for k in views[0]['contrasts'][a]:
            x = np.array([np.mean([v['contrasts'][a][k] for v in views if v['site']==site]) for site in base.SITES])
            intervals[a][k] = dict(site_difference=x.tolist(), mean=float(x.mean()),
                                   ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    old.write(OUT / 'readout.json', dict(result_source='fresh_run', registration=key, freeze=base.prior.sha(fp),
        summary=summary, intervals=intervals, views=views, original_2pct_unchanged=True,
        independent_confirmation=False, external_outcomes_read=False, deployment_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    args = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, f, cost, forecasts = register()
        if args.phase == 'freeze':
            freeze(key, source, f, cost, forecasts)
        else:
            score(key, source)
