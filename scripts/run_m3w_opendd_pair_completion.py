"""Train and evaluate factored missing-step utility at frozen vehicle slots."""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_support_rerank as rerank
from src.world_model import m3w_vehicle_pair_completion as model

learn, pair, base, old = rerank.learn, rerank.pair, rerank.base, rerank.old
np, torch, sha, write = rerank.np, rerank.torch, rerank.sha, rerank.write
OUT = learn.OUT.parent / 'opendd_pair_completion_v1'
LOCAL = learn.LOCAL.parent / OUT.name
ARMS = model.ARMS
CONTROLS = ('additive', 'joint')


def register():
    key, source, _, cost, forecasts = learn.register()
    p = learn.OUT / 'freeze.json'
    f = old.read(p)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require the frozen OpenDD vehicle pair study')
    reg = OUT / 'registration.json'
    refs = (Path(__file__), Path(model.__file__), Path(learn.__file__), Path(learn.model.__file__),
            Path(old.__file__), OUT / 'protocol.md', p, learn.OUT / 'registration.json')
    write(reg, dict(files={str(r.relative_to(base.prior.ROOT)): sha(r) for r in refs},
        arms=ARMS, controls=CONTROLS, steps=learn.STEPS, seeds=base.SEEDS, batch=old.BATCH, threads=4, workers=0,
        original_2pct_unchanged=True, external_outcomes_read=False), immutable=True)
    return sha(reg), source, f['cells'], cost, forecasts


def batches(d, m, cut, random_seed):
    old_batch = learn.donor_batch(d, m, cut, np.random.default_rng(random_seed))
    rng = np.random.default_rng(random_seed)
    w, y, mask = d['weight'], d['future'], d['valid']
    base_cv = m['cv']
    targets = {a: [] for a in ARMS}
    start = 0
    for lo, hi in m['slots'].values():
        row, family, i, j = learn.choices(m['peak'], lo, hi, rng)
        p = m['paths'][row, family]
        observed, _ = base.parent.paired(base_cv[row], p, y[row], mask[row])
        g = w[row]*observed.sum(1)
        radius = m['peak'][row, family]
        scale = 12.*w[lo]*(radius[i]+radius[j])[:, None]
        stop = start+len(i)
        np.testing.assert_array_equal(old_batch[0][start:stop].numpy(), m['x'][row[i], family[i]])
        np.testing.assert_array_equal(old_batch[1][start:stop].numpy(), m['x'][row[j], family[j]])
        for a in ARMS:
            targets[a].append(model.targets(old_batch[-1][start:stop].numpy(), g[i]+g[j], scale, a))
        start = stop
    if start != len(old_batch[-1]):
        raise ValueError('Derived labels are not aligned with the archived batch')
    return {a: (*old_batch[:3], torch.from_numpy(np.concatenate(t))) for a, t in targets.items()}


def fit(meta, rows):
    original = old.LOCAL
    try:
        old.LOCAL = LOCAL
        torch.manual_seed(meta['seed'])
        return old.fit(meta, rows, model.loss, model.PairValue(), learn.STEPS, old.BATCH)
    finally:
        old.LOCAL = original


@torch.inference_mode()
def choose(menu, archived, networks):
    for key in ('cv', 'ols4'):
        np.testing.assert_array_equal(menu[key], archived[key])
    np.testing.assert_array_equal(list(menu['slots']), archived['records'])
    np.testing.assert_array_equal(list(menu['slots'].values()), archived['bounds'])
    by_arm = {a: [] for a in ARMS}
    for lo, hi in menu['slots'].values():
        row, family, i, j = learn.choices(menu['peak'], lo, hi)
        features = [torch.from_numpy(x).split(4096) for x in learn.input_rows(menu, row, family, i, j)]
        for arm, net in networks.items():
            z = torch.cat([net(*b) for b in zip(*features, strict=True)]).numpy()
            rank = model.merit(z, arm)
            if not np.isfinite(rank).all():
                raise ValueError('Pair merit must be finite')
            k = rank.argmax()
            at = [i[k], j[k]]
            by_arm[arm].append((row[at], menu['paths'][row[at], family[at]], rank[k]))
    result = {}
    cv, rows = archived['cv'], archived['additive_rows']
    radius = np.linalg.norm(archived['additive_paths']-cv[rows], axis=-1).max(axis=-1)
    for arm, items in by_arm.items():
        rows, paths, scores = map(np.asarray, zip(*items, strict=True))
        np.testing.assert_allclose(np.linalg.norm(paths-cv[rows], axis=-1).max(axis=-1), radius,
                                   atol=1e-9, rtol=1e-9)
        result.update({f'{arm}_rows': rows, f'{arm}_paths': paths, f'{arm}_scores': scores})
    return result


def beat(**values):
    write(LOCAL / 'heartbeat.json', dict(pid=rerank.os.getpid(), **values))
    print(values, flush=True)


def freeze(key, source, cells, cost, forecasts):
    data, done = base.parent.load(source), {}
    for name, c in cells.items():
        cache = LOCAL / f'{name}.json'
        if cache.exists():
            r = old.read(cache)
            if (r['registration'] != key
                    or any(sha(LOCAL / p['file']) != p['sha256'] for p in r['packets'].values())
                    or any(sha(LOCAL / (x['identity']['name']+'.pt')) != x['sha256'] for x in r['weights'].values())):
                raise ValueError('Frozen cell changed')
            done[name] = r
            continue
        excluded, seed, cut = sorted(c['packets']), c['seed'], c['cut']
        donor = c['fit_sites']
        if set(donor) != set(base.SITES)-set(excluded):
            raise ValueError('Held or calibration location entered the fit')
        minibatches = {a: [] for a in ARMS}
        for s in donor:
            ex = (*excluded, s)
            tag = base.triple_id(ex, seed)
            net = base.load_net(base.LOCAL / f'{tag}.pt', cost['triple_checkpoints'][tag],
                                old.motion.motion.network(17), ex)
            d = data[s]
            group = batches(d, learn.menu(d, net), cut, [seed, base.SITES.index(s),
                                                           *map(base.SITES.index, excluded)])
            for arm in ARMS:
                minibatches[arm].append(group[arm])
        if [len(v[0]) for v in minibatches[ARMS[0]]] != c['donor_pairs']:
            raise ValueError('Training pair counts differ from the original study')
        weights, nets = {}, {}
        for a in ARMS:
            identity = dict(registration=key, name=f'{name}_{a}', seed=seed, excluded=excluded,
                            fit_sites=donor, role=a)
            nets[a], weights[a] = fit(identity, minibatches[a])
        tag = old.double_name(excluded, seed)
        forecaster = base.load_net(old.LOCAL / f'{tag}.pt', forecasts['double_excluded'][tag],
                                   old.motion.motion.network(17), excluded)
        packets = {}
        for s in excluded:
            with rerank.packet(learn.LOCAL, c['packets'][s]) as z:
                arrays = choose(learn.menu(data[s], forecaster), z, nets)
            path = LOCAL / f'{name}_{s}.npz'
            packets[s] = dict(file=path.name, sha256=base.parent.indices.save_npz(path, arrays))
        rules = {}
        for held, cal in (excluded, excluded[::-1]):
            rules[held] = {}
            with rerank.packet(learn.LOCAL, c['packets'][cal]) as a, rerank.packet(LOCAL, packets[cal]) as b:
                p = rerank.merge(a, b)
                for arm in ARMS:
                    r = pair.evaluate(pair.prediction(p, arm), p, data[cal], cut)
                    t, n = base.calibration(p[f'{arm}_scores'], r['dual'])
                    rules[held][arm] = dict(threshold=t, calibration_recordings=n)
        done[name] = dict(registration=key, weights=weights, packets=packets, rules=rules)
        write(cache, done[name], immutable=True)
        beat(phase='freeze', cell=name, complete=len(done))
    write(OUT / 'freeze.json', dict(registration=key, cells=done, original_2pct_unchanged=True,
                                   held_scoring_started=False), immutable=True)


def score(key, source, cells):
    freeze = OUT / 'freeze.json'
    f = old.read(freeze)
    if f['registration'] != key or set(f['cells']) != set(cells):
        raise ValueError('All pairs and thresholds must freeze before scoring')
    prior = old.read(learn.OUT / 'readout.json')
    old_views = {(r['site'], r['seed'], r['calibration_site']): r for r in prior['views']}
    data, views = base.parent.load(source), []
    for name, c in cells.items():
        for site, p in c['packets'].items():
            d, rule, row = data[site], c['rules'][site], f['cells'][name]
            cal = rule['calibration_site']
            with rerank.packet(learn.LOCAL, p) as a, rerank.packet(LOCAL, row['packets'][site]) as b:
                z = rerank.merge(a, b)
                cv = z['cv']
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
                pred, counts = {}, {}
                for a in (*ARMS, *CONTROLS):
                    pred[a] = pair.prediction(z, a)
                    v = pair.evaluate(pred[a], z, d, c['cut'])
                    setting = row['rules'][site][a] if a in ARMS else rule['arms'][a]
                    t = setting['threshold']
                    take = z[f'{a}_scores'] >= t if t is not None else np.zeros(len(v['dual']), bool)
                    counts[a] = dict(matched=len(take), matched_missing=int(v['missing'].sum()),
                        matched_risk=int(v['risk'].sum()), matched_dual=int(v['dual'].sum()),
                        released=int(take.sum()), released_missing=int((take & v['missing']).sum()),
                        released_risk=int((take & v['risk']).sum()), released_dual=int((take & v['dual']).sum()),
                        calibration=setting['calibration_recordings'])
                    if a in CONTROLS and counts[a] != old_views[site, c['seed'], cal]['arms'][a]:
                        raise ValueError('Original control counts did not replay')
                contrasts = {}
                for other in ('direct_aux', *CONTROLS):
                    v = pair.paired(pred['factorized'], pred[other], d, z['bounds'], mass)
                    v['dual_count'] = counts['factorized']['matched_dual']-counts[other]['matched_dual']
                    contrasts[other] = v
            views.append(dict(site=site, seed=c['seed'], calibration_site=cal, counts=counts, contrasts=contrasts))
        beat(phase='score', cell=name, views=len(views))
    if len(views) != 126:
        raise ValueError('Require all paired directions')
    draws = np.random.default_rng(2917).integers(len(base.SITES), size=(3000, len(base.SITES)))
    intervals = {}
    for arm, ref in views[0]['contrasts'].items():
        intervals[arm] = {}
        for key0 in ref:
            x = np.array([np.mean([v['contrasts'][arm][key0] for v in views if v['site'] == s]) for s in base.SITES])
            intervals[arm][key0] = dict(mean=float(x.mean()), ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist(),
                                        sites=x.tolist())
    summary = {}
    for a in (*ARMS, *CONTROLS):
        rows = [r['counts'][a] for r in views]
        active = [r for r in rows if r['released']]
        summary[a] = dict(**{k: sum(r[k] for r in rows) for k in rows[0]}, active=len(active),
                          all_dual=sum(r['released'] == r['released_dual'] for r in active))
    write(OUT / 'readout.json', dict(freeze_sha256=sha(freeze), result_source='fresh_run', directions=len(views),
        summary=summary, intervals=intervals, views=views, original_2pct_unchanged=True,
        external_outcomes_read=False, independent_confirmation=False, deployment_changed=False), immutable=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    args = parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, cells, cost, forecasts = register()
        if args.phase == 'freeze':
            freeze(key, source, cells, cost, forecasts)
        else:
            score(key, source, cells)
