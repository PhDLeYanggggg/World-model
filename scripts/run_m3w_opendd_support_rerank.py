"""Freeze and score support-aware OpenDD pair choices without new training."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_pair_learning as learn, run_m3w_vehicle_support_rank as support
from src.evaluation import m3w_opendd_support_rerank as model

pair, base, old, np, torch = learn.pair, learn.base, learn.old, learn.np, learn.torch
sha, write = base.prior.sha, base.prior.write_json
OUT = learn.OUT.parent / 'opendd_support_rerank_v1'
LOCAL = learn.LOCAL.parent / OUT.name
CONTROLS = ('additive', 'same_family', 'joint')
VARIANTS = dict(additive_support='additive', additive_shuffled='additive',
                family_support='same_family', joint_support='joint')


def register():
    key, source, _, _, forecasts = learn.register()
    support_key, support_source, _, _ = support.register()
    lp, sp = learn.OUT / 'freeze.json', support.OUT / 'freeze.json'
    paired, actors = old.read(lp), old.read(sp)
    if (source != support_source or paired['registration'] != key or actors['registration'] != support_key
            or not paired['held_choices_frozen'] or not actors['held_choices_frozen'] or actors['songdo_read']
            or len(paired['cells']) != 63 or set(paired['cells']) != set(actors['cells'])):
        raise ValueError('Both frozen development studies must use the same registered sources')
    files = (lp, sp, Path(__file__), Path(model.__file__), OUT / 'protocol.md', old.OUT / 'freeze.json',
             base.parent.OUT / 'cohorts.json')
    p = OUT / 'registration.json'
    write(p, dict(files={str(x.relative_to(base.prior.ROOT)): sha(x) for x in files},
                  original_2pct=True, controls=CONTROLS, variants=VARIANTS,
                  training_seeds=base.SEEDS, new_training=False, external_outcomes_read=False), immutable=True)
    return sha(p), source, paired['cells'], actors['cells'], forecasts


def packet(path, ref):
    f = path / ref['file']
    if sha(f) != ref['sha256']:
        raise ValueError(f'Changed choice packet: {f.name}')
    return np.load(f, allow_pickle=False)


@torch.inference_mode()
def choose(menu, archived, models, logits, shuffled):
    for k in ('cv', 'ols4'):
        np.testing.assert_array_equal(menu[k], archived[k])
    np.testing.assert_array_equal(list(menu['slots']), archived['records'])
    np.testing.assert_array_equal(list(menu['slots'].values()), archived['bounds'])
    result = {v: [] for v in VARIANTS}
    for n, (lo, hi) in enumerate(menu['slots'].values()):
        row, family, i, j = learn.choices(menu['peak'], lo, hi)
        x = [torch.from_numpy(v).split(4096) for v in learn.input_rows(menu, row, family, i, j)]
        scores = {a: learn.model.merit(torch.cat([net(*b) for b in zip(*x, strict=True)]).numpy())
                  for a, net in models.items()}
        same = family[i] == family[j]
        scores['same_family'] = np.where(same, scores['joint'], -np.inf)
        for a, s in scores.items():
            k = s.argmax()
            take = [i[k], j[k]]
            np.testing.assert_array_equal(row[take], archived[f'{a}_rows'][n])
            np.testing.assert_array_equal(menu['paths'][row[take], family[take]], archived[f'{a}_paths'][n])
        for arm, control in VARIANTS.items():
            rank = model.rank_with_support(scores['joint' if control == 'same_family' else control],
                                            row[i], row[j], shuffled if arm == 'additive_shuffled' else logits)
            if control == 'same_family':
                rank = np.where(same, rank, -np.inf)
            k = rank.argmax()
            take = [i[k], j[k]]
            if not np.isfinite(rank[k]):
                raise ValueError('Invalid selected pair')
            result[arm].append((row[take], menu['paths'][row[take], family[take]], rank[k]))
    arrays = {}
    cv = archived['cv']
    a = archived['additive_rows']
    radius = np.linalg.norm(archived['additive_paths']-cv[a], axis=-1).max(axis=-1)
    for arm, r in result.items():
        rows, paths, scores = map(np.asarray, zip(*r, strict=True))
        np.testing.assert_allclose(np.linalg.norm(paths-cv[rows], axis=-1).max(axis=-1), radius,
                                   atol=1e-9, rtol=1e-9)
        arrays.update({f'{arm}_rows': rows, f'{arm}_paths': paths, f'{arm}_scores': scores})
    return arrays


def beat(**kw):
    write(LOCAL / 'heartbeat.json', dict(pid=os.getpid(), **kw))
    print(kw, flush=True)


def freeze(key, source, cells, actors, forecasts):
    data, receipts = base.parent.load(source), {}
    for name, c in cells.items():
        ref, cache = actors[name], LOCAL / f'{name}.json'
        excluded, seed = sorted(c['packets']), c['seed']
        if ref['cut'] != c['cut'] or set(ref['identity']['excluded']) != set(excluded):
            raise ValueError('Easy cutoff or support model source exclusions changed')
        if cache.exists():
            cached = old.read(cache)
            if cached['registration'] != key:
                raise ValueError('Cached freeze belongs to a different study')
            for r in cached['packets'].values():
                with packet(LOCAL, r):
                    pass
            receipts[name] = cached
            continue
        support_net = base.load_net(support.LOCAL / f'{name}_support.pt', ref['checkpoint'],
                                    support.support_model(), excluded)
        models = {a: base.load_net(learn.LOCAL / f'{name}_{a}.pt', c['weights'][a],
                                   learn.model.PairValue(joint=a == 'joint'), excluded)
                  for a in ('additive', 'joint')}
        tag = old.double_name(excluded, seed)
        forecaster = base.load_net(old.LOCAL / f'{tag}.pt', forecasts['double_excluded'][tag],
                                   old.motion.motion.network(17), excluded)
        packets = {}
        for site in excluded:
            d = data[site]
            x, _ = support.features(d['history'], c['cut'][0])
            with torch.inference_mode():
                logits = support_net(torch.from_numpy(x)).squeeze(1).numpy()
            shuffled = model.shuffle_by_query(logits, d['query_counts'], [3227, seed, base.SITES.index(site)])
            with packet(learn.LOCAL, c['packets'][site]) as z:
                out = choose(learn.menu(d, forecaster), z, models, logits, shuffled)
            file = LOCAL / f'{name}_{site}.npz'
            packets[site] = dict(file=file.name, sha256=base.parent.indices.save_npz(file, out))
        rules = {}
        for held, cal in (excluded, excluded[::-1]):
            rules[held] = {}
            with packet(learn.LOCAL, c['packets'][cal]) as p, packet(LOCAL, packets[cal]) as v:
                z = merge(p, v)
                for arm in VARIANTS:
                    q = pair.evaluate(pair.prediction(z, arm), z, data[cal], c['cut'])
                    t, n = base.calibration(z[f'{arm}_scores'], q['dual'])
                    rules[held][arm] = dict(threshold=t, calibration_recordings=n)
        receipts[name] = dict(registration=key, packets=packets, rules=rules,
                              weights=dict(support=ref['checkpoint']['sha256'],
                                  **{a: c['weights'][a]['sha256'] for a in models}))
        write(cache, receipts[name], immutable=True)
        beat(phase='freeze', cell=name, done=len(receipts))
    write(OUT / 'freeze.json', dict(registration=key, cells=receipts, original_2pct_unchanged=True,
                                   held_scoring_started=False), immutable=True)


def merge(original, new):
    return {k: v[k] for v in (original, new) for k in v.files}


def score(key, source, cells):
    path = OUT / 'freeze.json'
    f = old.read(path)
    if f['registration'] != key or set(f['cells']) != set(cells):
        raise ValueError('Require complete choices and thresholds before held scoring')
    prior = old.read(learn.OUT / 'readout.json')
    lookup = {(v['site'], v['seed'], v['calibration_site']): v for v in prior['views']}
    data, views = base.parent.load(source), []
    for name, c in cells.items():
        row = f['cells'][name]
        for site, p in c['packets'].items():
            rule = c['rules'][site]
            d, cal = data[site], rule['calibration_site']
            with packet(learn.LOCAL, p) as old_z, packet(LOCAL, row['packets'][site]) as new_z:
                z = merge(old_z, new_z)
                cv = z['cv']
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(1))
                metrics, predictions = {}, {}
                for a in (*CONTROLS, *VARIANTS):
                    pred = predictions[a] = pair.prediction(z, a)
                    v = pair.evaluate(pred, z, d, c['cut'])
                    setting = rule['arms'][a] if a in CONTROLS else row['rules'][site][a]
                    t = setting['threshold']
                    use = z[f'{a}_scores'] >= t if t is not None else np.zeros(len(v['dual']), bool)
                    counts = metrics[a] = dict(matched=len(use), matched_missing=int(v['missing'].sum()),
                        matched_risk=int(v['risk'].sum()), matched_dual=int(v['dual'].sum()),
                        released=int(use.sum()), released_missing=int((use & v['missing']).sum()),
                        released_risk=int((use & v['risk']).sum()), released_dual=int((use & v['dual']).sum()),
                        calibration=setting['calibration_recordings'])
                    if a in CONTROLS and counts != lookup[site, c['seed'], cal]['arms'][a]:
                        raise ValueError('The archived held comparison did not replay')
                contrasts = {}
                for a, b in (*VARIANTS.items(), ('additive_support', 'additive_shuffled')):
                    r = pair.paired(predictions[a], predictions[b], d, z['bounds'], mass)
                    r['dual_count'] = metrics[a]['matched_dual']-metrics[b]['matched_dual']
                    r['support_count'] = metrics[b]['matched_missing']-metrics[a]['matched_missing']
                    contrasts[f'{a}_minus_{b}'] = r
            views.append(dict(site=site, seed=c['seed'], calibration_site=cal, arms=metrics, contrasts=contrasts))
        beat(phase='score', cell=name, views=len(views))
    if len(views) != 126:
        raise ValueError('Require all 126 original directions')
    draws = np.random.default_rng(2917).integers(len(base.SITES), size=(3000, len(base.SITES)))
    intervals = {}
    for k, v in views[0]['contrasts'].items():
        intervals[k] = {}
        for field in v:
            x = np.array([np.mean([r['contrasts'][k][field] for r in views if r['site'] == s])
                          for s in base.SITES])
            intervals[k][field] = dict(mean=float(x.mean()), sites=x.tolist(),
                                       ci95=np.quantile(x[draws].mean(1), [.025, .975]).tolist())
    summary = {}
    for a in views[0]['arms']:
        rows = [v['arms'][a] for v in views]
        summary[a] = {k: sum(r[k] for r in rows) for k in rows[0]}
        active = [r for r in rows if r['released']]
        summary[a].update(active=len(active), all_dual=sum(r['released'] == r['released_dual'] for r in active))
    write(OUT / 'readout.json', dict(result_source='fresh_run', weights='cached_verified', summary=summary,
          intervals=intervals, freeze_sha256=sha(path), directions=len(views), independent_confirmation=False,
          original_2pct_unchanged=True, external_outcomes_read=False, new_training=False), immutable=True)


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
        key, source, cells, actors, forecasts = register()
        if args.phase == 'freeze':
            freeze(key, source, cells, actors, forecasts)
        else:
            score(key, source, cells)
