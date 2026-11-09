"""Fit equal-capacity spacing ablations at the frozen OpenDD vehicle slots."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_m3w_opendd_vehicle_observed_geometry as inputs
from scripts import run_m3w_vehicle_pair_learning as learn
from src.world_model import m3w_vehicle_pair_geometry as model

pair, base, old, np, torch = learn.pair, learn.base, learn.old, learn.np, learn.torch
OUT = pair.OUT.parent / 'opendd_vehicle_pair_geometry_v1'
LOCAL = pair.LOCAL.parent / OUT.name
FIT_ARMS = ('motion', 'permuted', 'geometry')
ARMS = ('archived_additive', *FIT_ARMS)


def register():
    key, source, _, cost, forecasts = learn.register()
    parent = old.read(learn.OUT / 'freeze.json')
    if parent['registration'] != key or not parent['held_choices_frozen'] or len(parent['cells']) != 63:
        raise ValueError('Require the complete trained pair study')
    manifest_path = inputs.OUT / 'manifest.json'
    geo = old.read(manifest_path)
    cohort = old.read(base.parent.OUT / 'cohorts.json')
    if (not geo['past_coordinates_only'] or not geo['no_labels_exported']
            or set(geo['sites']) != set(base.SITES)):
        raise ValueError('Require the observed-only geometry export')
    for s, r in geo['sites'].items():
        if (r['cohort_sha256'] != cohort['sites'][s]['sha256']
                or r['sha256'] != base.prior.sha(inputs.LOCAL / f'{s}.npz')):
            raise ValueError('Observed geometry does not match the frozen cohort')
    refs = (Path(__file__), Path(model.__file__), OUT / 'protocol.md', learn.OUT / 'freeze.json',
            learn.OUT / 'registration.json', manifest_path, base.parent.OUT / 'cohorts.json')
    p = OUT / 'registration.json'
    old.write(p, dict(files={str(k.resolve().relative_to(base.prior.ROOT)): base.prior.sha(k) for k in refs},
        arms=ARMS, seeds=base.SEEDS, steps=learn.STEPS, workers=0, threads=4,
        external_outcomes_read=False, original_2pct=True))
    return base.prior.sha(p), source, parent, cost, forecasts


def origins(data):
    result = {}
    for site, d in data.items():
        with np.load(inputs.LOCAL / f'{site}.npz', allow_pickle=False) as f:
            a = f['observed_current_offsets']
        if a.shape != (len(d['history']), 2) or not np.isfinite(a).all():
            raise ValueError('Invalid alignment of query offsets')
        result[site] = a
    return result


def menu(d, net):
    return dict(learn.menu(d, net), history=d['history'])


def ordered_positions(d, observed, seed, site):
    return model.shuffle_origins(observed, d['query_counts'], [3227, seed, base.SITES.index(site)])


def augment(d, m, cut, rng_seed, observed, permuted):
    batch = learn.donor_batch(d, m, cut, np.random.default_rng(rng_seed))
    rng = np.random.default_rng(rng_seed)
    real, fake, pos = [], [], 0
    for lo, hi in m['slots'].values():
        row, family, i, j = learn.choices(m['peak'], lo, hi, rng)
        a, b, ka, kb = row[i], row[j], family[i], family[j]
        stop = pos+len(i)
        np.testing.assert_array_equal(batch[0][pos:stop].numpy(), m['x'][a, ka])
        np.testing.assert_array_equal(batch[1][pos:stop].numpy(), m['x'][b, kb])
        real.append(model.spacing(m, observed, a, b, ka, kb))
        fake.append(model.spacing(m, permuted, a, b, ka, kb))
        pos = stop
    if pos != len(batch[0]):
        raise ValueError('New features are not aligned with the original supervision')
    y, x = batch[-1], batch[-2]
    return {arm: (*batch[:2], torch.cat((x, torch.from_numpy(g)), dim=1), y) for arm, g in
            zip(FIT_ARMS, (np.zeros_like(x.numpy()), np.concatenate(fake), np.concatenate(real)), strict=True)}


def fit(identity, batches):
    original = old.LOCAL
    try:
        old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return old.fit(identity, batches, learn.model.loss, model.GeometryValue(), learn.STEPS, old.BATCH)
    finally:
        old.LOCAL = original


@torch.inference_mode()
def choose(m, models, archived, observed, permuted):
    np.testing.assert_array_equal(m['cv'], archived['cv'])
    np.testing.assert_array_equal(m['ols4'], archived['ols4'])
    np.testing.assert_array_equal(list(m['slots']), archived['records'])
    np.testing.assert_array_equal(list(m['slots'].values()), archived['bounds'])
    packet = {k: archived[k] for k in ('cv', 'ols4', 'records', 'bounds')}
    for k in ('paths', 'rows', 'scores'):
        packet[f'archived_additive_{k}'] = archived[f'additive_{k}']
    result = {arm: [] for arm in FIT_ARMS}
    for lo, hi in m['slots'].values():
        row, family, i, j = learn.choices(m['peak'], lo, hi)
        a, b, ka, kb = row[i], row[j], family[i], family[j]
        x, y, r = learn.input_rows(m, row, family, i, j)
        for arm, g in zip(FIT_ARMS, (np.zeros_like(r), model.spacing(m, permuted, a, b, ka, kb),
                                     model.spacing(m, observed, a, b, ka, kb)), strict=True):
            batch = tuple(torch.from_numpy(v).split(4096) for v in (x, y, np.concatenate((r, g), axis=1)))
            s = learn.model.merit(torch.cat([models[arm](*z) for z in zip(*batch, strict=True)]).numpy())
            k = int(s.argmax())
            q = [i[k], j[k]]
            if not np.isfinite(s[k]):
                raise ValueError('Invalid chosen score')
            result[arm].append((row[q], m['paths'][row[q], family[q]], s[k]))
    radius = np.linalg.norm(packet['archived_additive_paths']-packet['cv'][packet['archived_additive_rows']],
                            axis=-1).max(axis=-1)
    for arm, entries in result.items():
        for k, values in zip(('rows', 'paths', 'scores'), zip(*entries, strict=True), strict=True):
            packet[f'{arm}_{k}'] = np.asarray(values)
        rad = np.linalg.norm(packet[f'{arm}_paths']-packet['cv'][packet[f'{arm}_rows']], axis=-1).max(axis=-1)
        np.testing.assert_allclose(rad, radius, atol=1e-9, rtol=1e-9)
    return packet


def beat(**kw):
    base.prior.write_json(LOCAL / 'heartbeat.json', dict(pid=os.getpid(), **kw))
    print(kw, flush=True)


def freeze(key, source, parent, cost, forecasts):
    data, cells = base.parent.load(source), {}
    geo = origins(data)
    for name, c in parent['cells'].items():
        excluded, cut, seed = list(c['packets']), c['cut'], c['seed']
        receipt = LOCAL / f'{name}.json'
        if receipt.exists():
            r = old.read(receipt)
            if (r['registration'] != key
                    or any(base.prior.sha(LOCAL / f"{p['identity']['name']}.pt") != p['sha256'] for p in r['weights'].values())
                    or any(base.prior.sha(LOCAL / p['file']) != p['sha256'] for p in r['packets'].values())):
                raise ValueError('Cached ablation changed')
            cells[name] = r
            continue
        batches = {arm: [] for arm in FIT_ARMS}
        for donor in c['fit_sites']:
            if donor in excluded:
                raise ValueError('Held or calibration location in fit')
            sites = (*excluded, donor)
            tag = base.triple_id(sites, seed)
            net = base.load_net(base.LOCAL / f'{tag}.pt', cost['triple_checkpoints'][tag],
                                old.motion.motion.network(17), sites)
            d = data[donor]
            rng_seed = [seed, base.SITES.index(donor), *[base.SITES.index(s) for s in excluded]]
            b = augment(d, menu(d, net), cut, rng_seed, geo[donor], ordered_positions(d, geo[donor], seed, donor))
            for arm in FIT_ARMS:
                batches[arm].append(b[arm])
        if [len(b[0]) for b in batches['motion']] != c['donor_pairs']:
            raise ValueError('Training rows differ from the matched pair study')
        models, weights = {}, {}
        for arm in FIT_ARMS:
            identity = dict(name=f'{name}_{arm}', role=arm, registration=key, seed=seed,
                            fit_sites=c['fit_sites'], excluded=excluded)
            models[arm], weights[arm] = fit(identity, batches[arm])
        tag = old.double_name(excluded, seed)
        net = base.load_net(old.LOCAL / f'{tag}.pt', forecasts['double_excluded'][tag],
                            old.motion.motion.network(17), excluded)
        packets = {}
        for site, r in c['packets'].items():
            path = learn.LOCAL / r['file']
            if base.prior.sha(path) != r['sha256']:
                raise ValueError('Archived packet changed')
            d = data[site]
            with np.load(path, allow_pickle=False) as z:
                packet = choose(menu(d, net), models, z, geo[site], ordered_positions(d, geo[site], seed, site))
            p = LOCAL / f'{name}_{site}.npz'
            packets[site] = dict(file=p.name, sha256=base.parent.indices.save_npz(p, packet))
        rules = {}
        for held, cal in (excluded, excluded[::-1]):
            arms = {'archived_additive': c['rules'][held]['arms']['additive']}
            with np.load(LOCAL / packets[cal]['file'], allow_pickle=False) as z:
                for arm in FIT_ARMS:
                    v = pair.evaluate(pair.prediction(z, arm), z, data[cal], cut)
                    t, n = base.calibration(z[f'{arm}_scores'], v['dual'])
                    arms[arm] = dict(threshold=t, calibration_recordings=n)
            rules[held] = dict(calibration_site=cal, arms=arms)
        row = dict(registration=key, seed=seed, cut=cut, fit_sites=c['fit_sites'], weights=weights,
                   packets=packets, rules=rules, donor_pairs=c['donor_pairs'])
        old.write(receipt, row)
        cells[name] = row
        beat(phase='freeze', name=name, count=len(cells))
    old.write(OUT / 'freeze.json', dict(registration=key, cells=cells, held_choices_frozen=True))


def score(key, source):
    p = OUT / 'freeze.json'
    f = old.read(p)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require completed freeze')
    data, views = base.parent.load(source), []
    for name, c in f['cells'].items():
        for site, r in c['packets'].items():
            path = LOCAL / r['file']
            if base.prior.sha(path) != r['sha256'] or site in c['fit_sites']:
                raise ValueError('Frozen choice or source role changed')
            d = data[site]
            rule, arms = c['rules'][site], {}
            with np.load(path, allow_pickle=False) as z:
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(z['cv']-d['future'], axis=-1), 0.).sum(1))
                pred = {arm: pair.prediction(z, arm) for arm in ARMS}
                for arm in ARMS:
                    s = pair.evaluate(pred[arm], z, d, c['cut'])
                    t = rule['arms'][arm]['threshold']
                    use = z[f'{arm}_scores'] >= t if t is not None else np.zeros(len(s['dual']), bool)
                    arms[arm] = dict(matched=len(use), matched_risk=int(s['risk'].sum()),
                        matched_dual=int(s['dual'].sum()), matched_missing=int(s['missing'].sum()),
                        released=int(use.sum()), released_risk=int((use & s['risk']).sum()),
                        released_dual=int((use & s['dual']).sum()), released_missing=int((use & s['missing']).sum()),
                        calibration=rule['arms'][arm]['calibration_recordings'])
                contrasts = {arm: pair.paired(pred['geometry'], pred[arm], d, z['bounds'], mass) for arm in ARMS[:-1]}
                for arm in ARMS[:-1]:
                    contrasts[arm]['matched_dual_count'] = arms['geometry']['matched_dual']-arms[arm]['matched_dual']
            views.append(dict(site=site, seed=c['seed'], calibration_site=rule['calibration_site'],
                              arms=arms, contrasts=contrasts))
        beat(phase='score', name=name, count=len(views))
    summary = {arm: {k: sum(v['arms'][arm][k] for v in views) for k in views[0]['arms'][arm]} for arm in ARMS}
    for arm in ARMS:
        active = [v['arms'][arm] for v in views if v['arms'][arm]['released']]
        summary[arm].update(active=len(active), all_dual=sum(x['released']==x['released_dual'] for x in active))
    archived = old.read(learn.OUT / 'readout.json')['summary']['additive']
    if summary['archived_additive'] != archived:
        raise ValueError('Archived additive comparator does not replay')
    idx = np.random.default_rng(1739).integers(len(base.SITES), size=(3000, len(base.SITES)))
    intervals = {}
    for arm in ARMS[:-1]:
        intervals[arm] = {}
        for k in views[0]['contrasts'][arm]:
            x = np.array([np.mean([v['contrasts'][arm][k] for v in views if v['site']==site])
                          for site in base.SITES])
            intervals[arm][k] = dict(mean=float(x.mean()), sites=x.tolist(),
                                     ci95=np.quantile(x[idx].mean(1), [.025, .975]).tolist())
    old.write(OUT / 'readout.json', dict(result_source='fresh_run', registration=key, freeze=base.prior.sha(p),
        summary=summary, intervals=intervals, views=views, original_2pct_unchanged=True,
        external_outcomes_read=False, independent_confirmation=False, deployment_changed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'score'))
    a = parser.parse_args()
    LOCAL.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, source, parent, cost, forecasts = register()
        if a.phase == 'freeze':
            freeze(key, source, parent, cost, forecasts)
        else:
            score(key, source)
