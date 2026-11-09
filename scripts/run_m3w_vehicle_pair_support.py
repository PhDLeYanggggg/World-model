"""Freeze and score the two-edit OpenDD support/utility comparison."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_support_rank as rank
from src.evaluation.m3w_interaction_frontier import pair_reduction
from src.world_model import m3w_vehicle_pair_support as model

base, np, torch, old = rank.base, rank.np, rank.torch, rank.old
OUT = base.OUT.parent / 'opendd_vehicle_pair_support_v1'
LOCAL = base.LOCAL.parent / OUT.name
ARMS = model.ARMS


def register():
    key, source, cost, forecasters = rank.register()
    path = rank.OUT / 'freeze.json'
    support = old.read(path)
    if support['registration'] != key or not support['held_choices_frozen'] or len(support['cells']) != 63:
        raise ValueError('Require the completed recipient support experiment')
    refs = (Path(__file__), Path(model.__file__), OUT / 'protocol.md', path, rank.OUT / 'registration.json',
            Path(pair_reduction.__code__.co_filename), Path(base.policy.__code__.co_filename))
    p = OUT / 'registration.json'
    base.prior.write_json(p, dict(files={str(f.resolve().relative_to(base.prior.ROOT)): base.prior.sha(f) for f in refs},
        arms=ARMS, sites=base.SITES, seeds=base.SEEDS, same_count=2, original_2pct=True,
        songdo_read=False, reserved_outcomes_opened=False), immutable=True)
    return base.prior.sha(p), source, support, cost, forecasters


def beat(**kw):
    base.prior.write_json(LOCAL / 'heartbeat.json', dict(pid=os.getpid(), **kw))
    print(kw, flush=True)


def fixed_pairs(offsets, recording, peak):
    choice = {}
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        name = str(recording[lo])
        if recording[hi-1] != name:
            raise ValueError('Query spans a recording boundary')
        legal = peak[lo:hi] > 0.
        n = int(legal.any(axis=1).sum())
        if hi-lo >= 3 and (legal.sum(axis=0) >= 2).any() and (name not in choice or n > choice[name][0]):
            choice[name] = n, int(lo), int(hi)
    return {k: v[1:] for k, v in choice.items()}


@torch.inference_mode()
def propose(d, forecaster, factor, support, cut):
    cv, ols, offsets, paths, bx, units, _ = old.menu(d, old.forecast(forecaster, d))
    h, _ = rank.features(d['history'], cut[0])
    p = support(torch.from_numpy(h)).squeeze(1).sigmoid().numpy()
    gain, risk, _, conditional = [a.numpy() for a in factor.parts(torch.from_numpy(bx))]
    easy_harm = p[:, None]*conditional[..., 0]+conditional[..., 2]
    easy_mass = p[:, None]*conditional[..., 1]
    v = np.concatenate((gain, np.stack((risk[..., 0], easy_harm, risk[..., 1], easy_mass), axis=-1)), axis=-1)
    v = v*units[:, None, None]
    peak = np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1)
    queries = fixed_pairs(offsets, d['recording'], peak)
    if not queries:
        raise ValueError('No matched eligible recordings at this source')
    packet = dict(cv=cv, ols4=ols, records=np.array(list(queries), dtype='U20'), bounds=np.array(list(queries.values())))
    results = {a: [] for a in ARMS}
    for lo, hi in queries.values():
        best = model.choose(v[lo:hi], p[lo:hi], peak[lo:hi])
        if tuple(best) != ARMS:
            raise ValueError('Matched query lost an eligible control')
        for arm, (row, family, score) in best.items():
            row = row+lo
            results[arm].append((row, paths[row, family], score))
    for arm, entries in results.items():
        for kind, values in zip(('rows', 'paths', 'scores'), zip(*entries, strict=True), strict=True):
            packet[f'{arm}_{kind}'] = np.array(values)
    return packet


def prediction(packet, arm):
    pred = packet['cv'].copy()
    rows = packet[f'{arm}_rows']
    if len(np.unique(rows)) != rows.size:
        raise ValueError('Duplicate edited actor-window')
    pred[rows] = packet[f'{arm}_paths']
    return pred


def evaluate(pred, packet, d, cut):
    cv, ols, names = packet['cv'], packet['ols4'], packet['records']
    r = base.policy(pred, cv, cv, d['future'], d['valid'], d['recording'], easy_cut=cut[0], hard_cut=cut[1])
    rec = [r['recordings'][k] for k in names]
    if any(v['selected'] != 2 for v in rec):
        raise ValueError('Expected two edits in each matched recording')
    risk = np.array([v['supported'] for v in rec], dtype=bool)
    missing = np.array([v['completion_easy_risk_upper'] is None for v in rec])
    recs, idx = np.unique(d['recording'], return_inverse=True)
    ix = np.searchsorted(recs, names)
    np.testing.assert_array_equal(recs[ix], names)
    gain = []
    for ref in (cv, ols):
        _, lower = base.parent.paired(ref, pred, d['future'], d['valid'])
        gain.append(np.bincount(idx, weights=d['weight']*lower.sum(1))[ix])
    return dict(risk=risk, missing=missing, dual=risk & (gain[0] > 0.) & (gain[1] > 0.),
                gain_cv=gain[0], gain_ols=gain[1], p95=r['p95_known_ADE_pixel'])


def freeze(key, source, support, cost, forecasters):
    data = base.parent.load(source)
    cells = {}
    for name, cell in support['cells'].items():
        identity = cell['identity']
        pair, cut, seed = identity['excluded'], cell['cut'], identity['seed']
        receipt = LOCAL / f'{name}.json'
        if receipt.exists():
            result = old.read(receipt)
            if (result['registration'] != key
                    or any(base.prior.sha(LOCAL / p['file']) != p['sha256'] for p in result['packets'].values())):
                raise ValueError('Changed cached pair proposal')
            cells[name] = result
            continue
        risk_weights = cost['cells'][name]['weights']['factorized']
        forecaster_name = old.double_name(pair, seed)
        forecaster = base.load_net(old.LOCAL / f'{forecaster_name}.pt', forecasters['double_excluded'][forecaster_name],
                                   old.motion.motion.network(17), pair)
        factor = base.load_net(base.LOCAL / f'{name}_factorized.pt', risk_weights, base.model.FactorValue(), pair)
        predictor = base.load_net(rank.LOCAL / f'{name}_support.pt', cell['checkpoint'], rank.support_model(), pair)
        packets, rules = {}, {}
        for site in pair:
            packet = propose(data[site], forecaster, factor, predictor, cut)
            p = LOCAL / f'{name}_{site}.npz'
            packets[site] = dict(file=p.name, sha256=base.parent.indices.save_npz(p, packet))
        for held, cal in (pair, pair[::-1]):
            arms = {}
            with np.load(LOCAL / packets[cal]['file'], allow_pickle=False) as z:
                for arm in ARMS:
                    val = evaluate(prediction(z, arm), z, data[cal], cut)
                    t, n = base.calibration(z[f'{arm}_scores'], val['dual'])
                    arms[arm] = dict(threshold=t, calibration_recordings=n)
            rules[held] = dict(calibration_site=cal, arms=arms)
        result = dict(registration=key, fit_sites=identity['fit_sites'], seed=seed, cut=cut, packets=packets, rules=rules)
        old.write(receipt, result)
        cells[name] = result
        beat(phase='freeze', cell=name, count=len(cells))
    old.write(OUT / 'freeze.json', dict(registration=key, cells=cells, held_choices_frozen=True,
                                      songdo_read=False, original_2pct_unchanged=True))


def paired(treatment, control, d, bounds, mass):
    known, lower = base.parent.paired(control, treatment, d['future'], d['valid'])
    rel = [pair_reduction(treatment[None, lo:hi], control[lo:hi],
                          d['future'][lo:hi], d['valid'][lo:hi])[0] for lo, hi in bounds]
    return dict(ade_lower_pp=float(100.*d['weight'] @ lower.sum(1)/mass),
                ade_available_pp=float(100.*d['weight'] @ known.sum(1)/mass),
                relative_lower_native=float(np.mean(rel, axis=0)[1]))


def score(key, source):
    fp = OUT / 'freeze.json'
    f = old.read(fp)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require frozen pair proposals')
    data = base.parent.load(source)
    views = []
    for name, cell in f['cells'].items():
        for site, packet in cell['packets'].items():
            p = LOCAL / packet['file']
            if base.prior.sha(p) != packet['sha256'] or site in cell['fit_sites']:
                raise ValueError('Changed frozen choices or invalid source role')
            d = data[site]
            rules = cell['rules'][site]
            arms = {}
            with np.load(p, allow_pickle=False) as z:
                cv = z['cv']
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(axis=1))
                predictions = {arm: prediction(z, arm) for arm in ARMS}
                for arm in ARMS:
                    s = evaluate(predictions[arm], z, d, cell['cut'])
                    rule = rules['arms'][arm]
                    t = rule['threshold']
                    use = z[f'{arm}_scores'] >= t if t is not None else np.zeros(len(s['dual']), bool)
                    arms[arm] = dict(matched_recordings=len(use), matched_risk=int(s['risk'].sum()),
                        matched_dual=int(s['dual'].sum()), matched_missing=int(s['missing'].sum()),
                        calibrated_recordings=rule['calibration_recordings'], released_recordings=int(use.sum()),
                        released_risk=int((use & s['risk']).sum()), released_dual=int((use & s['dual']).sum()),
                        released_missing=int((use & s['missing']).sum()),
                        matched_gain_cv_pp=float(100.*s['gain_cv'].sum()/mass),
                        matched_gain_ols_cv_scaled_pp=float(100.*s['gain_ols'].sum()/mass),
                        p95_available_ade_native=s['p95'])
                contrasts = {control: paired(predictions['joint'], predictions[control], d, z['bounds'], mass)
                             for control in ('independent', 'uniform')}
            views.append(dict(site=site, seed=cell['seed'], calibration_site=rules['calibration_site'],
                              arms=arms, contrasts=contrasts))
        beat(phase='score', cell=name, count=len(views))
    count_keys = tuple(k for k, v in views[0]['arms'][ARMS[0]].items() if isinstance(v, int))
    summary = {arm: {k: sum(v['arms'][arm][k] for v in views) for k in count_keys} for arm in ARMS}
    for arm in ARMS:
        active = [v['arms'][arm] for v in views if v['arms'][arm]['released_recordings']]
        summary[arm].update(active_directions=len(active),
                            all_risk_directions=sum(r['released_recordings'] == r['released_risk'] for r in active),
                            all_dual_directions=sum(r['released_recordings'] == r['released_dual'] for r in active))
    rng = np.random.default_rng(2937)
    ix = rng.integers(len(base.SITES), size=(3000, len(base.SITES)))
    intervals = {}
    for arm in ('independent', 'uniform'):
        intervals[arm] = {}
        for metric in ('ade_lower_pp', 'ade_available_pp', 'relative_lower_native'):
            x = np.array([np.mean([r['contrasts'][arm][metric] for r in views if r['site'] == site])
                          for site in base.SITES])
            intervals[arm][metric] = dict(site=x.tolist(), mean=float(x.mean()),
                                         ci95=np.quantile(x[ix].mean(1), [.025, .975]).tolist())
    old.write(OUT / 'readout.json', dict(registration=key, freeze=base.prior.sha(fp), result_source='fresh_run',
        summary=summary, intervals=intervals, views=views, independent_confirmation=False,
        original_2pct_unchanged=True, songdo_read=False, reserved_outcomes_opened=False, deployment_changed=False))


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
        key, source, support, cost, forecasters = register()
        if args.phase == 'freeze':
            freeze(key, source, support, cost, forecasters)
        else:
            score(key, source)
