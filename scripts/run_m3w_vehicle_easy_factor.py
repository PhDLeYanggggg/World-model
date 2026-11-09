"""Fit factorized vehicle risk and calibrate on double-excluded OpenDD sites."""

import argparse
import fcntl
from itertools import combinations
import json
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_matched as old
from src.evaluation.m3w_selection_forecast_readout import policy
from src.world_model import m3w_vehicle_easy_factor as model

np, torch, parent, prior = old.np, old.torch, old.parent, old.prior
OUT = old.OUT.parent / 'opendd_vehicle_easy_factor_v1'
LOCAL = old.LOCAL.parent / OUT.name
SITES, SEEDS = old.SITES, old.SEEDS
ARMS = ('monolithic', 'factorized')


def register():
    key, source, _ = old.register()
    f = old.read(old.OUT / 'freeze.json')
    if f['registration'] != key or f['songdo_read'] or len(f['double_excluded']) != 63:
        raise ValueError('Require the verified double-excluded forecast bank')
    refs = (__file__, model.__file__, old.__file__, old.OUT / 'freeze.json',
            parent.OUT / 'cohorts.json', OUT / 'protocol.md')
    path = OUT / 'registration.json'
    old.write(path, dict(files={str(Path(p).relative_to(prior.ROOT)): prior.sha(Path(p)) for p in refs},
        sites=SITES, seeds=SEEDS, roles='all unordered double and triple exclusions; each double used in both directions',
        arms=ARMS, steps=old.VALUE_STEPS, forecast_steps=parent.STEPS, original_2pct=True,
        clock_unchanged=True, no_new_official_split=True, songdo_read=False, reserved_outcomes_opened=False))
    return prior.sha(path), source, f


def fit(*args):
    original = old.LOCAL
    old.LOCAL = LOCAL
    try:
        return old.fit(*args)
    finally:
        old.LOCAL = original


def triple_id(sites, seed):
    return '_'.join(sorted(sites)) + f'_{seed}_triple'


def load_net(path, ref, net, excluded):
    if prior.sha(path) != ref['sha256']:
        raise ValueError(f'Changed trained weight: {path.name}')
    state = torch.load(path, weights_only=True, map_location='cpu')
    if (state['identity'] != ref['identity'] or set(ref['identity']['excluded']) != set(excluded)
            or set(ref['identity']['fit_sites']) != set(SITES)-set(excluded) or state['step'] != ref['step']):
        raise ValueError(f'Checkpoint source roles do not match: {path.name}')
    net.load_state_dict(state['weights'])
    return net.eval()


def fixed_slots(offsets, recording, legal):
    """One fixed query per recording, picked by count of legal actors, then chronology."""
    chosen = {}
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        key = str(recording[lo])
        if recording[hi-1] != key:
            raise ValueError('Query spans a recording boundary')
        n = int(legal[lo:hi].any(axis=1).sum())
        if n >= 2 and (key not in chosen or n > chosen[key][0]):
            chosen[key] = (n, int(lo), int(hi))
    return {k: v[1:] for k, v in chosen.items()}


def proposals(value, paths, legal, slots):
    excess = value[..., 2:4] - .02*value[..., 4:6]
    rank = value[..., :2].min(axis=-1) - np.maximum(excess, 0.).sum(axis=-1)
    if not np.isfinite(rank).all():
        raise ValueError('Invalid score')
    ids, rows, scores, forecasts = [], [], [], []
    for record, (lo, hi) in slots.items():
        v = np.where(legal[lo:hi], rank[lo:hi], -np.inf)
        i, j = np.unravel_index(v.argmax(), v.shape)
        if not np.isfinite(v[i, j]):
            raise ValueError('Empty matched query')
        ids.append(record)
        rows.append(lo+i)
        scores.append(v[i, j])
        forecasts.append(paths[lo+i, j])
    return dict(record=np.array(ids, dtype='U20'), rows=np.array(rows, dtype=np.int64),
                scores=np.array(scores), paths=np.asarray(forecasts).reshape(-1, 12, 2))


def evaluate(p, cv, ols, d, cut):
    pred = cv.copy()
    pred[p['rows']] = p['paths']
    _, rec = np.unique(d['recording'], return_inverse=True)
    w = d['weight']
    r = policy(pred, cv, cv, d['future'], d['valid'], d['recording'], easy_cut=cut[0], hard_cut=cut[1])
    lower = []
    for ref in (cv, ols):
        _, z = parent.paired(ref, pred, d['future'], d['valid'])
        lower.append(np.bincount(rec, weights=w*z.sum(axis=1))[rec[p['rows']]])
    risk_ok = np.array([r['recordings'][k]['supported'] for k in p['record']], dtype=bool)
    easy = [r['recordings'][k]['completion_easy_risk_upper'] for k in p['record']]
    return dict(risk_ok=risk_ok, dual_ok=risk_ok & (lower[0] > 0.) & (lower[1] > 0.),
                lower_cv=lower[0], lower_ols=lower[1],
                easy_undefined=int(sum(x is None for x in easy)))


def calibration(scores, passed):
    """Largest score prefix where every edited calibration recording passes."""
    if not len(scores):
        return None, 0
    order = np.argsort(-scores, kind='stable')
    stop = np.flatnonzero(~passed[order])
    if not len(stop):
        return float(scores[order[-1]]), len(scores)
    bound = scores[order[stop[0]]]
    keep = scores > bound
    return (float(scores[keep].min()), int(keep.sum())) if keep.any() else (None, 0)


def batches_for(pair, seed, triples, data, cut):
    output = []
    for s in SITES:
        if s in pair:
            continue
        excluded = (*pair, s)
        tag = triple_id(excluded, seed)
        net = load_net(LOCAL / f'{tag}.pt', triples[tag], old.motion.motion.network(17), excluded)
        d = data[s]
        cv, ols, _, paths, x, unit, legal = old.menu(d, old.forecast(net, d))
        z = old.bridge.targets(cv, ols, paths, d['future'], d['valid'], cut[0], unit)
        unknown = (np.linalg.norm(paths-cv[:, None], axis=-1).max(axis=-1) / unit[:, None])
        unknown = np.where(d['valid'].any(axis=1)[:, None], 0., unknown).astype(np.float32)
        output.append(tuple(map(torch.from_numpy, (x[legal], z[legal], unknown[legal]))))
    return output


def make_value(arm):
    return old.bridge.BridgeValue() if arm == 'monolithic' else model.FactorValue()


def value_loss(net, data):
    x, y, u = data
    if isinstance(net, model.FactorValue):
        return model.factor_loss(net, x, y, u)
    return old.bridge.loss(net(x), y, torch.ones(len(x), dtype=torch.bool))


def freeze(key, source, f):
    data = parent.load(source)
    triples = {}
    for ex in combinations(SITES, 3):
        donors = [s for s in SITES if s not in ex]
        for seed in SEEDS:
            tag = triple_id(ex, seed)
            identity = dict(registration=key, name=tag, seed=seed, excluded=list(ex), fit_sites=donors,
                            role='four_site_vehicle_completion')
            torch.manual_seed(seed)
            _, triples[tag] = fit(identity, [data[s]['tensors'] for s in donors],
                lambda net, z: old.motion.loss(old.motion.predict(net, z[0], z[1]), *z[1:], completion=True),
                old.motion.motion.network(17), parent.STEPS, old.BATCH)
    cells = {}
    for pair in combinations(SITES, 2):
        donors = [s for s in SITES if s not in pair]
        cut = old.calibration_cut(data, donors)
        for seed in SEEDS:
            key0 = f'{pair[0]}_{pair[1]}_{seed}'
            cached = LOCAL / f'{key0}_receipt.json'
            if cached.exists():
                cell = old.read(cached)
                if (cell['registration'] != key or cell['excluded'] != list(pair)
                        or any(prior.sha(LOCAL / r['file']) != r['sha256'] for r in cell['directions'].values())):
                    raise ValueError('Frozen calibration packet changed')
                cells[key0] = cell
                continue
            batches = batches_for(pair, seed, triples, data, cut)
            weights, nets = {}, {}
            for arm in ARMS:
                name = f'{key0}_{arm}'
                ident = dict(registration=key, name=name, seed=seed, excluded=list(pair), fit_sites=donors, role=arm)
                torch.manual_seed(seed)
                nets[arm], weights[arm] = fit(ident, batches, value_loss, make_value(arm), old.VALUE_STEPS, old.BATCH)
            name = old.double_name(pair, seed)
            ref = f['double_excluded'][name]
            forecaster = load_net(old.LOCAL / f'{name}.pt', ref, old.motion.motion.network(17), pair)
            steps = {}
            for site in pair:
                d = data[site]
                cv, ols, offsets, paths, x, _, legal = old.menu(d, old.forecast(forecaster, d))
                slots = fixed_slots(offsets, d['recording'], legal)
                packets = dict(cv=cv, ols4=ols)
                for arm in ARMS:
                    with torch.inference_mode():
                        v = nets[arm](torch.from_numpy(x)).numpy()
                    p = proposals(v, paths, legal, slots)
                    packets.update({f'{arm}_{k}': a for k, a in p.items()})
                a, b = [packets[f'{k}_rows'] for k in ARMS]
                np.testing.assert_array_equal(packets[f'{ARMS[0]}_record'], packets[f'{ARMS[1]}_record'])
                q = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
                np.testing.assert_array_equal(q[a], q[b])
                np.testing.assert_allclose(np.linalg.norm(packets[f'{ARMS[0]}_paths']-cv[a], axis=-1).max(axis=-1),
                                           np.linalg.norm(packets[f'{ARMS[1]}_paths']-cv[b], axis=-1).max(axis=-1), atol=1e-9)
                path = LOCAL / f'{key0}_{site}_packet.npz'
                steps[site] = dict(file=path.name, sha256=parent.indices.save_npz(path, packets))
            rules = {}
            for held in pair:
                calibration_site = next(s for s in pair if s != held)
                p0 = steps[calibration_site]
                with np.load(LOCAL / p0['file'], allow_pickle=False) as z:
                    arm_rules = {}
                    for arm in ARMS:
                        p1 = {k: z[f'{arm}_{k}'] for k in ('record', 'rows', 'scores', 'paths')}
                        actual = evaluate(p1, z['cv'], z['ols4'], data[calibration_site], cut)
                        threshold, n = calibration(p1['scores'], actual['dual_ok'])
                        arm_rules[arm] = dict(threshold=threshold, calibration_edits=n,
                                              matched_calibration_edits=len(p1['rows']))
                    rules[held] = dict(calibration_site=calibration_site, arms=arm_rules)
            cell = dict(registration=key, excluded=list(pair), seed=seed, fit_sites=donors,
                        directions=steps, calibration=rules, weights=weights, cut=cut.tolist())
            old.write(cached, cell)
            cells[key0] = cell
            old.beat('freeze', pair=pair, seed=seed, cells=len(cells))
    old.write(OUT / 'freeze.json', dict(registration=key, triple_checkpoints=triples, cells=cells,
        held_choices_frozen=True, original_2pct=True, songdo_read=False))


def score(key, source):
    fp = OUT / 'freeze.json'
    f = old.read(fp)
    if f['registration'] != key or not f['held_choices_frozen'] or len(f['cells']) != 63:
        raise ValueError('Require all frozen choices before scoring')
    data = parent.load(source)
    results = []
    for name, cell in f['cells'].items():
        for site, ref in cell['directions'].items():
            p0 = LOCAL / ref['file']
            if prior.sha(p0) != ref['sha256'] or site in cell['fit_sites']:
                raise ValueError('Changed frozen choice')
            d = data[site]
            r = cell['calibration'][site]
            arms = {}
            with np.load(p0, allow_pickle=False) as z:
                cv, ols = z['cv'], z['ols4']
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(axis=1))
                for arm in ARMS:
                    v = {k: z[f'{arm}_{k}'] for k in ('record', 'rows', 'scores', 'paths')}
                    s = evaluate(v, cv, ols, d, cell['cut'])
                    cal = r['arms'][arm]
                    take = v['scores'] >= cal['threshold'] if cal['threshold'] is not None else np.zeros(len(v['rows']), dtype=bool)
                    arms[arm] = dict(matched_edits=len(take), matched_risk=int(s['risk_ok'].sum()),
                        matched_dual=int(s['dual_ok'].sum()), matched_easy_undefined=s['easy_undefined'],
                        matched_cv_gain_pp=float(100.*s['lower_cv'].sum()/mass),
                        calibrated_edits=int(take.sum()), calibrated_risk=int((take & s['risk_ok']).sum()),
                        calibrated_dual=int((take & s['dual_ok']).sum()),
                        calibrated_cv_gain_pp=float(100.*s['lower_cv'][take].sum()/mass), **cal)
            if arms[ARMS[0]]['matched_edits'] != arms[ARMS[1]]['matched_edits']:
                raise ValueError('Held count is not matched')
            results.append(dict(outer_site=site, calibration_site=r['calibration_site'], seed=cell['seed'], arms=arms))
        old.beat('score', pair=name, views=len(results))
    fields = ('matched_edits', 'matched_risk', 'matched_dual', 'matched_easy_undefined',
              'calibration_edits', 'calibrated_edits', 'calibrated_risk', 'calibrated_dual')
    summary = {a: {k: sum(r['arms'][a][k] for r in results) for k in fields} for a in ARMS}
    def interval(field):
        x = np.array([np.mean([r['arms']['factorized'][field]-r['arms']['monolithic'][field]
                               for r in results if r['outer_site'] == s]) for s in SITES])
        b = np.random.default_rng(3183).choice(x, (3000, len(x))).mean(axis=1)
        return dict(site_differences=x.tolist(), mean=float(x.mean()), ci95=np.quantile(b, [.025, .975]).tolist())
    contrasts = {k: interval(k) for k in ('matched_dual', 'matched_cv_gain_pp', 'calibrated_cv_gain_pp')}
    out = dict(registration=key, freeze=prior.sha(fp), result_source='fresh_run', role='nested_development_only',
        summary=summary, contrasts=contrasts, views=results, original_2pct_unchanged=True, songdo_read=False,
        independent_confirmation=False, reserved_outcomes_opened=False, deployment_changed=False)
    old.write(OUT / 'readout.json', out)
    print(json.dumps(dict(summary=summary, contrasts=contrasts), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=('freeze', 'score'))
    a = ap.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, src, f = register()
        if a.phase == 'freeze':
            freeze(key, src, f)
        else:
            score(key, src)
