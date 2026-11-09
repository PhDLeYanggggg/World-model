"""Rank OpenDD vehicle recipients by forecast-independent easy support."""

import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_vehicle_easy_factor as base
from src.evaluation.m3w_selection_forecast_readout import errors

np, torch, old = base.np, base.torch, base.old
OUT = base.OUT.parent / 'opendd_vehicle_support_rank_v1'
LOCAL = base.LOCAL.parent / OUT.name
ARMS = ('factorized', 'learned_support', 'prefix_support')
STEPS = 200


def features(history, cutoff):
    if history.ndim != 3 or history.shape[1:] != (8, 2) or not cutoff > 0:
        raise ValueError('Expected eight past samples and a positive donor cutoff')
    ds = np.diff(history, axis=1)
    speed = np.linalg.norm(ds, axis=2)
    accel = np.linalg.norm(np.diff(ds, axis=1), axis=2)
    backtest = []
    for k in (4, 5, 6):
        lead = np.arange(1, 9-k)[None, :, None]
        rollout = history[:, k-1:k] + lead*ds[:, k-2:k-1]
        backtest.append(np.linalg.norm(rollout-history[:, k:], axis=2).mean(axis=1))
    err = np.stack(backtest, axis=1)
    sizes = np.column_stack((speed, accel, err))/cutoff
    return np.log1p(sizes).astype(np.float32), 1./(1.+sizes[:, -3:].max(axis=1))


def support_label(cv, future, valid, cutoff):
    ade, _ = errors(cv, future, valid)
    return (ade > 0.) & (ade <= cutoff)


def support_model():
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(),
                               torch.nn.Linear(32, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))


def loss(net, batch):
    x, y = batch
    return torch.nn.functional.binary_cross_entropy_with_logits(net(x).squeeze(1), y)


def recipient_proposals(v, support, paths, legal, slots):
    # Keep the value-based candidate choice within each actor.
    excess = v[..., 2:4] - .02*v[..., 4:6]
    utility = v[..., :2].min(axis=-1) - np.maximum(excess, 0.).sum(axis=-1)
    if not np.isfinite(utility).all() or not np.isfinite(support).all():
        raise ValueError('Invalid support or value')
    masked = np.where(legal, utility, -np.inf)
    family = masked.argmax(axis=1)
    merit = masked[np.arange(len(masked)), family]
    records, rows, values = [], [], []
    for record, (lo, hi) in slots.items():
        ids = np.arange(lo, hi)
        ids = ids[np.isfinite(merit[ids])]
        top = ids[support[ids] == support[ids].max()]
        row = top[merit[top].argmax()]
        records.append(record)
        rows.append(row)
        values.append(merit[row])
    rows = np.asarray(rows, dtype=np.int64)
    values = np.asarray(values)
    return dict(record=np.asarray(records, dtype='U20'), rows=rows,
                paths=paths[rows, family[rows]], scores=support[rows]*np.maximum(values, 0.))


def register():
    key, source, forecasters = base.register()
    frozen_path = base.OUT / 'freeze.json'
    frozen = old.read(frozen_path)
    if frozen['registration'] != key or len(frozen['cells']) != 63 or not frozen['held_choices_frozen']:
        raise ValueError('Require the completed factorized development experiment')
    paths = (Path(__file__), OUT / 'protocol.md', frozen_path, base.OUT / 'registration.json',
             old.OUT / 'freeze.json', Path(errors.__code__.co_filename))
    p = OUT / 'registration.json'
    base.prior.write_json(p, dict(files={str(f.resolve().relative_to(base.prior.ROOT)): base.prior.sha(f) for f in paths},
        sites=base.SITES, seeds=base.SEEDS, arms=ARMS, steps=STEPS, original_2pct=True,
        songdo_read=False, reserved_outcomes_opened=False, workers=0, threads=4), immutable=True)
    return base.prior.sha(p), source, frozen, forecasters


def fit(identity, batches):
    before = old.LOCAL
    try:
        old.LOCAL = LOCAL
        torch.manual_seed(identity['seed'])
        return old.fit(identity, batches, loss, support_model(), STEPS, old.BATCH)
    finally:
        old.LOCAL = before


def beat(**fields):
    base.prior.write_json(LOCAL / 'heartbeat.json', dict(pid=os.getpid(), **fields))
    print(fields, flush=True)


def check_match(proposals, ref, cv, offsets):
    np.testing.assert_array_equal(proposals['record'], ref['record'])
    q = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    a, b = proposals['rows'], ref['rows']
    np.testing.assert_array_equal(q[a], q[b])
    np.testing.assert_allclose(np.linalg.norm(proposals['paths']-cv[a], axis=-1).max(axis=-1),
                               np.linalg.norm(ref['paths']-cv[b], axis=-1).max(axis=-1), atol=1e-9)


def packet_proposals(z, arm):
    return {k: z[f'{arm}_{k}'] for k in ('record', 'rows', 'paths', 'scores')}


def freeze(key, source, frozen, forecasters):
    data = base.parent.load(source)
    cells = {}
    for name, cell in frozen['cells'].items():
        pair, seed, cut = cell['excluded'], cell['seed'], cell['cut']
        receipt = LOCAL / f'{name}.json'
        if receipt.exists():
            done = old.read(receipt)
            if (done['identity']['registration'] != key
                    or base.prior.sha(LOCAL / f'{name}_support.pt') != done['checkpoint']['sha256']
                    or any(base.prior.sha(LOCAL / r['file']) != r['sha256'] for r in done['packets'].values())):
                raise ValueError('Cached fit or frozen packet changed')
            cells[name] = done
            continue
        batches = []
        for donor in cell['fit_sites']:
            if donor in pair:
                raise ValueError('Donor leaks calibration or held site')
            d = data[donor]
            cv, _ = base.parent.analytic(d['history'])
            x, _ = features(d['history'], cut[0])
            y = support_label(cv, d['future'], d['valid'], cut[0])
            batches.append(tuple(map(torch.from_numpy, (x, y.astype(np.float32)))))
        identity = dict(name=f'{name}_support', registration=key, seed=seed, excluded=pair,
                        fit_sites=cell['fit_sites'], role='actor_easy_reference_support')
        net, checkpoint = fit(identity, batches)
        f = old.double_name(pair, seed)
        forecast_net = base.load_net(old.LOCAL / f'{f}.pt', forecasters['double_excluded'][f],
                                     old.motion.motion.network(17), pair)
        vname = f'{name}_factorized'
        value_net = base.load_net(base.LOCAL / f'{vname}.pt', cell['weights']['factorized'],
                                  base.model.FactorValue(), pair)
        packets = {}
        for site in pair:
            d = data[site]
            cv, ols, offsets, paths, bx, _, legal = old.menu(d, old.forecast(forecast_net, d))
            x, proxy = features(d['history'], cut[0])
            slots = base.fixed_slots(offsets, d['recording'], legal)
            with torch.inference_mode():
                value = value_net(torch.from_numpy(bx)).numpy()
                probability = net(torch.from_numpy(x)).squeeze(1).sigmoid().numpy()
            ref = cell['directions'][site]
            p = base.LOCAL / ref['file']
            if base.prior.sha(p) != ref['sha256']:
                raise ValueError('Archived comparison packet changed')
            with np.load(p, allow_pickle=False) as z:
                np.testing.assert_array_equal(cv, z['cv'])
                np.testing.assert_array_equal(ols, z['ols4'])
                ref = packet_proposals(z, ARMS[0])
            out = dict(cv=cv, ols4=ols)
            for arm, score in zip(ARMS, (None, probability, proxy), strict=True):
                prop = ref if score is None else recipient_proposals(value, score, paths, legal, slots)
                check_match(prop, ref, cv, offsets)
                out.update({f'{arm}_{k}': a for k, a in prop.items()})
            path = LOCAL / f'{name}_{site}.npz'
            packets[site] = dict(file=path.name, sha256=base.parent.indices.save_npz(path, out))
        rules = {}
        for held, cal in (pair, pair[::-1]):
            arms = {ARMS[0]: cell['calibration'][held]['arms'][ARMS[0]]}
            with np.load(LOCAL / packets[cal]['file'], allow_pickle=False) as z:
                for arm in ARMS[1:]:
                    p = packet_proposals(z, arm)
                    s = base.evaluate(p, z['cv'], z['ols4'], data[cal], cut)
                    t, n = base.calibration(p['scores'], s['dual_ok'] & (p['scores'] > 0.))
                    arms[arm] = dict(threshold=t, calibration_edits=n)
            rules[held] = dict(calibration_site=cal, arms=arms)
        done = dict(identity=identity, checkpoint=checkpoint, packets=packets, rules=rules, cut=cut)
        old.write(receipt, done)
        cells[name] = done
        beat(phase='freeze', cell=name, complete=len(cells))
    base.prior.write_json(OUT / 'freeze.json', dict(registration=key, cells=cells,
        held_choices_frozen=True, songdo_read=False), immutable=True)


def score(key, source):
    freeze_path = OUT / 'freeze.json'
    frozen = old.read(freeze_path)
    parent = old.read(base.OUT / 'readout.json')
    if (frozen['registration'] != key or len(frozen['cells']) != 63 or not frozen['held_choices_frozen']
            or parent['freeze'] != base.prior.sha(base.OUT / 'freeze.json')):
        raise ValueError('Require complete frozen choices')
    data = base.parent.load(source)
    views = []
    for name, c in frozen['cells'].items():
        for site, r in c['packets'].items():
            p = LOCAL / r['file']
            if base.prior.sha(p) != r['sha256'] or site in c['identity']['fit_sites']:
                raise ValueError('Frozen evaluation source changed')
            d = data[site]
            rule = c['rules'][site]
            arms = {}
            with np.load(p, allow_pickle=False) as z:
                cv, ols = z['cv'], z['ols4']
                support = support_label(cv, d['future'], d['valid'], c['cut'][0])
                mass = float(d['weight'] @ np.where(d['valid'], np.linalg.norm(cv-d['future'], axis=-1), 0.).sum(axis=1))
                for arm in ARMS:
                    prop = packet_proposals(z, arm)
                    s = base.evaluate(prop, cv, ols, d, c['cut'])
                    present = support[prop['rows']]
                    if (~present).sum() != s['easy_undefined']:
                        raise ValueError('Support label does not match the original easy denominator')
                    t = rule['arms'][arm]['threshold']
                    use = prop['scores'] >= t if t is not None else np.zeros(len(present), bool)
                    arms[arm] = dict(matched=len(present), supported=int(present.sum()),
                        matched_risk=int(s['risk_ok'].sum()), matched_dual=int(s['dual_ok'].sum()),
                        matched_gain_pp=float(100.*s['lower_cv'].sum()/mass),
                        calibration_edits=rule['arms'][arm]['calibration_edits'],
                        released=int(use.sum()), released_unsupported=int((use & ~present).sum()),
                        released_risk=int((use & s['risk_ok']).sum()), released_dual=int((use & s['dual_ok']).sum()))
            views.append(dict(site=site, calibration_site=rule['calibration_site'],
                              seed=c['identity']['seed'], arms=arms))
        beat(phase='score', cell=name, views=len(views))
    fields = [k for k in views[0]['arms'][ARMS[0]] if k != 'matched_gain_pp']
    summary = {arm: {k: sum(v['arms'][arm][k] for v in views) for k in fields} for arm in ARMS}
    a = summary[ARMS[0]]
    b = parent['summary'][ARMS[0]]
    for k, k0 in (('matched', 'matched_edits'), ('matched_dual', 'matched_dual'),
                  ('matched_risk', 'matched_risk'), ('released', 'calibrated_edits'),
                  ('released_risk', 'calibrated_risk'), ('released_dual', 'calibrated_dual')):
        if a[k] != b[k0]:
            raise ValueError('Archived factorized comparison did not replay')
    contrasts = {}
    draws = np.random.default_rng(682).integers(len(base.SITES), size=(3000, len(base.SITES)))
    for arm in ARMS[1:]:
        contrasts[arm] = {}
        for k in ('supported', 'matched_dual', 'matched_gain_pp'):
            x = np.array([np.mean([v['arms'][arm][k]-v['arms'][ARMS[0]][k]
                                   for v in views if v['site'] == site]) for site in base.SITES])
            contrasts[arm][k] = dict(site_difference=x.tolist(), mean=float(x.mean()),
                ci95=np.quantile(x[draws].mean(axis=1), [.025, .975]).tolist())
    old.write(OUT / 'readout.json', dict(registration=key, freeze=base.prior.sha(freeze_path),
        result_source='fresh_run', summary=summary, contrasts=contrasts, views=views,
        original_2pct_unchanged=True, independent_confirmation=False, songdo_read=False,
        reserved_outcomes_opened=False, deployment_changed=False))


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
        key, source, frozen, forecasters = register()
        if args.phase == 'freeze':
            freeze(key, source, frozen, forecasters)
        else:
            score(key, source)
