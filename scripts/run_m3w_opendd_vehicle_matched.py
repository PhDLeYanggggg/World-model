"""Cross-fit matched selection on one frozen vehicle candidate, without Songdo."""

import argparse
from collections import Counter
import fcntl
from itertools import combinations
import json
from pathlib import Path
import platform
import sys

if sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_opendd_vehicle_residual as parent
from src.evaluation.m3w_camera_reference_capacity import linear_menu
from src.evaluation.m3w_interaction_frontier import aggregate_pair, denominators
from src.world_model import m3w_reference_bridge_policy as bridge
from src.world_model import m3w_vehicle_matched as matched

prior, motion, np, torch = parent.prior, parent.model, parent.np, parent.torch
OUT = parent.OUT.parent / 'opendd_vehicle_matched_v1'
LOCAL = parent.LOCAL.parent / OUT.name
SITES, SEEDS = parent.SITES, prior.SEEDS
VALUE_STEPS, SAVE, BATCH = 300, 100, 128


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    prior.write_json(path, obj, immutable=True)


def beat(phase, **extra):
    msg = dict(phase=phase, **extra)
    from os import getpid
    prior.write_json(LOCAL / 'heartbeat.json', dict(pid=getpid(), **msg))
    print(json.dumps(msg), flush=True)


def register():
    key, _ = parent.register()
    fp = parent.OUT / 'freeze.json'
    f = read(fp)
    if f['registration'] != key or f['songdo_read'] or len(f['checkpoints']) != 42:
        raise ValueError('Require fixed vehicle predictor and untouched external sources')
    refs = (Path(__file__), Path(bridge.__file__), Path(matched.__file__), Path(parent.__file__),
            parent.OUT / 'cohorts.json', fp, OUT / 'protocol.md')
    p = OUT / 'registration.json'
    write(p, dict(files={str(x.relative_to(prior.ROOT)): prior.sha(x) for x in refs},
        sites=SITES, seeds=SEEDS, crossfit_steps=parent.STEPS, value_steps=VALUE_STEPS, workers=0, threads=4,
        original_2pct=True, final_model_selected_on_held=False, songdo_read=False, confirmation_read=False))
    return prior.sha(p), key, f


def fit(identity, batches, step_fn, model, steps, batch_size):
    name = identity['name']
    p = LOCAL / f'{name}.pt'
    rng = np.random.default_rng([identity['seed'], *[SITES.index(s) for s in identity['excluded']]])
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    start = 0
    if p.exists():
        state = torch.load(p, weights_only=True, map_location='cpu')
        if state['identity'] != identity or not 0 <= state['step'] <= steps:
            raise ValueError(f'Cannot resume changed fit: {name}')
        start = state['step']
        model.load_state_dict(state['weights'])
        opt.load_state_dict(state['optim'])
        rng.bit_generator.state = state['rng']
    for step in range(start+1, steps+1):
        group = rng.integers(len(batches), size=batch_size)
        parts = []
        for i, data in enumerate(batches):
            ix = rng.integers(len(data[0]), size=int((group == i).sum()))
            parts.append(tuple(x[ix] for x in data))
        data = tuple(torch.cat(v) for v in zip(*parts, strict=True))
        opt.zero_grad()
        loss = step_fn(model, data)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss: {name}/{step}')
        loss.backward()
        opt.step()
        if step % SAVE == 0:
            tmp = p.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, weights=model.state_dict(), optim=opt.state_dict(),
                            rng=rng.bit_generator.state), tmp)
            tmp.replace(p)
            beat('fit', name=name, step=step, loss=float(loss.detach()))
    return model.eval(), dict(identity=identity, sha256=prior.sha(p), step=steps)


@torch.inference_mode()
def forecast(net, d):
    x, c = d['tensors'][:2]
    local = torch.cat([motion.predict(net, a, b) for a, b in zip(x.split(4096), c.split(4096), strict=True)]).numpy()
    return (local * d['scale'][:, None, None]) @ d['rotation'].transpose(0, 2, 1)


def menu(d, neural):
    h = d['history']
    cv, ols = parent.analytic(h)
    offsets = np.r_[0, d['query_counts'].cumsum()]
    raw = np.stack((ols, linear_menu(h, 12)[:, 2], neural), axis=1)
    paths = matched.same_radius(cv, raw, offsets)
    x, unit, legal, _ = bridge.features(h, cv, ols, paths)
    return cv, ols, offsets, paths, x, unit, legal


def calibration_cut(sites, donors):
    values = []
    for s in donors:
        d = sites[s]
        good = d['valid'].all(axis=1)
        cv, _ = parent.analytic(d['history'][good])
        val = np.linalg.norm(cv-d['future'][good], axis=-1).mean(axis=-1)
        values.append(val[val > 0.])
    return np.quantile(np.concatenate(values), [.25, .75])


def double_name(excluded, seed):
    return f'{"_".join(sorted(excluded))}_{seed}_double_excluded'


def freeze(key, data_key, parent_freeze):
    sites = parent.load(data_key)
    double, cells, preds = {}, {}, {}
    for pair in combinations(SITES, 2):
        donors = [s for s in SITES if s not in pair]
        for seed in SEEDS:
            name = double_name(pair, seed)
            identity = dict(registration=key, seed=seed, name=name, excluded=list(pair), fit_sites=donors,
                            role='five_source_vehicle_completion')
            torch.manual_seed(seed)
            net, proof = fit(identity, [sites[s]['tensors'] for s in donors],
                lambda net, z: motion.loss(motion.predict(net, z[0], z[1]), *z[1:], completion=True),
                motion.motion.network(17), parent.STEPS, BATCH)
            double[name] = proof
            for s in pair:
                preds[name, s] = forecast(net, sites[s])
    for held in SITES:
        d = sites[held]
        for seed in SEEDS:
            name = f'{held}_{seed}_value'
            targets = []
            for s in SITES:
                if s == held:
                    continue
                tag = double_name((held, s), seed)
                src = sites[s]
                donors = double[tag]['identity']['fit_sites']
                if held in donors or s in donors:
                    raise ValueError('Cost target predicted using its source or the outer held source')
                cv, ols, _, paths, x, unit, legal = menu(src, preds[tag, s])
                cut = calibration_cut(sites, donors)
                y = bridge.targets(cv, ols, paths, src['future'], src['valid'], cut[0], unit)
                if not np.isfinite(y).all():
                    raise ValueError('Nonfinite training cost target')
                targets.append(tuple(map(torch.from_numpy, (x[legal], y[legal]))))
            torch.manual_seed(seed)
            identity = dict(registration=key, seed=seed, name=name, excluded=[held], fit_sites=[s for s in SITES if s != held],
                            role='crossfit_vehicle_action_costs')
            net, proof = fit(identity, targets, lambda m, z: bridge.loss(m(z[0]), z[1], torch.ones(len(z[0]), dtype=torch.bool)),
                             bridge.BridgeValue(), VALUE_STEPS, BATCH)
            outer = f'{held}_{seed}_completion'
            chk = parent.LOCAL / f'{outer}.pt'
            orig = parent_freeze['checkpoints'][outer]
            if orig['sha256'] != prior.sha(chk) or held in orig['identity']['donors']:
                raise ValueError('Outer prediction does not exclude held source')
            state = torch.load(chk, weights_only=True, map_location='cpu')
            veh = motion.motion.network(17).eval()
            veh.load_state_dict(state['weights'])
            cv, ols, offsets, paths, x, unit, _ = menu(d, forecast(veh, d))
            with torch.inference_mode():
                value = net(torch.from_numpy(x)).numpy() * unit[:, None, None]
            chosen = matched.choose(cv, paths, value, offsets)
            p = LOCAL / f'{held}_{seed}_paths.npz'
            sha = parent.indices.save_npz(p, dict(cv=cv, ols4=ols, offsets=offsets, **chosen))
            cells[f'{held}_{seed}'] = dict(value=proof, outer_forecaster=orig['sha256'], sha256=sha)
            beat('freeze', site=held, seed=seed, edits=int((chosen['joint'] != cv).any(axis=(1, 2)).sum()))
    write(OUT / 'freeze.json', dict(registration=key, double_excluded=double, cells=cells, songdo_read=False,
        held_paths_frozen_before_score=True, forecast_updates=len(double)*parent.STEPS, value_updates=len(cells)*VALUE_STEPS))


def compare(pred, ref, cv, y, valid, q, w):
    mass = float(w @ np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(axis=1))
    if mass <= 0.:
        return None
    z = parent.paired(ref, pred, y, valid)[1]
    den = denominators(q, valid)
    pair = aggregate_pair(pred, ref, y, valid, q, den)
    return dict(lower_pp=float(100. * (w @ z.sum(axis=1))/mass), relative_lower_native=float(pair[1]) if den[1] else None)


def score(key, data_key):
    p = OUT / 'freeze.json'
    f = read(p)
    if f['registration'] != key or len(f['cells']) != 21 or not f['held_paths_frozen_before_score']:
        raise ValueError('Require the registered forecasts')
    sites = parent.load(data_key)
    results = {}
    for site, d in sites.items():
        cut = calibration_cut(sites, [s for s in SITES if s != site])
        names = d['recording']
        _, rec_index = np.unique(names, return_inverse=True)
        q = np.repeat(np.arange(len(d['query_counts'])), d['query_counts'])
        results[site] = {}
        for seed in SEEDS:
            tag = f'{site}_{seed}'
            r = LOCAL / f'{tag}_paths.npz'
            if prior.sha(r) != f['cells'][tag]['sha256']:
                raise ValueError('Frozen path hash changed')
            with np.load(r, allow_pickle=False) as z:
                a = {k: z[k] for k in z.files}
            cv, ols = a['cv'], a['ols4']
            np.testing.assert_array_equal(np.diff(a['offsets']), d['query_counts'])
            arms, paired = {}, {}
            for arm in matched.ARMS:
                arms[arm] = parent.metric(a[arm], cv, ols, d, cut)
            active = (a['joint'] != cv).any(axis=(1, 2))
            use = np.isin(rec_index, np.unique(rec_index[active]))
            for ref in matched.ARMS[1:]:
                paired[ref] = compare(a['joint'][use], a[ref][use], cv[use], d['future'][use], d['valid'][use],
                                      q[use], d['weight'][use]) if use.any() else None
            results[site][str(seed)] = dict(arms=arms, paired_on_active_recordings=paired,
                edits=int(active.sum()), active_recordings=int(len(np.unique(rec_index[active]))))
            beat('score', site=site, seed=seed, edits=int(active.sum()))
    rows = [r for s in results.values() for r in s.values()]
    summary = {a: dict(risk_passes=sum(r['arms'][a]['risk_passes'] for r in rows),
                       recording_seed_views=sum(r['arms'][a]['recordings'] for r in rows),
                       conservative_cv_gain_pp=float(np.mean([r['arms'][a]['cv_gain_known_lower_pp'][1] for r in rows])))
               for a in matched.ARMS}
    def ci(ref, metric):
        x = []
        for s in results.values():
            available = [z['paired_on_active_recordings'][ref] for z in s.values()]
            vals = [v[metric] for v in available if v is not None and v[metric] is not None]
            if vals:
                x.append(float(np.mean(vals)))
        n = len(x)
        b = np.random.default_rng(3183).choice(x, (3000, n)).mean(axis=1) if n >= 2 else None
        return dict(sites=n, mean=float(np.mean(x)) if n else None,
                    ci95=np.quantile(b, [.025, .975]).tolist() if b is not None else None)
    contrasts = {a: {m: ci(a, m) for m in ('lower_pp', 'relative_lower_native')} for a in matched.ARMS[1:]}
    result = dict(registration=key, freeze=prior.sha(p), results=results, summary=summary, contrasts=contrasts,
        edits_per_arm=sum(r['edits'] for r in rows), active_recording_seed_views=sum(r['active_recordings'] for r in rows),
        result_source='fresh_run', original_2pct_unchanged=True, songdo_read=False, independent_confirmation=False,
        reserved_outcomes_opened=False, deployment_changed=False, submission_ready=False)
    write(OUT / 'readout.json', result)
    print(json.dumps({k: result[k] for k in ('summary', 'contrasts', 'edits_per_arm', 'active_recording_seed_views')}, indent=2), flush=True)


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
        key, data_key, f = register()
        if a.phase == 'freeze':
            freeze(key, data_key, f)
        else:
            score(key, data_key)
