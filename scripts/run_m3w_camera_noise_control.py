"""Matched donor training and development-only camera scoring."""
import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_m3w_wepdtof_camera_capacity as pilot
from scripts import run_m3w_acme_gmu_camera as gmu
from scripts import run_m3w_acme_held_camera_probe as held
from src.evaluation.m3w_camera_reference_capacity import linear_menu
from src.world_model import m3w_sensor_noise_control as noise

study, base, np, torch = pilot.study, pilot.base, pilot.np, pilot.torch
OUT = study.OUT.parent/'camera_noise_control_v1'
LOCAL = study.PRIVATE.parent/OUT.name
ARMS = ('clean', 'noise')
STEPS, BATCH = 800, 128


def cameras():
    parent, windows = pilot.pilot()
    h, q, ids = study.camera_arrays(windows, pilot=True)
    streams = dict(wepdtof=dict(history=h, q=q, ids=ids, windows=windows, parent=parent))
    spec_path = gmu.PUBLIC/'registration.json'
    cmu_path = held.PUBLIC/'registration.json'
    cmu = study.read(cmu_path)['sites']['cmu']
    spec = study.read(spec_path)
    for name, p, data, factor in (
        ('gmu', spec_path, spec, gmu.CAMERA_FACTOR), ('cmu', cmu_path, cmu, cmu['factor'])
    ):
        source, rows, queries, ids, assets = [], [], [], [], []
        for j, start in enumerate(data['starts']):
            f = (gmu.PRIVATE/f'window_{start:05}.json' if name == 'gmu' else held.PRIVATE/f'cmu_{start:05}.json')
            saved = study.read(f)
            marker = 'registration' if name == 'gmu' else 'registration_sha256'
            if saved[marker] != base.sha(p):
                raise ValueError('ACME camera input does not match registration')
            frames = held.reassociate(held.association_frames(saved['frames'], start, factor))
            d = [{int(r[0]): gmu.feet(np.asarray(r[None, 1:5]))[0] for r in
                  np.asarray(f['rows'], dtype=float).reshape(-1, 7)} for f in frames]
            for k in sorted(set.intersection(*(set(z) for z in d))):
                rows.append([z[k] for z in d])
                queries.append(j)
                ids.append(k)
            assets.append(study.artifact(f))
            source.append((start, frames))
        streams[name] = dict(history=np.asarray(rows).reshape(-1, 8, 2), q=np.asarray(queries, dtype=int),
                             ids=np.asarray(ids, dtype=int), windows=source, assets=assets, spec=data, parent=p)
    return streams


def register():
    _, _, cohorts = study.register()
    streams = cameras()
    path = OUT/'registration.json'
    base.immutable(path, dict(code=[study.artifact(Path(p)) for p in (
            __file__, noise.__file__, study.spline.__file__, gmu.__file__, held.__file__,
            pilot.__file__, study.__file__, OUT/'protocol.md')],
        donors=cohorts, cameras={k:dict(parent=study.artifact(r['parent']), window_assets=r.get('assets', []),
            windows=len(r['windows']), online=len(r['history'])) for k, r in streams.items()},
        seeds=study.SEEDS, steps=STEPS, batch=BATCH, workers=0, threads=4,
        independent_confirmation=False, external_test_reopened=False, original_2pct=True))
    return base.sha(path), cohorts, streams


def training_data(cohorts):
    data = []
    for source in study.SITES:
        with np.load(study.checked(cohorts[source]), allow_pickle=False) as a:
            h, y = a['history'], a['future']
        valid = np.isfinite(y).all(2)
        take = valid.any(1)
        if not take.any():
            raise ValueError('Empty donor')
        data.append((h[take], y[take], valid[take]))
    return data


def fit(key, seed, arm, data):
    p = LOCAL/f'{arm}_{seed}.pt'
    identity = dict(registration=key, seed=seed, arm=arm)
    torch.manual_seed(seed)
    net = study.spline.Spline()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+1601)
    aug = np.random.default_rng(seed+1623)
    start, trace = 0, []
    if p.exists():
        ck = torch.load(p, map_location='cpu', weights_only=True)
        if ck['identity'] != identity or ck['step'] not in range(100, STEPS+1, 100):
            raise ValueError('Checkpoint identity mismatch')
        net.load_state_dict(ck['net'])
        opt.load_state_dict(ck['opt'])
        start, trace = ck['step'], ck['trace']
        rng.bit_generator.state, aug.bit_generator.state = ck['rng'], ck['aug']
    for i in range(start+1, STEPS+1):
        n = rng.multinomial(BATCH, np.full(len(data), 1./len(data)))
        at = [rng.integers(len(d[0]), size=k) for d, k in zip(data, n, strict=True)]
        h, y, valid = [np.concatenate([d[j][idx] for d, idx in zip(data, at, strict=True)]) for j in range(3)]
        if arm == 'noise':
            h, y = noise.perturb(h, y, aug)
        h, _, y = noise.normalized(h, y)
        x, scale = study.spline.designs(h)
        loss = study.spline.loss(net, tuple(torch.from_numpy(z) for z in x['ols_spline']),
            torch.from_numpy(scale), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(valid))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if i%100 == 0:
            trace.append(float(loss.detach()))
            tmp = p.with_suffix('.pending')
            torch.save(dict(identity=identity, step=i, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, aug=aug.bit_generator.state, trace=trace), tmp)
            tmp.replace(p)
            beat = dict(pid=os.getpid(), seed=seed, arm=arm, step=i, total=STEPS)
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(beat)+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            print(beat, flush=True)
    return net.eval(), dict(checkpoint=study.artifact(p), identity=identity, trace=trace, final_sampler=rng.bit_generator.state)


def freeze(key, cohorts, streams):
    data = training_data(cohorts)
    fits, by_source = {}, {s: {} for s in streams}
    for seed in study.SEEDS:
        models = {}
        for arm in ARMS:
            models[arm], fits[f'{seed}_{arm}'] = fit(key, seed, arm, data)
        if fits[f'{seed}_clean']['final_sampler'] != fits[f'{seed}_noise']['final_sampler']:
            raise ValueError('Paired training rows differ')
        for site, c in streams.items():
            h = c['history']
            normalized, scale, _ = noise.normalized(h)
            design, _ = study.spline.designs(normalized)
            x = tuple(torch.from_numpy(a) for a in design['ols_spline'])
            cv, ols4 = study.method.analytic(h)
            z = dict(cv=cv, ols4=ols4, median4=linear_menu(h, 12)[:, 2], q=c['q'], ids=c['ids'])
            with torch.inference_mode():
                for arm, net in models.items():
                    z[arm] = net(*x, False).numpy().astype(float)*scale[:, None, None]
            p = LOCAL/f'{site}_{seed}.npz'
            if p.exists():
                with np.load(p, allow_pickle=False) as previous:
                    if set(z) != set(previous.files):
                        raise ValueError('Frozen model bank changed')
                    for k in z:
                        np.testing.assert_array_equal(z[k], previous[k])
            else:
                tmp = p.with_suffix('.pending.npz')
                np.savez(tmp, **z)
                tmp.replace(p)
            by_source[site][str(seed)] = study.artifact(p)
    base.immutable(OUT/'freeze.json', dict(registration=key, fits=fits, forecasts=by_source,
        fit_steps=STEPS*len(study.SEEDS)*len(ARMS), no_camera_outcomes_read=True, selection=False))


def targets(site, c):
    h, q, ids = (c[k] for k in ('history', 'q', 'ids'))
    if site == 'wepdtof':
        return study.outcomes(c['windows'], h, q, ids, study.references(base.PILOT))[:3]
    if site == 'gmu':
        if base.sha(gmu.LABELS) != c['spec']['label_sha256']:
            raise ValueError('GMU labels changed')
        labels = gmu.label_tables()
    else:
        labels = held.label_tables(c['spec'])
    y = np.zeros((len(h), 12, 2))
    valid, matched = np.zeros((len(h), 12), bool), np.zeros(len(h), bool)
    for j, (start, frames) in enumerate(c['windows']):
        maps = []
        for t, f in enumerate(frames):
            tick = start+t*gmu.STRIDE
            points = [(lab, track[tick]) for lab, track in labels.items() if tick in track]
            maps.append(set(gmu.associate_labels(np.asarray(f['rows']).reshape(-1, 7), points)))
        match = dict(set.intersection(*maps))
        for k in np.flatnonzero(q == j):
            if ids[k] not in match:
                continue
            matched[k] = True
            track = labels[match[ids[k]]]
            for t in range(12):
                v = track.get(start+(8+t)*gmu.STRIDE)
                if v is not None:
                    y[k, t] = v-h[k, -1]
                    valid[k, t] = True
    return y, valid, matched


def score(key, streams):
    f = study.read(OUT/'freeze.json')
    if f['registration'] != key or not f['no_camera_outcomes_read']:
        raise ValueError('Require frozen camera predictions')
    rows = {}
    for site, c in streams.items():
        y, v, m = targets(site, c)
        if int(m.sum()) != {'wepdtof': 57, 'gmu': 8, 'cmu': 8}[site]:
            raise ValueError('Stable-match counts disagree with the earlier camera diagnostics')
        seeds = {}
        for seed in study.SEEDS:
            with np.load(study.checked(f['forecasts'][site][str(seed)]), allow_pickle=False) as a:
                np.testing.assert_array_equal(a['q'], c['q'])
                np.testing.assert_array_equal(a['ids'], c['ids'])
                p = {k:a[k] for k in ('cv', 'ols4', 'median4', *ARMS)}
            q = c['q']
            w = 1./np.bincount(q)[q]
            ref = float(w @ np.where(v, np.linalg.norm(p['cv']-y, axis=-1), 0.).sum(1))
            comp = {}
            for name in ('clean', 'cv', 'ols4', 'median4'):
                g, lower = study.paired(p[name], p['noise'], y, v)
                comp[name] = (100.*np.array([w@g.sum(1), w@lower.sum(1)])/ref).tolist()
            known = v.any(1)
            ade = {k:float((np.where(v, np.linalg.norm(x-y, axis=-1), 0.).sum(1)[known]/
                            v[known].sum(1)).mean()) for k, x in p.items()}
            end = v[:, -1]
            fde = {k:float(np.linalg.norm(x[end, -1]-y[end, -1], axis=-1).mean()) for k, x in p.items()}
            cut = study.read(study.OUT/'pilot_freeze.json')['cut_px'] if site == 'wepdtof' else None
            risk = {k:study.policy(x,p['cv'],p['cv'],y,v,np.repeat(site,len(q)),easy_cut=cut[0],
                     hard_cut=cut[1])['recordings'][site] for k,x in p.items()} if cut else None
            seeds[str(seed)] = dict(known_ade_px=ade, available_endpoint_fde_px=fde,
                noise_gain_known_lower_pp=comp, original_risk=risk)
        rows[site] = dict(online=len(v), stable=int(m.sum()), complete=int(v.all(1).sum()),
            known_steps=int(v.sum()), missing_rows=int((~v.any(1)).sum()), seeds=seeds)
        print(dict(site=site, stable=int(m.sum()), complete=int(v.all(1).sum()), noise_minus_clean_lower_pp=[
            r['noise_gain_known_lower_pp']['clean'][1] for r in seeds.values()]), flush=True)
    base.immutable(OUT/'readout.json', dict(registration=key, freeze=study.artifact(OUT/'freeze.json'),
        cameras=rows, exposed_development_only=True, original_2pct=True, independent_confirmation=False,
        external_test_reopened=False, deployment_changed=False, submission_ready=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze','score'))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, cohorts, streams = register()
        if args.phase == 'freeze':
            freeze(key, cohorts, streams)
        else:
            score(key, streams)


if __name__ == '__main__':
    main()
