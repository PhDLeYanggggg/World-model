"""Train causal velocity mixtures on OpenDD and score one exposed camera."""

import fcntl
import os
from pathlib import Path
import platform
import sys

if __name__ == '__main__' and sys.platform == 'darwin' and platform.machine() != 'arm64':
    raise SystemExit('Use the native arm64 .venv-pytorch interpreter before importing Torch')

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_camera_noise_control as old
from src.world_model import m3w_camera_velocity_bank as velocity

study, base, np, torch = old.study, old.base, old.np, old.torch
OUT = study.OUT.parent/'camera_velocity_mixer_v1'
LOCAL = study.PRIVATE.parent/OUT.name
ARMS = ('clean', 'noise', 'noise_preserve')
STEPS, BATCH = 800, 128


def register():
    parent, windows = old.pilot.pilot()
    donor = study.read(old.OUT/'registration.json')
    control = old.OUT/'freeze.json'
    frozen = study.read(control)
    cutpath = study.OUT/'pilot_freeze.json'
    cut = study.read(cutpath)
    if (frozen['registration'] != base.sha(old.OUT/'registration.json') or not frozen['no_camera_outcomes_read']
            or frozen['selection'] or cut['external_geometry_read'] or cut['pilot_camera_sha256'] != base.sha(parent)
            or donor['independent_confirmation'] or donor['external_test_reopened'] or not donor['original_2pct']
            or set(donor['donors']) != set(study.SITES)):
        raise ValueError('Expected the original donor and exposed camera assets')
    registration = OUT/'registration.json'
    base.immutable(registration, dict(source=base.PILOT, arms=ARMS, seeds=study.SEEDS, steps=STEPS,
        batch=BATCH, threads=4, workers=0, exposures='open development pilot only; no model or threshold selected here',
        assets=[study.artifact(p) for p in (Path(__file__), Path(velocity.__file__), Path(old.noise.__file__),
                 Path(old.__file__), old.OUT/'registration.json', control, cutpath, parent, OUT/'protocol.md')],
        donors=donor['donors'], held_outcomes_accessed=False, original_2pct=True))
    return base.sha(registration), donor['donors'], frozen['forecasts']['wepdtof'], windows, cut['cut_px']


def fit(key, seed, arm, donors):
    path = LOCAL/f'{arm}_{seed}.pt'
    identity = dict(registration=key, seed=seed, arm=arm)
    torch.manual_seed(seed)
    net = velocity.VelocityMixer()
    opt = torch.optim.AdamW(net.parameters(), lr=.002)
    rng = np.random.default_rng(seed+1601)
    aug = np.random.default_rng(seed+1623)
    start, trace = 0, []
    if path.exists():
        saved = torch.load(path, weights_only=True, map_location='cpu')
        if saved['identity'] != identity or saved['step'] not in range(100, STEPS+1, 100):
            raise ValueError('Training checkpoint differs from registration')
        net.load_state_dict(saved['net'])
        opt.load_state_dict(saved['opt'])
        start, trace = saved['step'], saved['trace']
        rng.bit_generator.state, aug.bit_generator.state = saved['rng'], saved['aug']
    for step in range(start+1, STEPS+1):
        take = rng.multinomial(BATCH, np.full(len(donors), 1./len(donors)))
        idx = [rng.integers(len(d[0]), size=k) for d, k in zip(donors, take, strict=True)]
        h, y, valid = [np.concatenate([d[j][i] for d, i in zip(donors, idx, strict=True)]) for j in range(3)]
        if arm != 'clean':
            h, y = old.noise.perturb(h, y, aug)
        x, v, scale = velocity.features(h)
        v = torch.from_numpy((v/scale[:, None, None]).astype(np.float32))
        y = torch.from_numpy(np.where(valid[..., None], y/scale[:, None, None], 0.).astype(np.float32))
        predict, _ = net(torch.from_numpy(x), v)
        loss, mae, guard = velocity.loss(predict, v[:, 0], y, torch.from_numpy(valid), preserve=arm.endswith('preserve'))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if step % 100 == 0:
            trace.append(dict(step=step, loss=float(loss.detach()), ade=float(mae.detach()), guard=float(guard.detach())))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=identity, step=step, net=net.state_dict(), opt=opt.state_dict(),
                            rng=rng.bit_generator.state, aug=aug.bit_generator.state, trace=trace), tmp)
            tmp.replace(path)
            heartbeat = dict(pid=os.getpid(), **identity, step=step, total=STEPS, **{k: trace[-1][k] for k in ('loss', 'ade', 'guard')})
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(heartbeat)+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            print(heartbeat, flush=True)
    return net.eval(), dict(checkpoint=study.artifact(path), trace=trace, sampler=rng.bit_generator.state,
                           augmentation_sampler=aug.bit_generator.state)


def forecast(key, donors, controls, h, q, ids):
    data = old.training_data(donors)
    x, v, _ = velocity.features(h)
    t = np.arange(1., 13.)[None, :, None]
    forecasts, fits = {}, {}
    for seed in study.SEEDS:
        with np.load(study.checked(controls[str(seed)]), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'], q)
            np.testing.assert_array_equal(z['ids'], ids)
            pred = dict(q=q, ids=ids, spline_clean=z['clean'], spline_noise=z['noise'])
            for k, path in zip(velocity.NAMES, v.transpose(1, 0, 2), strict=True):
                pred[k] = path[:, None]*t
            for k in ('cv', 'ols4', 'median4'):
                np.testing.assert_allclose(pred[k], z[k], rtol=1e-10, atol=1e-10)
        for arm in ARMS:
            net, fits[f'{arm}_{seed}'] = fit(key, seed, arm, data)
            with torch.inference_mode():
                _, w = net(torch.from_numpy(x), torch.from_numpy(v.astype(np.float32)))
            weight = w.numpy().astype(float)
            weight /= weight.sum(1, keepdims=True)
            pred[arm] = np.einsum('nk,nkd->nd', weight, v)[:, None]*t
            pred[f'{arm}_weights'] = weight
        if any(fits[f'{arm}_{seed}']['sampler'] != fits[f'clean_{seed}']['sampler'] for arm in ARMS):
            raise ValueError('Training sample draws differ across controls')
        if fits[f'noise_{seed}']['augmentation_sampler'] != fits[f'noise_preserve_{seed}']['augmentation_sampler']:
            raise ValueError('Perturbation differs across the two noisy controls')
        p = LOCAL/f'prediction_{seed}.npz'
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                if set(z.files) != set(pred):
                    raise ValueError('Frozen forecast keys changed')
                for k, a in pred.items():
                    np.testing.assert_array_equal(a, z[k])
        else:
            tmp = p.with_suffix('.pending.npz')
            np.savez(tmp, **pred)
            tmp.replace(p)
        forecasts[str(seed)] = study.artifact(p)
    base.immutable(OUT/'freeze.json', dict(registration=key, forecasts=forecasts, fits=fits,
        updates=STEPS*len(ARMS)*len(study.SEEDS), development_outcomes_not_read_during_fit=True,
        fixed_final_step=True, test_access=False))


def score(key, windows, cut, h, q, ids):
    file = OUT/'freeze.json'
    frozen = study.read(file)
    if frozen['registration'] != key or not frozen['development_outcomes_not_read_during_fit'] or frozen['test_access']:
        raise ValueError('Require all forecasters frozen before development scoring')
    y, valid, matches, _ = study.outcomes(windows, h, q, ids, study.references(base.PILOT))
    if (len(h), int(matches.sum()), int(valid.all(1).sum())) != (58, 57, 57):
        raise ValueError('Pilot roster differs from original camera evidence')
    results = {}
    for seed, artifact in frozen['forecasts'].items():
        with np.load(study.checked(artifact), allow_pickle=False) as z:
            arms = {k: z[k] for k in (*velocity.NAMES, 'spline_clean', 'spline_noise', *ARMS)}
            result = study.metrics(arms, y, valid, q, base.PILOT, cut, paired_controls=False)['arms']
            results[seed] = {k: {f: r[f] for f in ('ade_px', 'fde_px', 'p95_ade_px', 'known_vs_ols4', 'lower_vs_ols4',
                                                   'lower_vs_cv', 'edit_rate', 'dual_pass')} | dict(risk={
                f: r['risk'][f] for f in ('supported', 'completion_all_risk_upper', 'completion_easy_risk_upper',
                                          'known_easy_net_degradation')}) for k, r in result.items()}
            for arm in ARMS:
                results[seed][arm]['mean_velocity_weights'] = z[f'{arm}_weights'].mean(0).tolist()
        print(dict(seed=seed, ade={k: v['ade_px'] for k, v in results[seed].items()}), flush=True)
    base.immutable(OUT/'readout.json', dict(freeze_sha256=base.sha(file), rows=58, complete_futures=57, development_sources=1,
        seeds=results, forecast_grid=dict(observed=8, predicted=12, coordinate='pixel'), original_2pct=True,
        deployment_changed=False, external_test_reopened=False, independent_confirmation=False, submission_ready=False))


if __name__ == '__main__':
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        key, donors, controls, windows, cut = register()
        h, q, ids = study.camera_arrays(windows, pilot=True)
        forecast(key, donors, controls, h, q, ids)
        score(key, windows, cut, h, q, ids)
