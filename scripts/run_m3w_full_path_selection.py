"""Fit a cross-fitted candidate-value head and test it on the exposed camera."""
import argparse
import fcntl
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_camera_noise_control as camera
from src.world_model import m3w_full_path_selection as policy

study, base, np, torch = camera.study, camera.base, camera.np, camera.torch
OUT = camera.OUT.parent/'camera_full_path_selection_v1'
LOCAL = camera.LOCAL.parent/OUT.name
STEPS, BATCH = camera.STEPS, camera.BATCH
SITES = study.SITES
LEGACY = '48d75793a1510d9eeeaf8f2a8aeeaf84944abbbc7ed2330e1d6e9fb8d1aad565'


def register():
    parent, cohorts, cams = camera.register()
    fpath = camera.OUT/'freeze.json'
    parent_freeze = study.read(fpath)
    if parent_freeze['registration'] != parent or not parent_freeze['no_camera_outcomes_read']:
        raise ValueError('Require prior frozen deploy-time candidates')
    initial = OUT/'registration.json'
    if base.sha(initial) != LEGACY:
        raise ValueError('Missing original completed seed-17 forecaster registration')
    fp = OUT/'registration_v2.json'
    base.immutable(fp, dict(parent_freeze=study.artifact(fpath), donors=cohorts,
        completed_first_seed_forecaster_registration=study.artifact(initial),
        repair='Selector loss receives net(x); forecaster training and paths unchanged',
        sources=list(SITES), seeds=study.SEEDS, fits_exclude_prediction_site=True,
        code=[study.artifact(Path(p)) for p in (__file__, policy.__file__, study.costs.__file__,
            camera.__file__, camera.noise.__file__, study.spline.__file__, OUT/'protocol.md')],
        donor_steps=STEPS, selector_steps=STEPS, batch=BATCH, workers=0, threads=4,
        threshold_tuning=False, original_2pct=True, external_test_reopened=False,
        independent_confirmation=False))
    return base.sha(fp), cohorts, cams['wepdtof'], parent_freeze


def fit(key, name, seed, build, draw, loss, metadata):
    path = LOCAL/f'{name}.pt'
    # The first seed completed its producers before the selector adapter failed.
    producer_key = LEGACY if 'held' in metadata and seed == study.SEEDS[0] else key
    identity = dict(registration=producer_key, seed=seed, **metadata)
    torch.manual_seed(seed)
    net = build()
    opt = torch.optim.AdamW(net.parameters(), lr=.001)
    rng = np.random.default_rng(seed+1601)
    aug = np.random.default_rng(seed+1623)
    start, trace = 0, []
    if path.exists():
        s = torch.load(path, weights_only=True, map_location='cpu')
        if s['identity'] != identity or s['step'] not in range(100, STEPS+1, 100):
            raise ValueError('Unregistered checkpoint or training progress')
        net.load_state_dict(s['net'])
        opt.load_state_dict(s['opt'])
        rng.bit_generator.state, aug.bit_generator.state = s['rng'], s['aug']
        start, trace = s['step'], s['trace']
    for i in range(start+1, STEPS+1):
        l = loss(net, *draw(rng, aug))
        opt.zero_grad(set_to_none=True)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if i % 100 == 0:
            trace.append(float(l.detach()))
            tmp = path.with_suffix('.pending')
            torch.save(dict(identity=identity, net=net.state_dict(), opt=opt.state_dict(),
                            step=i, trace=trace, rng=rng.bit_generator.state, aug=aug.bit_generator.state), tmp)
            tmp.replace(path)
            beat = dict(pid=os.getpid(), name=name, step=i, total=STEPS)
            tmp = LOCAL/'heartbeat.pending'
            tmp.write_text(study.json.dumps(beat)+'\n')
            tmp.replace(LOCAL/'heartbeat.json')
            if i == 100 or i == STEPS:
                print(beat, flush=True)
    return net.eval(), dict(checkpoint=study.artifact(path), identity=identity, trace=trace,
                            sampler=rng.bit_generator.state)


def loader(cohorts):
    out = {}
    for site in SITES:
        with np.load(study.checked(cohorts[site]), allow_pickle=False) as z:
            h, y, offset = (z[k] for k in ('history', 'future', 'offsets'))
        q = np.repeat(np.arange(len(offset)-1), np.diff(offset))
        valid = np.isfinite(y).all(2)
        out[site] = (h, np.where(valid[..., None], y, 0.), valid, q)
    return out


def batcher(rows):
    def draw(rng):
        n = rng.multinomial(BATCH, np.full(len(rows), 1./len(rows)))
        ix = [rng.integers(len(a[0]), size=k) for a, k in zip(rows, n, strict=True)]
        return [np.concatenate([a[j][i] for a, i in zip(rows, ix, strict=True)]) for j in range(len(rows[0]))]
    return draw


def spline_draw(fitting, arm):
    draw = batcher(fitting)

    def batch(rng, aug):
        h, y, v = draw(rng)
        if arm == 'noise':
            h, y = camera.noise.perturb(h, y, aug)
        h, _, y = camera.noise.normalized(h, y)
        x, scale = study.spline.designs(h)
        return (tuple(torch.from_numpy(a) for a in x['ols_spline']), torch.from_numpy(scale),
                torch.from_numpy(y.astype(np.float32)), torch.from_numpy(v))
    return batch


@torch.inference_mode()
def path(net, history):
    h, scale, _ = camera.noise.normalized(history)
    x, _ = study.spline.designs(h)
    inputs = tuple(torch.from_numpy(z) for z in x['ols_spline'])
    p = np.concatenate([net(*(z[j:j+512] for z in inputs), False).numpy().astype(float)
                        for j in range(0, len(h), 512)])
    return p*scale[:, None, None]


def save_bank(dst, values):
    if dst.exists():
        with np.load(dst, allow_pickle=False) as z:
            if set(z.files) != set(values):
                raise ValueError('Frozen array schema changed')
            for k, x in values.items():
                np.testing.assert_array_equal(z[k], x)
    else:
        tmp = dst.with_suffix('.pending.npz')
        np.savez(tmp, **values)
        tmp.replace(dst)
    return study.artifact(dst)


def fit_and_freeze(key, cohorts, c, parent):
    data = loader(cohorts)
    fitting = {s:tuple(z[v.any(1)] for z in (h,y,v)) for s,(h,y,v,_) in data.items()}
    cv_histories = [np.linalg.norm(study.method.analytic(h)[0][v.all(1)]-y[v.all(1)], axis=-1).mean(1)
                    for h,y,v,q in data.values()]
    pos = np.concatenate(cv_histories)
    cut = float(np.quantile(pos[pos > 0], .25))
    fits, predictions, heads, camera_paths = {}, {}, {}, {}
    for seed in study.SEEDS:
        examples = []
        for site, (h, y, valid, q) in data.items():
            donors = [s for s in SITES if s != site]
            cv, ols = study.method.analytic(h)
            z = dict(q=q, cv=cv, ols4=ols, median4=camera.linear_menu(h, 12)[:,2])
            records = []
            for arm in camera.ARMS:
                net, info = fit(key, f'{site}_{seed}_{arm}', seed, study.spline.Spline,
                    spline_draw([fitting[s] for s in donors], arm), study.spline.loss,
                    dict(held=site, fit_sites=donors, arm=arm))
                fits[f'{site}_{seed}_{arm}'] = info
                records.append(info)
                z[arm] = path(net, h)
            if records[0]['sampler'] != records[1]['sampler']:
                raise ValueError('Noise and clean donor rows diverged')
            predictions[f'{site}_{seed}'] = save_bank(LOCAL/f'{site}_{seed}.npz', z)
            raw = np.stack([z[k] for k in ('ols4','median4','clean','noise')], axis=1)
            both = np.concatenate(policy.banks(cv, raw, q), axis=1)
            x, scale, legal, _ = study.costs.features(h,cv,ols,both)
            target = study.costs.targets(cv,ols,both,y,valid,cut,scale)
            use = legal.any(1)
            examples.append((x[use], target[use], legal[use]))
        draw = batcher(examples)

        def batch(rng, _):
            return tuple(torch.from_numpy(x) for x in draw(rng))

        def cost(net, x, y, legal):
            return study.costs.loss(net(x), y, legal)

        net, info = fit(key, f'head_{seed}', seed, study.costs.BridgeValue, batch, cost,
                       dict(fit_sites=list(SITES), paired_predictions='held_site', arm='value'))
        heads[str(seed)] = info
        with np.load(study.checked(parent['forecasts']['wepdtof'][str(seed)]), allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'], c['q'])
            np.testing.assert_array_equal(z['ids'], c['ids'])
            cv, ols = z['cv'], z['ols4']
            raw = np.stack([z[k] for k in ('ols4','median4','clean','noise')], axis=1)
            p = {k:z[k] for k in ('cv','ols4','median4','clean','noise','q','ids')}
        for label, b in zip(('raw','equal'), policy.banks(cv, raw, c['q']), strict=True):
            x, scale, legal, _ = study.costs.features(c['history'],cv,ols,b)
            with torch.inference_mode():
                value = net(torch.from_numpy(x)).numpy()*scale[:, None, None]
            for menu, n in (('analytic', 2),('expanded', 4)):
                r = policy.choices(c['history'],c['q'],cv,b[:,:n],value[:,:n],legal[:,:n],fixed=label=='equal')
                if label=='equal':
                    policy.matched(cv,c['q'],r)
                p.update({f'{menu}_{label}_{k}':v for k,v in r.items()})
        camera_paths[str(seed)] = save_bank(LOCAL/f'camera_{seed}.npz',p)
    base.immutable(OUT/'freeze.json', dict(registration=key, predictors=fits, donor_predictions=predictions,
        value_heads=heads, camera_predictions=camera_paths, configured_updates=(len(fits)+len(heads))*STEPS,
        donor_easy_cut=cut, camera_targets_not_read=True, no_camera_selection=True))


def score(key, c):
    path = OUT/'freeze.json'
    f = study.read(path)
    if f['registration'] != key or not f['camera_targets_not_read'] or not f['no_camera_selection']:
        raise ValueError('Require frozen camera paths')
    y, v, m = camera.targets('wepdtof', c)
    if len(y)!=58 or m.sum()!=57 or v.all(1).sum()!=57:
        raise ValueError('Exposed pilot roster changed')
    cut = study.read(study.OUT/'pilot_freeze.json')['cut_px']
    values = {}
    for seed in study.SEEDS:
        with np.load(study.checked(f['camera_predictions'][str(seed)]), allow_pickle=False) as a:
            np.testing.assert_array_equal(a['q'], c['q'])
            np.testing.assert_array_equal(a['ids'], c['ids'])
            p = {k:a[k] for k in a.files if k not in ('q','ids')}
        q = c['q']
        value = study.metrics(p,y,v,q,base.PILOT,cut,paired_controls=False)
        weight = 1./np.bincount(q)[q]
        comparisons = {}
        for menu in ('analytic','expanded'):
            name = f'{menu}_equal_'
            for k in policy.FIXED[1:]:
                ref, pred = p[name+k], p[name+'joint']
                a,b = study.paired(ref,pred,y,v)
                comparisons[name+k] = (100.*np.array([weight@a.sum(1),weight@b.sum(1)])/
                                       value['reference_mass']).tolist() if value['arms'][name+'joint']['risk']['selected'] else None
        values[str(seed)] = dict(**value, paired_joint_known_lower_pp=comparisons)
        print(dict(seed=seed, raw_edits=value['arms']['expanded_raw_joint']['risk']['selected'],
            raw_dual=value['arms']['expanded_raw_joint']['dual_pass'],
            equal=value['arms']['expanded_equal_joint']['lower_vs_ols4'],
            equal_vs_independent=comparisons['expanded_equal_independent']),flush=True)
    base.immutable(OUT/'readout.json',dict(registration=key, freeze=study.artifact(path), seeds=values,
        online=len(y), stable=int(m.sum()), complete=int(v.all(1).sum()), original_2pct=True,
        independent_confirmation=False, external_test_reopened=False, deployment_changed=False, submission_ready=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('freeze','score'))
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for d in (OUT,LOCAL):
        d.mkdir(parents=True,exist_ok=True)
    with (LOCAL/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,cohorts,c,parent = register()
        if a.phase=='freeze':
            fit_and_freeze(key,cohorts,c,parent)
        else:
            score(key,c)


if __name__ == '__main__':
    main()
