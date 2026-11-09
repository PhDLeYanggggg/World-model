"""Change only the risk estimate; do not relax the original risk rule."""
import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_full_path_selection as run
from src.world_model import m3w_signed_bridge_risk as signed

study, base, np, torch = run.study, run.base, run.np, run.torch


def register():
    key, cohorts, c, _ = run.register()
    parent = run.OUT/'freeze.json'
    f = study.read(parent)
    if f['registration'] != key or not f['camera_targets_not_read']:
        raise ValueError('Require frozen donor and camera predictions')
    p = run.OUT/'signed_registration.json'
    base.immutable(p, dict(parent=study.artifact(parent), code=study.artifact(Path(__file__)),
        model=study.artifact(Path(signed.__file__)), frozen_gain_head=True, donor_only=True,
        original_2pct=True, threshold_tuning=False, no_external_reopening=True,
        experiment='direct signed excess in place of positive harm minus positive reference predictions'))
    return base.sha(p), cohorts, c, f


def donor_examples(f, data, seed):
    for site, (h,y,v,q) in data.items():
        with np.load(study.checked(f['donor_predictions'][f'{site}_{seed}']),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'],q)
            cv,ols = z['cv'],z['ols4']
            b = np.stack([z[k] for k in ('ols4','median4','clean','noise')],axis=1)
        raw = np.concatenate(run.policy.banks(cv,b,q),axis=1)
        x,scale,legal,_ = study.costs.features(h,cv,ols,raw)
        target = signed.signed_target(study.costs.targets(cv,ols,raw,y,v,f['donor_easy_cut'],scale))
        on = legal.any(1)
        yield x[on],target[on],legal[on]


def freeze(key, cohorts, c, f):
    data = run.loader(cohorts)
    weights, files = {}, {}
    for seed in study.SEEDS:
        draw = run.batcher(list(donor_examples(f,data,seed)))

        def batch(rng,_):
            return tuple(torch.from_numpy(x) for x in draw(rng))

        def cost(net,x,y,legal):
            p = net(x)[legal]
            loss = torch.nn.functional.smooth_l1_loss(p,y[legal],beta=.01,reduction='none')
            return (loss*p.new_tensor([1.,2.])).mean()

        net, info = run.fit(key,f'signed_{seed}',seed,signed.SignedRisk,batch,cost,
                            dict(fit_sites=list(run.SITES),arm='signed_excess'))
        weights[str(seed)] = info
        old = f['value_heads'][str(seed)]
        ck = torch.load(study.checked(old['checkpoint']),map_location='cpu',weights_only=True)
        if ck['identity'] != old['identity'] or ck['step'] != run.STEPS:
            raise ValueError('Frozen gain weights changed')
        gain = study.costs.BridgeValue()
        gain.load_state_dict(ck['net'])
        gain.eval()
        with np.load(study.checked(f['camera_predictions'][str(seed)]),allow_pickle=False) as z:
            p = {k:z[k] for k in z.files}
        cv,ols,q = p['cv'],p['ols4'],p['q']
        np.testing.assert_array_equal(q,c['q'])
        np.testing.assert_array_equal(p['ids'],c['ids'])
        raw = np.stack([p[k] for k in ('ols4','median4','clean','noise')],axis=1)
        for label,b in zip(('raw','equal'),run.policy.banks(cv,raw,q),strict=True):
            x,scale,legal,_ = study.costs.features(c['history'],cv,ols,b)
            t = torch.from_numpy(x)
            with torch.inference_mode():
                value = gain(t).numpy()*scale[:,None,None]
                value[...,2:4] = net(t).numpy()*scale[:,None,None]
                value[...,4:6] = 0.
            a = run.policy.choices(c['history'],q,cv,b,value,legal,fixed=label=='equal')
            if label=='equal':
                run.policy.matched(cv,q,dict(original=p['expanded_equal_joint'],repaired=a['joint']))
            p[f'signed_{label}'] = a['joint']
        files[str(seed)] = run.save_bank(run.LOCAL/f'signed_camera_{seed}.npz',p)
    base.immutable(run.OUT/'signed_freeze.json',dict(registration=key,heads=weights,forecasts=files,
        steps=3*run.STEPS,camera_outcomes_read=False,gain_weights_unchanged=True))


def score(key,c):
    p = run.OUT/'signed_freeze.json'
    f = study.read(p)
    if f['registration'] != key or f['camera_outcomes_read'] or not f['gain_weights_unchanged']:
        raise ValueError('Require frozen signed policies')
    y,v,m = run.camera.targets('wepdtof',c)
    if len(y)!=58 or m.sum()!=57:
        raise ValueError('Development camera roster changed')
    cut = study.read(study.OUT/'pilot_freeze.json')['cut_px']
    result = {}
    for seed in study.SEEDS:
        with np.load(study.checked(f['forecasts'][str(seed)]),allow_pickle=False) as z:
            np.testing.assert_array_equal(z['q'],c['q'])
            np.testing.assert_array_equal(z['ids'],c['ids'])
            arms = {k:z[k] for k in ('cv','ols4','expanded_raw_joint','expanded_equal_joint','signed_raw','signed_equal')}
        r = study.metrics(arms,y,v,c['q'],base.PILOT,cut,paired_controls=False)['arms']
        result[str(seed)] = r
        print(dict(seed=seed,risk=r['signed_raw']['risk'],ols=r['signed_raw']['lower_vs_ols4'],
                   dual=r['signed_raw']['dual_pass']),flush=True)
    base.immutable(run.OUT/'signed_readout.json',dict(registration=key,freeze=study.artifact(p),
        seeds=result,original_2pct=True,development_only=True,independent_confirmation=False,
        external_test_reopened=False,deployment_changed=False))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('phase',choices=('freeze','score'))
    args = a.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (run.LOCAL/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,cohorts,c,f = register()
        if args.phase=='freeze':
            freeze(key,cohorts,c,f)
        else:
            score(key,c)
