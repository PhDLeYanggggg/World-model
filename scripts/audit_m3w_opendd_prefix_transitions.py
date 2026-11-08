"""Pair recording passes and support losses without refitting risk heads."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_m3w_opendd_velocity_flight_feasibility as feasibility
from scripts import run_m3w_opendd_prefix_risk as run
from src.evaluation.m3w_selection_forecast_readout import errors,policy


def transitions(d,p,masks,cut,hard):
    b = d['cv']
    use = (p!=b).any(axis=(1,2))
    orig = policy(p,b,b,d['y'],d['valid'],d['recordings'],easy_cut=cut,hard_cut=hard)['recordings']
    base,_ = errors(b,d['y'],d['valid'])
    candidate,_ = errors(p,d['y'],d['valid'])
    easy = np.isfinite(base) & (base>0.) & (base<=cut)
    harm = np.maximum(candidate-base,0.)
    out = {}
    for arm in ('local','prefix','joint'):
        alt = np.where(masks[arm][:,None,None],p,b)
        new = policy(alt,b,b,d['y'],d['valid'],d['recordings'],easy_cut=cut,hard_cut=hard)['recordings']
        total = Counter()
        for rec in orig:
            a,z = orig[rec],new[rec]
            at = d['recordings']==rec
            total['views'] += 1
            if a['supported'] and z['supported']:
                total['both_pass'] += 1
            elif a['supported']:
                total['lost_pass'] += 1
                total['lost_pass_undefined_easy'] += z['completion_easy_risk_upper'] is None
            elif z['supported']:
                total['rescued_pass'] += 1
                if feasibility.disposition(easy[at],use[at],base[at],harm[at])!='hindsight_singleton_feasible':
                    raise ValueError('Rescue contradicts fixed-candidate feasibility')
            else:
                total['both_fail'] += 1
        out[arm] = total
    return out


def main():
    import joblib
    import torch
    from src.evaluation import m3w_opendd_prefix_risk as prefix, m3w_opendd_velocity_calibration as calibrate
    from src.world_model import m3w_ols_replacement as motion, m3w_opendd_velocity_candidate as model

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run.np = run.calibration.np = run.velocity.np = run.calibration.parent.np = np
    run.calibration.torch = run.velocity.torch = run.calibration.parent.torch = torch
    run.calibration.model = run.velocity.model = model
    run.calibration.calibrate,run.calibration.parent.motion = calibrate,motion
    reg,source,frozen,_ = run.register()
    path = run.PUBLIC/'analysis.json'
    result = json.loads(path.read_text())
    if result['registration']!=reg or len(result['fits'])!=21:
        raise ValueError('Require completed registered prefix readout')
    sources = {s:run.data(s,run.prior.sha(run.prior.PUBLIC/'registration.json')) for s in run.SITES}
    total = {a:Counter() for a in ('local','prefix','joint')}
    for held,d in sources.items():
        train = [s for s in run.SITES if s!=held]
        cut = run.calibration.parent.easy_cut(sources,train)
        a = np.concatenate([sources[s]['cv_complete_ade'] for s in train])
        hard = float(np.quantile(a[a>0],.75))
        for view in source['sites'][held]:
            seed,alpha = view['seed'],view['selected']['uniform']
            forecast = run.calibration.frozen_forecast(sources,held,seed,frozen)
            p,_ = run.calibration.rollout(d,forecast,alpha,'uniform')
            ck = run.PRIVATE/f'{held}_{seed}.joblib'
            receipt = json.loads(ck.with_suffix('.json').read_text())
            if receipt['sha256']!=run.prior.sha(ck) or receipt['identity']!=dict(
                    registration=reg,held=held,seed=seed,train=train,alpha=alpha):
                raise ValueError('Risk-head identity mismatch')
            moments = joblib.load(ck).predict(run.features(d,p))*d['scale'][:,None]
            masks,_ = prefix.walk(d,p,moments)
            for arm,v in transitions(d,p,masks,cut,hard).items():
                total[arm].update(v)
    for arm,row in total.items():
        if row['both_pass']+row['rescued_pass']!=result['summary'][arm]['flight']['pass_count'] or (
                row['both_pass']+row['lost_pass']!=result['summary']['uniform']['flight']['pass_count']):
            raise ValueError('Paired recording states disagree with original evaluator')
    out = dict(scope='posthoc_paired_development',analysis_sha256=run.prior.sha(path),
               views={k:dict(v) for k,v in total.items()},official_confirmation=False)
    run.prior.write_json(run.PUBLIC/'flight_transitions.json',out,immutable=True)
    print(json.dumps(out,sort_keys=True))


if __name__=='__main__':
    main()
