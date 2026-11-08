"""Verify direct exchange choices and attribute recording pass transitions."""
from collections import Counter
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_direct_exchange as run


def choice_quality(groups, estimates, truth):
    out = {k:Counter() for k in run.ARMS}
    limits = np.r_[0,np.flatnonzero(np.diff(groups))+1,len(groups)]
    for lo,hi in zip(limits[:-1],limits[1:],strict=True):
        v, y = estimates[lo:hi],truth[lo:hi]
        good = (y[:,0]>0.) & (y[:,1:4]<=1e-12).all(1)
        safe = (v[:,1:4]<=0.).all(1)
        for name, mask in zip(run.ARMS,(np.ones(len(v),bool),safe,safe & (v[:,4]>=.75)),strict=True):
            score = np.where(mask,v[:,0],-np.inf)
            i = int(score.argmax())
            out[name]['queries'] += 1
            out[name]['oracle_possible'] += int(good.any())
            if score[i]>0.:
                out[name]['selected'] += 1
                out[name]['actual_positive'] += int(y[i,0]>0.)
                out[name]['actual_safe_positive'] += int(good[i])
    return out


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape
    from src.evaluation.m3w_selection_forecast_readout import policy

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg,_,_ = run.register()
    p = run.PUBLIC/'analysis.json'
    digest = run.prior.sha(p)
    read = json.loads(p.read_text())
    replay = json.loads((run.PUBLIC/'replay_execution.json').read_text())
    heads = json.loads((run.PUBLIC/'head_freeze.json').read_text())
    if (read['registration']!=reg or heads['registration']!=reg or
        replay['analysis_sha256']!=digest or not replay['fresh_exact_replay']):
        raise ValueError('Require the exact replayed direct exchange result')
    sources = {s:run.base.parent.site_data(s,torch,model,shape) for s in run.SITES}
    outer = run.base.shape_run.frozen.load_networks(torch,model)
    totals = {k:Counter() for k in run.ARMS}
    transitions = {k:Counter() for k in run.ARMS}
    rows = {}
    for idx,s in enumerate(run.SITES):
        d = sources[s]
        cv = d['cv']
        cuts = run.base.base.cuts_for([sources[k] for k in run.SITES if k!=s])
        rows[s] = []
        for seed in run.SEEDS:
            c = run.base.shape_run.candidate(d,outer[seed][idx],torch,model)
            _,z = run.base.parent.features(d,c['inc'])
            x,q,info,norm = run.actions(d,c,z)
            head,_,_ = run.fit(s,seed,reg,{},heads['heads'][f'{s}/{seed}'])
            est = head.predict(x)
            est[:,:4] *= norm[:,None]
            paths,counts = run.apply(cv,c['inc'],d['offsets'],q,info,est)
            x_truth,y,_ = run.actions(d,c,z,cuts[0])
            np.testing.assert_array_equal(x_truth,x)
            y[:,:4] *= norm[:,None]
            qual = choice_quality(q,est,y)
            half = cv+.5*(c['inc']-cv)

            def risk(p):
                return policy(p,cv,cv,d['y'],d['valid'],d['recordings'],
                              easy_cut=cuts[0],hard_cut=cuts[1])['recordings']

            ref = risk(half)
            changes = {}
            check = next(v['arms'] for v in read['by_location'][s] if v['seed']==seed)
            for name,pred in paths.items():
                if qual[name]['selected']!=counts[name]:
                    raise ValueError('Audit and actual choice differ')
                actual = risk(pred)
                if sum(v['supported'] for v in actual.values())!=check[name]['passes']:
                    raise ValueError('Frozen recording counts changed')
                changes[name] = Counter(run.base.transition(ref[k],v) for k,v in actual.items())
                totals[name].update(qual[name])
                transitions[name].update(changes[name])
            rows[s].append(dict(seed=seed,choice_quality=qual,recording_transitions=changes))
    run.prior.write_json(run.PUBLIC/'choice_attribution.json',dict(original_analysis_sha256=digest,
        posthoc_diagnostic=True,decisions_changed=False,choice_quality=totals,
        recording_transitions=transitions,by_location=rows),immutable=True)


if __name__=='__main__':
    main()
