"""Contrast sparse allocation with spreading the same query movement budget."""
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_causal_amplitude as run


def spread(cv,incumbent,joint,offsets):
    r = np.linalg.norm(incumbent-cv,axis=-1).max(1)
    used = np.linalg.norm(joint-cv,axis=-1).max(1)
    total = np.add.reduceat(r,offsets[:-1])
    budget = np.add.reduceat(used,offsets[:-1])
    if np.any(budget>total+1e-12):
        raise ValueError('Joint movement exceeds incumbent capacity')
    a = np.divide(budget,total,out=np.zeros_like(total),where=total>0).clip(0.,1.)
    result = cv+np.repeat(a,np.diff(offsets))[:,None,None]*(incumbent-cv)
    actual = np.add.reduceat(np.linalg.norm(result-cv,axis=-1).max(1),offsets[:-1])
    np.testing.assert_allclose(actual,budget,rtol=1e-7,atol=1e-14)
    return result


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, _ = run.register()
    p = run.PUBLIC/'analysis.json'
    digest = run.prior.sha(p)
    original = json.loads(p.read_text())
    replay = json.loads((run.PUBLIC/'replay_execution.json').read_text())
    freeze = json.loads((run.PUBLIC/'head_freeze.json').read_text())
    if (original['registration']!=reg or not original['radius_multiset_exact']
        or freeze['registration']!=reg or replay['analysis_sha256']!=digest or not replay['fresh_exact_replay']):
        raise ValueError('Require the replayed original study')
    src = {s:run.site_data(s,torch,model,shape) for s in run.SITES}
    outer = run.shape_run.frozen.load_networks(torch,model)
    rows, paired = {}, {}
    for i,s in enumerate(run.SITES):
        d = src[s]
        cv = d['cv']
        rows[s],paired[s] = [],[]
        cuts = run.base.cuts_for([src[k] for k in run.SITES if k!=s])
        for view in original['by_location'][s]:
            seed = view['seed']
            c = run.shape_run.candidate(d,outer[seed][i],torch,model)
            x,z = run.features(d,c['inc'])
            predictions = {}
            for a in run.ARMS:
                head, _, _ = run.fit(s,seed,a,reg,{},freeze['heads'][f'{s}/{seed}/{a}'])
                values = head.predict(x if a=='factor_local' else z)
                predictions[a] = run.method.reconstruct(values,cv,c['inc'],d['scale'],a!='direct')
            paths = run.method.rollout(cv,c['inc'],d['origin'],d['offsets'],d['recordings'],predictions)
            joint = paths['joint']
            paths.update(query_spread=spread(cv,c['inc'],joint,d['offsets']),fixed_half=cv+.5*(c['inc']-cv))
            report, controls = run.readout(d,paths,cuts)
            if report['joint']!=view['policies']['joint'] or controls!=view['matched']:
                raise ValueError('Post-hoc control changed the frozen joint result')
            keys = ('joint','query_spread','fixed_half')
            rows[s].append(dict(seed=seed,policies={k:report[k] for k in keys},matched=controls))
            support = (joint!=cv).any(axis=(1,2))
            paired[s].append(run.base.paired_summary(d,paths['query_spread'],joint,support)['lower_pp']
                             if support.any() else None)
    summary, _ = run.summary(rows)
    if any(v is None for seq in paired.values() for v in seq):
        matched = dict(status='required_view_not_estimable')
    else:
        x = np.array([np.mean(paired[s]) for s in run.SITES])
        draw = np.random.default_rng(817).integers(len(x),size=(3000,len(x)))
        matched = dict(status='estimated',point=float(x.mean()),ci95=np.quantile(x[draw].mean(1),[.025,.975]).tolist())
    run.prior.write_json(run.PUBLIC/'amplitude_spread.json',dict(original_analysis_sha256=digest,
        posthoc_diagnostic=True,query_mass_exact=True,edit_counts_not_matched=True,
        spread_vs_joint=matched,summary=summary,by_location=rows,decisions_changed=False),immutable=True)


if __name__=='__main__':
    main()
