"""Outcome-aware upper envelope for the exact registered swap menu."""
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_support_exchange as run


def exact_options(cv,inc,y,valid,cut):
    delta = inc-cv
    radius = np.linalg.norm(delta,axis=-1).max(1)
    i,j,a = run.method.options(radius)
    half = cv+.5*delta
    ref = np.linalg.norm(cv-y,axis=-1)
    old = np.linalg.norm(half-y,axis=-1)
    count = valid.sum(1).clip(1)
    known = valid.any(1)
    cv_ade = np.where(valid,ref,0.).sum(1)/count
    old_ade = np.where(valid,old,0.).sum(1)/count
    easy = known & (cv_ade>0.) & (cv_ade<=cut)

    def terms(at,signed):
        change = (signed/radius[at])[:,None,None]*delta[at]
        err = np.linalg.norm(half[at]+change-y[at],axis=-1)
        lower = np.where(valid[at],old[at]-err,-np.linalg.norm(change,axis=-1)).mean(1)
        ade = np.where(valid[at],err,0.).sum(1)/count[at]
        harm = (ade-cv_ade[at]).clip(0.)-(old_ade[at]-cv_ade[at]).clip(0.)
        debit = signed*~known[at]
        return np.column_stack((lower,harm+debit,harm*easy[at]+debit,(ade-old_ade[at])*easy[at]))

    return i,j,a,terms(i,a)+terms(j,-a)


def oracle(d,inc,cut):
    cv = d['cv']
    delta = inc-cv
    paths = {k:cv+.5*delta for k in ('utility_oracle','guarded_oracle')}
    counts = dict(queries_with_swap=0,queries_with_positive_gain=0,queries_with_safe_positive_gain=0,
                  total_options=0,safe_positive_options=0)
    r = np.linalg.norm(delta,axis=-1).max(1)
    for lo,hi in zip(d['offsets'][:-1],d['offsets'][1:],strict=True):
        sl = slice(lo,hi)
        i,j,a,v = exact_options(cv[sl],inc[sl],d['y'][sl],d['valid'][sl],cut)
        if not len(i):
            continue
        counts['queries_with_swap'] += 1
        counts['total_options'] += len(i)
        safe = (v[:,1:]<=1e-12).all(1)
        good = v[:,0]>0.
        counts['queries_with_positive_gain'] += bool(good.any())
        counts['queries_with_safe_positive_gain'] += bool((safe & good).any())
        counts['safe_positive_options'] += int((safe & good).sum())
        for name,score in (('utility_oracle',v[:,0]),('guarded_oracle',np.where(safe,v[:,0],-np.inf))):
            k = int(score.argmax())
            if score[k]>0.:
                u,w = lo+i[k],lo+j[k]
                paths[name][u] += a[k]/r[u]*delta[u]
                paths[name][w] -= a[k]/r[w]*delta[w]
    for p in paths.values():
        dist = np.linalg.norm(p-cv,axis=-1).max(1)
        np.testing.assert_array_equal(dist>0.,r>0.)
        np.testing.assert_allclose(np.add.reduceat(dist,d['offsets'][:-1]),
            .5*np.add.reduceat(r,d['offsets'][:-1]),rtol=1e-7,atol=1e-14)
    return paths,counts


def main():
    if platform.system()!='Darwin' or platform.machine()!='arm64':
        raise RuntimeError('Use native arm64 .venv-pytorch on this host')
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg,_ = run.register()
    p = run.PUBLIC/'analysis.json'
    digest = run.prior.sha(p)
    evidence = json.loads(p.read_text())
    replay = json.loads((run.PUBLIC/'replay_execution.json').read_text())
    if evidence['registration']!=reg or replay['analysis_sha256']!=digest or not replay['fresh_exact_replay']:
        raise ValueError('Replay causal arms before the outcome-aware upper envelope')
    sources = {s:run.parent.site_data(s,torch,model,shape) for s in run.SITES}
    outer = run.shape_run.frozen.load_networks(torch,model)
    rows = {}
    for n,s in enumerate(run.SITES):
        d = sources[s]
        cuts = run.base.cuts_for([sources[k] for k in run.SITES if k!=s])
        rows[s] = []
        for seed in run.SEEDS:
            c = run.shape_run.candidate(d,outer[seed][n],torch,model)
            paths,counts = oracle(d,c['inc'],cuts[0])
            paths.update(half=d['cv']+.5*(c['inc']-d['cv']))
            arms = run.readout(d,paths,cuts)
            old = next(v for v in evidence['by_location'][s] if v['seed']==seed)
            if arms['half']!=old['arms']['half']:
                raise ValueError('Fixed-half reference changed')
            if arms['guarded_oracle']['passes']<arms['half']['passes']:
                raise ValueError('Safe oracle should not lose a pass')
            rows[s].append(dict(seed=seed,arms=arms,counts=counts))
    count = {k:sum(v['counts'][k] for seq in rows.values() for v in seq)
             for k in rows[run.SITES[0]][0]['counts']}
    summary = {}
    draw = np.random.default_rng(817).integers(len(run.SITES),size=(3000,len(run.SITES)))
    for name in ('half','utility_oracle','guarded_oracle'):
        x = np.array([np.mean([v['arms'][name]['half']['lower_pp'] for v in rows[s]]) for s in run.SITES])
        summary[name] = dict(lower_pp=float(x.mean()),ci95=np.quantile(x[draw].mean(1),[.025,.975]).tolist(),
                             passes=sum(v['arms'][name]['passes'] for seq in rows.values() for v in seq))
    run.prior.write_json(run.PUBLIC/'oracle_headroom.json',dict(original_analysis_sha256=digest,
        posthoc_outcome_aware=True,not_a_policy=True,matched_all_recipients=True,
        summary=summary,counts=count,by_location=rows),immutable=True)


if __name__=='__main__':
    main()
