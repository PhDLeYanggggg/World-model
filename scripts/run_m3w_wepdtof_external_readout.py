"""Freeze the fixed-count controls and perform the single reserved-source readout."""
import argparse
import fcntl
from itertools import combinations,product
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_m3w_wepdtof_external_transfer as study

base,np,torch,method = study.base,study.np,study.torch,study.method
OUT,PRIVATE = study.OUT,study.PRIVATE
ACTIVE = tuple(f'fixed_{k}' for k in method.NAMES)
NAMES = study.ARMS+ACTIVE


def fixed(est,legal,q,h,cv,bank):
    picks = {k:np.full(len(q),-1,dtype=int) for k in ACTIVE}
    utility = est[...,:2]
    risk = est[...,2:4]-.02*est[...,4:6]
    raw = utility.min(2)-np.maximum(risk,0.).sum(2)
    for j in np.unique(q):
        ix = np.flatnonzero(q==j)
        rows = ix[legal[ix].all(1)]
        n = min(2,len(rows),len(ix)-1)
        if n<1:
            continue
        local = max(float(np.median(np.linalg.norm(np.diff(h[ix],axis=1),axis=-1)))*12.,1e-8)
        best = {k:(-np.inf,None) for k in ('joint','no_pair')}
        for recipients in combinations(rows,n):
            near = np.exp(-np.linalg.norm(h[recipients[0],-1]-h[recipients[-1],-1])/local)
            for arms in product(range(bank.shape[1]),repeat=n):
                a,b = np.array(tuple(zip(recipients,arms))).T
                u = utility[a,b].sum(0).min()-np.maximum(risk[a,b].sum(0),0.).sum()
                offsets = bank[a,b]-cv[a]
                penalty = near*np.linalg.norm(offsets[0]-offsets[-1],axis=1).mean()
                for k,val in (('joint',u-method.PAIR_WEIGHT*penalty),('no_pair',u)):
                    if val>best[k][0]:
                        best[k] = val,(a,b)
        rank = np.argsort(-raw[rows].max(1),kind='stable')[:n]
        families = []
        for k in range(bank.shape[1]):
            take = np.argsort(-raw[rows,k],kind='stable')[:n]
            families.append((raw[rows[take],k].sum(),rows[take],k))
        _,a,b = max(families,key=lambda v:v[0])
        selected = dict(joint=best['joint'][1],no_pair=best['no_pair'][1],
                        independent=(rows[rank],raw[rows[rank]].argmax(1)),uniform=(a,b))
        for k,(a,b) in selected.items():
            picks[f'fixed_{k}'][a]=b
    return method.apply(cv,bank,picks,q)


def register():
    key,cells,cohorts = study.register()
    p = OUT/'pilot_freeze.json'
    pilot = study.read(p)
    if pilot['registration']!=key or pilot['external_geometry_read'] or (OUT/'readout.json').exists():
        raise ValueError('Held outcomes must not have a strict-only readout')
    q = OUT/'evaluation_registration.json'
    base.immutable(q,dict(training_registration=key, pilot=study.artifact(p),
        files=[study.artifact(x) for x in (Path(__file__),OUT/'readout_amendment.md')],
        arms=NAMES, empty_matched_control='not_estimable', control_counts='min(two,eligible,agents minus one)',
        fixed_controls_deployment=False, original_2pct=True, selection_on_external=False))
    return base.sha(q),key,cells


def freeze(key,training,cells):
    p = OUT/'forecast_freeze.json'
    f = study.read(p)
    if (f['registration']!=training or not f['no_external_outcomes_read'] or f['source_groups']!=13
            or f['arms']!=list(study.ARMS) or (OUT/'external_readout.json').exists()):
        raise ValueError('Require original unscored external prediction freeze')
    nets = study.models(training,cells)
    c,windows = study.frozen_camera()
    result = []
    for r in f['rows']:
        clip = r['clip']
        h,q,ids = study.camera_arrays(windows[clip])
        with np.load(study.checked(r['artifact']),allow_pickle=False) as z:
            pred = {k:z[k] for k in z.files}
        np.testing.assert_array_equal(pred['q'],q);np.testing.assert_array_equal(pred['ids'],ids)
        for s,(net,_) in nets.items():
            cv,ols,p0 = (pred[f'{s}_{k}'] for k in ('cv','ols4','spline'))
            if len(h):
                bank = method.bank(cv,ols,p0,q)
                x,scale,legal,_ = study.costs.features(h,cv,ols,bank)
                with torch.inference_mode():
                    estimates = net(torch.from_numpy(x)).numpy()*scale[:,None,None]
                paths = fixed(estimates,legal,q,h,cv,bank)
            else:
                paths = {k:cv.copy() for k in ACTIVE}
            pred.update({f'{s}_{k}':v for k,v in paths.items()})
        dst = PRIVATE/f'{clip}_augmented.npz'
        if dst.exists():
            with np.load(dst,allow_pickle=False) as z:
                if set(z.files)!=set(pred):
                    raise ValueError('Different prediction arms')
                for k,v in pred.items():
                    np.testing.assert_array_equal(z[k],v)
        else:
            tmp = dst.with_suffix('.pending.npz')
            np.savez(tmp,**pred)
            tmp.replace(dst)
        result.append(dict(clip=clip,source=r['source'],online=len(h),artifact=study.artifact(dst)))
    base.immutable(OUT/'augmented_freeze.json',dict(evaluation_registration=key,camera=study.artifact(c),
        original_prediction_freeze=study.artifact(p), rows=result,groups=13,arms=NAMES,seeds=study.SEEDS,
        outcomes_unopened=True,original_2pct=True))


def augment_metrics(pred,y,v,q,name,cut):
    out = study.metrics(pred,y,v,q,name,cut)
    if not len(q):
        return out
    _,inv,count = np.unique(q,return_inverse=True,return_counts=True)
    w = 1./count[inv]
    for k in ('independent','uniform','no_pair'):
        if not out['arms']['joint']['risk']['selected']:
            out['comparisons'][k] = [None,None]
        a,b = pred[f'fixed_{k}'],pred['fixed_joint']
        mass = out['reference_mass']
        if out['arms']['fixed_joint']['risk']['selected'] and mass>0.:
            g,l = study.paired(a,b,y,v)
            out['comparisons'][f'fixed_{k}'] = (100.*np.array([w@g.sum(1),w@l.sum(1)])/mass).tolist()
        else:
            out['comparisons'][f'fixed_{k}'] = [None,None]
    return out


def summarize(rows):
    sites = sorted({r['source'] for r in rows})
    draws = 3000

    def ci(get):
        vals = []
        for s in sites:
            group = [get(v) for r in rows if r['source']==s for v in r['seeds'].values()]
            if all(x is not None for x in group):
                vals.append(float(np.mean(group)))
        if not vals:
            return dict(estimate=None,ci95=None,estimable_sources=0,total_sources=len(sites))
        a = np.array(vals)
        sims = np.random.default_rng(82317).choice(a,size=(draws,len(a)),replace=True).mean(1)
        return dict(estimate=float(a.mean()),ci95=np.quantile(sims,[.025,.975]).tolist(),
                    estimable_sources=len(a),total_sources=len(sites))

    flat = [z for r in rows for z in r['seeds'].values()]
    total = {k:sum(r[k] for r in rows) for k in ('online','matched','full','missing','reference_histories')}
    arms = {}
    for k in NAMES:
        get = lambda z:z['arms'].get(k,{})
        rs = [get(z).get('risk',{}) for z in flat]
        arms[k] = dict(**{f:ci(lambda z:get(z).get(f)) for f in (
            'lower_vs_cv','lower_vs_ols4','ade_px','fde_px','p95_ade_px','edit_rate')},
            pair_relative_error_px=ci(lambda z:z.get('pair_relative_error_px',{}).get(k)),
            edits=sum(x.get('selected',0) for x in rs),
            risk_passes=sum(x.get('supported',False) for x in rs),
            dual_passes=sum(get(z).get('dual_pass',False) for z in flat),
            undefined_easy=sum(x.get('completion_easy_risk_upper') is None for x in rs),
            worst_signed_easy_percent=max((100*x['known_easy_net_degradation'] for x in rs
                   if x.get('known_easy_net_degradation') is not None),default=None))
    comparisons = {k:ci(lambda z:z['comparisons'].get(k,[None,None])[1])
                   for k in ('independent','uniform','no_pair','fixed_independent','fixed_uniform','fixed_no_pair')}
    return dict(totals=total,arms=arms,matched_comparisons=comparisons,recording_seed_views=len(flat),
        sources=len(sites),bootstrap_draws=draws,bootstrap_unit='original source after averaging clips and seeds')


def score(key):
    path = OUT/'augmented_freeze.json'
    f = study.read(path)
    if (f['evaluation_registration']!=key or not f['outcomes_unopened'] or f['groups']!=13
            or f['arms']!=list(NAMES) or f['seeds']!=list(study.SEEDS)):
        raise ValueError('Require full augmented freeze before outcome access')
    c,windows = study.frozen_camera()
    if study.checked(f['camera'])!=c or set(windows)!={r['clip'] for r in f['rows']}:
        raise ValueError('Evaluation roster changed')
    study.checked(f['original_prediction_freeze'])
    receipt = OUT/'external_readout.json'
    cut = study.read(OUT/'pilot_freeze.json')['cut_px']
    key = base.sha(path)
    result = []
    for r in f['rows']:
        clip = r['clip']
        dest = PRIVATE/f'{clip}_augmented_score.json'
        if not dest.exists():
            if receipt.exists():
                raise ValueError('Refuse a second held evaluation')
            h,q,ids = study.camera_arrays(windows[clip])
            with np.load(study.checked(r['artifact']),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['q'],q); np.testing.assert_array_equal(z['ids'],ids)
                pred = {s:{k:z[f'{s}_{k}'] for k in NAMES} for s in study.SEEDS}
            y,v,m,ref = study.outcomes(windows[clip],h,q,ids,study.references(clip))
            values = {str(s):augment_metrics(a,y,v,q,clip,cut) for s,a in pred.items()}
            base.immutable(dest,dict(freeze=key,source=r['source'],clip=clip,online=len(h),matched=int(m.sum()),
                full=int(v.all(1).sum()),missing=int((~v.any(1)).sum()),reference_histories=ref,seeds=values))
        a = study.read(dest)
        if a['freeze']!=key or a['clip']!=clip:
            raise ValueError('Frozen outcome replay mismatch')
        result.append(a)
        study.beat(phase='one_shot_readout',clip=clip,done=len(result),total=len(f['rows']))
    s = summarize(result)
    base.immutable(receipt,dict(forecast_freeze=key,rows=result,summary=s, original_2pct=True,
        external_scoring_completed=True,external_tuning=False,dronecrowd_confirmation_access=False,
        deployment_changed=False,submission_ready=False))
    print(dict(totals=s['totals'],comparisons=s['matched_comparisons']),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','freeze','score'))
    phase = p.parse_args().phase
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with (PRIVATE/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        key,training,cells = register()
        if phase=='freeze':
            freeze(key,training,cells)
        elif phase=='score':
            score(key)


if __name__=='__main__':
    main()
