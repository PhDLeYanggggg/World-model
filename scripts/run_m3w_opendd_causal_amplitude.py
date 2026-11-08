"""Fit nested-support heads and replay the online variable-radius controller."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_radius_shape as shape_run
from src.evaluation import m3w_causal_amplitude as method

base, prior = shape_run.base, shape_run.prior
PUBLIC = shape_run.PUBLIC.parent/'opendd_causal_amplitude_v1'
PRIVATE = shape_run.PRIVATE.parent/PUBLIC.name
SITES, SEEDS = shape_run.SITES, shape_run.SEEDS
ARMS = ('direct','factor_local','factor_prefix')


def register():
    parent, donors = shape_run.register()
    flight = PUBLIC.parent/'opendd_flight_shape_risk_v1'
    files = [Path(__file__).resolve(),Path(method.__file__),PUBLIC/'protocol.md',
             shape_run.PUBLIC/'registration.json',flight/'analysis.json',flight/'fixed_radius_feasibility.json',
             ROOT/'src/evaluation/m3w_opendd_prefix_risk.py']
    p = PUBLIC/'registration.json'
    prior.write_json(p,dict(prior_registration=parent,sites=SITES,seeds=SEEDS,levels=method.LEVELS.tolist(),
        original_risk_budget=.02,external_access=False,dronecrowd_confirmation_access=False,
        source='opendd_exposed_development_only', files={str(f.relative_to(ROOT)):prior.sha(f) for f in files}),
        immutable=True)
    return prior.sha(p),donors


def site_data(site,torch,model,shape):
    d = shape_run.site_data(site,torch,model,shape)
    with np.load(prior.PRIVATE/f'{site}_cohort.npz',allow_pickle=False) as f:
        history, query_rec, start = f['history'],f['recording'],f['start']
    for rec in set(query_rec):
        if (np.diff(start[query_rec==rec])<=0).any():
            raise ValueError('Cannot run a chronological recording prefix')
    return dict(**d,history=history,origin=history[:,-1])


def features(d,inc):
    r = np.linalg.norm(inc-d['cv'],axis=-1)/d['scale'][:,None]
    vel = np.diff(d['history'],axis=1)
    denom = d['scale']
    smooth = np.linalg.norm(vel[:,-1]-vel.mean(1),axis=1)/denom
    sizes = np.diff(d['offsets'])
    own = np.column_stack((d['x'][:,:17].numpy(),r.max(1),r.mean(1),r[:,-1],smooth,
                          np.log1p(np.repeat(sizes,sizes))))
    prefix = method.prefix_features(d['history'],d['offsets'],d['recordings'],inc)
    return own.astype(np.float32),np.column_stack((own,prefix)).astype(np.float32)


def fit(held,seed,arm,reg,donor,expected):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    train = [s for s in SITES if s!=held]
    ident = dict(registration=reg,held=held,seed=seed,arm=arm,train=train)
    p = PRIVATE/f'{held}_{seed}_{arm}.joblib'
    receipt = p.with_suffix('.json')
    if p.exists() and receipt.exists():
        v = json.loads(receipt.read_text())
        if v['identity']!=ident or prior.sha(p)!=v['sha256'] or (expected and v!=expected):
            raise ValueError('Cached head provenance mismatch')
        return joblib.load(p),v,0
    if expected is not None:
        raise ValueError('Replay refuses to fit a missing head')
    x,y,w = [],[],[]
    for s in train:
        local,prefix,direct,factor = donor[s]
        x.append(local if arm=='factor_local' else prefix)
        y.append(direct if arm=='direct' else factor)
        w.append(np.full(len(local),1./len(local)))
    head = ExtraTreesRegressor(n_estimators=32,max_depth=7,min_samples_leaf=12,n_jobs=1,random_state=seed)
    head.fit(np.concatenate(x),np.concatenate(y),sample_weight=np.concatenate(w))
    tmp = p.with_suffix('.pending')
    joblib.dump(head,tmp,compress=1)
    tmp.replace(p)
    v = dict(identity=ident,sha256=prior.sha(p))
    prior.write_json(receipt,v,immutable=True)
    prior.write_json(PRIVATE/'heartbeat.json',dict(**ident,pid=os.getpid(),phase='head_complete'))
    return head,v,1


def readout(d, paths, cuts):
    from src.evaluation.m3w_selection_forecast_readout import policy
    from src.evaluation.m3w_opendd_prefix_risk import compression

    cv = d['cv']
    out = {}
    for name,p in paths.items():
        shift = np.linalg.norm(p-cv,axis=-1).max(1)
        support = shift>0
        r = policy(p,cv,cv,d['y'],d['valid'],d['recordings'],easy_cut=cuts[0],hard_cut=cuts[1])
        records = list(r['recordings'].values())
        geometry = 0.
        for lo,hi in zip(d['offsets'][:-1],d['offsets'][1:],strict=True):
            geometry += compression(d['origin'][lo:hi],cv[lo:hi],p[lo:hi],
                                    tuple(np.flatnonzero(support[lo:hi])))
        out[name] = dict(vs_cv=base.paired_summary(d,p,cv,support),edits=int(support.sum()),
            radius_mass=float(shift.sum()),passed=sum(v['supported'] for v in records),
            flights=len(records),undefined_selected_easy=sum(v['completion_easy_risk_upper'] is None for v in records),
            undefined_full_easy=sum(v['known_easy_net_degradation'] is None for v in records),
            available_ade=r['known_ADE_pixel'],available_fde=r['available_endpoint_FDE_diagnostic_pixel'],
            worst_defined_net_easy_pct=max((100*v['known_easy_net_degradation'] for v in records
                if v['known_easy_net_degradation'] is not None),default=None),
            pair_compression=geometry)
    joint = paths['joint']
    support = (joint!=cv).any(axis=(1,2))
    matched = {k:(base.paired_summary(d,joint,paths[k],support) if support.any() else None)
               for k in ('matched_independent','matched_additive')}
    return out,matched


def summary(rows):
    draws = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))

    def interval(values):
        defined = sum(x is not None for seq in values for x in seq)
        if defined!=len(SITES)*len(SEEDS):
            return dict(status='not_estimable',defined=defined,required=len(SITES)*len(SEEDS))
        z = np.asarray([np.mean(v) for v in values])
        return dict(status='estimated',point=float(z.mean()),ci95=np.quantile(z[draws].mean(1),[.025,.975]).tolist())

    out = {}
    for name in rows[SITES[0]][0]['policies']:
        nested = [[v['policies'][name] for v in rows[s]] for s in SITES]
        flat = [v for seq in nested for v in seq]
        out[name] = dict(vs_cv=interval([[v['vs_cv']['lower_pp'] for v in seq] for seq in nested]),
            **{k:sum(v[k] for v in flat) for k in ('edits','radius_mass','passed','flights',
                'undefined_selected_easy','undefined_full_easy','pair_compression')})
    matched = {k:interval([[None if v['matched'][k] is None else v['matched'][k]['lower_pp']
        for v in rows[s]] for s in SITES]) for k in ('matched_independent','matched_additive')}
    return out,matched


def run(reg,donors,replay):
    import torch
    from sklearn.metrics import average_precision_score
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    p,cp = PUBLIC/'analysis.json',PUBLIC/'head_freeze.json'
    expected = json.loads(cp.read_text()) if replay else None
    if replay and (expected['registration']!=reg or not p.is_file()):
        raise ValueError('Replay requires a completed matching readout')
    sources = {s:site_data(s,torch,model,shape) for s in SITES}
    outer,inner = shape_run.load_producers(torch,model,donors)
    cuts = {s:base.cuts_for([sources[k] for k in SITES if k!=s]) for s in SITES}
    heads,meta,fits = {},{},0
    for held in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {held}',flush=True)
        for seed in SEEDS:
            donor = {}
            for s,d in sources.items():
                if s==held:
                    continue
                c = shape_run.candidate(d,inner[tuple(sorted((s,held))),seed],torch,model)
                x,z = features(d,c['inc'])
                y,q = method.labels(d['cv'],d['y'],d['valid'],c['inc'],cuts[held][0],d['scale'])
                donor[s] = tuple(a[c['pool']] for a in (x,z,y,q))
            for arm in ARMS:
                key = f'{held}/{seed}/{arm}'
                head,info,n = fit(held,seed,arm,reg,donor,None if expected is None else expected['heads'][key])
                heads[held,seed,arm],meta[key] = head,info
                fits += n
    prior.write_json(cp,dict(registration=reg,heads=meta),immutable=True)
    rows = {}
    for i,s in enumerate(SITES):
        d = sources[s]
        rows[s] = []
        print(f'Scoring {s}',flush=True)
        prior.write_json(PRIVATE/'heartbeat.json',dict(pid=os.getpid(),phase='score',site=s))
        for seed in SEEDS:
            c = shape_run.candidate(d,outer[seed][i],torch,model)
            x,z = features(d,c['inc'])
            raw = {a:heads[s,seed,a].predict(x if a=='factor_local' else z) for a in ARMS}
            pred = {a:method.reconstruct(raw[a],d['cv'],c['inc'],d['scale'],a!='direct') for a in ARMS}
            paths = method.rollout(d['cv'],c['inc'],d['origin'],d['offsets'],d['recordings'],pred)
            paths.update(cv=d['cv'],incumbent=c['inc'])
            policies,matched = readout(d,paths,cuts[s])
            missing = ~d['valid'][c['pool']].any(1)
            diagnostics = {a:(float(average_precision_score(missing,raw[a][c['pool'],11])) if missing.any() else None)
                           for a in ('factor_local','factor_prefix')}
            rows[s].append(dict(seed=seed,policies=policies,matched=matched,unknown_rows=int(missing.sum()),
                               eligible_rows=len(missing),missing_ap=diagnostics))
    total,matched = summary(rows)
    prior.write_json(p,dict(registration=reg,heads=len(meta),summary=total,paired_joint=matched,by_location=rows,
        radius_multiset_exact=True,original_risk_unchanged=True,formal_confirmation=False,deployment=False,
        stage5c_executed=False,smc_enabled=False),immutable=True)
    receipt = PUBLIC/('replay_execution.json' if replay else 'execution.json')
    if not receipt.exists():
        prior.write_json(receipt,dict(registration=reg,analysis_sha256=prior.sha(p),new_heads_this_execution=fits,
            fresh_exact_replay=replay,pid=os.getpid(),torch_threads=4,workers=0,wall_seconds=round(monotonic()-t0,3)))
    prior.write_json(PRIVATE/'heartbeat.json',dict(pid=os.getpid(),phase='complete',replay=replay))
    print(prior.sha(p),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','run','replay'))
    phase = p.parse_args().phase
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg,donors = register()
        if phase=='register':
            print(reg)
        else:
            if platform.system()!='Darwin' or platform.machine()!='arm64':
                raise RuntimeError('Use native arm64 .venv-pytorch on this host')
            run(reg,donors,phase=='replay')
