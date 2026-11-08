"""Fit risk moments on nested forecasts and evaluate causal prefix allocation."""
import argparse
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_velocity_calibration as calibration

PUBLIC = calibration.PUBLIC.parent/'opendd_prefix_risk_v1'
PRIVATE = calibration.PRIVATE.parent/PUBLIC.name
prior = calibration.prior
velocity = calibration.velocity
SITES,SEEDS = calibration.SITES,calibration.SEEDS
NAMES = ('local','prefix','joint','matched_independent','matched_additive','uniform','cv')


def beat(phase, **kw):
    state = dict(phase=phase,pid=os.getpid(),**kw)
    prior.write_json(PRIVATE/'heartbeat.json',state)
    print(json.dumps(state),flush=True)


def register():
    source,frozen = calibration.register()
    path = calibration.PUBLIC/'analysis.json'
    result = json.loads(path.read_text())
    if (result['registration']!=source or result['official_confirmation'] or len(result['fresh_inner_fits'])!=63
        or prior.sha(path)!=json.loads((calibration.PUBLIC/'replay_execution.json').read_text())['analysis_sha256']):
        raise ValueError('Require verified calibration evidence')
    ck = {'+'.join(pair)+f'_{seed}':prior.sha(calibration.PRIVATE/f'{"+".join(pair)}_{seed}_self.pt')
          for pair in combinations(SITES,2) for seed in SEEDS}
    files = (Path(__file__).resolve(),PUBLIC/'protocol.md',Path(calibration.__file__),path,
        calibration.PUBLIC/'registration.json',calibration.PUBLIC/'replay_execution.json',
        ROOT/'src/evaluation/m3w_opendd_prefix_risk.py',ROOT/'src/evaluation/m3w_selection_forecast_readout.py',
        ROOT/'src/evaluation/m3w_native_metrics.py')
    p = PUBLIC/'registration.json'
    prior.write_json(p,dict(files={str(a.relative_to(ROOT)):prior.sha(a) for a in files},inner_checkpoint_sha256=ck,
        seeds=SEEDS,sites=SITES,prior_registration=source,official_risk_budget=.02,
        dronecrowd_confirmation_access=False,stage5c_executed=False,smc_enabled=False),immutable=True)
    return prior.sha(p),result,frozen,ck


def data(site,original):
    d = calibration.data(site,original)
    with np.load(prior.PRIVATE/f'{site}_cohort.npz',allow_pickle=False) as f:
        recording,start = f['recording'],f['start']
        for rec in set(recording):
            if (np.diff(start[recording==rec])<=0).any():
                raise ValueError('Flight query starts are not chronological')
        return {**d,'origin':f['history'][:,-1].copy()}


def inner_forecasts(sources,ident,bindings):
    out = {}
    for pair in combinations(SITES,2):
        key = '+'.join(pair)
        for seed in SEEDS:
            path = calibration.PRIVATE/f'{key}_{seed}_self.pt'
            if prior.sha(path)!=bindings[f'{key}_{seed}']:
                raise ValueError('Inner checkpoint changed')
            ck = torch.load(path,map_location='cpu',weights_only=True)
            expected = dict(registration=ident,held=key,arm='self',seed=seed,train=[s for s in SITES if s not in pair])
            if ck['identity']!=expected or ck['step']!=velocity.STEPS:
                raise ValueError('Incorrect inner fold or unfinished checkpoint')
            net = model.network(sources[pair[0]]['design'].shape[1])
            net.load_state_dict(ck['model'])
            for site in pair:
                out[pair,seed,site] = velocity.predict(net.eval(),sources[site],'self')
    return out


def features(d,p):
    scale = d['scale']
    delta = np.linalg.norm(p-d['cv'],axis=-1) / scale[:,None]
    ols = np.linalg.norm(d['ols']-d['cv'],axis=-1) / scale[:,None]
    dist = np.linalg.norm(p-d['ols'],axis=-1) / scale[:,None]
    n = np.repeat(np.diff(d['offsets']),np.diff(d['offsets']))
    x = np.column_stack((d['design'][:,:17].numpy(),delta.mean(1),delta.max(1),ols.mean(1),
        dist.mean(1),d['hindcast'].max(1),np.log1p(n)))
    if not np.isfinite(x).all():
        raise ValueError('Expected finite causal moment features')
    return x.astype(np.float32)


def labels(d,p,cut):
    ref,_ = errors(d['cv'],d['y'],d['valid'])
    err,_ = errors(p,d['y'],d['valid'])
    known = d['valid'].any(1)
    easy = known & (ref>0) & (ref<=cut)
    gain = np.where(known,ref-err,0.)
    harm = np.maximum(0.,-gain)
    unknown = np.where(known,0.,np.linalg.norm(p-d['cv'],axis=-1).max(1))
    values = np.column_stack((gain-unknown,harm+unknown,np.where(easy,harm,0.)+unknown,
                              np.where(known,ref,0.),np.where(easy,ref,0.)))
    return values/d['scale'][:,None]


def fit(sources,inner,held,seed,alpha,registration):
    train = [s for s in SITES if s!=held]
    name = f'{held}_{seed}'
    path = PRIVATE/f'{name}.joblib'
    receipt = path.with_suffix('.json')
    identity = dict(registration=registration,held=held,seed=seed,alpha=alpha,train=train)
    if path.exists() and receipt.exists():
        r = json.loads(receipt.read_text())
        if r['identity']!=identity or r['sha256']!=prior.sha(path):
            raise ValueError('Risk model checkpoint changed')
        return joblib.load(path),identity
    x,y,w = [],[],[]
    for site in train:
        pair = tuple(sorted((held,site)))
        cut = calibration.parent.easy_cut(sources,[s for s in SITES if s not in pair])
        d = sources[site]
        p,_ = calibration.rollout(d,inner[pair,seed,site],alpha,'uniform')
        x.append(features(d,p))
        y.append(labels(d,p,cut))
        w.append(np.full(len(p),1./len(p)))
    fit = ExtraTreesRegressor(n_estimators=32,max_depth=7,min_samples_leaf=12,n_jobs=1,random_state=seed)
    fit.fit(np.concatenate(x),np.concatenate(y),sample_weight=np.concatenate(w))
    tmp = path.with_suffix('.pending')
    joblib.dump(fit,tmp,compress=1)
    tmp.replace(path)
    prior.write_json(receipt,dict(identity=identity,sha256=prior.sha(path)),immutable=True)
    beat('head_checkpoint',held=held,seed=seed,fit_rows=sum(map(len,x)))
    return fit,identity


def eval_view(d,candidate,estimates,cut,hard):
    masks,geometry = prefix.walk(d,candidate,estimates)
    masks['uniform'] = (candidate!=d['cv']).any(axis=(1,2))
    masks['cv'] = np.zeros(len(candidate),dtype=bool)
    p = {a:np.where(m[:,None,None],candidate,d['cv']) for a,m in masks.items()}
    rows = {}
    for a in NAMES:
        value = readout.metrics(d,p[a],masks[a],cut)
        rows[a] = dict(cv_relative=value,flight=calibration.flight_risk(d,p[a],cut,hard),
                       summed_query_compression=geometry.get(a))
    count = np.add.reduceat(masks['joint'],d['offsets'][:-1])
    matched = {}
    for a in ('matched_independent','matched_additive'):
        if not np.array_equal(np.add.reduceat(masks[a],d['offsets'][:-1]),count):
            raise ValueError('Matched utility lost the joint query count')
        matched[a] = (readout.metrics(d,p['joint'],masks['joint'],cut,ref=p[a]) if count.sum() else None)
    return rows,matched


def summary(results):
    draws = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))
    rows = [results[s] for s in SITES]
    def estimate(x):
        v = np.asarray([np.mean([m['lower_pp'] for m in at]) for at in x])
        return dict(lower_pp=float(v.mean()),ci95_pp=np.quantile(v[draws].mean(1),[.025,.975]).tolist(),
                    positive_sites=int((v>0).sum()),worst_site_pp=float(v.min()))
    out = {}
    for arm in NAMES:
        views = [[a['policies'][arm] for a in at] for at in rows]
        flat = [a for at in views for a in at]
        cv = [v['cv_relative'] for v in flat]
        flight = [v['flight'] for v in flat]
        out[arm] = dict(**estimate([[a['cv_relative'] for a in at] for at in views]),
            switches=sum(m['switches'] for m in cv),worst_signed_easy_pct=max(m['signed_easy_pct'] for m in cv),
            worst_positive_easy_pct=max(m['positive_easy_pct'] for m in cv),
            ade=float(np.mean([m['ade'] for m in cv])),fde=float(np.mean([m['fde'] for m in cv])),
            flight={k:sum(f[k] for f in flight) for k in flight[0]},
            summed_query_compression=sum(v['summed_query_compression'] for v in flat)
                if flat[0]['summed_query_compression'] is not None else None)
    for arm in ('matched_independent','matched_additive'):
        x = [[v['matched'][arm] for v in at] for at in rows]
        defined = sum(m is not None for at in x for m in at)
        out[f'joint_vs_{arm}'] = (dict(status='estimated',**estimate(x)) if defined==len(SITES)*len(SEEDS)
            else dict(status='required_matched_view_not_estimable',defined_views=defined,required_views=len(SITES)*len(SEEDS)))
    return out


def run(registration,source,frozen,inner_bindings):
    start = monotonic()
    original = prior.sha(prior.PUBLIC/'registration.json')
    sources = {s:data(s,original) for s in SITES}
    inner = inner_forecasts(sources,source['registration'],inner_bindings)
    heads,identities = {},[]
    for held in SITES:
        for view in source['sites'][held]:
            seed,alpha = view['seed'],view['selected']['uniform']
            head,ident = fit(sources,inner,held,seed,alpha,registration)
            heads[held,seed] = head
            identities.append(ident)
    result = {}
    for held,d in sources.items():
        train = [s for s in SITES if s!=held]
        cut = calibration.parent.easy_cut(sources,train)
        a = np.concatenate([sources[s]['cv_complete_ade'] for s in train])
        hard = float(np.quantile(a[a>0],.75))
        result[held] = []
        for view in source['sites'][held]:
            seed,alpha = view['seed'],view['selected']['uniform']
            forecast = calibration.frozen_forecast(sources,held,seed,frozen)
            p,_ = calibration.rollout(d,forecast,alpha,'uniform')
            estimates = heads[held,seed].predict(features(d,p))*d['scale'][:,None]
            rows,matched = eval_view(d,p,estimates,cut,hard)
            if rows['uniform']['flight']!=view['policies']['uniform']['flight']:
                raise ValueError('Uniform replay disagrees with frozen flight risk')
            result[held].append(dict(seed=seed,policies=rows,matched=matched))
    path = PUBLIC/'analysis.json'
    replay = path.exists()
    prior.write_json(path,dict(registration=registration,fits=identities,sites=result,summary=summary(result),
        scope='development',independent_confirmation=False,deployment=False),immutable=True)
    beat('complete',elapsed_s=round(monotonic()-start,3),analysis_sha256=prior.sha(path),exact_replay=replay)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--register-only',action='store_true')
    args = p.parse_args()
    if sys.platform=='darwin' and platform.machine()!='arm64':
        raise SystemExit('Run native arm64 .venv-pytorch/bin/python, not Rosetta')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg = register()
        if args.register_only:
            print(reg[0])
        else:
            import joblib
            import numpy as np
            import torch
            from sklearn.ensemble import ExtraTreesRegressor
            from src.evaluation import m3w_opendd_prefix_risk as prefix
            from src.evaluation import m3w_opendd_velocity_calibration as calibrate, m3w_opendd_cv_abstention as readout
            from src.evaluation.m3w_selection_forecast_readout import errors,policy
            from src.world_model import m3w_ols_replacement as motion,m3w_opendd_velocity_candidate as model

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            calibration.np,calibration.torch,calibration.calibrate,calibration.model = np,torch,calibrate,model
            calibration.policy = policy
            calibration.parent.np,calibration.parent.torch,calibration.parent.motion = np,torch,motion
            velocity.np,velocity.torch,velocity.model = np,torch,model
            run(*reg)
