"""Directly learn the exact support-preserving two-target action."""
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
from scripts import run_m3w_opendd_support_exchange as base
from scripts import audit_m3w_opendd_exchange_headroom as oracle

prior = base.prior
PUBLIC = base.PUBLIC.parent/'opendd_direct_exchange_v1'
PRIVATE = base.PRIVATE.parent/PUBLIC.name
SITES, SEEDS = base.SITES,base.SEEDS
ARMS = ('direct_utility','direct_guard','event_guard')


def register():
    parent, donors = base.register()
    files = [Path(__file__).resolve(),Path(oracle.__file__),PUBLIC/'protocol.md']
    files += [base.PUBLIC/k for k in ('registration.json','analysis.json','replay_execution.json',
                                     'head_freeze.json','oracle_headroom.json')]
    p = PUBLIC/'registration.json'
    prior.write_json(p,dict(prior_registration=parent,sites=SITES,seeds=SEEDS,
        original_risk_budget=.02,external_access=False,dronecrowd_confirmation_access=False,
        source='opendd_exposed_development_only',files={str(f.relative_to(ROOT)):prior.sha(f) for f in files}),
        immutable=True)
    return prior.sha(p),parent,donors


def actions(d,c,x,cut=None):
    tables, y, group, info, norm = [],[],[],[],[]
    r, scale = c['r'],d['scale']
    for q,(lo,hi) in enumerate(zip(d['offsets'][:-1],d['offsets'][1:],strict=True)):
        sl = slice(lo,hi)
        if cut is None:
            i,j,amount = base.method.options(r[sl])
        else:
            i,j,amount,values = oracle.exact_options(d['cv'][sl],c['inc'][sl],d['y'][sl],d['valid'][sl],cut)
        if not len(i):
            continue
        i,j = i+lo,j+lo
        s = (scale[i]+scale[j])*.5
        separation = np.linalg.norm(d['origin'][i]-d['origin'][j],axis=1)
        tables.append(np.column_stack((x[i],x[j],x[i]-x[j],amount/r[i],amount/r[j],
                                       np.log1p(amount/s),np.log1p(separation/s))))
        if cut is not None:
            event = (values[:,0]>0.) & (values[:,1:]<=1e-12).all(1)
            y.append(np.column_stack((values/s[:,None],event)))
        group.append(np.full(len(i),q))
        info.append((i,j,amount))
        norm.append(s)
    X = np.concatenate(tables).astype(np.float32)
    if cut is not None:
        w = np.concatenate([np.full(len(a),1./len(a)/len(tables)) for a in tables])
        return X,np.concatenate(y),w
    return X, np.concatenate(group), tuple(np.concatenate(a) for a in zip(*info,strict=True)), np.concatenate(norm)


def fit(held,seed,reg,donor,expected):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    train = [s for s in SITES if s!=held]
    ident = dict(registration=reg,held=held,seed=seed,train=train)
    p = PRIVATE/f'{held}_{seed}.joblib'
    receipt = p.with_suffix('.json')
    if p.exists() and receipt.exists():
        v = json.loads(receipt.read_text())
        if v['identity']!=ident or prior.sha(p)!=v['sha256'] or (expected is not None and v!=expected):
            raise ValueError('Cached direct pair head changed')
        return joblib.load(p),v,0
    if expected is not None:
        raise ValueError('Replay cannot fit')
    x,y,w = (np.concatenate([donor[s][k] for s in train]) for k in range(3))
    head = ExtraTreesRegressor(n_estimators=24,max_depth=7,min_samples_leaf=24,
                               max_features=.65,n_jobs=1,random_state=seed)
    head.fit(x,y,sample_weight=w)
    tmp = p.with_suffix('.pending')
    joblib.dump(head,tmp,compress=1)
    tmp.replace(p)
    v = dict(identity=ident,sha256=prior.sha(p),actions=len(x))
    prior.write_json(receipt,v,immutable=True)
    prior.write_json(PRIVATE/'heartbeat.json',dict(**ident,pid=os.getpid(),phase='head_complete'))
    return head,v,1


def apply(cv,inc,offsets,group,info,estimate):
    i,j,amount = info
    r = np.linalg.norm(inc-cv,axis=-1).max(1)
    paths = {k:cv+.5*(inc-cv) for k in ARMS}
    counts = {k:0 for k in ARMS}
    limits = np.r_[0,np.flatnonzero(np.diff(group))+1,len(group)]
    for lo,hi in zip(limits[:-1],limits[1:],strict=True):
        v = estimate[lo:hi]
        guard = (v[:,1:4]<=0.).all(1)
        for k,ok in zip(ARMS,(np.ones(len(v),dtype=bool),guard,guard & (v[:,4]>=.75)),strict=True):
            score = np.where(ok,v[:,0],-np.inf)
            idx = int(score.argmax())
            if score[idx]<=0.:
                continue
            a,b,m = i[lo+idx],j[lo+idx],amount[lo+idx]
            paths[k][a] += m/r[a]*(inc[a]-cv[a])
            paths[k][b] -= m/r[b]*(inc[b]-cv[b])
            counts[k] += 1
    for p in paths.values():
        r_new = np.linalg.norm(p-cv,axis=-1).max(1)
        np.testing.assert_array_equal(r_new>0.,r>0.)
        np.testing.assert_allclose(np.add.reduceat(r_new,offsets[:-1]),
            .5*np.add.reduceat(r,offsets[:-1]),rtol=1e-7,atol=1e-14)
    return paths,counts


def run(reg,parent,donors,replay):
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    p,cp = PUBLIC/'analysis.json',PUBLIC/'head_freeze.json'
    expected = json.loads(cp.read_text()) if replay else None
    if replay and (expected['registration']!=reg or not p.is_file()):
        raise ValueError('Replay requires matching completed readout')
    sources = {s:base.parent.site_data(s,torch,model,shape) for s in SITES}
    outer,inner = base.shape_run.load_producers(torch,model,donors)
    cuts = {s:base.base.cuts_for([sources[k] for k in SITES if k!=s]) for s in SITES}
    old = json.loads((base.PUBLIC/'analysis.json').read_text())
    frozen_old = json.loads((base.PUBLIC/'head_freeze.json').read_text())
    if old['registration']!=parent or frozen_old['registration']!=parent:
        raise ValueError('Unexpected marginal control')
    heads,meta,fits = {},{},0
    for held in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {held}',flush=True)
        for seed in SEEDS:
            donor = {}
            if not replay:
                for s,d in sources.items():
                    if s==held:
                        continue
                    c = base.shape_run.candidate(d,inner[tuple(sorted((s,held))),seed],torch,model)
                    _,z = base.parent.features(d,c['inc'])
                    donor[s] = actions(d,c,z,cuts[held][0])
            key = f'{held}/{seed}'
            heads[held,seed],meta[key],n = fit(held,seed,reg,donor,None if expected is None else expected['heads'][key])
            fits += n
    prior.write_json(cp,dict(registration=reg,heads=meta),immutable=True)
    rows = {}
    for loc,s in enumerate(SITES):
        d = sources[s]
        cv = d['cv']
        rows[s] = []
        h = base.hindcast(d['history'],d['scale'])
        for seed in SEEDS:
            c = base.shape_run.candidate(d,outer[seed][loc],torch,model)
            _,z = base.parent.features(d,c['inc'])
            X,q,info,norm = actions(d,c,z)
            est = heads[s,seed].predict(X)
            est[:,:4] *= norm[:,None]
            paths,counts = apply(cv,c['inc'],d['offsets'],q,info,est)
            marginal,_,_ = base.fit(s,seed,parent,{},frozen_old['heads'][f'{s}/{seed}'])
            marginal_est = marginal.predict(z).reshape(-1,2,4)*d['scale'][:,None,None]
            m,allocation = base.method.rollout(cv,c['inc'],d['offsets'],marginal_est,h)
            paths.update(half=cv+.5*(c['inc']-cv),marginal_guard=m['guarded'])
            arms = base.readout(d,paths,cuts[s])
            check = next(v['arms'] for v in old['by_location'][s] if v['seed']==seed)
            if any(arms['marginal_guard'][k]!=check['guarded'][k] for k in ('cv','half','passes','undefined')):
                raise ValueError('Frozen marginal rule changed')
            contrast = {k:base.base.paired_summary(d,paths[k],paths['marginal_guard'],c['r']>0)['lower_pp']
                        for k in ARMS}
            rows[s].append(dict(seed=seed,arms=arms,allocation=dict(active_queries=dict(**counts,
                marginal_guard=allocation['active_queries']['guarded']),transferred_radius={}),
                guarded_vs_matched=None,vs_marginal=contrast))
    summary,_ = base.intervals(rows)
    draw = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))
    contrasts = {}
    for name in ARMS:
        v = np.array([np.mean([a['vs_marginal'][name] for a in rows[s]]) for s in SITES])
        contrasts[name] = dict(point=float(v.mean()),ci95=np.quantile(v[draw].mean(1),[.025,.975]).tolist())
    primary = summary['direct_guard']
    prior.write_json(p,dict(registration=reg,heads=len(meta),summary=summary,by_location=rows,
        paired_vs_marginal=contrasts,matched_all_recipients=True,matched_query_budget=True,
        primary_development_pass=(primary['vs_half']['ci95'][0]>0. and primary['passes']>=summary['half']['passes']),
        original_risk_unchanged=True,formal_confirmation=False,deployment=False,
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
        reg,parent,donors = register()
        if phase=='register':
            print(reg)
        else:
            if platform.system()!='Darwin' or platform.machine()!='arm64':
                raise RuntimeError('Use native arm64 .venv-pytorch on this host')
            run(reg,parent,donors,phase=='replay')
