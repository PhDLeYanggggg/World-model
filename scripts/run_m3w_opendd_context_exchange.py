"""Compare real past context and a same-query shuffled context control."""
import argparse
from collections import Counter
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
from scripts import run_m3w_opendd_direct_exchange as direct
from scripts.audit_m3w_opendd_direct_exchange import choice_quality
from src.evaluation import m3w_exchange_context as method

base, prior = direct.base,direct.prior
PUBLIC = direct.PUBLIC.parent/'opendd_context_exchange_v1'
PRIVATE = direct.PRIVATE.parent/PUBLIC.name
SITES, SEEDS = direct.SITES,direct.SEEDS
VELOCITY = base.shape_run.frozen.VELOCITY
ARMS = ('own','pedestrian','traffic','shuffled_traffic')


def register():
    parent,_,donors = direct.register()
    files = [Path(__file__).resolve(),Path(method.__file__),PUBLIC/'protocol.md',
        VELOCITY/'registration.json',VELOCITY/'replay_execution.json',
        direct.PUBLIC/'analysis.json',direct.PUBLIC/'choice_attribution.json',
        ROOT/'scripts/audit_m3w_opendd_direct_exchange.py']
    reg = prior.sha(VELOCITY/'registration.json')
    if json.loads((VELOCITY/'analysis.json').read_text())['registration']!=reg:
        raise ValueError('Unexpected parent context provenance')
    inputs = {}
    for site in SITES:
        p = prior.PRIVATE.parent/VELOCITY.name/f'{site}_context.npz'
        v = json.loads(p.with_suffix('.json').read_text())
        if v['registration']!=reg or v['sha256']!=prior.sha(p):
            raise ValueError(f'Past context changed: {site}')
        inputs[site] = v['sha256']
    p = PUBLIC/'registration.json'
    prior.write_json(p,dict(prior_registration=parent,sites=SITES,seeds=SEEDS,
        original_risk_budget=.02,dronecrowd_confirmation_access=False,external_access=False,
        source='opendd_exposed_development_only',context=inputs,
        files={str(f.relative_to(ROOT)):prior.sha(f) for f in files}),immutable=True)
    return prior.sha(p),donors,inputs


def site_data(site,torch,model,shape,digest):
    d = base.parent.site_data(site,torch,model,shape)
    p = prior.PRIVATE.parent/VELOCITY.name/f'{site}_context.npz'
    if prior.sha(p)!=digest:
        raise ValueError('Context bytes changed after registration')
    with np.load(p,allow_pickle=False) as f:
        context = f['context']
    if context.shape!=(len(d['history']),2,2,8,2):
        raise ValueError('Unexpected observation grid')
    np.testing.assert_array_equal(context[:,0],shape.neighbors(d['history'],d['offsets'])[:,0])
    z = [method.relative(d['history'],context[:,i],d['rot'],d['scale']) for i in range(2)]
    return dict(**d,pedestrian_context=z[0],traffic_context=z[1],
                motor_supported=np.isfinite(context[:,1,:,-1]).all(-1).any(1))


def features(d,inc):
    _, own = base.parent.features(d,inc)
    return method.feature_arms(own,d['pedestrian_context'],d['traffic_context'],d['offsets'])


def fit(held,seed,arm,reg,donor,expected):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    train = [s for s in SITES if s!=held]
    ident = dict(registration=reg,held=held,seed=seed,arm=arm,train=train)
    p = PRIVATE/f'{held}_{seed}_{arm}.joblib'
    receipt = p.with_suffix('.json')
    if p.exists() and receipt.exists():
        v = json.loads(receipt.read_text())
        if v['identity']!=ident or prior.sha(p)!=v['sha256'] or (expected is not None and v!=expected):
            raise ValueError('Cached context head changed')
        return joblib.load(p),v,0
    if expected is not None:
        raise ValueError('Replay cannot fit')
    x = np.concatenate([donor[s][arm][0] for s in train])
    y = np.concatenate([donor[s][arm][1] for s in train])
    w = np.concatenate([donor[s][arm][2] for s in train])
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


def paired(d,p,q,mask=None):
    from src.evaluation.m3w_opendd_location import paired as diff

    y,valid,cv = d['y'],d['valid'],d['cv']
    if mask is not None:
        y,valid,cv,p,q = (x[mask] for x in (y,valid,cv,p,q))
    mass = float(np.nansum(np.linalg.norm(cv-y,axis=-1)))
    return float(100*diff(q,p,y,valid)[1].sum()/mass) if mass>0 else None


def summarize(rows):
    rng = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))
    names = tuple(rows[SITES[0]][0]['paired'])
    out = {}
    for k in names:
        values = [[v['paired'][k] for v in rows[s]] for s in SITES]
        if any(a is None for seq in values for a in seq):
            out[k] = dict(status='not_estimable_on_all_sites',
                         defined_views=sum(a is not None for seq in values for a in seq))
        else:
            p = np.array([np.mean(seq) for seq in values])
            out[k] = dict(point=float(p.mean()),ci95=np.quantile(p[rng].mean(1),[.025,.975]).tolist(),
                          positive_locations=int((p>0).sum()))
    policies = {}
    for arm in ('half',*ARMS):
        views = [v['arms'][arm] for seq in rows.values() for v in seq]
        qual, transitions = Counter(),Counter()
        for v in views:
            qual.update(v['choice_quality'])
            transitions.update(v['recording_transitions'])
        policies[arm] = dict(passes=sum(v['passes'] for v in views),views=sum(v['views'] for v in views),
            undefined=sum(v['undefined'] for v in views),choice_quality=qual,recording_transitions=transitions)
    return policies,out


def readout(d,paths,quality,cuts):
    from src.evaluation.m3w_selection_forecast_readout import policy

    cv = d['cv']
    risks = {}
    for a,p in paths.items():
        risks[a] = policy(p,cv,cv,d['y'],d['valid'],d['recordings'],easy_cut=cuts[0],
                          hard_cut=cuts[1])['recordings']
    out = {}
    for a,v in risks.items():
        tr = Counter(base.transition(risks['half'][k],r) for k,r in v.items())
        out[a] = dict(passes=sum(r['supported'] for r in v.values()),views=len(v),
            undefined=sum(r['completion_easy_risk_upper'] is None for r in v.values()),
            choice_quality=quality.get(a,{}),recording_transitions=tr)
    pairs = {a+'_vs_half':paired(d,p,paths['half']) for a,p in paths.items() if a!='half'}
    pairs.update(pedestrian_vs_own=paired(d,paths['pedestrian'],paths['own']),
        traffic_vs_pedestrian=paired(d,paths['traffic'],paths['pedestrian']),
        traffic_vs_shuffled=paired(d,paths['traffic'],paths['shuffled_traffic']),
        traffic_vs_pedestrian_motor_supported=paired(d,paths['traffic'],paths['pedestrian'],d['motor_supported']))
    return out,pairs


def run(reg,donors,inputs,replay):
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    p,cp = PUBLIC/'analysis.json',PUBLIC/'head_freeze.json'
    expected = json.loads(cp.read_text()) if replay else None
    if replay and (expected['registration']!=reg or not p.is_file()):
        raise ValueError('Require a matching completed run')
    sources = {s:site_data(s,torch,model,shape,inputs[s]) for s in SITES}
    outer,inner = base.shape_run.load_producers(torch,model,donors)
    cuts = {s:base.base.cuts_for([sources[k] for k in SITES if k!=s]) for s in SITES}
    heads,meta,fits = {},{},0
    for held in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {held}',flush=True)
        for seed in SEEDS:
            data = {}
            if not replay:
                for s,d in sources.items():
                    if s==held:
                        continue
                    c = base.shape_run.candidate(d,inner[tuple(sorted((s,held))),seed],torch,model)
                    a = features(d,c['inc'])
                    x,y,w = direct.actions(d,c,a['own'],cuts[held][0])
                    data[s] = dict(own=(x,y,w))
                    for k in ARMS[1:]:
                        data[s][k] = direct.actions(d,c,a[k])[0],y,w
            for arm in ARMS:
                k = f'{held}/{seed}/{arm}'
                heads[held,seed,arm],meta[k],n = fit(held,seed,arm,reg,data,
                    None if expected is None else expected['heads'][k])
                fits += n
    prior.write_json(cp,dict(registration=reg,heads=meta),immutable=True)
    rows = {}
    fixed = json.loads((direct.PUBLIC/'analysis.json').read_text())['by_location']
    for i,s in enumerate(SITES):
        print(f'Scoring {s}',flush=True)
        d,rows[s] = sources[s],[]
        cv = d['cv']
        for seed in SEEDS:
            c = base.shape_run.candidate(d,outer[seed][i],torch,model)
            a = features(d,c['inc'])
            paths,choices,quality = dict(half=cv+.5*(c['inc']-cv)),{},{}
            for arm,x in a.items():
                X,q,info,norm = direct.actions(d,c,x)
                est = heads[s,seed,arm].predict(X)
                est[:,:4] *= norm[:,None]
                v,n = direct.apply(cv,c['inc'],d['offsets'],q,info,est)
                paths[arm] = v['direct_guard']
                choices[arm] = q,est,n,norm
            _,truth,_ = direct.actions(d,c,a['own'],cuts[s][0])
            for arm,(q,est,n,norm) in choices.items():
                y = truth.copy()
                y[:,:4] *= norm[:,None]
                quality[arm] = choice_quality(q,est,y)['direct_guard']
                if quality[arm]['selected']!=n['direct_guard']:
                    raise ValueError('Choice accounting differs')
            arms,comparisons = readout(d,paths,quality,cuts[s])
            ref = next(v for v in fixed[s] if v['seed']==seed)['arms']['half']
            if arms['half']['passes']!=ref['passes'] or arms['half']['undefined']!=ref['undefined']:
                raise ValueError('Fixed half reference did not reproduce')
            rows[s].append(dict(seed=seed,arms=arms,paired=comparisons))
    policies, contrasts = summarize(rows)
    success = (policies['traffic']['passes']>=policies['half']['passes'] and all(
        contrasts[k]['ci95'][0]>0 for k in ('traffic_vs_pedestrian','traffic_vs_shuffled')))
    prior.write_json(p,dict(registration=reg,heads=len(meta),policies=policies,contrasts=contrasts,
        by_location=rows,motor_supported_rows={s:int(d['motor_supported'].sum()) for s,d in sources.items()},
        total_rows=sum(len(d['cv']) for d in sources.values()),matched_query_budget=True,
        matched_all_recipients=True,primary_traffic_pass=success,formal_confirmation=False,deployment=False,
        original_risk_unchanged=True,stage5c_executed=False,smc_enabled=False),immutable=True)
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
        reg,donors,inputs = register()
        if phase=='register':
            print(reg)
        else:
            if platform.system()!='Darwin' or platform.machine()!='arm64':
                raise RuntimeError('Use native arm64 .venv-pytorch on this host')
            run(reg,donors,inputs,phase=='replay')
