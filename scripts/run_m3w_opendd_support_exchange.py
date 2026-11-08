"""Train and score support-preserving equal-budget motion exchanges."""
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
from scripts import run_m3w_opendd_causal_amplitude as parent
from scripts.audit_m3w_opendd_radius_risk import transition
from src.evaluation import m3w_support_exchange as method

prior, base, shape_run = parent.prior, parent.base, parent.shape_run
PUBLIC = parent.PUBLIC.parent/'opendd_support_exchange_v1'
PRIVATE = parent.PRIVATE.parent/PUBLIC.name
SITES, SEEDS = parent.SITES, parent.SEEDS


def register():
    parent_reg,donors = parent.register()
    files = [Path(__file__).resolve(),Path(method.__file__),PUBLIC/'protocol.md',
             parent.PUBLIC/'analysis.json',parent.PUBLIC/'amplitude_spread.json',
             parent.PUBLIC/'replay_execution.json',ROOT/'scripts/audit_m3w_opendd_radius_risk.py']
    p = PUBLIC/'registration.json'
    prior.write_json(p,dict(prior_registration=parent_reg,sites=SITES,seeds=SEEDS,
        original_risk_budget=.02,dronecrowd_confirmation_access=False,external_access=False,
        source='opendd_exposed_development_only',files={str(f.relative_to(ROOT)):prior.sha(f) for f in files}),
        immutable=True)
    return prior.sha(p),donors


def hindcast(h,scale):
    from src.world_model.m3w_ols_replacement import rollout

    gains = []
    for n in (4,5,6):
        cv,ols,_,_ = rollout(h[:,:n],8-n)
        y = h[:,n:]-h[:,n-1:n]
        gains.append((np.linalg.norm(cv-y,axis=-1)-np.linalg.norm(ols-y,axis=-1)).mean(1))
    return np.mean(gains,axis=0)/scale


def fit(held,seed,reg,donor,expected):
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor

    train = [s for s in SITES if s!=held]
    ident = dict(registration=reg,held=held,seed=seed,train=train)
    p = PRIVATE/f'{held}_{seed}.joblib'
    receipt = p.with_suffix('.json')
    if p.exists() and receipt.exists():
        v = json.loads(receipt.read_text())
        if v['identity']!=ident or prior.sha(p)!=v['sha256'] or (expected is not None and expected!=v):
            raise ValueError('Cached exchange head provenance mismatch')
        return joblib.load(p),v,0
    if expected is not None:
        raise ValueError('Replay refuses to fit')
    x = np.concatenate([donor[s][0] for s in train])
    y = np.concatenate([donor[s][1] for s in train])
    w = np.concatenate([np.full(len(donor[s][0]),1./len(donor[s][0])) for s in train])
    head = ExtraTreesRegressor(n_estimators=32,max_depth=7,min_samples_leaf=12,n_jobs=1,random_state=seed)
    head.fit(x,y,sample_weight=w)
    tmp = p.with_suffix('.pending')
    joblib.dump(head,tmp,compress=1)
    tmp.replace(p)
    v = dict(identity=ident,sha256=prior.sha(p))
    prior.write_json(receipt,v,immutable=True)
    prior.write_json(PRIVATE/'heartbeat.json',dict(**ident,pid=os.getpid(),phase='head_complete'))
    return head,v,1


def observed_numerators(d,p,cut):
    from src.evaluation.m3w_selection_forecast_readout import errors

    cv,y,valid = d['cv'],d['y'],d['valid']
    known = valid.any(1)
    pred,_ = errors(p,y,valid)
    ref,_ = errors(cv,y,valid)
    radius = np.linalg.norm(p-cv,axis=-1).max(1)
    delta = np.where(known,pred-ref,0.)
    easy = known & (ref>0.) & (ref<=cut)
    harm = delta.clip(0.)
    debit = radius[~known].sum()
    return np.array([harm.sum()+debit,harm[easy].sum()+debit,delta[easy].sum()])


def readout(d,paths,cuts):
    from src.evaluation.m3w_selection_forecast_readout import policy

    cv = d['cv']
    full = {}
    for name,p in paths.items():
        risk = policy(p,cv,cv,d['y'],d['valid'],d['recordings'],easy_cut=cuts[0],hard_cut=cuts[1])
        support = (p!=cv).any(axis=(1,2))
        full[name] = dict(risks=risk['recordings'],
            cv=base.paired_summary(d,p,cv,support),
            half=base.paired_summary(d,p,paths['half'],support),edits=int(support.sum()),
            worst_defined_easy=max((100*v['known_easy_net_degradation'] for v in risk['recordings'].values()
                if v['known_easy_net_degradation'] is not None),default=None))
    half = full['half']['risks']
    output = {}
    for name,row in full.items():
        records = list(row['risks'].values())
        status = Counter(transition(half[k],v) for k,v in row['risks'].items()) if name in method.ARMS else None
        output[name] = dict(cv=row['cv'],half=row['half'],edits=row['edits'],
            passes=sum(v['supported'] for v in records),views=len(records),
            undefined=sum(v['completion_easy_risk_upper'] is None for v in records),
            transitions=status,worst_defined_easy=row['worst_defined_easy'])
    return output


def intervals(rows):
    draw = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))

    def summarize(nested):
        n = sum(v is not None for seq in nested for v in seq)
        if n!=len(SITES)*len(SEEDS):
            return dict(status='not_estimable',defined=n,required=len(SITES)*len(SEEDS))
        x = np.array([np.mean(v) for v in nested])
        return dict(status='estimated',point=float(x.mean()),ci95=np.quantile(x[draw].mean(1),[.025,.975]).tolist(),
                    positive_locations=int((x>0).sum()))

    output = {}
    for a in rows[SITES[0]][0]['arms']:
        values = [[v['arms'][a] for v in rows[s]] for s in SITES]
        flat = [v for seq in values for v in seq]
        trans = Counter()
        for v in flat:
            if v['transitions']:
                trans.update(v['transitions'])
        output[a] = dict(**{k:sum(v[k] for v in flat) for k in ('edits','passes','views','undefined')},
            transitions=trans,vs_cv=summarize([[v['cv']['lower_pp'] for v in seq] for seq in values]),
            vs_half=summarize([[v['half']['lower_pp'] for v in seq] for seq in values]),
            active_queries=sum(v['allocation']['active_queries'].get(a,0) for seq in rows.values() for v in seq),
            transferred_radius=sum(v['allocation']['transferred_radius'].get(a,0.) for seq in rows.values() for v in seq),
            worst_defined_easy=max(v['worst_defined_easy'] for v in flat if v['worst_defined_easy'] is not None))
    paired = summarize([[v['guarded_vs_matched'] for v in rows[s]] for s in SITES])
    return output,paired


def run(reg,donors,replay):
    import torch
    from src.world_model import m3w_opendd_velocity_candidate as model
    from src.world_model import m3w_radius_shape as shape

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    t0 = monotonic()
    p,freeze = PUBLIC/'analysis.json',PUBLIC/'head_freeze.json'
    expected = json.loads(freeze.read_text()) if replay else None
    if replay and (expected['registration']!=reg or not p.is_file()):
        raise ValueError('Require exact completed run before replay')
    sources = {s:parent.site_data(s,torch,model,shape) for s in SITES}
    outer,inner = shape_run.load_producers(torch,model,donors)
    cuts = {s:base.cuts_for([sources[k] for k in SITES if k!=s]) for s in SITES}
    previous = json.loads((parent.PUBLIC/'amplitude_spread.json').read_text())['by_location']
    fits,meta,heads = 0,{},{}
    for held in SITES:
        print(f'{"Verifying" if replay else "Fitting"} {held}',flush=True)
        for seed in SEEDS:
            donor = {}
            for s,d in sources.items():
                if s==held:
                    continue
                c = shape_run.candidate(d,inner[tuple(sorted((s,held))),seed],torch,model)
                _,x = parent.features(d,c['inc'])
                y = method.labels(d['cv'],d['y'],d['valid'],c['inc'],d['scale'],cuts[held][0])
                donor[s] = x[c['pool']],y[c['pool']].reshape(-1,8)
            k = f'{held}/{seed}'
            heads[held,seed],meta[k],n = fit(held,seed,reg,donor,None if expected is None else expected['heads'][k])
            fits += n
    prior.write_json(freeze,dict(registration=reg,heads=meta),immutable=True)
    rows = {}
    for i,s in enumerate(SITES):
        d = sources[s]
        cv = d['cv']
        h = hindcast(d['history'],d['scale'])
        rows[s] = []
        for seed in SEEDS:
            c = shape_run.candidate(d,outer[seed][i],torch,model)
            _,x = parent.features(d,c['inc'])
            m = heads[s,seed].predict(x).reshape(-1,2,4)*d['scale'][:,None,None]
            paths, allocation = method.rollout(cv,c['inc'],d['offsets'],m,h)
            paths.update(half=cv+.5*(c['inc']-cv),cv=cv,incumbent=c['inc'])
            arms = readout(d,paths,cuts[s])
            old = next(v['policies']['fixed_half'] for v in previous[s] if v['seed']==seed)
            if arms['half']['passes']!=old['passed'] or arms['half']['cv']!=old['vs_cv']:
                raise ValueError('Fixed half did not reproduce')
            paired = (base.paired_summary(d,paths['guarded'],paths['matched_hindcast'],c['r']>0)['lower_pp']
                      if allocation['active_queries']['guarded'] else None)
            actual = observed_numerators(d,paths['guarded'],cuts[s][0])-observed_numerators(d,paths['half'],cuts[s][0])
            rows[s].append(dict(seed=seed,arms=arms,allocation=allocation,
                guarded_actual_risk_delta=actual.tolist(),guarded_vs_matched=paired))
    result,paired = intervals(rows)
    primary = result['guarded']
    prior.write_json(p,dict(registration=reg,heads=len(meta),summary=result,by_location=rows,
        guarded_vs_matched=paired,matched_query_budget=True,matched_all_recipients=True,
        primary_development_pass=(primary['vs_half']['ci95'][0]>0. and primary['passes']>=result['half']['passes']),
        original_risk_unchanged=True,formal_confirmation=False,deployment=False,
        stage5c_executed=False,smc_enabled=False),immutable=True)
    receipt = PUBLIC/('replay_execution.json' if replay else 'execution.json')
    if not receipt.exists():
        prior.write_json(receipt,dict(registration=reg,analysis_sha256=prior.sha(p),new_heads_this_execution=fits,
            fresh_exact_replay=replay,pid=os.getpid(),torch_threads=4,workers=0,wall_seconds=round(monotonic()-t0,3)))
    prior.write_json(PRIVATE/'heartbeat.json',dict(pid=os.getpid(),phase='complete',replay=replay))
    print(prior.sha(p),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run','replay'))
    phase = parser.parse_args().phase
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
