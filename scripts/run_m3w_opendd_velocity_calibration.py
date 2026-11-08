"""Calibrate and compare the frozen velocity candidate without outer-site labels."""
import argparse
import fcntl
from itertools import combinations
import json
from pathlib import Path
import platform
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_velocity_candidate as velocity

PUBLIC = velocity.PUBLIC.parent/'opendd_velocity_calibration_v1'
PRIVATE = velocity.PRIVATE.parent/PUBLIC.name
FROZEN = velocity.PRIVATE
parent = velocity.parent
prior = parent.prior
SEEDS = velocity.SEEDS
SITES = tuple(f'rdb{i}' for i in range(1,8))
ALPHAS = (.025,.05,.075,.10,.15,.25)
ARMS = ('uniform','agreement_cap')


def register():
    files = (Path(__file__).resolve(),PUBLIC/'protocol.md',Path(velocity.__file__),Path(parent.__file__),
        velocity.PUBLIC/'registration.json',velocity.PUBLIC/'analysis.json',velocity.PUBLIC/'replay_execution.json',
        prior.PUBLIC/'registration.json',ROOT/'src/world_model/m3w_opendd_velocity_candidate.py',
        ROOT/'src/evaluation/m3w_opendd_velocity_calibration.py',ROOT/'src/evaluation/m3w_opendd_cv_abstention.py',
        ROOT/'src/evaluation/m3w_selection_forecast_readout.py',ROOT/'src/evaluation/m3w_native_metrics.py')
    frozen = {f'{s}_{i}':prior.sha(FROZEN/f'{s}_{i}_self.pt') for s in SITES for i in SEEDS}
    p = PUBLIC/'registration.json'
    prior.write_json(p,dict(files={str(a.relative_to(ROOT)):prior.sha(a) for a in files},
        checkpoint_sha256=frozen,sites=SITES,seeds=SEEDS,alphas=ALPHAS,arms=ARMS,
        official_risk_budget=.02,dronecrowd_confirmation_access=False,stage5c_executed=False,smc_enabled=False),immutable=True)
    return prior.sha(p),frozen


def data(site,original):
    d = parent.prepare(site,original)
    with np.load(prior.PRIVATE/f'{site}_cohort.npz',allow_pickle=False) as f:
        h, recordings = f['history'],np.repeat(f['recording'],np.diff(f['offsets']))
    x,rot,scale = model.design(h,np.full((len(h),2,2,8,2),np.nan))
    return {**d,'design':torch.from_numpy(x),'rot':rot,'scale':scale,'recordings':recordings,
        'future_local':torch.from_numpy(((d['y']@rot)/scale[:,None,None]).astype(np.float32)),
        'cv_local':torch.from_numpy(((d['cv']@rot)/scale[:,None,None]).astype(np.float32)),
        'valid_torch':torch.from_numpy(d['valid'])}


def frozen_forecast(sources,held,seed,binding):
    path = FROZEN/f'{held}_{seed}_self.pt'
    if prior.sha(path)!=binding[f'{held}_{seed}']:
        raise ValueError('Frozen checkpoint SHA changed')
    ck = torch.load(path,weights_only=True,map_location='cpu')
    train = [s for s in SITES if s!=held]
    expected = dict(registration=prior.sha(velocity.PUBLIC/'registration.json'),held=held,train=train,seed=seed,arm='self')
    if ck['identity']!=expected or ck['step']!=velocity.STEPS:
        raise ValueError('Outer forecast is not source-excluded and complete')
    d = sources[held]
    net = model.network(d['design'].shape[1])
    net.load_state_dict(ck['model'])
    return velocity.predict(net.eval(),d,'self')


def rollout(d,pred,alpha,arm):
    common,cap = calibrate.prepare(d,pred)
    coef = np.zeros(len(common)) if alpha is None else alpha*common.astype(float)
    if arm=='agreement_cap':
        coef *= cap
    return calibrate.apply(d['cv'],pred,coef),coef


def select(sources,inner_bank,outer,seed,arm):
    trials = [dict(alpha=a,sites=[]) for a in ALPHAS]
    for inner in (s for s in SITES if s!=outer):
        pair = tuple(sorted((outer,inner)))
        d,pred = sources[inner],inner_bank[pair,seed,inner]
        cut = parent.easy_cut(sources,[s for s in SITES if s not in pair])
        for row in trials:
            f,c = rollout(d,pred,row['alpha'],arm)
            m = readout.metrics(d,f,c>0,cut)
            row['sites'].append(dict(site=inner,lower_pp=m['lower_pp'],positive_easy_pct=m['positive_easy_pct']))
    return calibrate.pick(trials),trials


def flight_risk(d,pred,cut,hard_cut):
    result = policy(pred,d['cv'],d['cv'],d['y'],d['valid'],d['recordings'],easy_cut=cut,hard_cut=hard_cut)
    r = list(result['recordings'].values())
    return dict(flights=len(r),pass_count=sum(a['supported'] for a in r),
        no_full_easy_reference=sum(a['known_easy_net_degradation'] is None for a in r),
        undefined_selected_easy=sum(a['completion_easy_risk_upper'] is None for a in r),
        defined_easy_excess=sum(a['completion_easy_risk_upper'] is not None and a['completion_easy_risk_upper']>.02 for a in r),
        defined_all_excess=sum(a['completion_all_risk_upper'] is not None and a['completion_all_risk_upper']>.02 for a in r))


def eval_policy(d,pred,alpha,arm,cut,hard_cut):
    candidate, coef = rollout(d,pred,alpha,arm)
    moved = (np.linalg.norm(candidate-d['cv'],axis=-1)>0).any(1)
    if not np.array_equal(moved,coef>0):
        raise ValueError('Effective edit count differs from coefficient support')
    out = dict(cv_relative=readout.metrics(d,candidate,moved,cut),flight=flight_risk(d,candidate,cut,hard_cut),
        mean_positive_alpha=float(coef[moved].mean()) if moved.any() else None,matched={})
    for name in ('minimum','hindcast'):
        old = calibrate.apply(d['cv'],calibrate.old_candidate(d,name),coef)
        if not np.array_equal((np.linalg.norm(old-d['cv'],axis=-1)>0).any(1),moved):
            raise ValueError('Matched old-bank recipients differ')
        out['matched'][name] = readout.metrics(d,candidate,moved,cut,ref=old) if moved.any() else None
    return out


def summary(records):
    result = {}
    draws = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))
    for arm in ('fixed_quarter',*ARMS):
        vals = [[r['policies'][arm] for r in records[s]] for s in SITES]
        cv = [[r['cv_relative'] for r in group] for group in vals]
        flat = [m for g in cv for m in g]
        v = np.asarray([np.mean([r['lower_pp'] for r in g]) for g in cv])
        d = dict(lower_pp=float(v.mean()),ci95_pp=np.quantile(v[draws].mean(1),[.025,.975]).tolist(),
            switches=sum(m['switches'] for m in flat),positive_sites=int((v>0).sum()),
            worst_site_pp=float(v.min()),worst_signed_easy_pct=max(m['signed_easy_pct'] for m in flat),
            worst_positive_easy_pct=max(m['positive_easy_pct'] for m in flat),
            ade=float(np.mean([m['ade'] for m in flat])),fde=float(np.mean([m['fde'] for m in flat])),matched={})
        flight = [r['flight'] for group in vals for r in group]
        d['flight'] = {k:sum(r[k] for r in flight) for k in flight[0]}
        for name in ('minimum','hindcast'):
            x = [[r['matched'][name] for r in group] for group in vals]
            active = [[m['lower_pp'] for m in g if m is not None] for g in x]
            if not all(len(a)==len(SEEDS) for a in active):
                d['matched'][name] = dict(status='required_matched_view_not_estimable',
                    defined_views=sum(map(len,active)),required_views=len(SITES)*len(SEEDS))
            else:
                w = np.asarray([np.mean(a) for a in active])
                d['matched'][name] = dict(status='estimated',lower_pp=float(w.mean()),
                    ci95_pp=np.quantile(w[draws].mean(1),[.025,.975]).tolist(),positive_sites=int((w>0).sum()))
        result[arm] = d
    return result


def run(registration,frozen):
    t = monotonic()
    original = prior.sha(prior.PUBLIC/'registration.json')
    sources = {s:data(s,original) for s in SITES}
    fits,inner = [],{}
    for pair in combinations(SITES,2):
        train = {s:sources[s] for s in SITES if s not in pair}
        for seed in SEEDS:
            net, meta = velocity.fit(train,'+'.join(pair),seed,'self',registration)
            if tuple(meta['train'])!=tuple(train):
                raise ValueError('Inner forecaster does not exclude both sites')
            for s in pair:
                inner[pair,seed,s] = velocity.predict(net,sources[s],'self')
            fits.append(meta)
    rows = {}
    for held,d in sources.items():
        train = [s for s in SITES if s!=held]
        cut = parent.easy_cut(sources,train)
        a = np.concatenate([sources[s]['cv_complete_ade'] for s in train])
        hard_cut = float(np.quantile(a[a>0],.75))
        rows[held] = []
        for seed in SEEDS:
            pred = frozen_forecast(sources,held,seed,frozen)
            view = dict(seed=seed,selected={},calibration={},policies={})
            view['policies']['fixed_quarter'] = eval_policy(d,pred,.25,'uniform',cut,hard_cut)
            for arm in ARMS:
                alpha, trials = select(sources,inner,held,seed,arm)
                view['selected'][arm] = alpha
                view['calibration'][arm] = trials
                view['policies'][arm] = eval_policy(d,pred,alpha,arm,cut,hard_cut)
            rows[held].append(view)
    path = PUBLIC/'analysis.json'
    replay = path.exists()
    prior.write_json(path,dict(registration=registration,fresh_inner_fits=fits,verified_outer_checkpoints=len(frozen),
        sites=rows,summary=summary(rows),scope='development',official_confirmation=False,deployment=False,
        runtime=dict(architecture=platform.machine(),torch=str(torch.__version__),threads=4,workers=0)),immutable=True)
    velocity.beat('complete',elapsed_s=round(monotonic()-t,3),analysis_sha256=prior.sha(path),exact_replay=replay)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-only',action='store_true')
    args = parser.parse_args()
    if sys.platform=='darwin' and platform.machine()!='arm64':
        raise SystemExit('Run native arm64 .venv-pytorch/bin/python, not Rosetta')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration,frozen = register()
        if args.register_only:
            print(registration)
        else:
            import numpy as np
            import torch
            from src.evaluation import m3w_opendd_velocity_calibration as calibrate, m3w_opendd_cv_abstention as readout
            from src.evaluation.m3w_selection_forecast_readout import policy
            from src.world_model import m3w_ols_replacement as motion, m3w_opendd_velocity_candidate as model

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            parent.np,parent.torch,parent.motion = np,torch,motion
            velocity.np,velocity.torch,velocity.model,velocity.PRIVATE = np,torch,model,PRIVATE
            run(registration,frozen)
