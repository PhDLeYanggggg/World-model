"""Fit source-held event heads and rerank the frozen local decision budget."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_opendd_prefix_risk as prefix

PUBLIC = prefix.PUBLIC.parent/'opendd_easy_support_v1'
PRIVATE = prefix.PRIVATE.parent/PUBLIC.name
prior = prefix.prior
cal = prefix.calibration
SITES,SEEDS = prefix.SITES,prefix.SEEDS
ARMS = ('local','mass_matched','event_matched','support_first')


def beat(phase,**kwargs):
    msg = dict(phase=phase,pid=os.getpid(),**kwargs)
    prior.write_json(PRIVATE/'heartbeat.json',msg)
    print(json.dumps(msg),flush=True)


def register():
    p_reg,source,outer,inner = prefix.register()
    p = prefix.PUBLIC/'analysis.json'
    result = json.loads(p.read_text())
    if result['registration']!=p_reg or result['independent_confirmation'] or len(result['fits'])!=21 or (
        prior.sha(p)!=json.loads((prefix.PUBLIC/'replay_execution.json').read_text())['analysis_sha256']):
        raise ValueError('Require verified prefix analysis')
    paths = (Path(__file__).resolve(),PUBLIC/'protocol.md',Path(prefix.__file__),p,
        prefix.PUBLIC/'registration.json',prefix.PUBLIC/'replay_execution.json',
        ROOT/'src/evaluation/m3w_opendd_easy_support.py',ROOT/'src/evaluation/m3w_opendd_prefix_risk.py',
        ROOT/'src/evaluation/m3w_selection_forecast_readout.py',ROOT/'src/evaluation/m3w_opendd_cv_abstention.py')
    binding = {}
    for i in result['fits']:
        k = f'{i["held"]}_{i["seed"]}'
        ck = prefix.PRIVATE/f'{k}.joblib'
        meta = json.loads(ck.with_suffix('.json').read_text())
        if meta['identity']!=i or meta['sha256']!=prior.sha(ck):
            raise ValueError('Frozen risk-head identity mismatch')
        binding[k] = meta['sha256']
    reg = PUBLIC/'registration.json'
    prior.write_json(reg,dict(files={str(a.relative_to(ROOT)):prior.sha(a) for a in paths},
        seed=SEEDS,sites=SITES,arms=ARMS,frozen_risk_sha256=binding,official_risk_budget=.02,
        dronecrowd_confirmation_access=False,stage5c_executed=False,smc_enabled=False),immutable=True)
    return prior.sha(reg),source,outer,inner,binding,result


def label(d,p,cut):
    ref,_ = errors(d['cv'],d['y'],d['valid'])
    value,_ = errors(p,d['y'],d['valid'])
    legal = (p!=d['cv']).any(axis=(1,2))
    easy = np.isfinite(ref)&(ref>0.)&(ref<=cut)
    return support.event(easy,legal,ref,value),legal


def fit(sources,inner,held,seed,alpha,registration):
    train = [s for s in SITES if s!=held]
    path = PRIVATE/f'{held}_{seed}.joblib'
    meta = path.with_suffix('.json')
    identity = dict(held=held,train=train,seed=seed,alpha=alpha,registration=registration)
    if path.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old['identity']!=identity or old['sha256']!=prior.sha(path):
            raise ValueError('Support-head resume identity changed')
        return joblib.load(path),identity
    xs,ys,ws = [],[],[]
    for site in train:
        pair = tuple(sorted((site,held)))
        d = sources[site]
        cut = cal.parent.easy_cut(sources,[s for s in SITES if s not in pair])
        p,_ = cal.rollout(d,inner[pair,seed,site],alpha,'uniform')
        y,legal = label(d,p,cut)
        if legal.any():
            xs.append(prefix.features(d,p)[legal])
            ys.append(y[legal])
            ws.append(np.full(int(legal.sum()),1./legal.sum()))
    y = np.concatenate(ys)
    if len(set(y))!=2:
        raise ValueError('Support classifier requires both event classes')
    head = ExtraTreesClassifier(n_estimators=32,max_depth=7,min_samples_leaf=12,n_jobs=1,random_state=seed)
    head.fit(np.concatenate(xs),y,sample_weight=np.concatenate(ws))
    tmp = path.with_suffix('.pending')
    joblib.dump(head,tmp,compress=1)
    tmp.replace(path)
    prior.write_json(meta,dict(identity=identity,sha256=prior.sha(path)),immutable=True)
    beat('support_checkpoint',held=held,seed=seed,rows=len(y),events=int(y.sum()))
    return head,identity


def prediction_quality(y,legal,prob):
    y,p = y[legal],prob[legal]
    return dict(rows=len(y),positives=int(y.sum()),prevalence=float(y.mean()) if len(y) else None,
                brier=float(np.mean((y-p)**2)) if len(y) else None,
                average_precision=float(average_precision_score(y,p)) if y.any() else None,
                auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None)


def risk_summary(records):
    vals = list(records.values())
    return dict(views=len(vals),passes=sum(r['supported'] for r in vals),
        undefined_easy=sum(r['completion_easy_risk_upper'] is None for r in vals),
        no_full_reference=sum(r['known_easy_net_degradation'] is None for r in vals),
        defined_easy_excess=sum(r['completion_easy_risk_upper'] is not None and r['completion_easy_risk_upper']>.02 for r in vals))


def evaluate(d,p,masks,cut,hard):
    pred = {a:np.where(m[:,None,None],p,d['cv']) for a,m in masks.items()}
    ref_counts = np.add.reduceat(masks['local'],d['offsets'][:-1])
    for arm in ('mass_matched','event_matched'):
        if not np.array_equal(np.add.reduceat(masks[arm],d['offsets'][:-1]),ref_counts):
            raise ValueError('Count-matched support rule changed query counts')
    risks = {a:policy(v,d['cv'],d['cv'],d['y'],d['valid'],d['recordings'],easy_cut=cut,hard_cut=hard)['recordings']
             for a,v in pred.items()}
    y,_ = label(d,p,cut)
    row = {}
    for arm in ARMS:
        m = masks[arm]
        pair = Counter()
        for r,before in risks['local'].items():
            after = risks[arm][r]
            if after['supported'] and not before['supported']:
                pair['rescues'] += 1
            if before['supported'] and not after['supported']:
                pair['losses'] += 1
            if before['completion_easy_risk_upper'] is None and after['completion_easy_risk_upper'] is not None:
                pair['newly_defined_easy'] += 1
            if before['completion_easy_risk_upper'] is not None and after['completion_easy_risk_upper'] is None:
                pair['lost_defined_easy'] += 1
        row[arm] = dict(cv_relative=readout.metrics(d,pred[arm],m,cut),flight=risk_summary(risks[arm]),
            transitions=dict(pair),selected_affordable_easy=int(y[m].sum()),
            paired_local=readout.metrics(d,pred[arm],m,cut,ref=pred['local']) if m.any() and masks['local'].any() else None)
    row['event_vs_mass'] = (readout.metrics(d,pred['event_matched'],masks['event_matched'],cut,ref=pred['mass_matched'])
                            if ref_counts.sum()>0 else None)
    return row


def summarize(results):
    idx = np.random.default_rng(817).integers(len(SITES),size=(3000,len(SITES)))
    def gain(groups):
        if any(m is None for site in groups for m in site):
            return dict(status='required_matched_view_not_estimable',defined_views=sum(m is not None for g in groups for m in g))
        v = np.asarray([np.mean([m['lower_pp'] for m in g]) for g in groups])
        return dict(status='estimated',lower_pp=float(v.mean()),ci95_pp=np.quantile(v[idx].mean(1),[.025,.975]).tolist(),
                    positive_sites=int((v>0).sum()),worst_site_pp=float(v.min()))
    groups = [results[s] for s in SITES]
    out = {}
    for arm in ARMS:
        values = [[v['policies'][arm] for v in g] for g in groups]
        flat = [v for g in values for v in g]
        cv = [[v['cv_relative'] for v in g] for g in values]
        all_cv = [v for g in cv for v in g]
        out[arm] = dict(cv_relative=gain(cv),paired_local=gain([[v['paired_local'] for v in g] for g in values]),
            switches=sum(v['switches'] for v in all_cv),selected_affordable_easy=sum(v['selected_affordable_easy'] for v in flat),
            ade=float(np.mean([v['ade'] for v in all_cv])),fde=float(np.mean([v['fde'] for v in all_cv])),
            worst_positive_easy_pct=max(v['positive_easy_pct'] for v in all_cv),
            flight={k:sum(v['flight'][k] for v in flat) for k in flat[0]['flight']},
            transitions={k:sum(v['transitions'].get(k,0) for v in flat) for k in (
                'rescues','losses','newly_defined_easy','lost_defined_easy')})
    out['event_vs_mass'] = gain([[v['policies']['event_vs_mass'] for v in g] for g in groups])
    metrics = {}
    for key in ('prevalence','average_precision','brier','auroc'):
        by_site = [[a['support_quality'][key] for a in g if a['support_quality'][key] is not None] for g in groups]
        v = [np.mean(g) for g in by_site if g]
        metrics[key] = dict(equal_site_mean=float(np.mean(v)) if v else None,sites=len(v))
    out['support_quality'] = metrics
    return out


def run(reg,source,outer,inner,binding,original):
    begin = monotonic()
    sources = {s:prefix.data(s,prior.sha(prior.PUBLIC/'registration.json')) for s in SITES}
    donors = prefix.inner_forecasts(sources,source['registration'],inner)
    heads,fit_info = {},[]
    for held in SITES:
        for r in source['sites'][held]:
            seed = r['seed']
            head,meta = fit(sources,donors,held,seed,r['selected']['uniform'],reg)
            heads[held,seed] = head
            fit_info.append(meta)
    rows = {}
    for held,d in sources.items():
        train = [s for s in SITES if s!=held]
        cut = cal.parent.easy_cut(sources,train)
        a = np.concatenate([sources[s]['cv_complete_ade'] for s in train])
        hard = float(np.quantile(a[a>0],.75))
        rows[held] = []
        for i,r in enumerate(source['sites'][held]):
            seed,alpha = r['seed'],r['selected']['uniform']
            p,_ = cal.rollout(d,cal.frozen_forecast(sources,held,seed,outer),alpha,'uniform')
            x = prefix.features(d,p)
            ck = prefix.PRIVATE/f'{held}_{seed}.joblib'
            if prior.sha(ck)!=binding[f'{held}_{seed}']:
                raise ValueError('Frozen risk-head weight changed')
            moments = joblib.load(ck).predict(x)*d['scale'][:,None]
            probs = heads[held,seed].predict_proba(x)[:,1]
            y,legal = label(d,p,cut)
            policy_rows = evaluate(d,p,support.walk(d,p,moments,probs),cut,hard)
            old = original['sites'][held][i]['policies']['local']
            if policy_rows['local']['flight']['passes']!=old['flight']['pass_count'] or (
                policy_rows['local']['cv_relative']['switches']!=old['cv_relative']['switches']):
                raise ValueError('Reference local policy failed replay')
            rows[held].append(dict(seed=seed,policies=policy_rows,support_quality=prediction_quality(y,legal,probs)))
    path = PUBLIC/'analysis.json'
    replay = path.exists()
    prior.write_json(path,dict(registration=reg,fresh_fits=fit_info,sites=rows,summary=summarize(rows),
        scope='development',independent_confirmation=False,official_safety_pass=False),immutable=True)
    beat('complete',elapsed_s=round(monotonic()-begin,3),analysis_sha256=prior.sha(path),exact_replay=replay)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--register-only',action='store_true')
    args = p.parse_args()
    if sys.platform=='darwin' and platform.machine()!='arm64':
        raise SystemExit('Run native arm64 .venv-pytorch/bin/python, not Rosetta')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registered = register()
        if args.register_only:
            print(registered[0])
        else:
            import joblib
            import numpy as np
            import torch
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.metrics import average_precision_score,roc_auc_score
            from src.evaluation import m3w_opendd_easy_support as support
            from src.evaluation import m3w_opendd_velocity_calibration as calibrate, m3w_opendd_cv_abstention as readout
            from src.evaluation.m3w_selection_forecast_readout import errors,policy
            from src.world_model import m3w_ols_replacement as motion,m3w_opendd_velocity_candidate as model

            torch.set_num_threads(4)
            torch.set_num_interop_threads(1)
            prefix.np=cal.np=prefix.velocity.np=cal.parent.np=np
            prefix.torch=cal.torch=prefix.velocity.torch=cal.parent.torch=torch
            prefix.model=cal.model=prefix.velocity.model=model
            cal.calibrate,cal.parent.motion=calibrate,motion
            run(*registered)
