"""Freeze and evaluate a centered-risk policy with retained-count controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 Python required before Torch import')
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import probe_m3w_signed_bias as bias
from src.world_model import m3w_centered_risk_policy as api
parent,base=bias.parent,bias.base
PUBLIC=parent.PUBLIC.parent/'european_centered_risk_policy_v1'
PRIVATE=parent.PRIVATE.parent/'european_centered_risk_policy_v1'
CONFIG='configs/m3w_european_centered_risk_policy_v1.json'
FILES=[CONFIG,'src/world_model/m3w_centered_risk_policy.py','scripts/run_m3w_centered_risk_policy.py',
       'tests/test_m3w_centered_risk_policy.py',str(PUBLIC.relative_to(ROOT)/'protocol.md')]
ARMS=parent.api.ARMS
POLICIES=('floor','raw_neural',*[a+'_'+p for a in ARMS for p in ('raw_independent','raw_joint',*api.POLICIES)])
CONTRASTS=[(a+'_centered_joint',a+'_'+p) for a in ARMS for p in ('matched_raw_joint','raw_joint','centered_independent')]


def beat(state,**kw):
    r=dict(state=state,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',r)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)


def identity():
    cfg=json.loads((ROOT/CONFIG).read_text());sp=bias.PUBLIC/'verification.json'
    assert base.digest(sp)==cfg['parent_seal_sha256']
    seal=json.loads(sp.read_text())
    for p,h in seal['source_bindings'].items():assert base.digest(ROOT/p)==h,p
    for p,h in seal['artifacts'].items():assert base.digest(bias.PUBLIC/p)==h,p
    assert cfg['groups']==108 and cfg['arms']==list(ARMS) and cfg['risk_budget']==.02 and cfg['node_limit']==256
    assert not any(cfg[k] for k in ('new_neural_training','new_parameter_fitting','threshold_search',
        'independent_roles_read','formal_primary_replaced','deployment_changed','stage5c_executed','smc_enabled'))
    return cfg,dict(parent_seal=base.artifact(sp),bindings={p:base.digest(ROOT/p) for p in FILES},
        policies=list(POLICIES),contrasts=CONTRASTS,groups=108,independent_roles_read=False,formal_primary_replaced=False)


def load():
    cfg,ident=identity();assert json.loads((PUBLIC/'registration.json').read_text())==json.loads(json.dumps(ident))
    base.inter.committed(PUBLIC/'registration.json')
    _,data,jobs,oid,_,_,_,_,_,pid=parent.load()
    parents={Path(r['path']).stem:r for r in json.loads((parent.PUBLIC/'decision_freeze.json').read_text())['groups']}
    biases={Path(r['path']).stem:r for r in json.loads((bias.PUBLIC/'completion.json').read_text())['groups']}
    return cfg,ident,data,jobs,oid,pid,parents,biases


def action_group(c,causal,pair,parent_ref,bias_ref,pid,cfg):
    for r in (parent_ref,bias_ref):assert base.artifact(ROOT/r['path'])==r
    old=json.loads((ROOT/parent_ref['path']).read_text());b=json.loads((ROOT/bias_ref['path']).read_text())
    assert b['parent_fit']==old['fit'] and not b['held_labels_used'] and not b['policy_changed']
    roles=b['roles'];base.floor_api.api.assert_roles(roles['producer_sites'],roles['controller_sites'],roles['training_sites'],roles['held_sites'])
    held,scores,pr=parent.predictions(c,causal,old['fit'],pid)
    assert base.artifact(ROOT/old['arrays']['path'])==old['arrays']
    with np.load(ROOT/old['arrays']['path'],allow_pickle=False) as z:cached={k:z[k].copy() for k in z.files}
    ids=c['ids'][held];np.testing.assert_array_equal(ids,cached['ids']);out=dict(ids=ids,eligible=cached['eligible'])
    name=c['name']+f'_pair{pair}';ridge=json.loads((base.floor_api.PRIVATE/'fits'/name/'complete.json').read_text())
    r=ridge['artifacts']['scores'];assert base.artifact(ROOT/r['path'])==r
    with np.load(ROOT/r['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'],ids)
        utility=(z['scores'][:,5].astype(float)-z['scores'][:,6].astype(float))/pr['cost_scale']
    keys=np.array([str(s)+'|'+str(r) for s,r in zip(causal['sites'][ids],causal['recordings'][ids])])
    stats={};checks={}
    for arm,p in scores.items():
        q=parent.api.head.parent.signed(p.astype(float))/pr['cost_scale'];delta=b['fits'][arm]['nonnegative_offset']
        fresh,stats[arm]=api.decisions(utility,q,delta,out['eligible'],keys,causal['frames'][ids],ids,node_limit=cfg['node_limit'])
        checks[arm]=api.check_queries(fresh,q,delta,out['eligible'],keys,causal['frames'][ids],ids)
        np.testing.assert_array_equal(cached[arm+'_independent'],out['eligible'] & (q<=0).all(1))
        out[arm+'_raw_independent']=cached[arm+'_independent'];out[arm+'_raw_joint']=cached[arm+'_joint']
        for k,v in fresh.items():out[arm+'_'+k]=v
    return out,dict(parent_action=parent_ref,bias_fit=bias_ref,utility_scores=r,
        solver=stats,checks=checks,future_fields_removed=True,held_outcomes_used=False)


def decide(cfg,ident,data,jobs,oid,pid,parents,biases,*,resume=False,replay=False,pilot=False):
    causal={k:data[k] for k in parent.CAUSAL_KEYS};refs=[];start=time.monotonic()
    for c in base.floor_api.contexts(causal,jobs,oid):
        for pair in range(6):
            name=c['name']+f'_pair{pair}';path=PRIVATE/'decisions'/(name+'.json');arr=path.with_suffix('.npz')
            if shutil.disk_usage(PRIVATE).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB and completed action groups')
            if path.exists() and resume and not replay:
                d=json.loads(path.read_text());assert d['identity']==json.loads(json.dumps(ident))
                assert d['parent_action']==parents[name] and d['bias_fit']==biases[name]
                assert base.artifact(ROOT/d['arrays']['path'])==d['arrays']
            else:
                if path.exists() and not replay:raise ValueError('Existing actions require resume or replay')
                arrays,meta=action_group(c,causal,pair,parents[name],biases[name],pid,cfg)
                parent.previous.descriptor.write_arrays(arr,arrays,replay)
                d=dict(identity=ident,arrays=base.artifact(arr),**meta);base.immutable_json(path,d)
            refs.append(base.artifact(path));beat('actions_replayed' if replay else 'actions_frozen',group=name,complete=len(refs))
            if pilot:
                size=arr.stat().st_size+path.stat().st_size
                estimate=size*cfg['groups']*3
                free=shutil.disk_usage(PRIVATE).free
                receipt=dict(identity=ident,group=name,seconds=time.monotonic()-start,
                    group_artifact_bytes=size,estimated_full_bytes_with_3x_safety=estimate,
                    free_disk_bytes=free,reserve_bytes=cfg['disk_reserve_bytes'],
                    storage_sufficient=free-estimate>cfg['disk_reserve_bytes'],
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid(),held_outcomes_used=False)
                base.immutable_json(PUBLIC/'pilot.json',receipt)
                if not receipt['storage_sufficient']:raise OSError('Pilot storage projection violates10GiB reserve')
                return
    assert len(refs)==108
    base.immutable_json(PUBLIC/'decision_freeze.json',dict(identity=ident,groups=refs,held_outcomes_used=False))
    base.immutable_json(PUBLIC/('prediction_replay.json' if replay else 'decision_runtime.json'),dict(identity=ident,
        groups=108,exact=replay,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pid=os.getpid(),future_fields_removed=True,new_neural_training=False,new_parameter_fitting=False))


def contrast(x,y):
    return dict(ADE_gain_percent=100*(1-x['error_sum']/y['error_sum']) if y['error_sum']>0 else None,
        easy_gain_floor_difference_pp=None if x['easy_gain_floor'] is None or y['easy_gain_floor'] is None else x['easy_gain_floor']-y['easy_gain_floor'],
        hard_gain_floor_difference_pp=None if x['hard_gain_floor'] is None or y['hard_gain_floor'] is None else x['hard_gain_floor']-y['hard_gain_floor'],
        selected_harm_reduction_pp=None if x['selected_positive_harm_ratio'] is None or y['selected_positive_harm_ratio'] is None else 100*(y['selected_positive_harm_ratio']-x['selected_positive_harm_ratio']),
        all_reference_harm_reduction_pp=100*(y['positive_harm_over_all_floor']-x['positive_harm_over_all_floor']),
        intervention_difference_pp=100*(x['intervention_rate']-y['intervention_rate']))


def evaluate(cfg,ident,data,jobs,oid,pid,parents,biases,*,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json');f=json.loads((PUBLIC/'decision_freeze.json').read_text())
    assert f['identity']==json.loads(json.dumps(ident));refs={Path(r['path']).stem:r for r in f['groups']}
    start=time.monotonic();rows=[];exchanges=[];solver=[]
    for c in base.floor_api.contexts({k:data[k] for k in parent.CAUSAL_KEYS},jobs,oid):
        cv,cf,(floor,ff),(neural,nf)=base.floor_api.costs(c,data,np.arange(len(c['ids'])))
        for pair in range(6):
            name=c['name']+f'_pair{pair}';r=refs[name];assert base.artifact(ROOT/r['path'])==r
            d=json.loads((ROOT/r['path']).read_text());solver.append(d['solver'])
            assert base.artifact(ROOT/d['arrays']['path'])==d['arrays']
            with np.load(ROOT/d['arrays']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            held=np.flatnonzero(np.isin(data['sites'][c['ids']],c['pairs'][pair][1]));np.testing.assert_array_equal(c['ids'][held],a['ids'])
            for site in c['pairs'][pair][1]:
                at=data['sites'][a['ids']]==site;ix=held[at];common=dict(group=name,site=site,seed=c['job']['old_identity']['seed'])
                for policy in POLICIES:
                    take=np.zeros(len(ix),bool) if policy=='floor' else np.ones(len(ix),bool) if policy=='raw_neural' else a[policy][at]
                    m=base.floor_api.metric(cv[ix],floor[ix],neural[ix],cf[ix],ff[ix],nf[ix],
                        np.where(take,neural[ix],floor[ix]),np.where(take,nf[ix],ff[ix]),take,data['valid'][c['ids'][ix]],
                        c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
                    m.update(parent.harm_accounting(floor[ix],neural[ix],np.isfinite(cv[ix]),take))
                    rows.append(dict(**common,policy=policy,metric=m))
                for new,old in CONTRASTS:
                    m=bias.diagnosis.api.exchange_accounting(floor[ix],neural[ix],a[old][at],a[new][at])
                    exchanges.append(dict(**common,policy=new+'_vs_'+old,metric=m))
        beat('held_development_scored',group=c['name'])
    sites=sorted(set(data['sites']))
    def reduce(rr):
        return {p:{k:base.inter.paired_localities([r for r in rr if r['policy']==p],sites,k,
            cfg['bootstrap_resamples'],cfg['bootstrap_seed']) for k in rr[0]['metric']} for p in sorted({r['policy'] for r in rr})}
    index={(r['group'],r['site'],r['policy']):r['metric'] for r in rows};contrasts=[]
    for new,old in CONTRASTS:
        for r in (v for v in rows if v['policy']==new):
            contrasts.append(dict(group=r['group'],site=r['site'],seed=r['seed'],policy=new+'_vs_'+old,
                metric=contrast(r['metric'],index[(r['group'],r['site'],old)])))
    summary=reduce(rows);paired=reduce(contrasts);gates={}
    worst={p:dict(risk_violating_views=sum(r['metric']['selected_positive_harm_ratio'] is not None and r['metric']['selected_positive_harm_ratio']>.02 for r in rows if r['policy']==p),
        undefined_selected_risk_views=sum(r['metric']['selected_positive_harm_ratio'] is None for r in rows if r['policy']==p),
        abstaining_views=sum(r['metric']['intervention_rate']==0 for r in rows if r['policy']==p),
        worst_easy_gain_CV=min(r['metric']['easy_gain_CV'] for r in rows if r['policy']==p and r['metric']['easy_gain_CV'] is not None)) for p in POLICIES}
    for arm in ARMS:
        key=arm+'_centered_joint_vs_'+arm+'_matched_raw_joint';m=paired[key]
        selected=[r['metric'] for r in rows if r['policy']==arm+'_centered_joint']
        positive=lambda k:m[k]['ci95'] is not None and m[k]['ci95'][0]>0
        g=dict(equal_count_confirmed=all(r['metric']['intervention_difference_pp']==0 for r in contrasts if r['policy']==key),
            equal_count_ADE_advantage=positive('ADE_gain_percent'),
            equal_count_all_reference_harm_reduction=positive('all_reference_harm_reduction_pp'),
            every_view_defined_selected_risk_within_2percent=all(v['selected_positive_harm_ratio'] is not None and v['selected_positive_harm_ratio']<=.02 for v in selected),
            every_view_easy_preserved=all(v['easy_gain_CV'] is not None and v['easy_gain_CV']>=-2 for v in selected),
            no_zero_CV_harm=all(v['zero_CV_harmed']==0 for v in selected))
        g['exploratory_screen_pass']=all(g.values());gates[arm]=g
    gates.update(formal_primary_replaced=False,independent_confirmation=False,calibration_certificate=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)
    out=dict(identity=ident,result_source='fresh_policy_actions_and_held_development_readout',
        forecasts_heads_offsets='cached_verified',summary=summary,paired=paired,exchanges=reduce(exchanges),worst_views=worst,
        by_seed={str(s):reduce([r for r in rows if r['seed']==s]) for s in (17,29,43)},gates=gates,
        solver={a:{p:{k:sum(v[a][p][k] for v in solver) for k in solver[0][a][p]} for p in ('centered','matched_raw')} for a in ARMS})
    old=json.loads((parent.PUBLIC/'summary.json').read_text())
    for r in ['floor',*[a+'_'+p for a in ARMS for p in ('raw_independent','raw_joint')]]:
        key=r.replace('_raw_','_')
        for k,v in old['summary'][key].items():assert summary[r][k]==v,(r,k)
    base.immutable_json(PRIVATE/'details.json',dict(rows=rows,contrasts=contrasts,exchanges=exchanges))
    base.immutable_json(PUBLIC/'summary.json',out);base.immutable_json(PUBLIC/'gates.json',gates)
    base.immutable_json(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),dict(identity=ident,
        exact=replay,summary=base.artifact(PUBLIC/'summary.json'),details=base.artifact(PRIVATE/'details.json'),
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['register','pilot','decide','replay','evaluate','replay_evaluate'],required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    if a.phase=='register':
        cfg,ident=identity();base.immutable_json(PUBLIC/'registration.json',ident);print(json.dumps(ident));return
    base.torch.set_num_threads(4);base.torch.set_num_interop_threads(1)
    with (PRIVATE/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat('started',phase=a.phase);v=load()
        if a.phase in ('pilot','decide','replay'):
            if a.phase=='decide':assert json.loads((PUBLIC/'pilot.json').read_text())['storage_sufficient']
            decide(*v,resume=a.resume,replay=a.phase=='replay',pilot=a.phase=='pilot')
        else:evaluate(*v,replay=a.phase=='replay_evaluate')
        beat('complete',phase=a.phase)


if __name__=='__main__':main()
