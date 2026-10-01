"""Registered source-only utility choice among three already frozen cost heads."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_source_checkpoint as parent
from scripts.report_m3w_source_checkpoint import locality_interval
from scripts import report_m3w_inner_separability as accounting
from scripts.export_m3w_easy_hurdle_create import closure
from src.world_model import m3w_source_policy_selection as api

NAME='european_source_policy_selection_v1'
PUBLIC=parent.PUBLIC.parent/NAME;PRIVATE=parent.PRIVATE.parent/NAME
CONFIG=ROOT/'configs'/('m3w_'+NAME+'.json')
base=parent.base;core=parent.api.core;digest=parent.digest;immutable=parent.immutable


def registration():
    cfg=json.loads(CONFIG.read_text());seal=parent.PUBLIC/'verification.json'
    assert digest(seal)==cfg['parent_seal_sha256']
    v=json.loads(seal.read_text())
    for k,h in v['source_bindings'].items():assert digest(ROOT/k)==h
    for k,h in v['artifacts'].items():assert digest(parent.PUBLIC/k)==h
    paths=closure(ROOT,['scripts.run_m3w_source_policy_selection'])
    paths += [CONFIG,PUBLIC/'protocol.md',ROOT/'tests/test_m3w_source_policy_selection.py']
    return cfg,dict(parent_seal_sha256=digest(seal),bindings={str(p.relative_to(ROOT)):digest(p) for p in paths},
                    parameter_updates=0,independent_roles_read=False)


def beat(**kw):
    row=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',row)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def fit_docs():
    freeze=json.loads((parent.PUBLIC/'training_freeze.json').read_text())
    out={}
    for ref in freeze['groups']:
        assert base.artifact(ROOT/ref['path'])==ref
        doc=json.loads((ROOT/ref['path']).read_text());assert base.artifact(ROOT/doc['checkpoint']['path'])==doc['checkpoint']
        out[Path(ref['path']).parent.name]=doc
    assert len(out)==72
    return out


def causal_action(p,moving,support,rec,frames,ids):
    return core.decisions(dict(affine=p,nonlinear=p),moving,support,rec,frames,ids)['affine']


def select(data,jobs,oid):
    fits=fit_docs();rows=[];causal={k:data[k] for k in parent.parent.parent.old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        for site in parent.parent.parent.sources(c):
            name=c['name']+'_fit_'+site;doc=fits[name];state=core.read_checkpoint(ROOT/doc['checkpoint']['path'])
            at,ids,x,env,y,_,identity=parent.parent.parent.training_arrays(c,data,site)
            assert identity==state['identity']['upstream'] and state['step']==2000
            _,val,partition=parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
            assert partition==state['partition']
            predictions,support=parent.api.paired_predictions(state,x[val],env[val])
            initial,other=core.predict(state,x[val],env[val],initial=True)
            np.testing.assert_array_equal(support,other)
            predictions=dict(mse=predictions['validation'],final=predictions['final'],initial=initial)
            vid=ids[val]
            actions={k:causal_action(p,c['moving'][at][val],support,data['recordings'][vid],data['frames'][vid],vid)
                     for k,p in predictions.items()}
            selection=api.choose(y[val],actions)
            rows.append(dict(group=name,source=site,checkpoint=doc['checkpoint'],partition=partition,
                source_validation_ids_hash=base.inter.array_hash(vid),selection=selection,
                validation_score_hashes={k:base.inter.array_hash(p) for k,p in predictions.items()},
                validation_action_hashes={k:base.inter.array_hash(v) for k,v in actions.items()}))
            if len(rows)%12==0:beat(state='source_validation_choice',sources=len(rows))
    assert len(rows)==72
    immutable(PUBLIC/'source_choices.json',dict(rows=rows,parameter_updates=0,independent_calibration=False))


def views(data,jobs,oid):
    base.inter.committed(PUBLIC/'source_choices.json')
    choices={r['group']:r for r in json.loads((PUBLIC/'source_choices.json').read_text())['rows']}
    for c,at,ids,p,_,pr,meta in parent.views(data,jobs,oid):
        choice=choices[c['name']+'_fit_'+meta['source']];selected=choice['selection']['selected']
        z=(c['x'][at]-pr['mean'])/pr['std'];support=np.sqrt(np.mean(z*z,axis=1))<=pr['support_limit']
        if selected=='fallback':new=np.zeros_like(p['validation'])
        elif selected=='initial':
            cp=choice['checkpoint'];assert base.artifact(ROOT/cp['path'])==cp
            state=core.read_checkpoint(ROOT/cp['path']);new,other=core.predict(state,c['x'][at],c['env'][at],initial=True)
            np.testing.assert_array_equal(support,other)
        else:new=p['validation' if selected=='mse' else 'final']
        scores=dict(affine=p['validation'],nonlinear=new)
        old=core.decisions(scores,c['moving'][at],support,data['recordings'][ids],data['frames'][ids],ids)
        actions={k:old[v] for k,v in [('mse','affine'),('selected','nonlinear'),('mse_matched','affine_matched'),('selected_matched','nonlinear_matched')]}
        if selected=='fallback':assert not actions['selected'].any()
        out=dict(view=meta['view'],site=meta['site'],source=meta['source'],seed=meta['seed'],selected_head=selected,
            ids_hash=meta['ids_hash'],score_hashes={k:base.inter.array_hash(v) for k,v in scores.items()},
            action_hashes={k:base.inter.array_hash(v) for k,v in actions.items()})
        yield c,at,ids,actions,out


def decide(data,jobs,oid):
    rows=[]
    for _,_,_,_,meta in views(data,jobs,oid):
        rows.append(meta)
        if len(rows)%36==0:beat(state='causal_decisions',views=len(rows))
    assert len(rows)==216
    immutable(PUBLIC/'decision_freeze.json',dict(rows=rows,causal_only=True,threshold_search=False))


def evaluate(data,jobs,oid,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json');started=time.monotonic()
    expected={r['view']:r for r in json.loads((PUBLIC/'decision_freeze.json').read_text())['rows']}
    rows=[];checks=0;queries=0
    for c,at,ids,actions,meta in views(data,jobs,oid):
        assert meta==expected[meta['view']]
        cv,cf,(floor,ff),(neural,nf)=base.floor_api.costs(c,data,at)
        known=np.isfinite(cv);easy=known&(cv>0)&(cv<=c['job']['design']['easy_cut'])
        _,groups=np.unique(np.rec.fromarrays([data['recordings'][ids],data['frames'][ids]]),return_inverse=True)
        count={k:np.bincount(groups,weights=a.astype(int)) for k,a in actions.items()}
        matched=np.minimum(count['mse'],count['selected'])
        np.testing.assert_array_equal(count['mse_matched'],matched);np.testing.assert_array_equal(count['selected_matched'],matched)
        queries+=len(matched)
        for policy,take in {**actions,'floor':np.zeros(len(ids),bool)}.items():
            m=base.floor_api.metric(cv,floor,neural,cf,ff,nf,np.where(take,neural,floor),np.where(take,nf,ff),take,
                data['valid'][ids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            chosen=known&easy&take;ref=float(floor[chosen].sum())
            m['selected_easy_positive_harm_ratio']=float(np.maximum(neural-floor,0)[chosen].sum())/ref if ref>0 else None
            checks+=accounting.audit_metric(m,cv,floor,neural,cf,ff,nf,take,data['valid'][ids],
                c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            rows.append(dict(**meta,policy=policy,metric=m))
        if len(rows)%180==0:beat(state='outcome_readout',views=len(rows)//5)
    assert len(rows)==1080
    immutable(PUBLIC/'readout.json',dict(rows=rows,independent_metric_checks=checks,query_count_checks=queries,
        parameter_updates=0,independent_confirmation=False,deployment_changed=False))
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),dict(seconds=time.monotonic()-started,
        exact_readout_replay=replay,all216_frozen_decisions_reverified=True))


def report(cfg,reg):
    assert json.loads((PUBLIC/'evaluation_replay.json').read_text())['exact_readout_replay']
    assert (PUBLIC/'source_choice_replay.json').exists()
    data=json.loads((PUBLIC/'readout.json').read_text());choices=json.loads((PUBLIC/'source_choices.json').read_text())['rows']
    metrics={(r['view'],r['policy']):r for r in data['rows']};contrasts=[]
    for (view,policy),r in metrics.items():
        if policy!='mse_matched':continue
        other=metrics[view,'selected_matched'];a=r['metric']['error_sum'];b=other['metric']['error_sum']
        contrasts.append(dict(site=r['site'],seed=r['seed'],gain=100*(a-b)/a if a>0 else None))
    ci=lambda pairs:locality_interval(pairs,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    policies={}
    for policy in ('mse','selected','mse_matched','selected_matched'):
        rows=[r for r in data['rows'] if r['policy']==policy]
        policies[policy]={k:ci([(r['site'],r['metric'][k]) for r in rows]) for k in
            ('all_gain_floor','easy_gain_floor','hard_gain_floor','FDE_gain_floor','intervention_rate')}
        for label,key in [('all','selected_positive_harm_ratio'),('easy','selected_easy_positive_harm_ratio')]:
            policies[policy][label+'_risk']=accounting.risk_coverage(rows,key)
        policies[policy]['worst_easy_degradation_percent']=max(-r['metric']['easy_gain_floor'] for r in rows
            if r['metric']['easy_gain_floor'] is not None)
    summary=dict(primary_matched_ADE_improvement=ci([(r['site'],r['gain']) for r in contrasts]),
        by_seed={str(s):ci([(r['site'],r['gain']) for r in contrasts if r['seed']==s]) for s in {r['seed'] for r in contrasts}},
        source_choices=dict(Counter(r['selection']['selected'] for r in choices)),policies=policies,
        independent_metric_checks=data['independent_metric_checks'],query_count_checks=data['query_count_checks'],
        new_neural_training=False,deployment_changed=False,independent_confirmation=False)
    immutable(PUBLIC/'summary.json',summary)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_m3w_source_policy_selection.py',
        'tests/test_m3w_source_checkpoint.py','tests/test_m3w_source_checkpoint_report.py','-q'],cwd=ROOT,capture_output=True,text=True)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(tests.stdout+tests.stderr)
    immutable(PUBLIC/'verification.json',dict(status='verified_development_policy_control',source_bindings=reg['bindings'],
        artifacts={p.name:digest(p) for p in PUBLIC.iterdir() if p.is_file() and p.name!='verification.json'},
        source_choices_replayed=True,full_readout_replayed=True,independent_metric_checks=data['independent_metric_checks'],
        independent_confirmation=False,deployment_changed=False,full_legacy_suite='not_run'))
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','select','replay_select','decide','evaluate','replay_evaluate','report'])
    args=p.parse_args();PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    cfg,reg=registration()
    if args.phase=='register':immutable(PUBLIC/'registration.json',reg);print('Registered source-only decision selection');return
    assert json.loads((PUBLIC/'registration.json').read_text())==reg;base.inter.committed(PUBLIC/'registration.json')
    core.torch.set_num_threads(4);core.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);beat(state='starting',phase=args.phase)
        if args.phase=='report':report(cfg,reg);return
        _,_,data,jobs,oid,_,_,_=parent.parent.parent.old.load()
        if args.phase in ('select','replay_select'):
            select(data,jobs,oid)
            if args.phase=='replay_select':immutable(PUBLIC/'source_choice_replay.json',dict(all72_exact=True))
        elif args.phase=='decide':decide(data,jobs,oid)
        else:evaluate(data,jobs,oid,args.phase=='replay_evaluate')


if __name__=='__main__':main()
