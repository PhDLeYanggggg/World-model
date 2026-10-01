"""Fixed-upstream43, crossed forest head seeds; no transfer-selected winner."""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_source_forest as parent
from src.world_model import m3w_crossed_seed as api

NAME='european_crossed_head_seed_v1'
PUBLIC=parent.PUBLIC.parent/NAME;PRIVATE=parent.PRIVATE.parent/NAME
CONFIG=ROOT/'configs'/('m3w_'+NAME+'.json')
base,core,inner=parent.base,parent.core,parent.inner
digest,immutable=parent.digest,parent.immutable


def registration():
    cfg=json.loads(CONFIG.read_text());parent.registration()
    seal=parent.PUBLIC/'verification.json';assert digest(seal)==cfg['parent_seal_sha256']
    v=json.loads(seal.read_text())
    for k,h in v['source_bindings'].items():assert digest(ROOT/k)==h
    for k,h in v['artifacts'].items():assert digest(parent.PUBLIC/k)==h
    assert cfg['estimator']==json.loads(parent.CONFIG.read_text())['estimator']
    paths=parent.closure(ROOT,['scripts.run_m3w_crossed_head_seed'])
    paths += [CONFIG,PUBLIC/'protocol.md',ROOT/'tests/test_m3w_crossed_seed.py']
    return cfg,dict(parent_seal_sha256=digest(seal),bindings={str(p.relative_to(ROOT)):digest(p) for p in paths},
        upstream_seed=43,head_seeds=[17,29,43],independent_roles_read=False)


def beat(**kw):
    d=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    base.inter.json_write(PRIVATE/'heartbeat.json',d)
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
    print(json.dumps(d),flush=True)


def guard(cfg):
    if shutil.disk_usage(PRIVATE).free<cfg['disk_reserve_bytes']+32*2**20:
        raise OSError('Preserve original10GiB reserve; use CREATE if necessary')


def contexts(data,jobs,oid):
    causal={k:data[k] for k in inner.old.parent.CAUSAL_KEYS}
    for c in base.floor_api.contexts(causal,jobs,oid):
        if c['job']['old_identity']['seed']==43:
            yield c


def train(cfg,data,jobs,oid,pilot=False,resume=False,replay=False):
    originals=parent.forest_docs();refs=[];began=time.monotonic()
    for c in contexts(data,jobs,oid):
        for site in inner.sources(c):
            name=c['name']+'_fit_'+site;original=originals[name];assert original['seed']==43
            old=joblib.load(ROOT/original['checkpoint']['path'])
            neural=core.read_checkpoint(ROOT/original['identity']['parent_checkpoint']['path'])
            at,ids,x,env,y,_,upstream=inner.training_arrays(c,data,site)
            assert upstream==neural['identity']['upstream']
            tr,val,partition=parent.parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
            assert partition==original['partition']==neural['partition']
            core.exact(old['preprocess'],neural['preprocess'])
            for seed in cfg['new_head_seeds']:
                guard(cfg);group=name+'_head'+str(seed)
                home=PRIVATE/('replay' if replay else 'heads')/group
                cp,done=home/'checkpoint.joblib',home/'complete.json'
                identity=dict(registration_sha256=digest(PUBLIC/'registration.json'),
                    group=group,base_group=name,upstream=upstream,upstream_seed=43,head_seed=seed,
                    parent_forest=original['checkpoint'],parent_neural=original['identity']['parent_checkpoint'],partition=partition)
                if done.exists() and resume and not replay:
                    doc=json.loads(done.read_text());assert doc['identity']==identity
                    assert base.artifact(cp)==doc['checkpoint']
                else:
                    state=parent.api.fit(x[tr],env[tr],y[tr],data['sites'][ids][tr],data['recordings'][ids][tr],
                        data['frames'][ids][tr],old['preprocess'],settings=cfg['estimator'],seed=seed,identity=identity,
                        path=cp,heartbeat=lambda **kw:beat(state='crossed_training',group=group,**kw),
                        resume=resume and not replay,stop_at=cfg['pilot_trees'] if pilot else None)
                    if pilot:
                        eligible=[d for d in originals.values() if d['seed']==43]
                        ns=[d['partition']['train_rows'] for d in eligible for _ in cfg['new_head_seeds']]
                        leaf,depth=cfg['estimator']['min_samples_leaf'],cfg['estimator']['max_depth']
                        nodes=[min(2**(depth+1)-1,max(1,2*(n//leaf)-1)) for n in ns]
                        structure=state['model'].estimators_[0].tree_.__getstate__()
                        size=structure['nodes'].dtype.itemsize+structure['values'][0].nbytes
                        bound=math.ceil((sum(nodes)+2*max(nodes))*cfg['estimator']['trees']*size*1.01)
                        bound+=(len(nodes)+2)*2**20+32*2**20
                        free=shutil.disk_usage(PRIVATE).free
                        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
                        immutable(PUBLIC/'pilot.json',dict(group=group,trees=cfg['pilot_trees'],fit_seconds=state['seconds'],
                            elapsed_seconds=time.monotonic()-began,peak_RSS_bytes=rss,checkpoint_bytes=cp.stat().st_size,
                            node_payload_bytes=size,conservative_storage_bytes=bound,free_bytes=free,
                            disk_reserve_bytes=cfg['disk_reserve_bytes'],storage_sufficient=free-bound>cfg['disk_reserve_bytes'],
                            memory_feasible=rss<40*2**30,estimated_total_fit_seconds=state['seconds']*8*sum(ns)/int(tr.sum()),
                            estimated_runtime_not_measurement=True))
                        return
                    api.assert_shared_fit(old,state)
                    if replay:
                        ref=joblib.load(PRIVATE/'heads'/group/'checkpoint.joblib');parent.api.exact_forest(ref,state)
                        np.testing.assert_array_equal(parent.api.predict(ref,x[val],env[val])[0],parent.api.predict(state,x[val],env[val])[0])
                        immutable(PUBLIC/'fit_replay.json',dict(group=group,exact_except_elapsed=True,
                            seconds=time.monotonic()-began,all48_retrained=False));return
                    valid=parent.validation(c,at,ids,x,env,y,val,neural,state,data)
                    doc=dict(identity=identity,source=site,upstream_seed=43,head_seed=seed,partition=partition,
                        checkpoint=base.artifact(cp),checkpoint_bytes=cp.stat().st_size,trees=len(state['model'].estimators_),
                        seconds=state['seconds'],trace=state['trace'],input_hashes=state['input_hashes'],
                        same_input_hashes_as_parent43=True,validation=valid)
                    immutable(done,doc)
                refs.append(base.artifact(done));beat(state='fit_complete',fits=len(refs),group=group)
    assert len(refs)==48
    cached={k:v['checkpoint'] for k,v in originals.items() if v['seed']==43};assert len(cached)==24
    immutable(PUBLIC/'training_freeze.json',dict(new_groups=refs,cached43=cached,new_fits=48,cached_fits=24,
        seconds=time.monotonic()-began,result_source='fresh_run48_cached_verified24',independent_roles_read=False))


def docs():
    base.inter.committed(PUBLIC/'training_freeze.json')
    freeze=json.loads((PUBLIC/'training_freeze.json').read_text());out={}
    for ref in freeze['new_groups']:
        assert base.artifact(ROOT/ref['path'])==ref
        doc=json.loads((ROOT/ref['path']).read_text())
        assert base.artifact(ROOT/doc['checkpoint']['path'])==doc['checkpoint']
        out[(doc['identity']['base_group'],doc['head_seed'])]=doc
    for group,doc in parent.forest_docs().items():
        if doc['seed']!=43:continue
        assert doc['checkpoint']==freeze['cached43'][group];out[(group,43)]=doc
    assert len(out)==72
    return out


def views(data,jobs,oid):
    fits=docs();old={r['view']:r for r in json.loads((parent.PUBLIC/'decision_freeze.json').read_text())['rows']}
    for c in contexts(data,jobs,oid):
        states={};passes={}
        for source in inner.sources(c):
            group=c['name']+'_fit_'+source
            states[source]={s:joblib.load(ROOT/fits[group,s]['checkpoint']['path']) for s in api.SEEDS}
            for seed in (17,29):api.assert_shared_fit(states[source][43],states[source][seed])
            passes[source]={s:fits[group,s]['validation']['completion_screen']['finite_completion_supported'] for s in api.SEEDS}
        for pair,(fitting,outer) in enumerate(c['pairs']):
            for source in fitting:
                target=next(s for s in fitting if s!=source)
                assert not {source,target}&(set(c['producer_sites'])|set(c['controller_sites'])|set(outer))
                at=np.flatnonzero(data['sites'][c['ids']]==target);ids=c['ids'][at]
                p={};support=None
                for s in api.SEEDS:
                    p[s],check=parent.api.predict(states[source][s],c['x'][at],c['env'][at])
                    if support is not None:np.testing.assert_array_equal(support,check)
                    support=check
                act=api.actions(p,c['moving'][at],support,data['recordings'][ids],data['frames'][ids],ids,passes[source])
                view=c['name']+f'_pair{pair}_from_'+source;prev=old[view]
                assert base.inter.array_hash(ids)==prev['ids_hash']
                assert base.inter.array_hash(p[43])==prev['prediction_hashes']['forest']
                for a,b in [('seed43','forest'),('screen43','screened')]:
                    assert base.inter.array_hash(act[a])==prev['action_hashes'][b]
                meta=dict(view=view,site=target,source=source,upstream_seed=43,ids_hash=prev['ids_hash'],
                    source_screen={str(s):v for s,v in passes[source].items()},
                    prediction_hashes={str(s):base.inter.array_hash(v) for s,v in p.items()},
                    action_hashes={k:base.inter.array_hash(v) for k,v in act.items()},parent43_exact=True)
                yield c,at,ids,p,act,states[source][43]['preprocess'],meta


def decide(data,jobs,oid):
    rows=[]
    for *_,meta in views(data,jobs,oid):
        rows.append(meta)
        if len(rows)%12==0:beat(state='freeze_actions',views=len(rows))
    assert len(rows)==72
    immutable(PUBLIC/'decision_freeze.json',dict(rows=rows,head_views=216,causal_only=True,all_parent43_exact=True))


def evaluate(data,jobs,oid,replay=False):
    base.inter.committed(PUBLIC/'decision_freeze.json');began=time.monotonic()
    frozen={r['view']:r for r in json.loads((PUBLIC/'decision_freeze.json').read_text())['rows']}
    old={(r['view'],r['policy']):r['metric'] for r in json.loads((parent.PUBLIC/'readout.json').read_text())['rows']}
    rows=[];quality=[];components=[];checks=queries=parent_checks=0
    for c,at,ids,p,act,pr,meta in views(data,jobs,oid):
        assert meta==frozen[meta['view']]
        cv,cf,(floor,ff),(neural,nf)=base.floor_api.costs(c,data,at)
        y=core.targets(cv,floor,neural,c['job']['design']['easy_cut']);known=np.isfinite(cv)
        easy=known&(cv>0)&(cv<=c['job']['design']['easy_cut'])
        _,group=np.unique(np.rec.fromarrays([data['recordings'][ids],data['frames'][ids]]),return_inverse=True)
        count={k:np.bincount(group,weights=v.astype(int)) for k,v in act.items()}
        for s in (17,29):
            match=np.minimum(count['seed43'],count['seed'+str(s)])
            np.testing.assert_array_equal(count['ref43_matched'+str(s)],match)
            np.testing.assert_array_equal(count['seed'+str(s)+'_matched'],match);queries+=len(match)
        for s in api.SEEDS:
            quality.append(dict(view=meta['view'],site=meta['site'],head_seed=s,
                score=parent.quality(p[s],y,pr,data,ids)))
            components.append(dict(view=meta['view'],site=meta['site'],source=meta['source'],head_seed=s,
                accounting=api.component_accounting(y,p[s],act['seed'+str(s)])))
        for policy,take in {**act,'floor':np.zeros(len(ids),bool)}.items():
            m=base.floor_api.metric(cv,floor,neural,cf,ff,nf,np.where(take,neural,floor),np.where(take,nf,ff),take,
                data['valid'][ids],c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            chosen=easy&take;ref=float(floor[chosen].sum())
            m['selected_easy_positive_harm_ratio']=float(np.maximum(neural-floor,0)[chosen].sum())/ref if ref>0 else None
            checks+=parent.accounting.audit_metric(m,cv,floor,neural,cf,ff,nf,take,data['valid'][ids],
                c['job']['design']['easy_cut'],c['job']['design']['hard_cut'])
            if policy in ('seed43','screen43','floor'):
                original={'seed43':'forest','screen43':'screened','floor':'floor'}[policy]
                core.exact(m,old[meta['view'],original]);parent_checks+=1
            rows.append(dict(**meta,policy=policy,metric=m))
        if len(quality)%36==0:beat(state='readout',views=len(quality)//3)
    assert len(rows)==792 and len(quality)==216 and parent_checks==216
    immutable(PUBLIC/'readout.json',dict(rows=rows,quality=quality,components=components,
        independent_metric_checks=checks,query_count_checks=queries,parent43_metric_views_exact=parent_checks,
        independent_confirmation=False,deployment_changed=False))
    immutable(PUBLIC/('evaluation_replay.json' if replay else 'evaluation_runtime.json'),dict(
        seconds=time.monotonic()-began,full_readout_exact_replay=replay,all_parent43_actions_scores_metrics_exact=True))


def report(cfg,reg):
    assert json.loads((PUBLIC/'fit_replay.json').read_text())['exact_except_elapsed']
    assert json.loads((PUBLIC/'evaluation_replay.json').read_text())['full_readout_exact_replay']
    fits=docs();d=json.loads((PUBLIC/'readout.json').read_text());rows=d['rows']
    ci=lambda a:parent.locality_interval(a,cfg['bootstrap_resamples'],cfg['bootstrap_seed'])
    results={}
    m={(r['view'],r['policy']):r for r in rows}
    q={(r['view'],r['head_seed']):r for r in d['quality']}
    for s in api.SEEDS:
        source=[v for (g,k),v in fits.items() if k==s]
        result=dict(source_validation_MSE=ci([(v['source'],v['validation']['scores']['forest']) for v in source]),
            source_screen_pass=sum(v['validation']['completion_screen']['finite_completion_supported'] for v in source),policies={})
        for policy in ('seed'+str(s),'screen'+str(s)):
            rr=[r for r in rows if r['policy']==policy]
            result['policies'][policy]={k:ci([(r['site'],r['metric'][k]) for r in rr]) for k in
                ('all_gain_floor','easy_gain_floor','hard_gain_floor','FDE_gain_floor','intervention_rate')}
            z=result['policies'][policy]
            z['all_risk']=parent.accounting.risk_coverage(rr,'selected_positive_harm_ratio')
            z['easy_risk']=parent.accounting.risk_coverage(rr,'selected_easy_positive_harm_ratio')
            z['worst_whole_easy_degradation_percent']=max(-r['metric']['easy_gain_floor'] for r in rr if r['metric']['easy_gain_floor'] is not None)
        if s!=43:
            result['matched_ADE_improvement_over43']=ci([(r['site'],100*(r['metric']['error_sum']-
                m[v,'seed'+str(s)+'_matched']['metric']['error_sum'])/r['metric']['error_sum'])
                for (v,p),r in m.items() if p=='ref43_matched'+str(s)])
            result['transferred_MSE_difference_over43']=ci([(r['site'],r['score']-q[v,43]['score']) for (v,k),r in q.items() if k==s])
        results[str(s)]=result
    failures=[r for r in rows if r['policy']=='screen43' and api.risk_status(r['metric']['selected_easy_positive_harm_ratio'])=='violating']
    assert len(failures)==6
    persistence=[]
    for r in failures:
        view=r['view'];one=dict(view=view,source=r['source'],site=r['site'],head_seeds={})
        for s in api.SEEDS:
            raw=m[view,'seed'+str(s)]['metric'];screen=m[view,'screen'+str(s)]['metric']
            one['head_seeds'][str(s)]=dict(raw_risk=raw['selected_easy_positive_harm_ratio'],
                screened_risk=screen['selected_easy_positive_harm_ratio'],
                raw_status=api.risk_status(raw['selected_easy_positive_harm_ratio']),
                screened_status=api.risk_status(screen['selected_easy_positive_harm_ratio']),
                source_screen_pass=r['source_screen'][str(s)])
        persistence.append(one)
    fresh=[v for (g,k),v in fits.items() if k!=43]
    summary=dict(heads=results,parent_failure_persistence=persistence,new_fits=48,cached_verified_fits=24,
        fixed_upstream_seed=43,no_best_seed_selection=True,
        new_fit_seconds=sum(v['seconds'] for v in fresh),new_checkpoint_bytes=sum(v['checkpoint_bytes'] for v in fresh),
        independent_metric_checks=d['independent_metric_checks'],query_count_checks=d['query_count_checks'],
        parent43_metric_views_exact=d['parent43_metric_views_exact'],
        result_source='fresh_run48_heads_and216_head_views_cached_verified24_heads',
        independent_confirmation=False,deployment_changed=False)
    immutable(PUBLIC/'summary.json',summary)
    test=subprocess.run([sys.executable,'-m','pytest','tests/test_m3w_crossed_seed.py','tests/test_m3w_source_forest.py',
        'tests/test_m3w_source_checkpoint.py','tests/test_m3w_source_checkpoint_report.py',
        'tests/test_m3w_unknown_outcome_bounds.py','-q'],cwd=ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    (PUBLIC/'scoped_tests.txt').write_text(test.stdout+test.stderr)
    immutable(PUBLIC/'verification.json',dict(status='verified_fixed_upstream_seed_control',source_bindings=reg['bindings'],
        artifacts={p.name:digest(p) for p in PUBLIC.iterdir() if p.is_file() and p.name!='verification.json'},
        first_new_fit_exact_replay=True,full_readout_exact_replay=True,independent_confirmation=False,
        deployment_changed=False,full_legacy_suite='not_run'))
    print(json.dumps(dict(new_fits=48,cached_fits=24,head_views=216,verified=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','pilot','train','replay_fit','decide','evaluate','replay_evaluate','report'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    cfg,reg=registration()
    if a.phase=='register':immutable(PUBLIC/'registration.json',reg);print('Registered crossed cost-head seeds');return
    assert json.loads((PUBLIC/'registration.json').read_text())==reg;base.inter.committed(PUBLIC/'registration.json')
    core.torch.set_num_threads(4);core.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard(cfg);beat(state='starting',phase=a.phase)
        if a.phase=='report':report(cfg,reg);return
        _,_,data,jobs,oid,_,_,_=inner.old.load()
        if a.phase in ('pilot','train','replay_fit'):
            if a.phase=='train':
                pilot=json.loads((PUBLIC/'pilot.json').read_text());assert pilot['storage_sufficient'] and pilot['memory_feasible']
            train(cfg,data,jobs,oid,a.phase=='pilot',a.resume,a.phase=='replay_fit')
        elif a.phase=='decide':decide(data,jobs,oid)
        else:evaluate(data,jobs,oid,a.phase=='replay_evaluate')


if __name__=='__main__':main()
