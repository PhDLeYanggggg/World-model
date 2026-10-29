"""One registered donor-objective repair with actual fixed-policy readout."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import time
from scripts import run_m3w_coupled_support as parent
from src.evaluation import m3w_easy_constrained as api
from src.evaluation import m3w_easy_constrained_policy as policy

ROOT,sha,once=parent.ROOT,parent.sha,parent.once;pooled=parent.pooled
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_easy_constrained_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_easy_constrained_v1'
REG=PUBLIC/'registration.json'
APPROVAL=PUBLIC.parent/'european_gate_contract_v1/explicit_user_confirmation_20261029.md'


def register():
    result=parent.admission()
    files=[Path(__file__),ROOT/'src/evaluation/m3w_easy_constrained.py',
        ROOT/'src/evaluation/m3w_easy_constrained_policy.py',ROOT/'tests/test_m3w_easy_constrained.py',
        ROOT/'tests/test_m3w_easy_constrained_policy.py',ROOT/'tests/test_m3w_easy_constrained_runner.py',
        PUBLIC/'protocol.md',APPROVAL,parent.REG,parent.PUBLIC/'complete.json',parent.PUBLIC/'summary.json',
        ROOT/'src/evaluation/m3w_selected_tail_audit.py',ROOT/'src/world_model/m3w_unknown_outcome_bounds.py',
        ROOT/'src/world_model/m3w_temporal_auxiliary_readout.py',ROOT/'src/world_model/m3w_component_calibration.py',
        ROOT/'src/world_model/m3w_scaled_risk_readout.py']
    once(REG,dict(bindings={str(p.resolve().relative_to(ROOT)):sha(p) for p in files},views=72,
        contexts=sorted(result[0]['contexts']),seeds=result[0]['seeds'],k=32,primary=api.PRIMARY,
        changed_factor='donor_easy_quadratic_constraint',risk_budget=.02,matched_risk_required=False,
        risk_MSE_secondary=True,held_inference=False,independent_roles_read=False,
        whole_pipeline_OOF=False,deployment_changed=False,neural_updates=0))
    return result


def admission():
    result=register()
    for rel in json.loads(REG.read_text())['bindings']:pooled.previous.previous.committed(ROOT/rel)
    pooled.previous.previous.committed(REG);return result


def runtime():
    pooled.previous.previous.train.torch.set_num_threads(4)
    pooled.previous.previous.train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def beat(**kw):
        event=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(event)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if kw.get('state')!='analogue_queries':print(json.dumps(event),flush=True)
        if time.monotonic()-start>43200:raise TimeoutError('12h boundary;resume unchanged protocol')
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('Preserve10GiB reserve')
    beat(state='admitted');return start,beat


def diagnostic(out,take,y,p,scale):
    np=api.np;d={k:v[take] for k,v in out['channel'].items()};known=np.isfinite(y).all(1)
    result=dict(local_supported=int(out['supported'][take].sum()),nested_supported=int(out['nested_supported'][take].sum()),
        donor_conflicts=int(d['conflict'].sum()),
        reduced_alpha=int((d['alpha']<d['parent_alpha']-1e-12).sum()),alpha_zero=int((d['alpha']==0).sum()),
        alpha_one=int((d['alpha']==1).sum()),alpha_sum=float(d['alpha'].sum()),
        donor_easy_constraint_violations=int((d['after'][:,1]>d['before'][:,1]+1e-10*np.maximum(1,d['before'][:,1])).sum()),
        channel_loss={k:d[k].sum(0).tolist() for k in ('before','parent_after','after','easy_only_after')},
        channel_sha256={k:api.base.fingerprint(v) for k,v in d.items()},arms={},strata={})
    for a,v in p.items():
        result['arms'][a]=dict(easy_exceeds_all=int((v[:,4]>v[:,1]+1e-7).sum()))
    masks=dict(donor_conflict=d['conflict']>0,no_donor_conflict=d['conflict']==0,
        reduced_alpha=d['alpha']<d['parent_alpha']-1e-12,unchanged_alpha=d['alpha']>=d['parent_alpha']-1e-12,
        alpha_zero=d['alpha']==0,alpha_positive=d['alpha']>0)
    for name,m in masks.items():
        obs=m&known
        result['strata'][name]=dict(rows=int(m.sum()),known=int(obs.sum()),unknown=int((m&~known).sum()),
            signed_SSE={a:{t:float(((((v[obs,col]-.02*v[obs,ref])-(y[obs,col]-.02*y[obs,ref]))/scale)**2).sum())
                for t,col,ref in (('all',1,2),('easy',4,3))} for a,v in p.items()})
    return result


def execute(phase,resume=False):
    reg,_,ancestor=admission();states=pooled.previous.previous.models(reg)
    _,origins=pooled.previous.previous.parent.models(reg)
    olddone=json.loads((parent.PUBLIC/'complete.json').read_text());oldrefs={Path(r['path']).name:r for r in olddone['groups']}
    pilot,check=phase=='pilot',phase=='check';finish=PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not check:raise ValueError('Completed phase immutable')
    if phase=='run' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Pilot required')
    start,beat=runtime();rows=[];receipts=[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        contexts=sorted(reg['contexts'].items())
        for context,refs in contexts[:1] if pilot else contexts:
            packets,held=pooled.LOADER.packets(refs,ancestor,beat);del held
            bank,spans,_=pooled.api.pool(packets)
            bank['scale']=api.np.concatenate([api.np.full(len(packets[s][0]),packets[s][-1]['scale']) for s in sorted(packets)])
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                paths={s:PUBLIC/'groups'/f'{context}_{s}_seed{seed}.json' for s in sorted(packets)}
                if resume and not check and not pilot and all(p.exists() for p in paths.values()):
                    fresh=[json.loads(p.read_text()) for p in paths.values()]
                    if any(r['registration_sha256']!=sha(REG) for r in fresh):raise ValueError('Resume binding changed')
                else:
                    values,queries,_=pooled.predictions(context,seed,packets,origins,states)
                    q=api.np.concatenate([queries[s] for s in sorted(packets)])
                    frozen=api.np.concatenate([values[s]['origin'] for s in sorted(packets)])
                    out=api.retrieve(bank,q,frozen,heartbeat=beat);fresh=[]
                    for source in sorted(packets):
                        oldpath=parent.PUBLIC/'groups'/paths[source].name
                        if sha(oldpath)!=oldrefs[oldpath.name]['sha256']:raise ValueError('Prior result changed')
                        old=json.loads(oldpath.read_text());lo,hi=spans[source]
                        take=(out['indices']>=lo)&(out['indices']<hi);ix=out['indices'][take]-lo
                        args=packets[source];_,env,y,_,_,_,_,pr=args
                        p={'origin':values[source]['origin'][ix]}
                        full={'origin':values[source]['origin']}
                        for arm in api.MODES:
                            p[arm]=p['origin'].copy();p[arm][:,[1,4]]=out['predictions'][arm][take]
                            full[arm]=full['origin'].copy();full[arm][ix]=p[arm]
                        references=[r for r in out['references'] if r['query_source']==source]
                        if references!=old['references']:raise ValueError('Reference bank changed')
                        for arm,prior in (('origin','origin'),('parent_blend','nested_support_blend')):
                            if api.base.fingerprint(p[arm])!=old['prediction_sha256'][prior]:raise ValueError('Frozen prediction changed')
                        scores={a:api.base.score(y[ix],v,env[ix],pr['scale'],api.np.ones(len(ix),bool)) for a,v in p.items()}
                        for arm,prior in (('origin','origin'),('parent_blend','nested_support_blend')):
                            if scores[arm]!=old['scores'][prior]:raise ValueError('Prior score changed')
                        diag=diagnostic(out,take,y[ix],p,pr['scale'])
                        if diag['donor_easy_constraint_violations']:raise AssertionError('Donor easy bound failed')
                        pol,selected=policy.evaluate(args,full,pooled.api.eligible(args))
                        if not api.np.array_equal(selected['origin'],queries[source]):raise AssertionError('Original policy changed')
                        if pilot:continue
                        row=dict(context=context,source=source,seed=seed,scope='fit',registration_sha256=sha(REG),
                            parent_group_sha256=sha(oldpath),references=references,scores=scores,diagnostics=diag,policy=pol,
                            prior_MSE=dict(independent_cost={t:old['scores']['independent_cost'][t]['full_cohort_MSE'] for t in ('all','easy')},uniform=old['prior_uniform_MSE']),
                            prediction_sha256={a:api.base.fingerprint(v) for a,v in p.items()},
                            action_sha256={a:api.base.fingerprint(v) for a,v in selected.items()},
                            result_source='fresh_run_donor_constraint_fit_and_fitting_policy_readout',neural_updates=0,
                            whole_pipeline_OOF=False,held_inference=False,independent_roles_read=False,deployment_changed=False)
                        if check:
                            if row!=json.loads(paths[source].read_text()):raise ValueError('Full replay differs')
                        else:once(paths[source],row)
                        fresh.append(row)
                    if pilot:
                        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            views=4,queries=len(out['indices']),supported=int(out['supported'].sum()),
                            donor_conflicts=int(out['channel']['conflict'].sum()),frozen_controls_exact=True,
                            policy_and_diagnostic_paths_executed=True,engineering_pass=True,registration_sha256=sha(REG)))
                        beat(state='pilot_complete');return
                rows.extend(fresh)
                receipts.extend(dict(path=str(paths[r['source']].relative_to(ROOT)),sha256=sha(paths[r['source']])) for r in fresh)
                beat(state='verified_views' if check else 'fit_views',views=len(rows),total=72)
        summary=api.summarize(rows,pooled.previous.previous.parent.expected(reg));summary['policy_readout']=policy.summarize(rows)
        if check:
            done=json.loads(finish.read_text())
            if done['groups']!=receipts or done['summary_sha256']!=sha(PUBLIC/'summary.json') or summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(finish,dict(groups=receipts,registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json'),pid=os.getpid(),
                seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                neural_updates=0,held_inference=False,independent_roles_read=False,deployment_changed=False))
        beat(state='verified' if check else 'run_complete',policy_budget_supported=summary['policy_readout']['primary_full_roster_budget_supported'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','pilot','run','check'));p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='register':register()
    else:execute(args.action,args.resume)
