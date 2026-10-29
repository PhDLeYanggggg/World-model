"""Registered coupled risk fitting; no query or independent-role optimization."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import time

from scripts import run_m3w_local_cost_analogue as parent
from src.evaluation import m3w_coupled_support as api

ROOT,sha,once=parent.ROOT,parent.sha,parent.once
pooled=parent.pooled
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_coupled_support_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_coupled_support_v1'
REG=PUBLIC/'registration.json'


def register():
    result=parent.admission()
    files=[Path(__file__),ROOT/'src/evaluation/m3w_coupled_support.py',
           ROOT/'tests/test_m3w_coupled_support.py',ROOT/'tests/test_m3w_coupled_support_runner.py',
           PUBLIC/'protocol.md',parent.REG,parent.PUBLIC/'complete.json',parent.PUBLIC/'summary.json',
           parent.PUBLIC/'verification.json',parent.PUBLIC/'failure_analysis.md']
    once(REG,dict(bindings={str(p.resolve().relative_to(ROOT)):sha(p) for p in files},
        contexts=sorted(result[0]['contexts']),seeds=result[0]['seeds'],views=72,k=32,primary=api.PRIMARY,
        changed_factors=['joint_quadratic_constraint','donor_record_excluded_convex_blend'],
        query_policy_changed=False,normalization_changed=False,shuffle_seed=20261029,
        held_inference=False,independent_roles_read=False,whole_pipeline_OOF=False,
        neural_updates=0,deployment_changed=False))
    return result


def admission():
    result=register()
    for rel in json.loads(REG.read_text())['bindings']: pooled.previous.previous.committed(ROOT/rel)
    pooled.previous.previous.committed(REG)
    return result


def runtime():
    pooled.previous.previous.train.torch.set_num_threads(4)
    pooled.previous.previous.train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def beat(**kw):
        event=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(event)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        if kw.get('state')!='analogue_queries':print(json.dumps(event),flush=True)
        if time.monotonic()-start>43200:raise TimeoutError('12h boundary; resume unchanged protocol')
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('Preserve10GiB reserve')
    beat(state='admitted');return start,beat


def diagnostic(out,take,y,p,env,scale):
    np=api.np;local=out['supported'][take];d={k:v[take] for k,v in out['diagnostics'].items()}
    known=np.isfinite(y).all(1)
    result=dict(local_supported=int(local.sum()),local_unsupported=int((~local).sum()),
        nested_supported=int(d['nested_supported'].sum()),alpha_zero=int((d['alpha']==0).sum()),
        alpha_one=int((d['alpha']==1).sum()),alpha_sum=float(d['alpha'].sum()),
        quantiles={k:parent.np_quantiles(v[local]) for k,v in d.items()},
        diagnostic_sha256={k:api.base.fingerprint(v) for k,v in d.items()},
        nested_loss_before=float(d['nested_loss_before'].sum()),nested_loss_after=float(d['nested_loss_after'].sum()),
        nested_loss_is_training_objective_not_generalization=True,arms={},slices={})
    bad=p['independent_cost'][:,4]>p['independent_cost'][:,1]+1e-7
    low_record=d['record_ESS']<2;low_row=d['effective_donors']<2
    result['overlap']={f'inconsistent_{b}_low_record_{c}':int((local & (bad==b) & (low_record==c)).sum())
                       for b in (False,True) for c in (False,True)}
    result['row_leverage_overlap']={f'inconsistent_{b}_low_row_{c}':int((local & (bad==b) & (low_row==c)).sum())
                       for b in (False,True) for c in (False,True)}
    slices={'one_reference_record':local & (d['distinct_records']==1),
            'multiple_reference_records':local & (d['distinct_records']>1),
            'record_ESS_below2':local & low_record,'record_ESS_atleast2':local & ~low_record,
            'alpha_zero':d['alpha']==0,'alpha_positive':d['alpha']>0}
    for name,mask in slices.items():
        observed=mask & known
        result['slices'][name]=dict(rows=int(mask.sum()),known=int(observed.sum()),unknown=int((mask & ~known).sum()),
            envelope_over_scale_sum=float((env[mask]/scale).sum()),
            original_easy_fraction_sum=float((p['origin'][mask,4]/np.maximum(env[mask],1e-150)).sum()),
            signed_SSE={a:{t:float(((((v[observed,col]-.02*v[observed,ref])-
                (y[observed,col]-.02*y[observed,ref]))/scale)**2).sum())
                for t,col,ref in (('all',1,2),('easy',4,3))} for a,v in p.items()})
    for arm,v in p.items():
        support=local if arm=='independent_cost' else np.ones(len(y),bool);obs=known & support
        result['arms'][arm]=dict(easy_exceeds_all=int((v[support,4]>v[support,1]+1e-7).sum()))
        for t,col in (('all',1),('easy',4)):
            event=y[obs,col]>0;err=((v[obs,col]-y[obs,col])/scale)**2
            result['arms'][arm][t]=dict(events=int(event.sum()),event_harm_SSE=float(err[event].sum()),
                non_event_harm_SSE=float(err[~event].sum()),
                predicted_harm_on_events=float(v[obs,col][event].sum()/scale),
                predicted_harm_on_non_events=float(v[obs,col][~event].sum()/scale))
    return result


def execute(phase,resume=False):
    reg,_,ancestor=admission();states=pooled.previous.previous.models(reg)
    _,origins=pooled.previous.previous.parent.models(reg)
    previous=json.loads((parent.PUBLIC/'complete.json').read_text())
    oldrefs={Path(r['path']).name:r for r in previous['groups']}
    pilot,check=phase=='pilot',phase=='check';finish=PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not check:raise ValueError('Completed phase immutable')
    if phase=='run' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Pilot must pass first')
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
                    if any(r['registration_sha256']!=sha(REG) for r in fresh):raise ValueError('Resume freeze changed')
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
                        _,env,y,_,_,_,_,pr=packets[source]
                        p={'origin':values[source]['origin'][ix]}
                        for arm in api.MODES:
                            p[arm]=p['origin'].copy();p[arm][:,[1,4]]=out['predictions'][arm][take]
                        references=[r for r in out['references'] if r['query_source']==source]
                        if references!=old['references']:raise ValueError('Reference bank/normalizer changed')
                        for arm,prior in (('origin','origin'),('independent_cost','local_signed32')):
                            if api.base.fingerprint(p[arm])!=old['prediction_sha256'][prior]:raise ValueError('Prior predictions changed')
                        support=out['supported'][take]
                        scores={a:api.base.score(y[ix],v,env[ix],pr['scale'],support if a=='independent_cost'
                            else api.np.ones(len(ix),bool)) for a,v in p.items()}
                        for arm,prior in (('origin','origin'),('independent_cost','local_signed32')):
                            if scores[arm]!=old['scores'][prior]:raise ValueError('Prior score changed')
                        d=diagnostic(out,take,y[ix],p,env[ix],pr['scale'])
                        if any(d['arms'][a]['easy_exceeds_all'] or scores[a]['benefit_harm_envelope_exceedances'] for a in api.JOINT_ARMS):
                            raise AssertionError('Joint constraints failed')
                        if pilot:continue
                        row=dict(context=context,source=source,seed=seed,scope='fit',registration_sha256=sha(REG),
                            parent_group_sha256=sha(oldpath),references=references,scores=scores,diagnostics=d,
                            prior_uniform_MSE={t:old['scores']['selected_nearest32'][t]['full_cohort_MSE'] for t in ('all','easy')},
                            prediction_sha256={a:api.base.fingerprint(v) for a,v in p.items()},
                            local_supported_sha256=api.base.fingerprint(support),result_source='fresh_run_joint_donor_blend_fit',
                            neural_updates=0,whole_pipeline_OOF=False,held_inference=False,independent_roles_read=False,deployment_changed=False)
                        if check:
                            if row!=json.loads(paths[source].read_text()):raise ValueError('Full replay differs')
                        else:once(paths[source],row)
                        fresh.append(row)
                    if pilot:
                        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-start,
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,views=4,
                            queries=len(out['indices']),supported=int(out['supported'].sum()),
                            frozen_controls_exact=True,joint_constraints_pass=True,diagnostic_writer_executed=True,
                            engineering_pass=bool(out['supported'].all()),scientific_screen_evaluated=False,registration_sha256=sha(REG)))
                        beat(state='pilot_complete');return
                rows.extend(fresh)
                receipts.extend(dict(path=str(paths[r['source']].relative_to(ROOT)),sha256=sha(paths[r['source']])) for r in fresh)
                beat(state='verified_views' if check else 'fit_views',views=len(rows),total=72)
        summary=api.summarize(rows,pooled.previous.previous.parent.expected(reg))
        if check:
            done=json.loads(finish.read_text())
            if done['groups']!=receipts or done['summary_sha256']!=sha(PUBLIC/'summary.json') or summary!=json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Summary replay differs')
        else:
            once(PUBLIC/'summary.json',summary)
            once(finish,dict(groups=receipts,registration_sha256=sha(REG),summary_sha256=sha(PUBLIC/'summary.json'),
                pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                neural_updates=0,held_inference=False,independent_roles_read=False,deployment_changed=False))
        beat(state='verified' if check else 'run_complete',magnitude_screen=summary['magnitude_screen_pass'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','pilot','run','check'))
    p.add_argument('--resume',action='store_true');args=p.parse_args();
    if args.action=='register':register()
    else:execute(args.action,args.resume)
