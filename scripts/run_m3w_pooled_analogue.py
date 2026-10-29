"""Audit and execute a frozen pooled-reference fitting information control."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import time

from scripts import run_m3w_selected_analogue as previous
from src.evaluation import m3w_pooled_analogue as api

ROOT,sha,once = previous.ROOT,previous.sha,previous.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_pooled_analogue_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_pooled_analogue_v1'
REG = PUBLIC/'registration.json'
LOADER = previous.previous.parent.parent.run.parent.prior


def runtime():
    previous.previous.train.torch.set_num_threads(4)
    previous.previous.train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True); began=time.monotonic()
    def beat(**kw):
        d=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        p=PRIVATE/'heartbeat.tmp';p.write_text(json.dumps(d)+'\n');os.replace(p,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        if d.get('state')!='analogue_queries':print(json.dumps(d),flush=True)
        if time.monotonic()-began>43200:raise TimeoutError('12h boundary; resume unchanged scope')
        if shutil.disk_usage(ROOT).free<10*2**30:raise OSError('Preserve10GiB reserve')
    return began,beat


def audit():
    reg,_,ancestor=previous.admission(); began,beat=runtime();contexts={}
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context,refs in sorted(reg['contexts'].items()):
            packets,held=LOADER.packets(refs,ancestor,beat);del held
            _,_,census=api.pool(packets);contexts[context]=census
        once(PUBLIC/'input_audit.json',dict(contexts=contexts,source_count=len({s for c in contexts.values() for s in c}),
            compatible_schema=True,all_recordings_have32donors=all(v['pooled_supported_all_recordings'] for c in contexts.values() for v in c.values()),
            coordinate_invariance_claim=False,feature_width=380,observed_scale_column=354,
            old_registration_sha256=sha(previous.REG),pid=os.getpid(),seconds=time.monotonic()-began,
            architecture=platform.machine(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            result_source='fresh_run_input_only_TRAIN_packet_audit',held_inference=False,scientific_scores_computed=False))
        beat(state='input_audit_complete')


def register():
    reg,cfg,ancestor=previous.admission();audit=json.loads((PUBLIC/'input_audit.json').read_text())
    if not audit['compatible_schema'] or not audit['all_recordings_have32donors']:
        raise ValueError('Compatible supported reference bank required before registration')
    files=[Path(__file__),ROOT/'src/evaluation/m3w_pooled_analogue.py',ROOT/'tests/test_m3w_pooled_analogue.py',
           PUBLIC/'protocol.md',PUBLIC/'input_audit.json',ROOT/'src/world_model/m3w_floor_relative.py',
           ROOT/'src/world_model/m3w_european_source_intervention.py',ROOT/'src/world_model/m3w_native_gain_harm.py']
    once(REG,dict(bindings={str(p.resolve().relative_to(ROOT)):sha(p) for p in files},
        parent_registration_sha256=sha(previous.REG),parent_complete_sha256=sha(previous.PUBLIC/'complete.json'),
        contexts=sorted(reg['contexts']),seeds=reg['seeds'],views=72,k=32,shuffle_seed=20261029,
        changed_factor='reference_bank_only_with_source_qualified_recording_keys',
        query_policy_changed=False,held_inference=False,independent_roles_read=False,new_neural_training=False,
        whole_pipeline_OOF=False,deployment_changed=False))
    return reg,cfg,ancestor


def admission():
    out=register()
    for rel in json.loads(REG.read_text())['bindings']:previous.previous.committed(ROOT/rel)
    previous.previous.committed(REG)
    return out


def predictions(context,seed,packets,origins,states):
    origin=previous.previous.train.api.core.read_checkpoint(ROOT/origins[context,seed,'self']['checkpoint']['path'])
    estimation=previous.previous.train.api.core.read_checkpoint(ROOT/states[context,seed,'estimation']['checkpoint']['path'])
    values={};queries={};oldrows={}
    complete=json.loads((previous.PUBLIC/'complete.json').read_text())
    receipts={Path(r['path']).name:r for r in complete['groups']}
    for source,args in sorted(packets.items()):
        x,env,_,_,_,recs,frames,pr=args
        for state in (origin,estimation):
            previous.previous.train.api.core.exact(pr,state['anchor_state']['source_preprocess'][source])
            for name,a in zip(('x','envelope','primary','temporal','sites','recordings','frames'),args[:-1]):
                if previous.api.fingerprint(a)!=state['input_hashes'][source][name]:raise ValueError('Input changed')
        p={a:previous.previous.train.api.predict(s,source,x,env,recs,frames)
           for a,s in (('origin',origin),('estimation',estimation))}
        q=previous.previous.readout.actions(p['origin'],api.eligible(args))
        path=previous.PUBLIC/'groups'/f'{context}_{source}_seed{seed}.json'
        if sha(path)!=receipts[path.name]['sha256']:raise ValueError('Prior group changed')
        old=json.loads(path.read_text())
        if previous.api.fingerprint(q)!=old['parent_cohort_sha256']:raise ValueError('Query selection changed')
        for arm in p:
            if previous.api.fingerprint(p[arm][q])!=old['prediction_sha256'][arm]:raise ValueError('Frozen control prediction changed')
        values[source],queries[source],oldrows[source]=p,q,old
    return values,queries,oldrows


def execute(phase,resume):
    reg,_,ancestor=admission();states=previous.previous.models(reg);_,origins=previous.previous.parent.models(reg)
    pilot=phase=='pilot';checking=phase=='check';finish=PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not checking:raise ValueError('Completed phase immutable')
    if phase=='run' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Real pilot required')
    began,beat=runtime();rows=[];receipts=[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for context,refs in sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items()):
            packets,held=LOADER.packets(refs,ancestor,beat);del held
            bank,spans,census=api.pool(packets)
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                paths={s:PUBLIC/'groups'/f'{context}_{s}_seed{seed}.json' for s in packets}
                if resume and not checking and not pilot and all(p.exists() for p in paths.values()):
                    fresh=[json.loads(paths[s].read_text()) for s in sorted(paths)]
                    if any(r['registration_sha256']!=sha(REG) for r in fresh):raise ValueError('Resume binding changed')
                else:
                    values,queries,oldrows=predictions(context,seed,packets,origins,states)
                    q=api.np.concatenate([queries[s] for s in sorted(packets)])
                    out=api.retrieve(bank,q,heartbeat=beat);fresh=[]
                    if pilot:
                        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-began,
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            query_rows=len(out['indices']),supported=int(out['supported'].sum()),views=4,
                            engineering_pass=bool(out['supported'].all()),scientific_screen_evaluated=False,
                            frozen_predictions_exact=True,registration_sha256=sha(REG)))
                        beat(state='pilot_complete');return
                    for source in sorted(packets):
                        lo,hi=spans[source];take=(out['indices']>=lo)&(out['indices']<hi)
                        ix=out['indices'][take]-lo;support=out['supported'][take]
                        p={a:v[ix] for a,v in values[source].items()}
                        for arm in api.ARMS[2:]:
                            p[arm]=p['origin'].copy();p[arm][:,[1,4]]=out['predictions'][arm][take]
                        x,env,y,_,_,_,_,pr=packets[source]
                        row=dict(context=context,source=source,seed=seed,scope='fit',registration_sha256=sha(REG),
                            parent_cohort_sha256=oldrows[source]['parent_cohort_sha256'],
                            prior_predictions_exact=True,references=[v for v in out['references'] if v['query_source']==source],
                            prediction_sha256={a:previous.api.fingerprint(v) for a,v in p.items()},
                            supported_sha256=previous.api.fingerprint(support),
                            scores={a:previous.api.score(y[ix],v,env[ix],pr['scale'],
                                api.np.ones(len(ix),bool) if a in ('origin','estimation') else support) for a,v in p.items()},
                            whole_pipeline_OOF=False,held_inference=False,independent_roles_read=False,deployment_changed=False,
                            result_source='fresh_run_pooled_fitting_analogue')
                        if checking:
                            if row!=json.loads(paths[source].read_text()):raise ValueError('Full retrieval replay differs')
                        else:once(paths[source],row)
                        fresh.append(row)
                rows.extend(fresh)
                receipts.extend(dict(path=str(paths[r['source']].relative_to(ROOT)),sha256=sha(paths[r['source']])) for r in fresh)
                beat(state='verified_views' if checking else 'fit_views',views=len(rows),total=72)
        result=api.summarize(rows,previous.previous.parent.expected(reg))
        if checking:
            done=json.loads(finish.read_text())
            if done['groups']!=receipts or done['summary_sha256']!=sha(PUBLIC/'summary.json') or result!=json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Full summary replay differs')
        else:
            once(PUBLIC/'summary.json',result)
            once(finish,dict(groups=receipts,summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(REG),
                pid=os.getpid(),seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                neural_updates=0,held_inference=False,independent_roles_read=False,deployment_changed=False))
        beat(state='verified' if checking else 'run_complete',signal_screen=result['conditional_signal_screen_pass'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('audit','register','pilot','run','check'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.action=='audit':audit()
    elif args.action=='register':register()
    else:execute(args.action,args.resume)
