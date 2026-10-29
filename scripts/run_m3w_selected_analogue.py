"""Fit-only recording-excluded analogue probe with immutable checkpoints."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import time

from scripts import run_m3w_risk_decoupling as previous
from src.evaluation import m3w_selected_analogue as api

ROOT, sha, once = previous.ROOT, previous.sha, previous.once
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_selected_analogue_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_selected_analogue_v1'
REG = PUBLIC/'registration.json'


def register():
    reg, cfg, ancestor = previous.admission()
    paths = [Path(__file__), ROOT/'src/evaluation/m3w_selected_analogue.py',
        ROOT/'tests/test_m3w_selected_analogue.py', PUBLIC/'protocol.md', PUBLIC/'prior_controls.md']
    once(REG, dict(bindings={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths},
        parent_registration_sha256=sha(previous.REG), parent_complete_sha256=sha(previous.PUBLIC/'complete.json'),
        parent_freeze_sha256=sha(previous.PUBLIC/'training_freeze.json'),
        contexts=sorted(reg['contexts']), seeds=reg['seeds'], k=32, shuffle_seed=20261029,
        views=72, same_recording_reference=False, query_label_input=False,
        whole_pipeline_OOF=False, held_inference=False, independent_roles_read=False,
        new_neural_training=False, deployment_changed=False))
    return reg, cfg, ancestor


def admission():
    out=register()
    for rel in json.loads(REG.read_text())['bindings']: previous.committed(ROOT/rel)
    previous.committed(REG)
    return out


def runtime():
    previous.train.torch.set_num_threads(4); previous.train.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    def beat(**kw):
        value=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        path=PRIVATE/'heartbeat.tmp'; path.write_text(json.dumps(value)+'\n'); os.replace(path,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        if value.get('state') != 'analogue_queries': print(json.dumps(value),flush=True)
        if time.monotonic()-start > 43200: raise TimeoutError('12h resume boundary; no smaller scientific scope')
        if shutil.disk_usage(ROOT).free < 10*2**30: raise OSError('Preserve10GiB reserve')
    return start,beat


def view(context,source,seed,args,origin,estimation,beat,score=True):
    x,env,y,_,sites,recs,frames,pr=args
    for state in (origin,estimation):
        previous.train.api.core.exact(pr,state['anchor_state']['source_preprocess'][source])
        for name,a in zip(('x','envelope','primary','temporal','sites','recordings','frames'),args[:-1]):
            if previous.train.api.fingerprint(a) != state['input_hashes'][source][name]:
                raise ValueError('Original fitting packet changed')
    predictions={a:previous.train.api.predict(s,source,x,env,recs,frames)
                 for a,s in (('origin',origin),('estimation',estimation))}
    key=f'{context}_{source}_seed{seed}'
    old_path=previous.PUBLIC/'readout'/(key+'.json')
    completed=json.loads((previous.PUBLIC/'complete.json').read_text())
    ref=next(r for r in completed['readouts'] if r['path']==str(old_path.relative_to(ROOT)))
    if sha(old_path) != ref['sha256']: raise ValueError('Prior readout changed')
    old=json.loads(old_path.read_text())
    for arm,p in predictions.items():
        if previous.train.api.fingerprint(p) != old['prediction_hashes'][arm]:
            raise ValueError('Frozen original/estimation prediction replay differs')
    eligible=previous.train.api.parent.previous.prior.eligible(args)
    query=previous.readout.actions(predictions['origin'],eligible)
    if previous.train.api.fingerprint(query) != old['result']['common_cohort_hash']:
        raise ValueError('Original fixed cohort changed')
    retrieved=api.retrieve(x,env,y,recs,eligible,query,heartbeat=beat)
    ix=retrieved['indices']; supported=retrieved['supported']
    subset={a:p[ix] for a,p in predictions.items()}
    for arm in api.MODES:
        p=subset['origin'].copy(); p[:,[1,4]]=retrieved['predictions'][arm]; subset[arm]=p
    base=dict(context=context,source=source,seed=seed,scope='fit',
        registration_sha256=sha(REG),parent_cohort_sha256=old['result']['common_cohort_hash'],
        prior_predictions_exact=True,references=retrieved['references'],
        prediction_sha256={a:api.fingerprint(p) for a,p in subset.items()},
        supported_sha256=api.fingerprint(supported),whole_pipeline_OOF=False,
        held_inference=False,independent_roles_read=False,deployment_changed=False,
        result_source='fresh_run_recording_excluded_fitting_analogue')
    if score:
        base['scores']={a:api.score(y[ix],p,env[ix],pr['scale'],
            api.np.ones(len(ix),bool) if a in ('origin','estimation') else supported) for a,p in subset.items()}
    else:
        base.update(query_rows=len(ix),supported=int(supported.sum()),unknown_queries=int((~api.np.isfinite(y[ix]).all(1)).sum()))
    return base


def execute(phase,resume):
    reg,_,ancestor=admission(); docs=previous.models(reg); _,origins=previous.parent.models(reg)
    pilot=phase=='pilot'; checking=phase=='check'; finish=PUBLIC/('pilot.json' if pilot else 'complete.json')
    if finish.exists() and not checking: raise ValueError('Completed phase immutable')
    if phase=='run' and not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real source pilot required')
    began,beat=runtime(); rows=[]; refs=[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        items=sorted(reg['contexts'].items())[:1] if pilot else sorted(reg['contexts'].items())
        for context,context_items in items:
            packets,held=previous.parent.parent.run.parent.prior.packets(context_items,ancestor,beat); del held
            for seed in reg['seeds'][:1] if pilot else reg['seeds']:
                origin=previous.train.api.core.read_checkpoint(ROOT/origins[context,seed,'self']['checkpoint']['path'])
                estimation=previous.train.api.core.read_checkpoint(ROOT/docs[context,seed,'estimation']['checkpoint']['path'])
                sources=sorted(packets)[:1] if pilot else sorted(packets)
                for source in sources:
                    key=f'{context}_{source}_seed{seed}'
                    path=PUBLIC/'groups'/(key+'.json')
                    if path.exists() and resume and not checking and not pilot:
                        row=json.loads(path.read_text())
                        if row['registration_sha256'] != sha(REG): raise ValueError('Resume binding changed')
                        beat(state='verified_checkpoint_reused',view=key)
                    else:
                        start=time.monotonic(); row=view(context,source,seed,packets[source],origin,estimation,beat,score=not pilot)
                        if pilot:
                            once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-began,
                                view_seconds=time.monotonic()-start,query_rows=row['query_rows'],supported=row['supported'],
                                reference_recordings=len(row['references']),same_recording_donors=0,
                                prior_predictions_exact=True,engineering_pass=True,
                                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                estimated_72_view_seconds=(time.monotonic()-start)*72,
                                registration_sha256=sha(REG),scientific_screen_evaluated=False))
                            beat(state='pilot_complete'); return
                        if checking:
                            if json.loads(path.read_text()) != row: raise ValueError('Full retrieval replay differs')
                        else:
                            if path.exists(): raise ValueError('Explicit resume required')
                            once(path,row)
                    rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                    beat(state='verified_view' if checking else 'fit_view',views=len(rows),total=72)
        result=api.summarize(rows,previous.parent.expected(reg))
        if checking:
            done=json.loads(finish.read_text())
            if done['groups'] != refs or done['summary_sha256'] != sha(PUBLIC/'summary.json') or result != json.loads((PUBLIC/'summary.json').read_text()):
                raise ValueError('Complete summary replay differs')
        else:
            once(PUBLIC/'summary.json',result)
            once(finish,dict(groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),registration_sha256=sha(REG),
                pid=os.getpid(),seconds=time.monotonic()-began,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                neural_updates=0,held_inference=False,independent_roles_read=False,deployment_changed=False))
        beat(state='verified' if checking else 'run_complete',signal_screen=result['conditional_signal_screen_pass'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('register','pilot','run','check'))
    parser.add_argument('--resume',action='store_true'); args=parser.parse_args()
    if args.action=='register': register()
    else: execute(args.action,args.resume)
