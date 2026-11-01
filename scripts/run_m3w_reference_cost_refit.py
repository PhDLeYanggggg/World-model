"""Paired real cost fits with complete query coverage and learnable references."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required before Torch')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_query_easy_support as previous
from scripts import run_m3w_free_allocation_actor as parent
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_reference_cost_refit as api
from src.world_model import m3w_refit_cost_allocation as decoder
from src.evaluation import m3w_reference_cost_refit as evaluation
from src.evaluation.m3w_metric_bundle import pack

NAME='european_reference_cost_refit_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;CONFIG=ROOT/'configs/m3w_reference_cost_refit_v1.json';REG=PUBLIC/'registration.json'
sha,once,write=parent.sha,parent.once,parent.write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_reference_cost_refit.py',ROOT/'src/world_model/m3w_refit_cost_allocation.py',
    ROOT/'src/evaluation/m3w_reference_cost_refit.py',ROOT/'tests/test_m3w_reference_cost_refit.py',ROOT/'tests/test_m3w_refit_cost_allocation.py',PUBLIC/'protocol.md')


def admission(register=False):
    values,_=previous.admission();cfg=json.loads(CONFIG.read_text());checked=json.loads((previous.PUBLIC/'verification.json').read_text())
    if not checked['exact72_groups_and_remaining_query_results'] or checked['summary_sha256']!=sha(previous.PUBLIC/'summary.json'):raise ValueError('Verified necessary-support obstruction required')
    if (cfg['arms']!=list(api.ARMS) or cfg['seeds']!=[17,29,43] or cfg['training']['minimum_updates']!=2000
            or cfg['training']['largest_source_cycles']!=4 or cfg['risk_budget']!=.02 or cfg['parameters_per_head']!=31119
            or cfg['reference_MSE_weight']!=1. or cfg['reference_log_scale_clip']!=20.
            or any(cfg[k] for k in ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):raise ValueError('Registered real coverage budget and unchanged roles required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},previous_complete_sha256=sha(previous.PUBLIC/'complete.json'),
        previous_summary_sha256=sha(previous.PUBLIC/'summary.json'),previous_verification_sha256=sha(previous.PUBLIC/'verification.json'),
        parent_cost_training_freeze_sha256=sha(parent.previous.previous.PUBLIC/'training_freeze.json'),
        contexts=json.loads(parent.REG.read_text())['contexts'],seeds=cfg['seeds'],new_cost_fits=36,
        reference_trainable_capacity_difference=262,total_parameters=31119,warm_start_not_from_scratch=True,
        primary='learned_minus_locked_full_joint_completion_lower',normalizers_and_features_frozen=True,
        source_role='exposed_SOURCE_fitting_resubstitution_only',new_trajectory_training=False,
        independent_roles_opened=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Frozen cost-refit definition changed')
            parent.source.frozen.BASE.engine.committed(ROOT/rel)
        parent.source.frozen.BASE.engine.committed(REG)
    return cfg,values


def runtime(cfg):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume original checkpoints')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hphase boundary;resume same budget')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiBreserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['private_checkpoint_cap_bytes']:raise OSError('1GiBprivate checkpoint cap')
    def beat(**kw):
        guard();v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0,resource_probing=False)
    return start,beat,guard


def inputs(values,beat):
    _,sv,store,banks,models,controls=values;old,main,_=parent.resources();original=aggregate_groups(parent.NAME)
    for context,items in sorted(sv[2]['contexts'].items()):
        packets,changed=parent.previous.previous.context_packets(context,items,sv,store,banks,beat)
        for seed in values[0]['seeds']:
            data,ranges,_=parent.source.prepare(context,seed,packets,changed,store,models,old,main,beat)
            ref=controls[context,seed];warm=api.parent.read_checkpoint(ROOT/ref['path']);api.parent.exact(warm['input_hashes'],data['input_hashes'])
            neutral=api.base.predict(warm,data['causal'])
            for source,packet in packets.items():
                prior=original[f'{context}_{source}_seed{seed}_fit.json']
                if api.parent.digest(neutral[ranges[source]])!=prior['cost_prediction_sha256'] or api.parent.digest(packet['ids'])!=prior['ids_sha256']:raise ValueError('Original cost inference/rows differ')
            yield context,seed,data,ranges,packets,warm,ref,neutral
            del data,warm,neutral
        del packets,changed


def checkpoint(context,seed,arm,phase='heads'):return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data,warm,ref,context,seed,arm,cfg,phase,beat,guard,**kw):
    return api.fit(data,warm=warm,arm=arm,settings=cfg['training'],identity=dict(context=context,registration_sha256=sha(REG),warm_checkpoint=ref),
        path=checkpoint(context,seed,arm,phase),heartbeat=lambda **event:beat(context=context,seed=seed,**event),guard=guard,**kw)


def equal(a,b):
    for k in a:
        if k!='seconds':api.parent.exact(a[k],b[k])


def pilot():
    cfg,values=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed pilot immutable')
    gen=inputs(values,beat);context,seed,data,_,_,warm,ref,neutral=next(gen);states=[];cells=[]
    for arm in api.ARMS:
        m=api.initialize(warm,arm);dummy=dict(arm=arm,seed=seed,settings=cfg['training'],dimension=warm['dimension'],model=m.state_dict())
        api.np.testing.assert_array_equal(api.predict(dummy,data['causal']),neutral)
        fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=50)
        recovered=fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=100,resume=True)
        direct=fit(data,warm,ref,context,seed,arm,cfg,'pilot_direct',beat,guard,stop_at=100);equal(recovered,direct)
        states.append(recovered);cells.append(dict(arm=arm,seconds=direct['seconds'],parameters=direct['total_parameters'],
            trainable_parameters=direct['trainable_parameters'],neutral_native_predictions_exact=True,recovery_exact=True,
            first_trace=direct['trace'][0],final_trace=direct['trace'][-1]))
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400,engineering_pass=True,cells=cells,
        conservative_primary_projection_seconds=max(x['seconds'] for x in cells)/100*8192*36,
        projection_excludes_preparation_refits_and_readout=True,registration_sha256=sha(REG)))
    beat(state='real400update_reference_pilot_complete')


def train(resume=False):
    cfg,values=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Complete training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Real pilot required')
    refs=[];refit_updates=0;first=sorted(json.loads(REG.read_text())['contexts'])[0]
    for context,seed,data,_,_,warm,ref,_ in inputs(values,beat):
        states=[]
        for arm in api.ARMS:
            path=checkpoint(context,seed,arm);s=fit(data,warm,ref,context,seed,arm,cfg,'heads',beat,guard,resume=resume and path.exists())
            if s['step']!=s['budget'] or s['total_parameters']!=31119 or int(s['sampler']['counts'].min())<4:raise ValueError('Complete registered cost budget required')
            if context==first and seed==17:
                rp=checkpoint(context,seed,arm,'full_refit');equal(s,fit(data,warm,ref,context,seed,arm,cfg,'full_refit',beat,guard,resume=resume and rp.exists()))
                refit_updates+=s['step'];once(PUBLIC/f'first_full_refit_{arm}.json',dict(scientific_state_exact=True,updates=s['step'],checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(rp)))
            receipt=PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
            once(receipt,dict(context=context,seed=seed,arm=arm,step=s['step'],budget=s['budget'],parameters=s['total_parameters'],trainable_parameters=s['trainable_parameters'],
                checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),seconds=s['seconds'],input_hashes=s['input_hashes'],draw_hash=s['draw_hash'],
                minimum_query_visits=int(s['sampler']['counts'].min()),never_drawn_queries=int((s['sampler']['counts']==0).sum()),source_query_counts=s['source_query_counts'],
                source_completed_cycles=s['sampler']['completed'],first_trace=s['trace'][0],final_trace=s['trace'][-1],unknown_rows_drawn=s['unknown_rows_drawn'],
                unknown_rows_with_supervised_labels=0,warm_checkpoint=ref,registration_sha256=sha(REG)))
            refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));states.append(s);beat(state='cost_head_complete',fits=len(refs),total=36,context=context,seed=seed,arm=arm)
        api.assert_matched(*states)
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=refs,primary_optimizer_updates=sum(json.loads((ROOT/r['path']).read_text())['step'] for r in refs),exact_full_refit_updates=refit_updates,registration_sha256=sha(REG)))
    beat(state='all36_reference_cost_fits_frozen')


def evaluate(verify=False,resume=False):
    cfg,values=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed readout immutable')
    fitted={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Original cost fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('Frozen model changed')
        fitted[r['context'],r['seed'],r['arm']]=r
    if len(fitted)!=36:raise ValueError('All36cost endpoints required')
    original=aggregate_groups(parent.NAME);rows=[];refs=[];ids=set();records=set()
    for context,seed,data,ranges,packets,_,_,_ in inputs(values,beat):
        states={a:api.parent.read_checkpoint(ROOT/fitted[context,seed,a]['checkpoint']['path']) for a in api.ARMS};api.assert_matched(*states.values())
        predictions={a:api.predict(s,data['causal']) for a,s in states.items()}
        job=next(j for j in values[1][4] if j['key']==context.split('_controller')[0])
        for name,packet in sorted(packets.items()):
            key=f'{context}_{name}_seed{seed}_fit.json';path=PUBLIC/'groups'/key;progress=PRIVATE/'progress'/(key+'.pt.gz');old=original[key]
            p={a:v[ranges[name]] for a,v in predictions.items()}
            cached=api.parent.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['zero_joint']
            if api.parent.digest(cached)!=old['action_sha256']['zero_joint']:raise ValueError('Original parent full actions changed')
            if path.exists() and resume and not verify:row=json.loads(path.read_text())
            else:
                if verify:
                    saved=api.parent.read_checkpoint(progress)
                    if saved['prediction_sha256']!={a:api.parent.digest(v) for a,v in p.items()}:raise ValueError('New cost inference changed')
                    choices,ranks,info=saved['choices'],saved['ranks'],saved['allocation']
                else:
                    partial=PRIVATE/'allocation_progress'/(key+'.pt.gz')
                    choices,ranks,info=decoder.allocate(p,packet['allowed'],packet['recordings'],packet['frames'],packet['ids'],cached,seconds=cfg['solver_seconds_per_query'],
                        checkpoint=lambda v:api.parent.save_checkpoint(partial,v),heartbeat=lambda **kw:beat(state='refit_allocation',context=context,source=name,seed=seed,**kw),
                        resume_state=api.parent.read_checkpoint(partial) if resume and partial.exists() else None)
                    api.parent.save_checkpoint(progress,dict(choices=choices,ranks=ranks,allocation=info,prediction_sha256={a:api.parent.digest(v) for a,v in p.items()}))
                result=evaluation.readout(packet,choices,ranks,info,p,job['design']['easy_cut'],values[1][3]['origin'][packet['ids']],data['rms'])
                if result['policies']['parent_joint']!=old['result']['policies']['zero_joint']:raise ValueError('Original parent scores changed')
                row=dict(context=context,source=name,seed=seed,result=result,matched_training=True,
                    minimum_gradient_query_visits=min(int(s['sampler']['counts'].min()) for s in states.values()),original_parent_exact=True,
                    ids_sha256=api.parent.digest(packet['ids']),parent_prediction_sha256=old['cost_prediction_sha256'],
                    prediction_sha256={a:api.parent.digest(v) for a,v in p.items()},action_sha256={a:api.parent.digest(v) for a,v in choices.items()},
                    training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),registration_sha256=sha(REG))
                if verify:
                    if row!=json.loads(path.read_text()):raise ValueError('Complete inference/frozen-action scoring differs')
                else:once(path,row)
            rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));ids.update(map(int,packet['ids']));records.update((name,str(r)) for r in packet['recordings'])
            beat(state='cost_refit_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('All original targets and recordings required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical cost-refit summary differs')
        once(PUBLIC/'verification.json',dict(all72_new_cost_inference_frozen_actions_and_scores_exact=True,original_parent_controls_exact=True,optimization_replayed=False,
            summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**runtime,groups=refs,training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),summary_sha256=sha(PUBLIC/'summary.json')));pack(ROOT,PUBLIC)
    write(PRIVATE/('verify_runtime.json' if verify else 'eval_runtime.json'),runtime);beat(state='all72_verified' if verify else 'all72_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','train','eval','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    for d in (PRIVATE,PRIVATE/'progress',PRIVATE/'allocation_progress'):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':admission(True)
        elif a.phase=='pilot':pilot()
        elif a.phase=='train':train(a.resume)
        else:evaluate(a.phase=='verify',a.resume)
