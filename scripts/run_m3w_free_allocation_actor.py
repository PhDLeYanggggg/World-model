"""Separate unbounded routing from frozen costs,with matched complete SOURCE controls."""
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
from scripts import run_m3w_decision_cost_critic as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_free_allocation_actor as api
from src.world_model import m3w_actor_allocation as decoder
from src.evaluation import m3w_free_actor_readout as evaluation
from src.evaluation.m3w_metric_bundle import pack

np=api.np;source=previous.source
NAME='european_free_allocation_actor_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_free_allocation_actor_v1.json';REG=PUBLIC/'registration.json'
sha,once,write=previous.sha,previous.once,previous.write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_free_allocation_actor.py',ROOT/'src/world_model/m3w_actor_allocation.py',
    ROOT/'src/evaluation/m3w_free_actor_readout.py',ROOT/'tests/test_m3w_free_allocation_actor.py',ROOT/'tests/test_m3w_actor_allocation.py',PUBLIC/'protocol.md')


def admission(register=False):
    _,values,store,banks,models,controls=previous.admission();cfg=json.loads(CONFIG.read_text())
    checked=json.loads((previous.PUBLIC/'verification.json').read_text());support=json.loads((previous.PUBLIC/'soft_support_diagnostic.json').read_text())
    if (cfg['arms']!=list(api.ARMS) or cfg['seeds']!=[17,29,43] or cfg['risk_budget']!=.02
            or cfg['training']['steps']!=2000 or cfg['training']['width']!=32 or cfg['parameters_per_actor']!=4321
            or not checked['all72_frozen_actions_inference_and_scores_exact'] or checked['summary_sha256']!=sha(previous.PUBLIC/'summary.json')
            or support['summary']['records_with_any_strictly_positive_relaxed_lower_bound']!=444
            or any(cfg[k] for k in ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Verified diagnosed parent and original risk/data roles required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_verification_sha256=sha(previous.PUBLIC/'verification.json'),parent_summary_sha256=sha(previous.PUBLIC/'summary.json'),
        parent_support_sha256=sha(previous.PUBLIC/'soft_support_diagnostic.json'),
        cost_training_freeze_sha256=sha(previous.previous.PUBLIC/'training_freeze.json'),contexts=json.loads(previous.REG.read_text())['contexts'],
        primary='selected_context_joint_minus_zero_gate_joint_full_completion_lower',
        changed_factor='separate_free_routing_actor_with_frozen_costs_and_causal_query_constraints',
        within_actor_control='self_context_vs_selected_context_same_features_capacity_draws_budget',
        new_actor_fits=36,primary_optimizer_updates=72000,exact_refit_updates=4000,pilot_optimizer_updates=400,
        forecast_producer_seed=43,critic_seeds=cfg['seeds'],cost_controls=18,forecasts_and_costs_frozen=True,
        role='exposed_SOURCE_fitting_resubstitution_only',own_outcomes_read=False,
        independent_roles_opened=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Frozen allocation-actor definition changed:'+rel)
            source.frozen.BASE.engine.committed(ROOT/rel)
        source.frozen.BASE.engine.committed(REG)
    return cfg,values,store,banks,models,controls


def runtime(cfg):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume the fixed free-actor roster')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h phase limit;resume explicit checkpoints')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['private_checkpoint_cap_bytes']:raise OSError('1GiB checkpoint cap')
    def beat(**event):
        guard();v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event);write(PRIVATE/'heartbeat.json',v)
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
        print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),device='cpu',threads=4,interop_threads=1,workers=0,resource_probing=False)
    return start,beat,guard


def prepare(context,seed,packets,changed,store,models,controls,old,main,original,beat,*,preprocess=None,fitting=False):
    data,ranges,_=source.prepare(context,seed,packets,changed,store,models,old,main,beat)
    state=api.parent.read_checkpoint(ROOT/controls[context,seed]['path']);api.parent.exact(state['input_hashes'],data['input_hashes'])
    p=api.prior.base.predict(state,data['causal']);teacher=np.full(len(p),-1,np.int64);zeros={}
    for name,packet in sorted(packets.items()):
        key=f'{context}_{name}_seed{seed}_fit.json';ref=original[key]
        choice=api.parent.read_checkpoint(previous.previous.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
        z=dict(independent=choice['released_independent'],joint=choice['released_joint'])
        if (api.parent.digest(p[ranges[name]])!=ref['prediction_sha256']['released_benefit']
                or api.parent.digest(z['joint'])!=ref['action_sha256']['released_joint']):raise ValueError('Old full control changed')
        teacher[ranges[name]]=z['joint'];zeros[name]=z
    c=api.causal_inputs(data['causal'],state,p,teacher,preprocess=preprocess,fitting=fitting)
    prepared=api.training_data(c,data)
    if c['features'][api.ARMS[0]].shape[-1]!=133:raise ValueError('Frozen133-field actor schema required')
    beat(state='complete_actor_inputs_prepared',context=context,seed=seed,rows=len(p),queries=len(c['groups']),
        features_per_action=133,all7actions_preserved=True,frozen_moment_hash=api.parent.digest(p),normalization_fitted=fitting)
    return prepared,ranges,zeros


def checkpoint(context,seed,arm,phase='heads'):return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data,cfg,context,seed,arm,phase,beat,guard,**kw):
    return api.fit(data,arm=arm,seed=seed,settings=cfg['training'],decision=cfg['decision'],
        identity=dict(registration_sha256=sha(REG),context=context,cost_seed=seed),path=checkpoint(context,seed,arm,phase),
        heartbeat=lambda **event:beat(context=context,seed=seed,**event),guard=guard,**kw)


def equal(a,b):
    for k in a:
        if k!='seconds':api.parent.exact(a[k],b[k])


def resources():return (aggregate_groups(source.frozen.source.parent.NAME),aggregate_groups(source.frozen.NAME),aggregate_groups(previous.previous.NAME))


def pilot():
    cfg,values,store,banks,models,controls=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed pilot immutable')
    old,main,original=resources();context,items=sorted(values[2]['contexts'].items())[0]
    packets,changed=previous.previous.context_packets(context,items,values,store,banks,beat)
    data,ranges,zero=prepare(context,17,packets,changed,store,models,controls,old,main,original,beat,fitting=True)
    c=data['causal'];states=[];cells=[]
    for arm in api.ARMS:
        m=api.AllocationActor(133,cfg['training']['width'],17)
        dummy=dict(arm=arm,dimension=133,settings=cfg['training'],seed=17,model=m.state_dict(),preprocess=c['preprocess'],temperature=c['temperature'])
        u=api.predict(dummy,c);np.testing.assert_array_equal(u,c['native_gain'])
        for name,packet in packets.items():np.testing.assert_array_equal(decoder.independent(c['prediction'][ranges[name]],packet['allowed'],u[ranges[name]]),zero[name]['independent'])
        fit(data,cfg,context,17,arm,'pilot',beat,guard,stop_at=50)
        recovered=fit(data,cfg,context,17,arm,'pilot',beat,guard,stop_at=100,resume=True)
        direct=fit(data,cfg,context,17,arm,'pilot_direct',beat,guard,stop_at=100);equal(recovered,direct)
        if recovered['parameters']!=4321 or not recovered['unknown_bound_rows_drawn']:raise ValueError('Full real actor updates required')
        states.append(recovered);cells.append(dict(arm=arm,parameters=4321,zero_actor_exact=True,interrupted_recovery_exact=True,
            seconds=direct['seconds'],first_trace=recovered['trace'][0],final_trace=recovered['trace'][-1]))
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400,cells=cells,engineering_pass=True,zero_frozen_cost_control_exact=True,
        rough72000update_training_seconds=max(v['seconds'] for v in cells)*20*36,projection_excludes_admission_preparation_refit_readout=True,
        own_policy_outcomes_read=False,registration_sha256=sha(REG)))
    beat(state='real400update_actor_pilot_complete')


def train(resume=False):
    cfg,values,store,banks,models,controls=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Real pilot required')
    old,main,original=resources();refs=[];first=sorted(values[2]['contexts'])[0]
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed=previous.previous.context_packets(context,items,values,store,banks,beat)
        for seed in cfg['seeds']:
            data,_,_=prepare(context,seed,packets,changed,store,models,controls,old,main,original,beat,fitting=True);states=[]
            for arm in api.ARMS:
                path=checkpoint(context,seed,arm);s=fit(data,cfg,context,seed,arm,'heads',beat,guard,resume=resume and path.exists())
                if s['step']!=2000 or s['parameters']!=4321:raise ValueError('Complete fixed actor budget required')
                if context==first and seed==17:
                    rp=checkpoint(context,seed,arm,'full_refit');equal(s,fit(data,cfg,context,seed,arm,'full_refit',beat,guard,resume=resume and rp.exists()))
                    once(PUBLIC/('first_full_refit_'+arm+'.json'),dict(scientific_state_exact=True,updates=2000,
                        checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(rp)))
                receipt=PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                once(receipt,dict(context=context,seed=seed,arm=arm,step=s['step'],parameters=s['parameters'],
                    checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),input_hashes=s['input_hashes'],draw_hash=s['draw_hash'],
                    seconds=s['seconds'],first_trace=s['trace'][0],final_trace=s['trace'][-1],unknown_bound_rows_drawn=s['unknown_bound_rows_drawn'],
                    unknown_rows_with_supervised_labels=0,frozen_cost_checkpoint=controls[context,seed],registration_sha256=sha(REG)))
                refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));states.append(s)
                beat(state='actor_head_complete',heads=len(refs),total=36,context=context,seed=seed,arm=arm)
            api.assert_matched(*states);del data,states
        del packets,changed
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=refs,primary_optimizer_updates=72000,exact_refit_updates=4000,cached_frozen_cost_heads=18,registration_sha256=sha(REG)))
    beat(state='all36_free_actors_frozen')


def evaluate(verify=False,resume=False):
    cfg,values,store,banks,models,controls=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed result immutable')
    fitted={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('New fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text());cp=r['checkpoint']
        if sha(ROOT/cp['path'])!=cp['sha256']:raise ValueError('New actor changed')
        fitted[r['context'],r['seed'],r['arm']]=cp
    if len(fitted)!=36:raise ValueError('All36 frozen actors required')
    old,main,original=resources();legacy=aggregate_groups(previous.NAME);done=aggregate_groups(NAME) if verify else {};rows=[];refs=[];ids=set();records=set()
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed=previous.previous.context_packets(context,items,values,store,banks,beat)
        job=next(j for j in values[4] if j['key']==context.split('_controller')[0])
        for seed in cfg['seeds']:
            states={a:api.parent.read_checkpoint(ROOT/fitted[context,seed,a]['path']) for a in api.ARMS};api.assert_matched(*states.values())
            data,ranges,zero=prepare(context,seed,packets,changed,store,models,controls,old,main,original,beat,preprocess=states[api.ARMS[0]]['preprocess'])
            for s in states.values():api.parent.exact(s['input_hashes'],data['input_hashes'])
            c=data['causal'];utilities={a:api.predict(s,c) for a,s in states.items()}
            for name,packet in sorted(packets.items()):
                key=f'{context}_{name}_seed{seed}_fit.json';path=PUBLIC/'groups'/key;progress=PRIVATE/'progress'/(key+'.pt.gz')
                p=c['prediction'][ranges[name]];u={a:v[ranges[name]] for a,v in utilities.items()};reference=original[key]
                if path.exists() and not verify:
                    if not resume:raise ValueError('Explicit partial scoring resume required')
                    row=json.loads(path.read_text())
                else:
                    if verify:
                        saved=api.parent.read_checkpoint(progress)
                        if saved['utility_sha256']!={a:api.parent.digest(v) for a,v in u.items()}:raise ValueError('Frozen utility inference changed')
                        choices,ranks,info=saved['choices'],saved['ranks'],saved['allocation']
                    else:
                        partial=PRIVATE/'allocation_progress'/(key+'.pt.gz')
                        choices,ranks,info=decoder.allocate(p,packet['allowed'],u,packet['recordings'],packet['frames'],packet['ids'],zero[name],
                            seconds=cfg['solver_seconds_per_query'],checkpoint=lambda v:api.parent.save_checkpoint(partial,v),
                            heartbeat=lambda **kw:beat(state='actor_allocation',context=context,source=name,seed=seed,**kw),
                            resume_state=api.parent.read_checkpoint(partial) if resume and partial.exists() else None)
                        api.parent.save_checkpoint(progress,dict(choices=choices,ranks=ranks,allocation=info,utility_sha256={a:api.parent.digest(v) for a,v in u.items()}))
                    archived=api.parent.read_checkpoint(previous.PRIVATE/'progress'/(key+'.pt.gz'))['choices']['released_joint']
                    if api.parent.digest(archived)!=legacy[key]['action_sha256']['released_joint']:raise ValueError('Legacy direct policy changed')
                    choices=dict(choices,decision_v1_joint=archived);ranks=dict(ranks,decision_v1_joint=np.zeros(len(archived)))
                    result=evaluation.readout(packet,choices,ranks,info,job['design']['easy_cut'],values[3]['origin'][packet['ids']])
                    for a in ('independent','joint'):
                        if result['policies']['zero_'+a]!=reference['result']['policies']['released_'+a]:raise ValueError('Original full control score changed')
                    if result['policies']['decision_v1_joint']!=legacy[key]['result']['policies']['decision_joint']:raise ValueError('Legacy full policy score changed')
                    row=dict(context=context,source=name,seed=seed,result=result,old_controls_exact=True,matched_training_verified=True,cost_moments_exact=True,
                        ids_sha256=api.parent.digest(packet['ids']),cost_prediction_sha256=api.parent.digest(p),utility_sha256={a:api.parent.digest(v) for a,v in u.items()},
                        action_sha256={a:api.parent.digest(v) for a,v in choices.items()},registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),
                        held_scored=False,independent_roles_opened=False)
                    if verify:
                        if row!=done[key]:raise ValueError('Frozen actions/inference/full scoring differ')
                    else:once(path,row)
                ids.update(map(int,packet['ids']));records.update((name,str(r)) for r in packet['recordings']);rows.append(row)
                refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));beat(state='actor_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
            del data,states,utilities
        del packets,changed
    if len(ids)!=149510 or len(records)!=74:raise ValueError('Original complete target cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical summary differs')
        once(PUBLIC/'verification.json',dict(all72_frozen_actions_inference_and_scores_exact=True,all72_zero_and_legacy_controls_exact=True,
            optimization_replayed=False,summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),training_freeze_sha256=sha(PUBLIC/'training_freeze.json')))
        pack(ROOT,PUBLIC)
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_verified' if verify else 'all72_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','train','eval','verify'));p.add_argument('--resume',action='store_true');args=p.parse_args()
    for d in (PRIVATE,PRIVATE/'progress',PRIVATE/'allocation_progress'):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(True)
        elif args.phase=='pilot':pilot()
        elif args.phase=='train':train(args.resume)
        else:evaluate(args.phase=='verify',args.resume)
