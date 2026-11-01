"""Train and evaluate matched coherent cost critics on the unchanged SOURCE roster."""
import argparse
import copy
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
from scripts import run_m3w_coherent_cost_projection as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from scripts import report_m3w_coherent_cost_projection as prior_report
from src.world_model import m3w_own_decision_cost as api
from src.evaluation import m3w_own_decision_cost as evaluation
from src.evaluation.m3w_metric_bundle import pack

NAME='european_own_decision_cost_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;CONFIG=ROOT/'configs/m3w_own_decision_cost_v1.json';REG=PUBLIC/'registration.json'
sha,once,write=previous.sha,previous.once,previous.write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_own_decision_cost.py',ROOT/'src/evaluation/m3w_own_decision_cost.py',
    ROOT/'tests/test_m3w_own_decision_cost.py',ROOT/'tests/test_m3w_own_decision_evidence.py',PUBLIC/'protocol.md')


def admission(register=False):
    _,values,fits=previous.admission();cfg=json.loads(CONFIG.read_text());prior=prior_report.build()
    if prior['summary']['groups']!=72 or not prior['replay']['all72_projection_inference_and_frozen_action_scores_exact']:
        raise ValueError('Complete verified predecessor required')
    if (cfg['arms']!=list(api.ARMS) or cfg['seeds']!=[17,29,43] or cfg['risk_budget']!=.02
            or cfg['training']['minimum_updates']!=2000 or cfg['training']['largest_source_cycles']!=4
            or cfg['training']['solver_seconds_per_query']!=2. or cfg['solver_seconds_per_query']!=2.
            or cfg['parameters_per_head']!=31119 or any(cfg[k] for k in
                ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Unchanged complete coverage,budget,active capacity and data roles required')
    parent_files={name:sha(previous.PUBLIC/name) for name in ('registration.json','complete.json','summary.json','verification.json','metric_bundle.json')}
    dependency_paths=('src/world_model/m3w_reference_cost_refit.py','src/world_model/m3w_selected_query_critic.py',
        'src/world_model/m3w_query_unit_critic.py','src/world_model/m3w_refit_cost_allocation.py','src/world_model/m3w_frame_action_allocation.py')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},parent_files=parent_files,
        dependencies={p:sha(ROOT/p) for p in dependency_paths},contexts=json.loads(previous.REG.read_text())['contexts'],
        seeds=cfg['seeds'],new_cost_fits=36,total_and_trainable_parameters=31119,
        primary='own_choice_minus_fixed_choice_joint_full_completion_lower',
        secondary='own_choice_joint_minus_frozen_projected_learned_joint',
        warm_start='same_preceding_learned_reference_cost_head_for_both_arms',
        shared_decoder='differentiable_available_candidate_reference_mean_and_benefit_cap',
        factor='query_constraint_residual_targets_on_fixed_causal_initial_or_detached_own_native_joint_choices',
        normalizers_and_features_frozen=True,source_role='exposed_SOURCE_fitting_resubstitution_only',
        new_trajectory_training=False,independent_roles_opened=False,deployment_changed=False,
        submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        reg=json.loads(REG.read_text())
        if reg['parent_files']!=parent_files:raise ValueError('Preceding complete evidence changed')
        for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():
            if sha(ROOT/rel)!=digest:raise ValueError('Registered training/allocator definition changed')
            previous.parent.parent.source.frozen.BASE.engine.committed(ROOT/rel)
        previous.parent.parent.source.frozen.BASE.engine.committed(REG)
    return cfg,values,fits


def runtime(cfg):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged training or allocation checkpoints')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hphase boundary;resume unchanged budget')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiBreserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['private_checkpoint_cap_bytes']:raise OSError('1GiBprivate checkpoint cap')
    def beat(**kw):
        guard();v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
        write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0,resource_probing=False,new_cost_training=True)
    return start,beat,guard


def inputs(values,fits,beat):
    old=aggregate_groups(previous.NAME)
    for context,seed,data,ranges,packets,_,_,_ in previous.parent.inputs(values,beat):
        ref=fits[context,seed,'learned_references'];warm=api.parent.read_checkpoint(ROOT/ref['checkpoint']['path'])
        api.parent.exact(warm['input_hashes'],data['input_hashes'])
        frozen=previous.projection.project(previous.api.predict(warm,data['causal']),data['causal']['allowed'])
        for name in packets:
            row=old[f'{context}_{name}_seed{seed}_fit.json']
            if api.parent.digest(frozen[ranges[name]])!=row['projected_prediction_sha256']['learned_references']:
                raise ValueError('Frozen predecessor native cost control changed')
        yield context,seed,data,ranges,packets,warm,ref,frozen


def checkpoint(context,seed,arm,phase='heads'):return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data,warm,ref,context,seed,arm,cfg,phase,beat,guard,**kw):
    return api.fit(data,warm=warm,arm=arm,settings=cfg['training'],
        identity=dict(context=context,registration_sha256=sha(REG),warm_checkpoint=ref['checkpoint']),
        path=checkpoint(context,seed,arm,phase),heartbeat=lambda **event:beat(context=context,seed=seed,**event),guard=guard,**kw)


def pilot():
    cfg,values,fits=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'pilot.json').exists():raise ValueError('Complete pilot immutable')
    gen=inputs(values,fits,beat);context,seed,data,ranges,packets,warm,ref,frozen=next(gen);cells=[];states=[]
    neutral=api.predict(dict(warm,settings=cfg['training']),data['causal'])
    difference=float(api.np.max(abs(neutral-frozen)/data['causal']['scales'][:,None,None]))
    if difference>1e-10:raise ValueError('Torch projection must retain predecessor numerical semantics')
    for arm in api.ARMS:
        fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=50)
        recovered=fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=100,resume=True)
        direct=fit(data,warm,ref,context,seed,arm,cfg,'pilot_direct',beat,guard,stop_at=100)
        api.assert_replay(recovered,direct);states.append(direct)
        cells.append(dict(arm=arm,seconds=direct['seconds'],parameters=direct['total_parameters'],
            trainable_parameters=direct['trainable_parameters'],recovery_scientific_state_exact=True,
            training_solver_calls=direct['training_solver_calls'],training_solver_seconds=direct['training_solver_seconds'],
            first_trace=direct['trace'][0],final_trace=direct['trace'][-1]))
    api.assert_matched(*states)
    name=sorted(packets)[0];packet=packets[name];groups=api.allocation.base.query_groups(packet['recordings'],packet['frames'],packet['ids'])
    choose=sorted(set(range(min(12,len(groups))))|set(sorted(range(len(groups)),key=lambda i:(-len(groups[i]),i))[:6]))
    ix=api.np.sort(api.np.concatenate([groups[i] for i in choose]));p={a:api.predict(s,data['causal'])[ranges[name]][ix] for a,s in zip(api.ARMS,states)}
    args=(p,packet['allowed'][ix],packet['recordings'][ix],packet['frames'][ix],packet['ids'][ix],api.np.full(len(ix),-1,api.np.int64));saved=[]
    class Stop(Exception):pass
    def interrupt(v):saved.append(copy.deepcopy(v));raise Stop()
    try:api.allocate(*args,seconds=cfg['solver_seconds_per_query'],checkpoint=interrupt)
    except Stop:pass
    recovered=api.allocate(*args,seconds=cfg['solver_seconds_per_query'],resume_state=saved[0]);direct=api.allocate(*args,seconds=cfg['solver_seconds_per_query'])
    api.parent.exact(recovered,direct)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400,engineering_pass=True,cells=cells,torch_numpy_initial_max_normalized_difference=difference,
        actual_query_recovery_exact=True,allocation_queries=len(choose),allocation_rows=len(ix),maximum_agents=max(len(groups[i]) for i in choose),
        projected_primary_fit_seconds=sum(x['seconds'] for x in cells)/100*104160,
        projection_is_one_context_estimate_excludes_preparation_refits_and_evaluation=True,
        new_policy_outcomes_scored=False,registration_sha256=sha(REG)))
    beat(state='real_own_decision_training_and_recovery_pilot_complete')


def train(resume=False):
    cfg,values,fits=admission();start,beat,guard=runtime(cfg)
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Complete training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Actual training pilot required')
    refs=[];refit_updates=0;first=sorted(json.loads(REG.read_text())['contexts'])[0]
    for context,seed,data,_,_,warm,ref,_ in inputs(values,fits,beat):
        states=[]
        for arm in api.ARMS:
            path=checkpoint(context,seed,arm);s=fit(data,warm,ref,context,seed,arm,cfg,'heads',beat,guard,resume=resume and path.exists())
            if s['step']!=s['budget'] or s['trainable_parameters']!=31119 or int(s['sampler']['counts'].min())<4:raise ValueError('Full registered training budget required')
            if context==first and seed==17:
                rp=checkpoint(context,seed,arm,'full_refit');other=fit(data,warm,ref,context,seed,arm,cfg,'full_refit',beat,guard,resume=resume and rp.exists())
                api.assert_replay(s,other);refit_updates+=s['step']
                once(PUBLIC/f'first_full_refit_{arm}.json',dict(scientific_state_exact=True,updates=s['step'],
                    wall_time_excluded=True,checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(rp)))
            receipt=PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
            once(receipt,dict(context=context,seed=seed,arm=arm,step=s['step'],budget=s['budget'],parameters=s['total_parameters'],
                trainable_parameters=s['trainable_parameters'],checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                seconds=s['seconds'],input_hashes=s['input_hashes'],draw_hash=s['draw_hash'],
                minimum_query_visits=int(s['sampler']['counts'].min()),never_drawn_queries=int((s['sampler']['counts']==0).sum()),
                source_query_counts=s['source_query_counts'],source_completed_cycles=s['sampler']['completed'],
                training_solver_calls=s['training_solver_calls'],training_solver_seconds=s['training_solver_seconds'],
                solver_reason_counts=s['solver_reason_counts'],selected_query_counts=s['selected_query_counts'],
                first_trace=s['trace'][0],final_trace=s['trace'][-1],unknown_rows_drawn=s['unknown_rows_drawn'],
                unknown_rows_with_supervised_labels=0,warm_checkpoint=ref['checkpoint'],registration_sha256=sha(REG)))
            refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));states.append(s)
            beat(state='own_decision_cost_head_complete',fits=len(refs),total=36,context=context,seed=seed,arm=arm)
        api.assert_matched(*states)
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=refs,primary_optimizer_updates=sum(json.loads((ROOT/r['path']).read_text())['step'] for r in refs),
        exact_full_refit_updates=refit_updates,registration_sha256=sha(REG)))
    beat(state='all36_own_decision_cost_fits_frozen')


def evaluate(verify=False,resume=False):
    cfg,values,fits=admission();start,beat,_=runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Complete readout immutable')
    fitted={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('New training receipt changed')
        r=json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('Frozen new cost weights changed')
        fitted[r['context'],r['seed'],r['arm']]=r
    if len(fitted)!=36:raise ValueError('All36new cost endpoints required')
    old=aggregate_groups(previous.NAME);rows=[];refs=[];ids=set();records=set()
    for context,seed,data,ranges,packets,_,_,_ in inputs(values,fits,beat):
        states={a:api.parent.read_checkpoint(ROOT/fitted[context,seed,a]['checkpoint']['path']) for a in api.ARMS};api.assert_matched(*states.values())
        for s in states.values():api.parent.exact(s['input_hashes'],data['input_hashes'])
        predictions={a:api.predict(s,data['causal']) for a,s in states.items()}
        job=next(j for j in values[1][4] if j['key']==context.split('_controller')[0])
        for name,packet in sorted(packets.items()):
            key=f'{context}_{name}_seed{seed}_fit.json';path=PUBLIC/'groups'/key;progress=PRIVATE/'progress'/(key+'.pt.gz');prior=old[key]
            p={a:v[ranges[name]] for a,v in predictions.items()};saved=api.parent.read_checkpoint(previous.PRIVATE/'progress'/(key+'.pt.gz'))
            control=saved['choices']['learned_joint'];control_rank=saved['ranks']['learned_joint']
            if api.parent.digest(control)!=prior['action_sha256']['learned_joint']:raise ValueError('Frozen projected control choices changed')
            if path.exists() and resume and not verify:
                row=json.loads(path.read_text())
                if row['registration_sha256']!=sha(REG) or row['prediction_sha256']!={a:api.parent.digest(v) for a,v in p.items()}:raise ValueError('Completed resumed group changed')
            else:
                if verify:
                    saved=api.parent.read_checkpoint(progress)
                    if saved['prediction_sha256']!={a:api.parent.digest(v) for a,v in p.items()}:raise ValueError('New coherent cost inference changed')
                    choices,ranks,info=saved['choices'],saved['ranks'],saved['allocation']
                else:
                    partial=PRIVATE/'allocation_progress'/(key+'.pt.gz')
                    choices,ranks,info=api.allocate(p,packet['allowed'],packet['recordings'],packet['frames'],packet['ids'],control,
                        seconds=cfg['solver_seconds_per_query'],checkpoint=lambda v:api.parent.save_checkpoint(partial,v),
                        heartbeat=lambda **kw:beat(state='own_decision_allocation',context=context,source=name,seed=seed,**kw),
                        resume_state=api.parent.read_checkpoint(partial) if resume and partial.exists() else None)
                    ranks['frozen_projected_joint']=control_rank.copy()
                    api.parent.save_checkpoint(progress,dict(choices=choices,ranks=ranks,allocation=info,prediction_sha256={a:api.parent.digest(v) for a,v in p.items()}))
                api.np.testing.assert_array_equal(choices['frozen_projected_joint'],control);api.np.testing.assert_array_equal(ranks['frozen_projected_joint'],control_rank)
                result=evaluation.readout(packet,choices,ranks,info,p,job['design']['easy_cut'],values[1][3]['origin'][packet['ids']],data['rms'])
                if result['policies']['frozen_projected_joint']!=prior['result']['policies']['learned_joint']:raise ValueError('Original full projected control score changed')
                for rec,v in result['recordings'].items():
                    if v['frozen_projected_joint']!=prior['result']['recordings'][rec]['learned_joint']:raise ValueError('Original recording control changed')
                row=dict(context=context,source=name,seed=seed,result=result,matched_training=True,original_control_exact=True,
                    minimum_gradient_query_visits=min(int(s['sampler']['counts'].min()) for s in states.values()),
                    coherence={a:previous.coherence(v,packet['allowed'],packet['scales']) for a,v in p.items()},
                    ids_sha256=api.parent.digest(packet['ids']),prediction_sha256={a:api.parent.digest(v) for a,v in p.items()},
                    action_sha256={a:api.parent.digest(c) for a,c in choices.items()},registration_sha256=sha(REG),
                    training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),parent_complete_sha256=sha(previous.PUBLIC/'complete.json'))
                if verify:
                    if row!=json.loads(path.read_text()):raise ValueError('Full coherent inference and frozen-action scores differ')
                else:once(path,row)
            rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));ids.update(map(int,packet['ids']));records.update((name,str(r)) for r in packet['recordings'])
            beat(state='own_decision_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('Full original cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    receipt=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical summary differs')
        once(PUBLIC/'verification.json',dict(**receipt,all72_new_cost_inference_and_frozen_action_scores_exact=True,
            original_full_and_record_control_exact=True,optimization_replayed=False,summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**receipt,groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),training_freeze_sha256=sha(PUBLIC/'training_freeze.json')));pack(ROOT,PUBLIC)
    beat(state='all72_own_decision_verified' if verify else 'all72_own_decision_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train','eval','verify'));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    for d in (PUBLIC,PRIVATE,PRIVATE/'progress',PRIVATE/'allocation_progress'):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(True)
        elif args.phase=='pilot':pilot()
        elif args.phase=='train':train(args.resume)
        else:evaluate(args.phase=='verify',args.resume)
