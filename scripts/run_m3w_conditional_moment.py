"""Matched complete-bank conditional-cost factor identification on SOURCE only."""
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

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_boundary_guard as prior
from src.world_model import m3w_conditional_moment as api
from src.evaluation import m3w_conditional_moment as evaluation
from src.evaluation.m3w_metric_bundle import pack

NAME='european_conditional_moment_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;CONFIG=ROOT/'configs/m3w_conditional_moment_v1.json';REG=PUBLIC/'registration.json'
sha,once,write=prior.sha,prior.once,prior.write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_conditional_moment.py',ROOT/'src/evaluation/m3w_conditional_moment.py',
       ROOT/'tests/test_m3w_conditional_moment.py',ROOT/'tests/test_m3w_conditional_moment_evidence.py',PUBLIC/'protocol.md')
CONTROLS={'frozen_moment_guarded':'moment_guarded','frozen_boundary_guarded':'boundary_guarded',
          'frozen_tdef_guarded':'frozen_tdef_guarded','frozen_own_joint':'frozen_own_joint','tdef_raw':'tdef_raw'}


def admission(register=False):
    pc,*args=prior.admission();cfg=json.loads(CONFIG.read_text())
    replay=json.loads((prior.PUBLIC/'verification.json').read_text())
    if (cfg['arms']!=list(api.ARMS) or cfg['seeds']!=[17,29,43] or cfg['risk_budget']!=.02
            or cfg['parameters_per_head']!=31381 or cfg['training']!=pc['training']
            or cfg['occurrence_loss_weight']!=1 or cfg['conditional_loss_weight']!=1
            or not replay['all72_new_cost_inference_and_action_scores_exact']
            or any(cfg[k] for k in ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Matched complete coverage and unchanged original risk contract required')
    parents={n:sha(prior.PUBLIC/n) for n in ('registration.json','training_freeze.json','complete.json','summary.json','verification.json','metric_bundle.json')}
    deps=('src/world_model/m3w_own_decision_cost.py','src/world_model/m3w_reference_cost_refit.py',
          'src/world_model/m3w_selected_query_critic.py','src/world_model/m3w_easy_hurdle.py',
          'src/world_model/m3w_hurdle_risk.py','src/world_model/m3w_current_bank_deferral.py',
          'src/evaluation/m3w_current_guard_support.py','scripts/run_m3w_boundary_guard.py')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},dependencies={p:sha(ROOT/p) for p in deps},
        parent_files=parents,contexts=json.loads(prior.REG.read_text())['contexts'],seeds=cfg['seeds'],fits=36,
        parameters_each=31381,primary='identified_minus_marginal_full_completion_lower',
        reused_method='old_easy_hurdle_occurrence_and_conditional_cost_supervision_not_a_new_hurdle_method',
        new_application='shared_easy_factor_for_full_coherent7action_five_moment_bank_complete_causal_query_coverage',
        warm_encoder='same_completed_own_choice_encoder_new_equal_fitting_prior_initialized_heads',
        both_training_query_losses='fixed_original_causal_initial_choices',frozen_ranking='current_bank_final_TDEF',
        source_role='exposed_SOURCE_fitting_resubstitution_only',source_forecaster_seed=43,new_trajectory_training=False,
        independent_roles_opened=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    reg=json.loads(REG.read_text())
    if reg['parent_files']!=parents:raise ValueError('Parent results changed')
    for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():
        if sha(ROOT/rel)!=digest:raise ValueError('Registered conditional-factor definition changed')
        if not register:prior.prior.parent.previous.parent.parent.source.frozen.BASE.engine.committed(ROOT/rel)
    if not register:prior.prior.parent.previous.parent.parent.source.frozen.BASE.engine.committed(REG)
    return cfg,*args


def runtime(cfg,phase):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume the same checkpoint')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hphase boundary;resume unchanged budget')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiB disk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['private_checkpoint_cap_bytes']:raise OSError('1GiB private checkpoint cap')
    def beat(**event):
        guard();v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**event)
        write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0,resource_probing=False)
    return start,beat,guard


def inputs(args,beat):
    for context,seed,data,ranges,packets,warm,ref,risk,score in prior.inputs(*args,beat):
        job=next(j for j in args[0][1][4] if j['key']==context.split('_controller')[0])
        prepared=api.prepare(data,packets,ranges,job['design']['easy_cut'])
        yield context,seed,prepared,ranges,packets,warm,ref,risk,score


def checkpoint(context,seed,arm,phase='heads'):return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data,warm,ref,context,seed,arm,cfg,phase,beat,guard,**kwargs):
    return api.fit(data,warm=warm,arm=arm,settings=cfg['training'],
        identity=dict(context=context,registration_sha256=sha(REG),encoder_warm_checkpoint=ref['checkpoint']),
        path=checkpoint(context,seed,arm,phase),heartbeat=lambda **kw:beat(context=context,seed=seed,**kw),guard=guard,**kwargs)


def pilot(resume=False):
    cfg,*args=admission();start,beat,guard=runtime(cfg,'pilot')
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed pilot immutable')
    context,seed,data,_,_,warm,ref,_,score=next(inputs(args,beat));states=[];cells=[]
    for arm in api.ARMS:
        path=checkpoint(context,seed,arm,'pilot')
        if not path.exists():fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=50)
        recovered=fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=100,resume=True)
        rp=checkpoint(context,seed,arm,'pilot_direct')
        direct=fit(data,warm,ref,context,seed,arm,cfg,'pilot_direct',beat,guard,stop_at=100,resume=resume and rp.exists())
        api.assert_replay(recovered,direct);a=api.predict(recovered,data['causal']);b=api.predict(direct,data['causal'])
        api.np.testing.assert_array_equal(a,b)
        prior.prior.api.base.exact(prior.prior.api.choices(score,data['causal']['allowed'],a),prior.prior.api.choices(score,data['causal']['allowed'],b))
        if not any(not api.torch.equal(v,direct['initial_model'][k]) for k,v in direct['model'].items()):raise ValueError('Real optimizer updates required')
        states.append(direct);cells.append(dict(arm=arm,seconds=direct['seconds'],parameters=direct['parameters'],
            recovery_scientific_state_exact=True,complete_prediction_and_choice_recovery_exact=True,
            first_trace=direct['trace'][0],final_trace=direct['trace'][-1]))
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400,engineering_pass=True,cells=cells,projected_main_fit_seconds=sum(c['seconds'] for c in cells)/100*104160,
        projection_excludes_preparation_refits_and_readout=True,new_policy_outcomes_scored=False,registration_sha256=sha(REG)))
    beat(state='real400update_conditional_pilot_complete')


def train(resume=False):
    cfg,*args=admission();start,beat,guard=runtime(cfg,'train')
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Actual resource/recovery pilot required')
    refs=[];refits=0;first=sorted(json.loads(REG.read_text())['contexts'])[0]
    for context,seed,data,_,_,warm,ref,_,_ in inputs(args,beat):
        states=[]
        for arm in api.ARMS:
            path=checkpoint(context,seed,arm);s=fit(data,warm,ref,context,seed,arm,cfg,'heads',beat,guard,resume=resume and path.exists())
            if s['step']!=s['budget'] or s['parameters']!=31381 or s['sampler']['counts'].min()<4:raise ValueError('Complete registered budget and capacity required')
            for k in ('draw_hash','budget','step','unknown_rows_drawn','row_draws'):api.base.exact(s[k],warm[k])
            if context==first and seed==17:
                rp=checkpoint(context,seed,arm,'full_refit');other=fit(data,warm,ref,context,seed,arm,cfg,'full_refit',beat,guard,resume=resume and rp.exists())
                api.assert_replay(s,other);refits+=s['step']
                once(PUBLIC/f'first_full_refit_{arm}.json',dict(scientific_state_exact=True,updates=s['step'],wall_time_excluded=True,
                    checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(rp)))
            receipt=PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
            once(receipt,dict(context=context,seed=seed,arm=arm,step=s['step'],budget=s['budget'],parameters=s['parameters'],
                checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),seconds=s['seconds'],
                input_hashes=s['input_hashes'],draw_hash=s['draw_hash'],minimum_query_visits=int(s['sampler']['counts'].min()),
                source_query_counts=s['source_query_counts'],priors=s['priors'],first_trace=s['trace'][0],final_trace=s['trace'][-1],
                unknown_rows_drawn=s['unknown_rows_drawn'],unknown_rows_with_supervised_labels=0,row_draws=s['row_draws'],
                warm_checkpoint=ref['checkpoint'],prior_query_draws_exact=True,registration_sha256=sha(REG)))
            refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));states.append(s)
            beat(state='conditional_head_complete',fits=len(refs),total=36,context=context,seed=seed,arm=arm)
        api.assert_matched(*states)
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=refs,primary_optimizer_updates=sum(json.loads((ROOT/r['path']).read_text())['step'] for r in refs),exact_full_refit_updates=refits,registration_sha256=sha(REG)))
    beat(state='all36_conditional_heads_frozen')


def evaluate(verify=False,resume=False):
    cfg,*args=admission();start,beat,_=runtime(cfg,'verify' if verify else 'eval')
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed readout immutable')
    fitted={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('New receipt changed')
        r=json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('New frozen weights changed')
        fitted[r['context'],r['seed'],r['arm']]=r
    if len(fitted)!=36:raise ValueError('All36heads required')
    old=prior.prior.parent.aggregate_groups(prior.NAME);rows=[];refs=[];ids=set();records=set()
    for context,seed,data,ranges,packets,_,_,_,score in inputs(args,beat):
        states={a:api.base.read_checkpoint(ROOT/fitted[context,seed,a]['checkpoint']['path']) for a in api.ARMS};api.assert_matched(*states.values())
        for s in states.values():api.base.exact(s['input_hashes'],data['input_hashes'])
        predictions={a:api.predict(s,data['causal'],components=True) for a,s in states.items()}
        job=next(j for j in args[0][1][4] if j['key']==context.split('_controller')[0])
        for name,packet in sorted(packets.items()):
            key=f'{context}_{name}_seed{seed}_fit.json';path=PUBLIC/'groups'/key;progress=PRIVATE/'progress'/(key+'.pt.gz');parent=old[key]
            p={a:v[0][ranges[name]] for a,v in predictions.items()};q={a:v[1][ranges[name]] for a,v in predictions.items()};choices={};ranks={}
            for a in api.ARMS:choices[a+'_guarded'],ranks[a+'_guarded']=prior.prior.api.choices(score[ranges[name]],packet['allowed'],p[a])['guarded']
            saved=api.base.read_checkpoint(prior.PRIVATE/'progress'/(key+'.pt.gz'))
            for a,b in CONTROLS.items():
                choices[a]=saved['choices'][b];ranks[a]=saved['ranks'][b]
                if api.base.digest(choices[a])!=parent['action_sha256'][b]:raise ValueError('Original frozen action changed')
            coherent=all((v[:,:,0]<=v[:,:,2]+1e-10).all() and (v[:,:,3]<=v[:,:,2]+1e-10).all() and (v[:,:,4]<=v[:,:,1]+1e-10).all()
                and (v[:,:,:2].sum(2)<=packet['envelope']+1e-10).all() for v in p.values())
            feasible=coherent and all(prior.prior.api.parent.allocation.base.feasible(p[a],packet['allowed'],choices[a+'_guarded']) for a in api.ARMS)
            if not feasible:raise ValueError('Coherent costs and unchanged predicted guard required before labels')
            payload=dict(choices=choices,ranks=ranks,prediction_sha256={a:api.base.digest(v) for a,v in p.items()},event_probability_sha256={a:api.base.digest(v) for a,v in q.items()})
            if verify:api.base.exact(api.base.read_checkpoint(progress),payload)
            elif not(path.exists() and resume):api.base.save_checkpoint(progress,payload)
            result=evaluation.readout(packet,choices,ranks,p,q,job['design']['easy_cut'],args[0][1][3]['origin'][packet['ids']])
            for a,b in CONTROLS.items():
                if result['policies'][a]!=parent['result']['policies'][b]:raise ValueError('Original full control changed')
                for rec,v in result['recordings'].items():
                    if v[a]!=parent['result']['recordings'][rec][b]:raise ValueError('Original record control changed')
            row=dict(context=context,source=name,seed=seed,result=result,matched_training=True,original_control_exact=True,
                guarded_predicted_feasible=bool(feasible),minimum_gradient_query_visits=min(int(s['sampler']['counts'].min()) for s in states.values()),
                ids_sha256=api.base.digest(packet['ids']),prediction_sha256=payload['prediction_sha256'],event_probability_sha256=payload['event_probability_sha256'],
                action_sha256={a:api.base.digest(c) for a,c in choices.items()},registration_sha256=sha(REG),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),parent_complete_sha256=sha(prior.PUBLIC/'complete.json'))
            if verify or(path.exists() and resume):
                if row!=json.loads(path.read_text()):raise ValueError('Complete new inference/old-control replay differs')
            else:once(path,row)
            rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            ids.update(map(int,packet['ids']));records.update((name,str(r)) for r in packet['recordings'])
            beat(state='conditional_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('Original full cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    receipt=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical summary differs')
        once(PUBLIC/'verification.json',dict(**receipt,all72_new_moment_probability_inference_and_action_scores_exact=True,original_full_and_record_controls_exact=True,
            optimization_replayed=False,summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**receipt,groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),training_freeze_sha256=sha(PUBLIC/'training_freeze.json')));pack(ROOT,PUBLIC)
    beat(state='all72_conditional_verified' if verify else 'all72_conditional_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train','eval','verify'));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    for d in (PUBLIC,PRIVATE,PRIVATE/'progress'):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(True)
        elif args.phase=='pilot':pilot(args.resume)
        elif args.phase=='train':train(args.resume)
        else:evaluate(args.phase=='verify',args.resume)
