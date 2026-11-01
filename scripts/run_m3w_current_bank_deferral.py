"""Run the registered current-bank expected-ADE versus published TDEF control."""
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
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_own_decision_cost as parent
from scripts import report_m3w_own_decision_cost as parent_report
from src.world_model import m3w_current_bank_deferral as api
from src.evaluation import m3w_current_bank_deferral as evaluation
from src.evaluation.m3w_metric_bundle import pack

NAME='european_current_bank_deferral_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;CONFIG=ROOT/'configs/m3w_current_bank_deferral_v1.json';REG=PUBLIC/'registration.json'
sha,once,write=parent.sha,parent.once,parent.write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_current_bank_deferral.py',ROOT/'src/evaluation/m3w_current_bank_deferral.py',
       ROOT/'tests/test_m3w_current_bank_deferral.py',ROOT/'tests/test_m3w_current_bank_deferral_evidence.py',PUBLIC/'protocol.md')


def admission(register=False):
    pc,values,fits=parent.admission();prior=parent_report.build();cfg=json.loads(CONFIG.read_text())
    if (cfg['arms']!=list(api.ARMS) or cfg['seeds']!=[17,29,43] or cfg['risk_budget']!=.02
            or cfg['parameters_per_head']!=30726 or cfg['training']!={k:v for k,v in pc['training'].items() if k!='solver_seconds_per_query'}
            or any(cfg[k] for k in ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Same complete causal inputs,coverage,budget and risk contract required')
    parent_files={n:sha(parent.PUBLIC/n) for n in ('registration.json','training_freeze.json','summary.json','complete.json','verification.json','metric_bundle.json')}
    deps=('src/world_model/m3w_regression_deferral.py','src/world_model/m3w_selected_query_critic.py','src/world_model/m3w_reference_cost_refit.py')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},parent_files=parent_files,
        dependencies={p:sha(ROOT/p) for p in deps},contexts=json.loads(parent.REG.read_text())['contexts'],
        seeds=cfg['seeds'],fits=36,parameters_each=30726,primary='guarded_tdef_minus_ade_full_completion_lower',
        method='Mao_Mohri_Zhong_ICML2024_Eq3_loss_adapter_not_original_architecture_or_theorem_reproduction',
        all_causal_inputs_and_preprocessing_frozen=True,frozen_guard='own_choice_original_pointwise_two2percent_rules',
        source_forecaster_seed=43,source_role='exposed_SOURCE_fitting_resubstitution_only',
        independent_roles_opened=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        reg=json.loads(REG.read_text())
        if reg['parent_files']!=parent_files:raise ValueError('Complete own-decision evidence changed')
        for rel,digest in dict(reg['bindings'],**reg['dependencies']).items():
            if sha(ROOT/rel)!=digest:raise ValueError('Registered baseline definition changed')
            parent.previous.parent.parent.source.frozen.BASE.engine.committed(ROOT/rel)
        parent.previous.parent.parent.source.frozen.BASE.engine.committed(REG)
    fitted={}
    for ref in prior['training']['freeze']['fits']:
        row=json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256']:raise ValueError('Parent cost weights changed')
        fitted[row['context'],row['seed'],row['arm']]=row
    return cfg,values,fits,fitted


def runtime(cfg,phase):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume same definition/checkpoint')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hphase boundary')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiBdisk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['private_checkpoint_cap_bytes']:raise OSError('1GiBprivate checkpoint cap')
    def beat(**kw):
        guard();v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**kw)
        write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0,resource_probing=False)
    return start,beat,guard


def inputs(values,fits,fitted,beat):
    groups=parent.aggregate_groups(parent.NAME)
    for context,seed,data,ranges,packets,warm,ref,_ in parent.inputs(values,fits,beat):
        row=fitted[context,seed,'own_choice'];state=api.base.read_checkpoint(ROOT/row['checkpoint']['path'])
        api.base.exact(state['input_hashes'],data['input_hashes']);risk=parent.api.predict(state,data['causal'])
        for name in packets:
            if api.base.digest(risk[ranges[name]])!=groups[f'{context}_{name}_seed{seed}_fit.json']['prediction_sha256']['own_choice']:
                raise ValueError('Frozen risk inference differs')
        yield context,seed,api.training_data(data,packets,ranges),ranges,packets,warm,ref,risk


def checkpoint(context,seed,arm,phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_{arm}.pt.gz'


def fit(data,warm,ref,context,seed,arm,cfg,phase,beat,guard,**kw):
    return api.fit(data,warm=warm,arm=arm,settings=cfg['training'],
        identity=dict(context=context,registration_sha256=sha(REG),warm_checkpoint=ref['checkpoint']),
        path=checkpoint(context,seed,arm,phase),heartbeat=lambda **event:beat(context=context,seed=seed,**event),guard=guard,**kw)


def pilot(resume=False):
    cfg,values,fits,fitted=admission();start,beat,guard=runtime(cfg,'pilot')
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed pilot immutable')
    context,seed,data,_,_,warm,ref,risk=next(inputs(values,fits,fitted,beat));cells=[];states=[]
    for arm in api.ARMS:
        path=checkpoint(context,seed,arm,'pilot')
        if not path.exists():fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=50)
        recovered=fit(data,warm,ref,context,seed,arm,cfg,'pilot',beat,guard,stop_at=100,resume=True)
        direct_path=checkpoint(context,seed,arm,'pilot_direct')
        direct=fit(data,warm,ref,context,seed,arm,cfg,'pilot_direct',beat,guard,stop_at=100,resume=resume and direct_path.exists())
        api.assert_replay(recovered,direct);states.append(direct)
        a,b=api.predict(recovered,data['causal']),api.predict(direct,data['causal'])
        api.np.testing.assert_array_equal(a,b)
        api.base.exact(api.choices(a,data['causal']['allowed'],risk),api.choices(b,data['causal']['allowed'],risk))
        if not any(not api.torch.equal(v,direct['initial_model'][k]) for k,v in direct['model'].items()):raise ValueError('Actual optimizer update required')
        cells.append(dict(arm=arm,seconds=direct['seconds'],parameters=direct['parameters'],scientific_recovery_exact=True,
            complete_causal_prediction_and_choice_recovery_exact=True,first_trace=direct['trace'][0],final_trace=direct['trace'][-1]))
    api.assert_matched(*states)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        actual_optimizer_updates=400,engineering_pass=True,cells=cells,projected_main_fit_seconds=sum(c['seconds'] for c in cells)/100*104160,
        projection_excludes_preparation_refits_and_readout=True,new_policy_outcomes_scored=False,registration_sha256=sha(REG)))
    beat(state='real400update_current_bank_pilot_complete')


def train(resume=False):
    cfg,values,fits,fitted=admission();start,beat,guard=runtime(cfg,'train')
    if (PUBLIC/'training_freeze.json').exists():raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']:raise ValueError('Actual recovery/resource pilot required')
    refs=[];refit_updates=0;first=sorted(json.loads(REG.read_text())['contexts'])[0]
    for context,seed,data,_,_,warm,ref,_ in inputs(values,fits,fitted,beat):
        states=[]
        for arm in api.ARMS:
            path=checkpoint(context,seed,arm);s=fit(data,warm,ref,context,seed,arm,cfg,'heads',beat,guard,resume=resume and path.exists())
            if s['step']!=s['budget'] or s['parameters']!=30726 or s['sampler']['counts'].min()<4:raise ValueError('Full matched budget/capacity/coverage required')
            parent_ref=fitted[context,seed,'own_choice'];p=api.base.read_checkpoint(ROOT/parent_ref['checkpoint']['path'])
            for k in ('draw_hash','budget','step','unknown_rows_drawn','row_draws'):api.base.exact(s[k],p[k])
            if context==first and seed==17:
                other_path=checkpoint(context,seed,arm,'full_refit')
                other=fit(data,warm,ref,context,seed,arm,cfg,'full_refit',beat,guard,resume=resume and other_path.exists())
                api.assert_replay(s,other);refit_updates+=s['step']
                once(PUBLIC/f'first_full_refit_{arm}.json',dict(scientific_state_exact=True,updates=s['step'],wall_time_excluded=True,
                    checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(other_path)))
            receipt=PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
            once(receipt,dict(context=context,seed=seed,arm=arm,step=s['step'],budget=s['budget'],parameters=s['parameters'],
                checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),seconds=s['seconds'],
                input_hashes=s['input_hashes'],draw_hash=s['draw_hash'],minimum_query_visits=int(s['sampler']['counts'].min()),
                source_query_counts=s['source_query_counts'],source_completed_cycles=s['sampler']['completed'],
                first_trace=s['trace'][0],final_trace=s['trace'][-1],unknown_rows_drawn=s['unknown_rows_drawn'],
                unknown_rows_with_supervised_labels=0,row_draws=s['row_draws'],warm_checkpoint=ref['checkpoint'],
                prior_query_draws_exact=True,registration_sha256=sha(REG)))
            refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)));states.append(s)
            beat(state='current_bank_head_complete',fits=len(refs),total=36,context=context,seed=seed,arm=arm)
        api.assert_matched(*states)
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        fits=refs,primary_optimizer_updates=sum(json.loads((ROOT/r['path']).read_text())['step'] for r in refs),
        exact_full_refit_updates=refit_updates,registration_sha256=sha(REG)))
    beat(state='all36_current_bank_heads_frozen')


def evaluate(verify=False,resume=False):
    cfg,values,fits,fitted=admission();start,beat,_=runtime(cfg,'verify' if verify else 'eval')
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed readout immutable')
    trained={}
    for ref in json.loads((PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Frozen fit receipt changed')
        r=json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/r['checkpoint']['path'])!=r['checkpoint']['sha256']:raise ValueError('Frozen baseline weights changed')
        trained[r['context'],r['seed'],r['arm']]=r
    if len(trained)!=36:raise ValueError('All36heads required')
    old=parent.aggregate_groups(parent.NAME);rows=[];refs=[];ids=set();records=set()
    for context,seed,data,ranges,packets,_,_,risk in inputs(values,fits,fitted,beat):
        states={a:api.base.read_checkpoint(ROOT/trained[context,seed,a]['checkpoint']['path']) for a in api.ARMS}
        api.assert_matched(*states.values())
        for s in states.values():api.base.exact(s['input_hashes'],data['input_hashes'])
        predictions={a:api.predict(s,data['causal']) for a,s in states.items()}
        job=next(j for j in values[1][4] if j['key']==context.split('_controller')[0])
        for name,packet in sorted(packets.items()):
            key=f'{context}_{name}_seed{seed}_fit.json';path=PUBLIC/'groups'/key;progress=PRIVATE/'progress'/(key+'.pt.gz');prior=old[key]
            score={a:v[ranges[name]] for a,v in predictions.items()};cost=risk[ranges[name]];choices={};ranks={}
            for arm in api.ARMS:
                for mode,(c,r) in api.choices(score[arm],packet['allowed'],cost).items():choices[arm+'_'+mode],ranks[arm+'_'+mode]=c,r
            frozen=api.base.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))
            for mode in ('independent','joint'):
                previous='own_'+mode;current='frozen_'+previous;c=frozen['choices'][previous]
                if api.base.digest(c)!=prior['action_sha256'][previous]:raise ValueError('Frozen parent choices changed')
                at=api.np.flatnonzero(c>=0);r=api.np.zeros(len(c));r[at]=cost[at,c[at],0]-cost[at,c[at],1]
                api.np.testing.assert_array_equal(r,frozen['ranks'][previous]);choices[current]=c.copy();ranks[current]=r
            predicted_feasible=all(api.parent.allocation.base.feasible(cost,packet['allowed'],choices[a+'_guarded']) for a in api.ARMS)
            if not predicted_feasible:raise ValueError('No predicted risk violation may escape frozen guard')
            if verify:
                saved=api.base.read_checkpoint(progress);api.base.exact(saved,dict(choices=choices,ranks=ranks,score_sha256={a:api.base.digest(v) for a,v in score.items()}))
            elif not (path.exists() and resume):
                api.base.save_checkpoint(progress,dict(choices=choices,ranks=ranks,score_sha256={a:api.base.digest(v) for a,v in score.items()}))
            result=evaluation.readout(packet,choices,ranks,job['design']['easy_cut'],values[1][3]['origin'][packet['ids']])
            for mode in ('independent','joint'):
                a='own_'+mode;b='frozen_'+a
                if result['policies'][b]!=prior['result']['policies'][a]:raise ValueError('Frozen full parent score differs')
                for rec,v in result['recordings'].items():
                    if v[b]!=prior['result']['recordings'][rec][a]:raise ValueError('Frozen recording score differs')
            row=dict(context=context,source=name,seed=seed,result=result,matched_training=True,original_control_exact=True,
                guarded_predicted_feasible=True,minimum_gradient_query_visits=min(int(s['sampler']['counts'].min()) for s in states.values()),
                ids_sha256=api.base.digest(packet['ids']),score_sha256={a:api.base.digest(v) for a,v in score.items()},
                action_sha256={a:api.base.digest(c) for a,c in choices.items()},registration_sha256=sha(REG),
                training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),parent_complete_sha256=sha(parent.PUBLIC/'complete.json'))
            if verify or (path.exists() and resume):
                if row!=json.loads(path.read_text()):raise ValueError('Complete new inference/frozen action score differs')
            else:once(path,row)
            rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));ids.update(map(int,packet['ids']));records.update((name,str(r)) for r in packet['recordings'])
            beat(state='current_bank_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('Original full target/record cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    receipt=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical public summary differs')
        once(PUBLIC/'verification.json',dict(**receipt,all72_new_score_inference_and_action_scores_exact=True,original_full_and_record_controls_exact=True,
            optimization_replayed=False,summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**receipt,groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),training_freeze_sha256=sha(PUBLIC/'training_freeze.json')));pack(ROOT,PUBLIC)
    beat(state='all72_current_bank_verified' if verify else 'all72_current_bank_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train','eval','verify'));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    for d in (PUBLIC,PRIVATE,PRIVATE/'progress'):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(True)
        elif args.phase=='pilot':pilot(args.resume)
        elif args.phase=='train':train(args.resume)
        else:evaluate(args.phase=='verify',args.resume)
