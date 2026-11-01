"""Registered matched seven-action decision-cost fitting and fixed SOURCE readout."""
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

if platform.system() == 'Darwin' and platform.machine() != 'arm64': raise RuntimeError('Native arm64 required before Torch')
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_coupled_benefit_risk as previous
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_decision_cost_critic as api
from src.evaluation import m3w_decision_cost_readout as evaluation
from src.evaluation.m3w_metric_bundle import pack

np = api.np; source = previous.source
NAME = 'european_decision_cost_critic_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE = ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG = ROOT/'configs/m3w_decision_cost_critic_v1.json'; REG = PUBLIC/'registration.json'
sha, once, write = previous.sha, previous.once, previous.write
FILES = (Path(__file__), CONFIG, ROOT/'src/world_model/m3w_decision_cost_critic.py',
    ROOT/'src/evaluation/m3w_decision_cost_readout.py', ROOT/'tests/test_m3w_decision_cost_critic.py', PUBLIC/'protocol.md')


def admission(register=False):
    oldcfg, values, store, banks, models, _ = previous.admission()
    cfg = json.loads(CONFIG.read_text()); check = json.loads((previous.PUBLIC/'verification.json').read_text())
    if (cfg['training'] != oldcfg['training'] or cfg['seeds'] != [17, 29, 43] or cfg['risk_budget'] != .02
            or cfg['decision'] != dict(cost_weight=1., dual_every=100, dual_learning_rate=1., dual_maximum=100.)
            or not check['all72groups_exact'] or check['summary_sha256'] != sha(previous.PUBLIC/'summary.json')
            or any(cfg[k] for k in ('held_scored','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Verified complete matched regression controls and original task required')
    controls = {}
    for ref in json.loads((previous.PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Old fit receipt changed')
        row = json.loads((ROOT/ref['path']).read_text())
        if row['mode'] == 'released_benefit':
            cp = row['checkpoint']
            if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('Old regression checkpoint changed')
            controls[row['context'],row['seed']] = cp
    if len(controls) != 18: raise ValueError('All18 fixed regression heads required')
    once(REG, dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(previous.REG), parent_training_freeze_sha256=sha(previous.PUBLIC/'training_freeze.json'),
        parent_verification_sha256=sha(previous.PUBLIC/'verification.json'), contexts=json.loads(previous.REG.read_text())['contexts'],
        changed_factor='add_expected_seven_action_decision_cost_with_whole_SOURCE_record_dual_updates',
        cached_verified_regression_controls=18, fresh_neural_cost_fits=18, primary_optimizer_updates=36000,
        exact_refit_updates=2000, real_pilot_updates=300, forecaster_seed=43,
        primary='decision_minus_regression_full_joint_completion_lower', inference_outcomes_used=False,
        own_outcomes_read_at_registration=False, role='exposed_SOURCE_fitting_resubstitution_only',
        independent_roles_opened=False, deployment_changed=False, submission_ready=False, stage5c_executed=False, smc_enabled=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if sha(ROOT/rel) != digest: raise ValueError('Frozen definition changed:'+rel)
            source.frozen.BASE.engine.committed(ROOT/rel)
        source.frozen.BASE.engine.committed(REG)
    return cfg, values, store, banks, models, controls


def runtime(cfg):
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1); start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume frozen decision-cost roster')))
    def guard():
        if time.monotonic()-start > cfg['hard_runtime_limit_seconds']: raise TimeoutError('12h boundary; explicit resume')
        if shutil.disk_usage(ROOT).free < cfg['disk_reserve_bytes']: raise OSError('10GiB reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz')) > cfg['private_checkpoint_cap_bytes']: raise OSError('1GiB private checkpoint cap')
    def beat(**event):
        guard(); row = dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        write(PRIVATE/'heartbeat.json',row)
        with (PRIVATE/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),device='cpu',threads=4,interop_threads=1,workers=0,resource_probing=False)
    return start,beat,guard


def checkpoint(context,seed,phase='heads'):
    return PRIVATE/phase/f'{context}_seed{seed}_decision.pt.gz'


def fit(data,cfg,context,seed,phase,beat,guard,**kw):
    return api.fit(data,seed=seed,settings=cfg['training'],decision=cfg['decision'],
        identity=dict(registration_sha256=sha(REG),context=context,cost_seed=seed),path=checkpoint(context,seed,phase),
        heartbeat=lambda **event:beat(context=context,seed=seed,**event),guard=guard,**kw)


def equal(left,right):
    for k in left:
        if k != 'seconds': api.parent.exact(left[k],right[k])


def pilot():
    cfg,values,store,banks,models,_ = admission(); start,beat,guard = runtime(cfg)
    if (PUBLIC/'pilot.json').exists(): raise ValueError('Completed pilot immutable')
    old,main = aggregate_groups(source.frozen.source.parent.NAME),aggregate_groups(source.frozen.NAME)
    context,items = sorted(values[2]['contexts'].items())[0]
    packets,changed = previous.context_packets(context,items,values,store,banks,beat)
    data,_,_ = source.prepare(context,17,packets,changed,store,models,old,main,beat)
    control = fit(data,cfg,context,17,'pilot_regression',beat,guard,stop_at=100,regression_control=True)
    cached = api.parent.read_checkpoint(previous.checkpoint(context,17,'released_benefit','pilot'))
    api.assert_matched(control,cached); api.parent.exact(control['model'],cached['model'])
    fit(data,cfg,context,17,'pilot',beat,guard,stop_at=50)
    resumed = fit(data,cfg,context,17,'pilot',beat,guard,stop_at=100,resume=True)
    direct = fit(data,cfg,context,17,'pilot_direct',beat,guard,stop_at=100)
    equal(resumed,direct); api.assert_matched(resumed,cached)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,real_optimizer_updates=300,
        control100updates_exact=True,interrupted_recovery_exact=True,parameters=resumed['parameters'],
        first_trace=resumed['trace'][0],last_trace=resumed['trace'][-1],
        rough36000update_training_seconds=direct['seconds']*360,
        projection_excludes_admission_context_preparation_refit_and_readout=True,engineering_pass=True,
        own_policy_outcomes_read=False,registration_sha256=sha(REG)))
    beat(state='real300update_pilot_complete')


def train(resume=False):
    cfg,values,store,banks,models,controls = admission(); start,beat,guard = runtime(cfg)
    if (PUBLIC/'training_freeze.json').exists(): raise ValueError('Completed training immutable')
    if not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']: raise ValueError('Real pilot required')
    old,main = aggregate_groups(source.frozen.source.parent.NAME),aggregate_groups(source.frozen.NAME); refs=[]
    first = sorted(values[2]['contexts'])[0]
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed = previous.context_packets(context,items,values,store,banks,beat)
        for seed in cfg['seeds']:
            data,_,_ = source.prepare(context,seed,packets,changed,store,models,old,main,beat)
            path = checkpoint(context,seed); state = fit(data,cfg,context,seed,'heads',beat,guard,resume=resume and path.exists())
            control = api.parent.read_checkpoint(ROOT/controls[context,seed]['path']); api.assert_matched(state,control)
            if state['step'] != 2000 or state['parameters'] != 30857: raise ValueError('Complete matched update budget required')
            if context == first and seed == 17:
                rp=checkpoint(context,seed,'full_refit')
                equal(state,fit(data,cfg,context,seed,'full_refit',beat,guard,resume=resume and rp.exists()))
                once(PUBLIC/'first_full_refit.json',dict(scientific_state_exact=True,updates=2000,
                    checkpoint_sha256=sha(path),refit_checkpoint_sha256=sha(rp)))
            receipt=PUBLIC/'fits'/f'{context}_seed{seed}.json'
            once(receipt,dict(context=context,seed=seed,step=state['step'],parameters=state['parameters'],
                checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                input_hashes=state['input_hashes'],draw_hash=state['draw_hash'],probability_temperature=state['probability_temperature'],
                record_count=len(state['record_keys']),maximum_dual=float(state['dual'].max()),
                first_trace=state['trace'][0],final_trace=state['trace'][-1],seconds=state['seconds'],
                unknown_rows_drawn=state['unknown_rows_drawn'],unknown_rows_with_supervised_labels=0,
                unknown_bounds_used_in_decision_objective=True,old_control=controls[context,seed],
                registration_sha256=sha(REG),paired_features_initialization_draws_updates_exact=True))
            refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)))
            beat(state='decision_head_complete',heads=len(refs),total=18,context=context,seed=seed)
            del data,state,control
        del packets,changed
    once(PUBLIC/'training_freeze.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,fits=refs,
        fresh_optimizer_updates=36000,exact_refit_updates=2000,cached_verified_controls=18,
        registration_sha256=sha(REG),independent_roles_opened=False))
    beat(state='all18_decision_heads_frozen')


def evaluate(verify=False,resume=False):
    cfg,values,store,banks,models,controls = admission(); start,beat,guard = runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify: raise ValueError('Completed results immutable')
    freeze=json.loads((PUBLIC/'training_freeze.json').read_text()); fitted={}
    for ref in freeze['fits']:
        if sha(ROOT/ref['path']) != ref['sha256']: raise ValueError('Fit receipt changed')
        row=json.loads((ROOT/ref['path']).read_text()); cp=row['checkpoint']
        if sha(ROOT/cp['path']) != cp['sha256']: raise ValueError('New head changed')
        fitted[row['context'],row['seed']]=cp
    if len(fitted)!=18: raise ValueError('All18 frozen new heads required')
    old,main=aggregate_groups(source.frozen.source.parent.NAME),aggregate_groups(source.frozen.NAME)
    original=aggregate_groups(previous.NAME); completed=aggregate_groups(NAME) if verify else {}; rows=[]; refs=[]; ids=set(); recs=set()
    for context,items in sorted(values[2]['contexts'].items()):
        packets,changed=previous.context_packets(context,items,values,store,banks,beat)
        job=next(j for j in values[4] if j['key']==context.split('_controller')[0])
        for seed in cfg['seeds']:
            data,ranges,_=source.prepare(context,seed,packets,changed,store,models,old,main,beat)
            new=api.parent.read_checkpoint(ROOT/fitted[context,seed]['path']); control=api.parent.read_checkpoint(ROOT/controls[context,seed]['path'])
            api.assert_matched(new,control); api.parent.exact(new['input_hashes'],data['input_hashes'])
            pred=api.predict(new,data['causal']); oldpred=api.base.predict(control,data['causal'])
            for name,packet in sorted(packets.items()):
                key=f'{context}_{name}_seed{seed}_fit.json'; path=PUBLIC/'groups'/key; progress=PRIVATE/'progress'/(key+'.pt.gz')
                reference=original[key]; p=dict(protected_benefit=oldpred[ranges[name]],released_benefit=pred[ranges[name]])
                if (api.parent.digest(packet['ids'])!=reference['ids_sha256']
                        or api.parent.digest(p['protected_benefit'])!=reference['prediction_sha256']['released_benefit']):
                    raise ValueError('Exact original prediction/row controls required')
                if path.exists() and not verify:
                    if not resume: raise ValueError('Explicit partial readout resume required')
                    row=json.loads(path.read_text())
                    if row['registration_sha256']!=sha(REG): raise ValueError('Partial registration changed')
                else:
                    oldchoices=api.parent.read_checkpoint(previous.PRIVATE/'progress'/(key+'.pt.gz'))['choices']
                    controlchoices=dict(set_independent=oldchoices['released_independent'],set_joint=oldchoices['released_joint'])
                    if verify:
                        saved=api.parent.read_checkpoint(progress)
                        choices,ranks,info=saved['choices'],saved['ranks'],saved['allocation']
                        if saved['prediction_sha256']!={a:api.parent.digest(v) for a,v in p.items()}: raise ValueError('Frozen prediction changed')
                    else:
                        partial=PRIVATE/'allocation_progress'/(key+'.pt.gz')
                        choices,ranks,info=previous.decoder.allocate(p,packet['allowed'],packet['recordings'],packet['frames'],packet['ids'],controlchoices,
                            seconds=cfg['solver_seconds_per_query'],checkpoint=lambda v:api.parent.save_checkpoint(partial,v),
                            heartbeat=lambda **kw:beat(state='decision_allocation',context=context,source=name,seed=seed,**kw),
                            resume_state=api.parent.read_checkpoint(partial) if resume and partial.exists() else None)
                        api.parent.save_checkpoint(progress,dict(choices=choices,ranks=ranks,allocation=info,
                            prediction_sha256={a:api.parent.digest(v) for a,v in p.items()}))
                    for mode in ('independent','joint'):
                        np.testing.assert_array_equal(choices['protected_'+mode],oldchoices['released_'+mode])
                    result=evaluation.readout(packet,choices,ranks,info,p,job['design']['easy_cut'],values[3]['origin'][packet['ids']],data['rms'])
                    if result['policies']['regression_joint']!=reference['result']['policies']['released_joint']: raise ValueError('Old full-policy score differs')
                    row=dict(context=context,source=name,seed=seed,result=result,old_control_exact=True,paired_training_verified=True,
                        ids_sha256=reference['ids_sha256'],prediction_sha256={a:api.parent.digest(v) for a,v in p.items()},
                        action_sha256={a:api.parent.digest(v) for a,v in choices.items()},registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),held_scored=False,independent_roles_opened=False)
                    if verify:
                        if row!=completed[key]: raise ValueError('Frozen actions/inference/original full scoring differ')
                    else: once(path,row)
                ids.update(map(int,packet['ids'])); recs.update((name,str(r)) for r in packet['recordings'])
                rows.append(row); refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                beat(state='decision_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
            del data,new,control,pred,oldpred
        del packets,changed
    if len(ids)!=149510 or len(recs)!=74: raise ValueError('Original complete cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg)
    summary.update(unique_target_rows=len(ids),unique_recordings=len(recs))
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()): raise ValueError('Summary changed')
        once(PUBLIC/'verification.json',dict(all72_frozen_actions_inference_and_scores_exact=True,all72_old_controls_exact=True,
            optimization_replayed=False,summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary)
        once(PUBLIC/'complete.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=refs,
            summary_sha256=sha(PUBLIC/'summary.json'),training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),registration_sha256=sha(REG)))
        pack(ROOT,PUBLIC)
    write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),dict(pid=os.getpid(),
        seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='all72_verified' if verify else 'all72_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','pilot','train','eval','verify')); p.add_argument('--resume',action='store_true')
    args=p.parse_args(); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register': admission(True)
        elif args.phase=='pilot': pilot()
        elif args.phase=='train': train(args.resume)
        else: evaluate(args.phase=='verify',args.resume)
