"""Register,measure and evaluate zero-step cost coherence without opening new roles."""
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
from scripts import run_m3w_reference_cost_refit as parent
from scripts.verify_m3w_reference_refit_coherence import coherence
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.world_model import m3w_coherent_cost_projection as projection
from src.evaluation import m3w_coherent_cost_projection as evaluation
from src.evaluation.m3w_metric_bundle import pack

api,decoder=parent.api,parent.decoder
NAME='european_coherent_cost_projection_v1';PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME;CONFIG=ROOT/'configs/m3w_coherent_cost_projection_v1.json';REG=PUBLIC/'registration.json'
sha,once,write=parent.sha,parent.once,parent.write
FILES=(Path(__file__),CONFIG,ROOT/'src/world_model/m3w_coherent_cost_projection.py',ROOT/'src/evaluation/m3w_coherent_cost_projection.py',
    ROOT/'tests/test_m3w_coherent_cost_projection.py',ROOT/'tests/test_m3w_coherent_projection_controls.py',PUBLIC/'protocol.md')


def admission(register=False):
    _,values=parent.admission();cfg=json.loads(CONFIG.read_text())
    checked=json.loads((parent.PUBLIC/'verification.json').read_text());diag=json.loads((parent.PUBLIC/'coherence_verification.json').read_text())
    if (not checked['all72_new_cost_inference_frozen_actions_and_scores_exact'] or not checked['original_parent_controls_exact']
            or checked['summary_sha256']!=sha(parent.PUBLIC/'summary.json') or not diag['prediction_bytes_unchanged']
            or diag['models']!=36 or diag['verification_sha256']!=sha(parent.PUBLIC/'verification.json')
            or diag['executable_sha256']!=sha(ROOT/'scripts/verify_m3w_reference_refit_coherence.py')):
        raise ValueError('Complete verified preceding cost refit and nonintervening diagnosis required')
    if (cfg['seeds']!=[17,29,43] or cfg['risk_budget']!=.02 or cfg['solver_seconds_per_query']!=2.
            or cfg['reference_pool']!='equal_mean_of_causally_available_candidates_per_agent'
            or cfg['benefit_projection']!='minimum_of_old_benefit_and_shared_reference'
            or not cfg['harm_and_easy_harm_unchanged']
            or any(cfg[k] for k in ('new_training','independent_roles_opened','deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Registered zero-step intervention and original roles/budgets required')
    fits={}
    for ref in json.loads((parent.PUBLIC/'training_freeze.json').read_text())['fits']:
        if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Original cost fit receipt changed')
        f=json.loads((ROOT/ref['path']).read_text())
        if sha(ROOT/f['checkpoint']['path'])!=f['checkpoint']['sha256']:raise ValueError('Original frozen cost weights changed')
        fits[f['context'],f['seed'],f['arm']]=f
    if len(fits)!=36:raise ValueError('All36 original cost endpoints required')
    previous={f:sha(parent.PUBLIC/f) for f in ('registration.json','training_freeze.json','complete.json','summary.json','verification.json','coherence_verification.json')}
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},parent_files=previous,
        contexts=json.loads(parent.REG.read_text())['contexts'],seeds=cfg['seeds'],frozen_cost_models=36,
        primary='projected_learned_joint_minus_frozen_unprojected_learned_joint_full_completion_lower',
        source_role='exposed_SOURCE_fitting_resubstitution_only',new_training=False,
        independent_roles_opened=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if not register:
        reg=json.loads(REG.read_text())
        if reg['parent_files']!=previous:raise ValueError('Complete parent result changed')
        for rel,digest in reg['bindings'].items():
            if sha(ROOT/rel)!=digest:raise ValueError('Frozen projection definition changed')
            parent.parent.source.frozen.BASE.engine.committed(ROOT/rel)
        parent.parent.source.frozen.BASE.engine.committed(REG)
    return cfg,values,fits


def runtime(cfg):
    api.torch.set_num_threads(4);api.torch.set_num_interop_threads(1);start=time.monotonic()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume the same projection definition')))
    def guard():
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hphase boundary;resume unchanged')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('10GiBdisk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['private_checkpoint_cap_bytes']:raise OSError('1GiBcheckpoint cap')
    def beat(**kw):
        guard();v=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
        write(PRIVATE/'heartbeat.json',v);print(json.dumps(v),flush=True)
    beat(state='runtime_ready',architecture=platform.machine(),threads=4,interop_threads=1,workers=0,new_training=False,resource_probing=False)
    return start,beat,guard


def frozen_controls(raw,saved,old):
    if set(raw)!=set(api.ARMS):raise ValueError('Only the two frozen native prediction arms are accepted')
    choices={};ranks={}
    for prefix,arm in zip(('locked','learned'),api.ARMS):
        if api.parent.digest(raw[arm])!=old['prediction_sha256'][arm] or saved['prediction_sha256'][arm]!=old['prediction_sha256'][arm]:
            raise ValueError('Original native cost predictions changed')
        name=prefix+'_joint';c=saved['choices'][name].copy()
        if api.parent.digest(c)!=old['action_sha256'][name]:raise ValueError('Original full control actions changed')
        at=api.np.flatnonzero(c>=0);rank=api.np.zeros(len(c));rank[at]=raw[arm][at,c[at],0]-raw[arm][at,c[at],1]
        api.np.testing.assert_array_equal(rank,saved['ranks'][name])
        choices['raw_'+name]=c;ranks['raw_'+name]=rank
    return choices,ranks


def inputs(values,fits,beat):
    for context,seed,data,ranges,packets,_,_,_ in parent.inputs(values,beat):
        states={a:api.parent.read_checkpoint(ROOT/fits[context,seed,a]['checkpoint']['path']) for a in api.ARMS}
        api.assert_matched(*states.values())
        for s in states.values():api.parent.exact(s['input_hashes'],data['input_hashes'])
        raw={a:api.predict(s,data['causal']) for a,s in states.items()}
        projected={a:projection.project(p,data['causal']['allowed']) for a,p in raw.items()}
        yield context,seed,data,ranges,packets,raw,projected


def pilot():
    cfg,values,fits=admission();start,beat,_=runtime(cfg)
    if (PUBLIC/'pilot.json').exists():raise ValueError('Completed resource probe immutable')
    gen=inputs(values,fits,beat);context,seed,_,ranges,packets,raw,projected=next(gen)
    name=sorted(packets)[0];packet=packets[name];key=f'{context}_{name}_seed{seed}_fit.json'
    old=aggregate_groups(parent.NAME)[key];saved=api.parent.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))
    controls,_=frozen_controls({a:v[ranges[name]] for a,v in raw.items()},saved,old)
    groups=decoder.base.query_groups(packet['recordings'],packet['frames'],packet['ids'])
    choose=sorted(set(range(min(12,len(groups))))|set(sorted(range(len(groups)),key=lambda i:(-len(groups[i]),i))[:6]))
    ix=api.np.sort(api.np.concatenate([groups[i] for i in choose]));p={a:v[ranges[name]][ix] for a,v in projected.items()}
    args=(p,packet['allowed'][ix],packet['recordings'][ix],packet['frames'][ix],packet['ids'][ix],controls['raw_learned_joint'][ix])
    checkpoints=[]
    class Stop(Exception):pass
    def interrupt(v):checkpoints.append(copy.deepcopy(v));raise Stop()
    try:decoder.allocate(*args,seconds=cfg['solver_seconds_per_query'],checkpoint=interrupt)
    except Stop:pass
    recovered=decoder.allocate(*args,seconds=cfg['solver_seconds_per_query'],resume_state=checkpoints[0])
    before=time.monotonic();direct=decoder.allocate(*args,seconds=cfg['solver_seconds_per_query']);seconds=time.monotonic()-before
    api.parent.exact(recovered,direct)
    once(PUBLIC/'pilot.json',dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        direct_decoder_seconds=seconds,queries=len(choose),rows=len(ix),maximum_query_agents=max(len(groups[i]) for i in choose),
        rough112596query_decoder_projection_seconds=seconds/len(choose)*112596,
        one_source_size_biased_estimate_not_completed_readout=True,actual_query_recovery_exact=True,
        new_policy_outcomes_scored=False,new_training=False,diagnostics=direct[2],registration_sha256=sha(REG)))
    beat(state='actual_projection_resource_probe_complete')


def evaluate(verify=False,resume=False):
    cfg,values,fits=admission();start,beat,_=runtime(cfg)
    if (PUBLIC/'complete.json').exists() and not verify:raise ValueError('Completed readout immutable')
    if not verify and not json.loads((PUBLIC/'pilot.json').read_text())['actual_query_recovery_exact']:raise ValueError('Actual recovery probe required')
    old_groups=aggregate_groups(parent.NAME);rows=[];refs=[];ids=set();records=set()
    for context,seed,data,ranges,packets,raw_full,projected_full in inputs(values,fits,beat):
        job=next(j for j in values[1][4] if j['key']==context.split('_controller')[0])
        for name,packet in sorted(packets.items()):
            key=f'{context}_{name}_seed{seed}_fit.json';path=PUBLIC/'groups'/key
            raw={a:v[ranges[name]] for a,v in raw_full.items()};projected={a:v[ranges[name]] for a,v in projected_full.items()}
            previous=old_groups[key];cache=api.parent.read_checkpoint(parent.PRIVATE/'progress'/(key+'.pt.gz'))
            controls,control_ranks=frozen_controls(raw,cache,previous)
            projected_hash={a:api.parent.digest(v) for a,v in projected.items()};progress=PRIVATE/'progress'/(key+'.pt.gz')
            if path.exists() and resume and not verify:
                row=json.loads(path.read_text())
                if row['registration_sha256']!=sha(REG) or row['projected_prediction_sha256']!=projected_hash:raise ValueError('Completed resumed group changed')
            else:
                if verify:
                    saved=api.parent.read_checkpoint(progress)
                    if saved['projected_prediction_sha256']!=projected_hash:raise ValueError('Projection inference changed')
                    choices,ranks,info=saved['choices'],saved['ranks'],saved['allocation']
                else:
                    partial=PRIVATE/'allocation_progress'/(key+'.pt.gz')
                    choices,ranks,info=decoder.allocate(projected,packet['allowed'],packet['recordings'],packet['frames'],packet['ids'],controls['raw_learned_joint'],
                        seconds=cfg['solver_seconds_per_query'],checkpoint=lambda v:api.parent.save_checkpoint(partial,v),
                        heartbeat=lambda **kw:beat(state='coherent_cost_allocation',context=context,source=name,seed=seed,**kw),
                        resume_state=api.parent.read_checkpoint(partial) if resume and partial.exists() else None)
                    choices.pop('parent_joint');ranks.pop('parent_joint');choices.update(controls);ranks.update(control_ranks)
                    api.parent.save_checkpoint(progress,dict(choices=choices,ranks=ranks,allocation=info,projected_prediction_sha256=projected_hash))
                for c in controls:api.np.testing.assert_array_equal(choices[c],controls[c]);api.np.testing.assert_array_equal(ranks[c],control_ranks[c])
                result=evaluation.readout(packet,choices,ranks,info,raw,projected,job['design']['easy_cut'],values[1][3]['origin'][packet['ids']],data['rms'])
                for prefix,arm in zip(('locked','learned'),api.ARMS):
                    if result['policies']['raw_'+prefix+'_joint']!=previous['result']['policies'][prefix+'_joint']:
                        raise ValueError('Original full control scores changed')
                    if result['cost_MSE']['raw_'+prefix]!=previous['result']['calibration_MSE'][arm]:raise ValueError('Original cost error changed')
                    for rec,v in result['recordings'].items():
                        if v['raw_'+prefix+'_joint']!=previous['result']['recordings'][rec][prefix+'_joint']:raise ValueError('Original record control changed')
                costs=dict(zip(evaluation.COSTS,(raw['locked_references'],raw['learned_references'],projected['locked_references'],projected['learned_references'])))
                row=dict(context=context,source=name,seed=seed,result=result,original_controls_exact=True,projection_contract_verified=True,
                    coherence={a:coherence(v,packet['allowed'],packet['scales']) for a,v in costs.items()},
                    ids_sha256=api.parent.digest(packet['ids']),raw_prediction_sha256=previous['prediction_sha256'],
                    projected_prediction_sha256=projected_hash,action_sha256={a:api.parent.digest(c) for a,c in choices.items()},
                    registration_sha256=sha(REG),parent_complete_sha256=sha(parent.PUBLIC/'complete.json'),new_training=False)
                if verify:
                    if row!=json.loads(path.read_text()):raise ValueError('Complete new inference/frozen-action scores differ')
                else:once(path,row)
            rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)));ids.update(map(int,packet['ids']));records.update((name,str(r)) for r in packet['recordings'])
            beat(state='projection_group_complete',groups=len(rows),total=72,context=context,source=name,seed=seed)
    if len(ids)!=149510 or len(records)!=74:raise ValueError('Full original target/record cohort required')
    summary=evaluation.summarize(rows,json.loads(REG.read_text())['contexts'],cfg);summary.update(unique_target_rows=len(ids),unique_recordings=len(records))
    phase_receipt=dict(pid=os.getpid(),seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()):raise ValueError('Canonical projection summary differs')
        once(PUBLIC/'verification.json',dict(**phase_receipt,all72_projection_inference_and_frozen_action_scores_exact=True,
            original_full_and_record_controls_exact=True,optimization_replayed=False,
            summary_sha256=sha(PUBLIC/'summary.json'),complete_sha256=sha(PUBLIC/'complete.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(PUBLIC/'complete.json',dict(**phase_receipt,groups=refs,summary_sha256=sha(PUBLIC/'summary.json')));pack(ROOT,PUBLIC)
    beat(state='all72_projection_verified' if verify else 'all72_projection_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','eval','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    for d in (PRIVATE,PRIVATE/'progress',PRIVATE/'allocation_progress'):d.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':admission(True)
        elif a.phase=='pilot':pilot()
        else:evaluate(a.phase=='verify',a.resume)
