"""Complete-region training repair; frozen causal inference and SOURCE protocol."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_policy_region_cost as parent
from src.world_model import m3w_balanced_region_cost as api

NAME='european_balanced_region_cost_v1'
PUBLIC=parent.PUBLIC.parent/NAME;PRIVATE=parent.PRIVATE.parent/NAME
CONFIG=ROOT/'configs/m3w_balanced_region_cost_v1.json';REG=PUBLIC/'registration.json'
DIAG=parent.PUBLIC.parent/'european_region_gradient_v1'
sha,once,json_write,fingerprint=parent.sha,parent.once,parent.json_write,parent.fingerprint
FILES=(CONFIG,Path(__file__),ROOT/'src/world_model/m3w_balanced_region_cost.py',
    ROOT/'src/world_model/m3w_exact_region_cost.py',ROOT/'tests/test_m3w_balanced_region_cost.py',
    ROOT/'tests/test_m3w_exact_region_cost.py',PUBLIC/'protocol.md',DIAG/'summary.json',DIAG/'complete.json',DIAG/'verification.json')


def load():
    values=parent.admission();cfg=json.loads(CONFIG.read_text());oldcfg=values[0]
    diagnostic=json.loads((DIAG/'verification.json').read_text())
    if (cfg['training']!=oldcfg['training'] or cfg['seeds']!=oldcfg['seeds'] or cfg['region_weight']!=1
            or cfg['risk_budget']!=.02 or not diagnostic['exact_replay'] or diagnostic['states']!=54
            or diagnostic['summary_sha256']!=sha(DIAG/'summary.json')):
        raise ValueError('Matched training and verified54-state gradient evidence required')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        parent_verification_sha256=sha(parent.PUBLIC/'verification.json'),
        fits=54,steps_per_fit=2000,parameters=26309,region_weight=1.,region_chunk_size=512,
        change=cfg['change'],same_base_initialization_and_draws=True,extra_auxiliary_compute_reported=True,
        risk_budget=.02,independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    return cfg,values[1:],parent.models()


def admission():
    values=load();reg=json.loads(REG.read_text())
    for rel,digest in reg['bindings'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Registered complete-region experiment changed')
        parent.parent.committed(ROOT/rel)
    parent.parent.committed(REG);return values


def runtime(cfg):
    parent.torch.set_num_threads(4);parent.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiBdisk reserve')
        if sum(p.stat().st_size for p in PRIVATE.rglob('*.pt.gz'))>cfg['checkpoint_cap_bytes']:
            raise OSError('Private checkpoint cap exceeded')
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12hboundary;resume same roster')
    def beat(**event):
        guard();value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        json_write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume checkpoint')))
    guard();return started,beat,guard


def prepare(values,context,items,beat,include_held=False):
    pc,reg,source,jobs,identity,teachers,packed,polygons=values
    return parent.old.scene_packets(parent.parent.prepare_context(context,items,source,jobs,identity,beat,include_held=include_held),
        source,packed,polygons)


def neutral(resume=False):
    cfg,values,controls=admission();started,beat,guard=runtime(cfg)
    if (PUBLIC/'neutral_replay.json').exists():raise ValueError('Completed neutral immutable')
    context,items=sorted(values[1]['contexts'].items())[0]
    packets=prepare(values,context,items,beat)['fit'];data=parent.old.training_data(packets)
    regions=parent.region_data(packets,values[5][context,17,'physical_neural']['checkpoint'],'physical_neural')
    state=api.fit(data,arm='physical_neural',settings=cfg['training'],seed=17,
        identity=dict(registration_sha256=sha(REG)),path=PRIVATE/'neutral'/'checkpoint.pt.gz',regions=regions,
        region_weight=0,chunk_size=cfg['region_chunk_size'],heartbeat=beat,guard=guard,resume=resume)
    reference=parent.parent.api.read_checkpoint(ROOT/values[5][context,17,'physical_neural']['checkpoint']['path'])
    parent.api.assert_neutral(reference,state)
    once(PUBLIC/'neutral_replay.json',dict(pid=os.getpid(),seconds=time.monotonic()-started,updates=2000,
        all_legacy_scientific_state_except_identity_runtime_exact=True,region_weight=0,registration_sha256=sha(REG)))
    beat(state='neutral2000_exact')


def train(pilot=False,resume=False):
    cfg,values,controls=admission();finish=PUBLIC/('pilot.json' if pilot else 'training_freeze.json')
    if finish.exists():raise ValueError('Completed phase immutable')
    if not pilot and (not json.loads((PUBLIC/'pilot.json').read_text())['engineering_pass']
            or not json.loads((PUBLIC/'neutral_replay.json').read_text())['all_legacy_scientific_state_except_identity_runtime_exact']):
        raise ValueError('Real recovery and full disabled neutral required')
    started,beat,guard=runtime(cfg);refs=[];times=[];auxiliary_rows=0
    contexts=sorted(values[1]['contexts'].items())[:1] if pilot else sorted(values[1]['contexts'].items())
    for context,items in contexts:
        packets=prepare(values,context,items,beat)['fit'];data=parent.old.training_data(packets)
        for seed in cfg['seeds'][:1] if pilot else cfg['seeds']:
            for arm in parent.parent.api.ARMS:
                regions=parent.region_data(packets,values[5][context,seed,arm]['checkpoint'],arm)
                path=PRIVATE/('pilot' if pilot else 'heads')/f'{context}_seed{seed}_{arm}.pt.gz'
                kw=dict(data=data,arm=arm,settings=cfg['training'],seed=seed,
                    identity=dict(context=context,registration_sha256=sha(REG)),path=path,regions=regions,
                    region_weight=1.,chunk_size=cfg['region_chunk_size'],heartbeat=beat,guard=guard)
                if pilot and not path.exists():api.fit(**kw,stop_at=cfg['pilot_resume_at'])
                state=api.fit(**kw,stop_at=cfg['pilot_steps'] if pilot else None,resume=path.exists() and (pilot or resume))
                if pilot:
                    direct=path.with_name(path.stem+'_direct.pt.gz')
                    reference=api.fit(**{**kw,'path':direct},stop_at=cfg['pilot_steps'],resume=direct.exists() and resume)
                    for key in state:
                        if key!='seconds':parent.parent.api.exact(state[key],reference[key])
                    times.append(state['seconds'])
                else:
                    reference=parent.parent.api.read_checkpoint(ROOT/controls[context,seed,arm]['checkpoint']['path'])
                    for key in ('initial_model','preprocess','draw_hash','sampler_rng','torch_rng','parameters','input_hashes','region_arrays'):
                        parent.parent.api.exact(state[key],reference[key])
                    receipt=PUBLIC/'fits'/f'{context}_seed{seed}_{arm}.json'
                    once(receipt,dict(context=context,seed=seed,arm=arm,step=state['step'],parameters=state['parameters'],
                        checkpoint=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                        seconds=state['seconds'],first_trace=state['trace'][0],final_trace=state['trace'][-1],
                        auxiliary_teacher_rows=state['auxiliary_teacher_rows'],auxiliary_forward_rows=state['auxiliary_forward_rows'],
                        base_row_draws=state['row_draws'],draw_hash=state['draw_hash'],
                        initialization_preprocessing_base_draws_and_teacher_regions_exact=True,
                        region_order=state['exact_region_order'],registration_sha256=sha(REG),result_source='fresh_native_torch_complete_region_learning'))
                    refs.append(dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)))
                    auxiliary_rows+=state['auxiliary_teacher_rows']
                    beat(state='fit_complete',fits=len(refs),total=54,context=context,seed=seed,arm=arm)
        del packets,data
    if pilot:
        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-started,engineering_pass=True,
            actual_updates=120,exact_resume_all_arms=True,projected_primary_optimizer_seconds=sum(times)*100*18,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,held_scored=False))
    else:
        if len(refs)!=54:raise ValueError('All54endpoints required')
        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-started,fits=refs,count=54,optimizer_updates=108000,
            registration_sha256=sha(REG),inner_held_scored=False,auxiliary_teacher_rows=auxiliary_rows,
            auxiliary_forward_rows=2*auxiliary_rows,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            independent_roles_opened=False))
    beat(state='pilot_complete' if pilot else 'training_complete')


def models():
    parent.parent.committed(PUBLIC/'training_freeze.json');frozen=json.loads((PUBLIC/'training_freeze.json').read_text())
    if frozen['count']!=54 or frozen['inner_held_scored'] or frozen['registration_sha256']!=sha(REG):
        raise ValueError('Committed54endpoint freeze before readout required')
    result={}
    for ref in frozen['fits']:
        row=json.loads((ROOT/ref['path']).read_text());key=row['context'],row['seed'],row['arm']
        if sha(ROOT/ref['path'])!=ref['sha256'] or sha(ROOT/row['checkpoint']['path'])!=row['checkpoint']['sha256'] or key in result:
            raise ValueError('Unique unchanged completed model receipt required')
        result[key]=row
    return result


def evaluate(verify=False,resume=False):
    cfg,values,controls=admission();learned=models();finish=PUBLIC/'complete.json'
    if finish.exists() and not verify:raise ValueError('Completed readout immutable')
    started,beat,guard=runtime(cfg);rows=[];refs=[];pc=values[0];reg=values[1]
    _,origin_refs=parent.parent.old.direct.ORIGINS.models(reg)
    expected=parent.aggregates.aggregate_groups(parent.NAME)
    for context,items in sorted(reg['contexts'].items()):
        scopes=prepare(values,context,items,beat,True)
        for seed in cfg['seeds']:
            states={a:parent.parent.api.read_checkpoint(ROOT/learned[context,seed,a]['checkpoint']['path']) for a in parent.parent.api.ARMS}
            prior={a:parent.parent.api.read_checkpoint(ROOT/controls[context,seed,a]['checkpoint']['path']) for a in parent.parent.api.ARMS}
            origin=parent.parent.old.base.core.read_checkpoint(ROOT/origin_refs[context,seed,'self']['checkpoint']['path'])
            for scope,packets in scopes.items():
                original=parent.parent.old.direct.origin_predictions(origin,{s:p['args'] for s,p in packets.items()})
                for name,packet in sorted(packets.items()):
                    path=PUBLIC/'groups'/f'{context}_{name}_seed{seed}_{scope}.json'
                    if path.exists() and not (verify or resume):raise ValueError('Explicit partial readout resume required')
                    old=expected[path.name];choices={};ranks={};predicted={}
                    for arm in parent.parent.api.ARMS:
                        p=parent.old.api.predict(states[arm],packet['x'],packet['scene'],packet['envelope'],packet['scales']);predicted[arm]=p
                        for multiplier in (1.,*pc['sensitivity_harm_multipliers']):
                            key=arm if multiplier==1 else arm+'_harm'+str(int(multiplier))
                            choices[key],ranks[key]=parent.parent.api.select_actions(p,packet['allowed'],arm=arm,harm_multiplier=multiplier)
                        p=parent.old.api.predict(prior[arm],packet['x'],packet['scene'],packet['envelope'],packet['scales'])
                        key=arm+'_stochastic_parent';choices[key],ranks[key]=parent.parent.api.select_actions(p,packet['allowed'],arm=arm)
                        if fingerprint(choices[key])!=old['action_sha256'][arm]:raise ValueError('Frozen stochastic comparator differs')
                    op=original[name];take=parent.parent.old.base.actions(op,parent.parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:,6]
                    choices['frozen_origin_binary']=parent.np.where(take,6,-1);ranks['frozen_origin_binary']=op[:,0]-op[:,1]
                    choices['floor']=parent.np.full(len(take),-1,parent.np.int64);ranks['floor']=parent.np.zeros(len(take))
                    result=parent.readout.group_readout(packet,choices,ranks,predicted)
                    for arm in parent.parent.api.ARMS:
                        if result['policies'][arm+'_stochastic_parent']!=old['result']['policies'][arm]:
                            raise ValueError('Frozen stochastic policy metric replay differs')
                    a,b=choices['physical_neural_stochastic_parent'],choices['physical_neural']
                    aa,bb=parent.readout.matched_choices(a,b,ranks['physical_neural_stochastic_parent'],ranks['physical_neural'],packet['recordings'],packet['frames'],packet['ids'])
                    result['exact_minus_stochastic']=dict(full=parent.readout.paired_bounds(packet,a,b),matched=parent.readout.paired_bounds(packet,aa,bb),
                        matched_count=int((aa>=0).sum()),matched_role='utility_only_not_safety')
                    value=dict(context=context,source=name,seed=seed,scope=scope,result=result,
                        action_sha256={a:fingerprint(c) for a,c in choices.items()},registration_sha256=sha(REG),
                        training_freeze_sha256=sha(PUBLIC/'training_freeze.json'),result_source='fresh_SOURCE_complete_region_readout',independent_roles_opened=False)
                    if verify:
                        if value!=json.loads(path.read_text()):raise ValueError('Exact complete-region readout differs')
                    else:once(path,value)
                    rows.append(value);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            beat(state='readout_progress',context=context,views=len(rows),total=144)
        del scopes
    summary=parent.readout.summarize(rows,pc)
    for scope in summary['scopes']:
        selected=[r for r in rows if r['scope']==scope]
        summary['scopes'][scope]['exact_minus_stochastic']={m+'_'+b:parent.readout.interval(selected,
            lambda r,m=m,b=b:r['exact_minus_stochastic'][m][b],pc) for m in ('full','matched') for b in ('known_pp','lower_pp','upper_pp')}
    summary.update(complete_region_auxiliary=True,same_base_draws_extra_compute=True)
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()) or refs!=json.loads(finish.read_text())['groups']:
            raise ValueError('Exact complete-region summary differs')
        once(PUBLIC/'verification.json',dict(exact_replay=True,groups=144,summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json',summary);once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),
            registration_sha256=sha(REG),independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),dict(pid=os.getpid(),seconds=time.monotonic()-started))
    beat(state='readout_verified' if verify else 'readout_complete',full_budget=summary['primary_full_roster_budget_supported'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('register','pilot','neutral','train','evaluate','verify'));p.add_argument('--resume',action='store_true');a=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.phase=='register':load()
        elif a.phase=='neutral':neutral(a.resume)
        elif a.phase in ('pilot','train'):train(a.phase=='pilot',a.resume)
        else:evaluate(a.phase=='verify',a.resume)
