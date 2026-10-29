"""Fixed fitting-only bank controls; no new training, threshold or held-role use."""
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
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts import run_m3w_multiaction_forest as parent
from src.evaluation import m3w_cost_bank_factorization as api
from src.evaluation import m3w_multiaction_cost_readout as readout
from src.evaluation.m3w_multiaction_forest_readout import subsets,pooled_subsets

NAME='european_cost_bank_factorization_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
CONFIG=ROOT/'configs/m3w_cost_bank_factorization_v1.json';REG=PUBLIC/'registration.json'
sha,once,json_write=parent.sha,parent.once,parent.json_write
FILES=(CONFIG,Path(__file__),ROOT/'src/evaluation/m3w_cost_bank_factorization.py',
    ROOT/'tests/test_m3w_cost_bank_factorization.py',PUBLIC/'protocol.md',PUBLIC/'prior_controls.md')
CONTRASTS=('factorized','mixed_neural_restricted','suppressed_neural_expert','competition_lost')


def load():
    cfg=json.loads(CONFIG.read_text());pc,values=parent.admission()
    if cfg['seeds']!=pc['seeds'] or cfg['risk_budget']!=.02 or cfg['families']!=['forest','neural']:
        raise ValueError('Frozen comparison and unchanged risk budget required')
    once(REG,dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in FILES},
        parent_registration_sha256=sha(parent.REG),parent_training_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        parent_summary_sha256=sha(parent.PUBLIC/'summary.json'),parent_verification_sha256=sha(parent.PUBLIC/'verification.json'),
        fitting_cost_verification_sha256=sha(parent.PUBLIC/'fitting_cost_verification.json'),
        static_neural_freeze_sha256=sha(parent.parent.PUBLIC/'training_freeze.json'),
        contexts=sorted(values[2]['contexts']),seeds=cfg['seeds'],families=cfg['families'],groups=144,
        scope='original_fitting_only',held_scored=False,new_optimizer_updates=0,new_trees=0,
        comparison='same_mixed_estimator_restriction_vs_distinct_frozen_expert_composition',
        composition_cost='two_estimators_not_parameter_or_compute_matched_to_one',
        independent_roles_opened=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    return cfg,values


def admission():
    cfg,values=load()
    for rel,digest in json.loads(REG.read_text())['bindings'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Registered fitting-only control changed: '+rel)
        parent.parent.parent.committed(ROOT/rel)
    parent.parent.parent.committed(REG);return cfg,values


def runtime(cfg):
    torch.set_num_threads(4);torch.set_num_interop_threads(1);started=time.monotonic()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume completed immutable groups')))
    def guard():
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Preserve10GiB reserve')
        if time.monotonic()-started>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h boundary; resume same roster')
    def beat(**event):
        guard();value=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**event)
        json_write(PRIVATE/'heartbeat.json',value)
        with (PRIVATE/'events.jsonl').open('a') as stream:stream.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    guard();return started,beat,guard


def summarize(rows,pc):
    output={}
    for family in ('forest','neural'):
        selected=[r for r in rows if r['family']==family]
        if len(selected)!=72 or len({(r['context'],r['source'],r['seed']) for r in selected})!=72:
            raise ValueError('All72fitting source/context/seed views required per family')
        names=list(selected[0]['result']['policies']);safety={}
        for name in names:
            ps=[r['result']['policies'][name]['policy'] for r in selected]
            records=[record[name] for r in selected for record in r['result']['recordings'].values()]
            safety[name]=dict(required_views=len(ps),supported_views=sum(p['budget_status']['supported'] for p in ps),
                required_record_views=len(records),supported_record_views=sum(p['budget_status']['supported'] for p in records),
                undefined_easy_record_views=sum(p['easy_selected_risk_upper'] is None for p in records),
                known_all_risk_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk']>.02+1e-12 for p in records),
                known_easy_risk_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk']>.02+1e-12 for p in records),
                selected=sum(p['selected_count'] for p in ps),selected_unknown=sum(p['selected_unknown'] for p in ps))
        getter=lambda r:r['bank_diagnostics']
        gate={}
        for estimator in ('mixed','dedicated'):
            gate[estimator]={key:sum(getter(r['result'])[estimator][key] for r in selected)
                for key in ('queries','eligible','unavailable','passing')}
            for key in ('overlapping_failures','exclusive_failures','failure_bit_patterns'):
                keys=sorted({k for r in selected for k in getter(r['result'])[estimator][key]})
                gate[estimator][key]={k:sum(getter(r['result'])[estimator][key].get(k,0) for r in selected) for k in keys}
        for key in ('neural_expert_pass_mixed_gate_reject','mixed_gate_pass_physical_competition_lost',
                    'mixed_gate_pass_and_neural_selected','mixed_gate_pass_neural_expert_reject'):
            gate[key]=sum(getter(r['result'])[key] for r in selected)
        gate['suppression_components']={key:sum(getter(r['result'])['neural_expert_pass_mixed_reject_components'][key]
            for r in selected) for key in api.FAILURES}
        output[family]=dict(groups=72,unique_fitting_recordings=len({(r['source'],key)
            for r in selected for key in r['result']['recordings']}),safety=safety,gate6=gate,
            accuracy={name:readout.interval(selected,lambda r,n=name:r['policies'][n]['known_ADE_gain_floor_percent'],pc) for name in names},
            FDE={name:readout.interval(selected,lambda r,n=name:r['policies'][n]['known_FDE_gain_floor_percent'],pc) for name in names},
            contrasts={name:{mode+'_'+bound:readout.interval(selected,
                lambda r,n=name,m=mode,b=bound:r['bank_contrasts'][n][m][b],pc)
                for mode in ('full','matched') for bound in ('known_pp','lower_pp','upper_pp')} for name in CONTRASTS},
            subsets=pooled_subsets(selected))
    return dict(result_source='fresh_frozen_fitting_bank_factorization_not_new_training',families=output,
        optimizer_updates=0,new_trees=0,groups=144,scope='fitting_resubstitution_only',held_scored=False,
        independent_confirmation=False,whole_pipeline_OOF=False,independent_roles_opened=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False)


def execute(verify=False,resume=False):
    cfg,values=admission();finish=PUBLIC/'complete.json'
    if finish.exists() and not verify:raise ValueError('Completed fitting diagnosis immutable; use verify')
    learned,neural=parent.models(),parent.parent.models()
    started,beat,guard=runtime(cfg);expected=parent.aggregate_groups(parent.NAME)
    expected_hash={Path(r['path']).name:r['sha256'] for r in json.loads((parent.PUBLIC/'complete.json').read_text())['groups']}
    _,origin_refs=parent.parent.parent.old.direct.ORIGINS.models(values[2]);rows,refs=[],[]
    for context,items in sorted(values[2]['contexts'].items()):
        scopes=parent.context_data(context,items,values,beat,held=False)
        if set(scopes)!={'fit'}:raise ValueError('Fitting-only scope required')
        packets=scopes['fit'];job=next(j for j in values[4] if j['key']==context.split('_controller')[0])
        for seed in cfg['seeds']:
            states=dict(forest={a:parent.joblib.load(ROOT/learned[context,seed,a]['checkpoint']['path']) for a in parent.parent.parent.api.ARMS},
                neural={a:parent.parent.parent.api.read_checkpoint(ROOT/neural[context,seed,a]['checkpoint']['path']) for a in parent.parent.parent.api.ARMS})
            pr=states['forest']['physical_only']['preprocess']
            for bank in states.values():
                for state in bank.values():parent.parent.parent.api.exact(pr,state['preprocess'])
            origin=parent.parent.parent.old.base.core.read_checkpoint(ROOT/origin_refs[context,seed,'self']['checkpoint']['path'])
            original=parent.parent.parent.old.direct.origin_predictions(origin,{s:p['args'] for s,p in packets.items()})
            for name,packet in sorted(packets.items()):
                oldkey=f'{context}_{name}_seed{seed}_fit.json';old=expected[oldkey]
                for family,bank in states.items():
                    path=PUBLIC/'groups'/f'{context}_{name}_seed{seed}_{family}_fit.json'
                    if path.exists() and not (verify or resume):raise ValueError('Explicit group resume required')
                    kw=packet['x'],packet['scene'],packet['envelope'],packet['scales']
                    predict=parent.api.predict if family=='forest' else parent.parent.api.predict
                    prediction={a:predict(state,*kw) for a,state in bank.items()}
                    choices,ranks,predicted,diagnosis=api.policies(prediction,packet['allowed'])
                    for arm in parent.parent.parent.api.ARMS:
                        oldname=arm if family=='forest' else arm+'_neural_parent'
                        if parent.api.fingerprint(choices[arm])!=old['action_sha256'][oldname]:
                            raise ValueError('Frozen original fitting choices changed')
                    op=original[name]
                    take=parent.parent.parent.old.base.actions(op,parent.parent.parent.old.base.shared.eligible(packet['args'])) & packet['allowed'][:,6]
                    choices['frozen_origin_binary']=np.where(take,6,-1);ranks['frozen_origin_binary']=op[:,0]-op[:,1]
                    result=readout.group_readout(packet,choices,ranks,predicted)
                    for arm in parent.parent.parent.api.ARMS:
                        oldname=arm if family=='forest' else arm+'_neural_parent'
                        if result['policies'][arm]!=old['result']['policies'][oldname]:raise ValueError('Original fitting metrics changed')
                    result['bank_diagnostics']=diagnosis;result['bank_contrasts']={}
                    for control in CONTRASTS:
                        left,right=choices['physical_neural'],choices[control]
                        aa,bb=readout.matched_choices(left,right,ranks['physical_neural'],ranks[control],
                            packet['recordings'],packet['frames'],packet['ids'])
                        result['bank_contrasts'][control]=dict(full=readout.paired_bounds(packet,left,right),
                            matched=readout.paired_bounds(packet,aa,bb),matched_count=int((aa>=0).sum()),
                            matched_role='utility_only_not_safety')
                    result['subsets']=subsets(packet,choices,job['design']['easy_cut'])
                    value=dict(context=context,source=name,seed=seed,family=family,scope='fit',result=result,
                        action_sha256={k:parent.api.fingerprint(v) for k,v in choices.items()},
                        prediction_sha256={k:parent.api.fingerprint(v) for k,v in predicted.items()},
                        parent_group_sha256=expected_hash[oldkey],registration_sha256=sha(REG),
                        result_source='fresh_fitting_only_inference_cached_verified_frozen_estimators',held_scored=False)
                    if verify:
                        if value!=json.loads(path.read_text()):raise ValueError('Exact fitting bank control differs')
                    else:once(path,value)
                    rows.append(value);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
            beat(state='bank_control_progress',context=context,seed=seed,groups=len(rows),total=144)
        del scopes,packets
    summary=summarize(rows,values[1])
    if verify:
        if summary!=json.loads((PUBLIC/'summary.json').read_text()) or refs!=json.loads(finish.read_text())['groups']:
            raise ValueError('Exact bank control summary differs')
        once(PUBLIC/'verification.json',dict(exact_replay=True,groups=144,summary_sha256=sha(PUBLIC/'summary.json')))
    else:
        once(PUBLIC/'summary.json',summary)
        once(finish,dict(pid=os.getpid(),seconds=time.monotonic()-started,groups=refs,summary_sha256=sha(PUBLIC/'summary.json'),
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,held_scored=False,
            new_optimizer_updates=0,new_trees=0,independent_roles_opened=False))
    json_write(PRIVATE/('verification_runtime.json' if verify else 'runtime.json'),dict(pid=os.getpid(),
        seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    beat(state='bank_control_verified' if verify else 'bank_control_complete',groups=144,held_scored=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':load()
        else:execute(args.phase=='verify',args.resume)
