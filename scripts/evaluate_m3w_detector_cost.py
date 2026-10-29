"""Frozen matched detector-cost readout and independent exact inference replay."""
import argparse
import fcntl
import json
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':raise RuntimeError('Native arm64 required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_m3w_detector_cost as run
from scripts.report_m3w_feature_cost_evidence import aggregate_groups
from src.evaluation import m3w_detector_cost_readout as api
from src.evaluation import m3w_completion_cost_readout as score
from src.evaluation import m3w_multiaction_cost_readout as base

REG=run.PUBLIC/'readout_registration.json'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_detector_cost_readout.py',
       ROOT/'tests/test_m3w_detector_cost_readout.py',run.PUBLIC/'protocol.md')


def admission(register=False):
    cfg,values,old,store=run.admission();learned=run.models()
    run.once(REG,dict(bindings={str(p.relative_to(ROOT)):run.sha(p) for p in FILES},
        training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
        store_sha256=store.complete_sha256,required_groups=288,primary='history_physical_neural',
        scope='original_fitting_only',held_scored=False,independent_roles_opened=False,deployment_changed=False))
    if not register:
        for rel,digest in json.loads(REG.read_text())['bindings'].items():
            if run.sha(ROOT/rel)!=digest:raise ValueError('Registered detector readout changed')
            run.parent.parent.parent.committed(ROOT/rel)
        run.parent.parent.parent.committed(REG)
    return cfg,values,old,store,learned


def execute(verify=False,resume=False):
    cfg,values,old,store,learned=admission();finish=run.PUBLIC/'complete.json'
    if finish.exists() and not verify:raise ValueError('Completed readout immutable; use verify')
    started,beat,guard=run.runtime(cfg);rows=[];refs=[]
    previous=aggregate_groups(run.parent.NAME)
    engine=run.parent.parent.parent
    _,origins=engine.old.direct.ORIGINS.models(values[2])
    for context,items in sorted(values[2]['contexts'].items()):
        packets=run.parent.packets(context,items,values,beat)
        original=run.parent.api.training_data(packets,run.parent.parent)
        prepared={arm:run.data_for(packets,store,arm,original) for arm in cfg['representations']}
        job=next(j for j in values[4] if j['key']==context.split('_controller')[0])
        for seed in cfg['seeds']:
            states={(arm,bank):run.parent.api.base.read_checkpoint(ROOT/learned[context,seed,bank,arm]['checkpoint']['path'])
                for arm in cfg['representations'] for bank in cfg['banks']}
            for (arm,bank),state in states.items():
                run.parent.api.base.exact(prepared[arm]['preprocess'],state['preprocess'])
                run.parent.api.base.exact(prepared[arm]['input_hashes'],state['input_hashes'])
            prior={bank:run.parent.api.base.read_checkpoint(ROOT/old[context,seed,bank,'known_only']['checkpoint']['path']) for bank in cfg['banks']}
            origin=engine.old.base.core.read_checkpoint(ROOT/origins[context,seed,'self']['checkpoint']['path'])
            originals=engine.old.direct.origin_predictions(origin,{s:p['args'] for s,p in packets.items()})
            for source,packet in sorted(packets.items()):
                guard();predictions={};choices={};ranks={}
                for arm in cfg['representations']:
                    extra=store.read(packet['ids'],arm)
                    for bank in cfg['banks']:
                        pred=run.api.predict(states[arm,bank],packet['x'],packet['scene'],extra,packet['envelope'],packet['scales'])
                        predictions[arm,bank]=pred
                        choices[arm,bank],ranks[arm,bank]=run.parent.api.base.select_actions(pred,packet['allowed'],arm=bank)
                    choices[arm,'neural_only'],ranks[arm,'neural_only']=run.parent.api.base.select_actions(
                        predictions[arm,'physical_neural'],packet['allowed'],arm='neural_only')
                oldrow=previous[f'{context}_{source}_seed{seed}_known_only_fit.json']
                for bank in cfg['banks']:
                    p=run.parent.api.predict(prior[bank],packet['x'],packet['scene'],packet['envelope'],packet['scales'])
                    np.testing.assert_array_equal(p,predictions['disabled',bank])
                    if engine.fingerprint(choices['disabled',bank])!=oldrow['action_sha256'][bank]:
                        raise ValueError('Disabled frozen known-only action hash changed')
                op=originals[source]
                take=engine.old.base.actions(op,engine.old.base.shared.eligible(packet['args'])) & packet['allowed'][:,6]
                for arm in cfg['representations']:
                    path=run.PUBLIC/'groups'/f'{context}_{source}_seed{seed}_{arm}_fit.json'
                    if path.exists() and not(verify or resume):raise ValueError('Explicit partial readout resume required')
                    current={bank:choices[arm,bank] for bank in (*cfg['banks'],'neural_only')}
                    order={bank:ranks[arm,bank] for bank in (*cfg['banks'],'neural_only')}
                    current['frozen_origin_binary'],order['frozen_origin_binary']=np.where(take,6,-1),op[:,0]-op[:,1]
                    current['floor'],order['floor']=np.full(len(take),-1,np.int64),np.zeros(len(take))
                    result=base.group_readout(packet,current,order,{bank:predictions[arm,bank] for bank in cfg['banks']})
                    if arm=='disabled':
                        for bank in cfg['banks']:
                            if result['policies'][bank]!=oldrow['result']['policies'][bank]:
                                raise ValueError('Exact original known-only scores changed')
                    result['representation_contrasts']={control:{bank:score.paired(packet,
                        choices[control,bank],choices[arm,bank],ranks[control,bank],ranks[arm,bank])
                        for bank in cfg['banks']} for control in cfg['representations']}
                    result['subsets']=score.subsets(packet,current,job['design']['easy_cut'])
                    result['neural_only_reference']='current_mixed_head_restricted_to_neural_action_not_separate_training'
                    value=dict(context=context,source=source,seed=seed,representation=arm,scope='fit',result=result,
                        action_sha256={n:engine.fingerprint(c) for n,c in current.items()},
                        prediction_sha256={b:engine.fingerprint(predictions[arm,b]) for b in cfg['banks']},
                        registration_sha256=run.sha(REG),training_freeze_sha256=run.sha(run.PUBLIC/'training_freeze.json'),
                        result_source='fresh_run_frozen_paired_detector_cost_inference',held_scored=False,independent_roles_opened=False)
                    if verify:
                        if value!=json.loads(path.read_text()):raise ValueError('Exact detector inference replay differs')
                    else:run.once(path,value)
                    rows.append(value);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=run.sha(path)))
            beat(state='detector_readout_progress',context=context,seed=seed,groups=len(rows),total=288)
        del packets,original,prepared
    summary=api.summarize(rows,values[2]['contexts'],values[1])
    if verify:
        if summary!=json.loads((run.PUBLIC/'summary.json').read_text()) or refs!=json.loads(finish.read_text())['groups']:
            raise ValueError('Complete detector summary replay differs')
        run.once(run.PUBLIC/'verification.json',dict(exact_replay=True,groups=288,summary_sha256=run.sha(run.PUBLIC/'summary.json')))
    else:
        run.once(run.PUBLIC/'summary.json',summary)
        run.once(finish,dict(pid=run.os.getpid(),seconds=time.monotonic()-started,peak_RSS_bytes=run.parent.rss_bytes(),
            groups=refs,summary_sha256=run.sha(run.PUBLIC/'summary.json'),held_scored=False,independent_roles_opened=False))
    run.write(run.PRIVATE/('verification_runtime.json' if verify else 'readout_runtime.json'),
        dict(pid=run.os.getpid(),seconds=time.monotonic()-started,peak_RSS_bytes=run.parent.rss_bytes()))
    beat(state='detector_readout_verified' if verify else 'detector_readout_complete',groups=288,
        fitting_advancement_gate=summary['fitting_advancement_gate'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='register':admission(register=True)
        else:execute(args.phase=='verify',args.resume)
