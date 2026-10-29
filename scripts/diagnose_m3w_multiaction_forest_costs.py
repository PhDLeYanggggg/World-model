"""Frozen fitting-only cross-estimator selected-region cost diagnosis, no tuning."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_multiaction_forest as run
from src.evaluation.m3w_policy_region_shift import compare_regions

METRICS=('harm_prediction_ratio','easy_harm_prediction_ratio',
    'reference_inflation_ratio','easy_reference_inflation_ratio')


def conditional_summary(records,estimator,pool):
    groups=[r['diagnosis'][estimator][pool] for r in records]
    return dict(required_record_views=len(groups),action_views=sum(g['action_views'] for g in groups),
        known_action_views=sum(g['known_action_views'] for g in groups),
        unknown_action_views=sum(g['unknown_action_views'] for g in groups),
        ratio={key:dict(median=statistics.median(v) if (v:=[g[key] for g in groups if g[key] is not None]) else None,
            finite_record_views=len(v),undefined_record_views=len(groups)-len(v),
            condition='strictly_positive_observed_denominator_not_full_roster_safety') for key in METRICS})


def execute(verify=False):
    bindings=(Path(__file__),run.PUBLIC/'mechanism_protocol.md',
        ROOT/'src/evaluation/m3w_policy_region_shift.py',ROOT/'src/evaluation/m3w_selected_cost_diagnosis.py',
        ROOT/'tests/test_m3w_forest_fitting_diagnosis.py')
    for path in bindings:run.parent.parent.committed(path)
    cfg,values=run.admission();models,neural=run.models(),run.parent.models()
    started,beat,guard=run.runtime(cfg);expected=run.aggregate_groups(run.NAME);records=[]
    for context,items in sorted(values[2]['contexts'].items()):
        packets=run.context_data(context,items,values,beat)['fit']
        for seed in cfg['seeds']:
            for arm,actions in run.parent.parent.api.ARMS.items():
                forest=run.joblib.load(ROOT/models[context,seed,arm]['checkpoint']['path'])
                nn=run.parent.parent.api.read_checkpoint(ROOT/neural[context,seed,arm]['checkpoint']['path'])
                for name,packet in sorted(packets.items()):
                    kw=packet['x'],packet['scene'],packet['envelope'],packet['scales']
                    p=run.api.predict(forest,*kw);q=run.parent.api.predict(nn,*kw)
                    fc,_=run.parent.parent.api.select_actions(p,packet['allowed'],arm=arm)
                    nc,_=run.parent.parent.api.select_actions(q,packet['allowed'],arm=arm)
                    prior=expected[f'{context}_{name}_seed{seed}_fit.json']
                    if (run.api.fingerprint(fc)!=prior['action_sha256'][arm]
                            or run.api.fingerprint(nc)!=prior['action_sha256'][arm+'_neural_parent']):
                        raise ValueError('Exact frozen fitting choices required')
                    for recording in sorted(set(packet['recordings'])):
                        at=packet['recordings']==recording
                        args=packet['y'][at],packet['allowed'][at]
                        a=compare_regions(p[at],*args,fc[at],nc[at],packet['scales'][at],actions)
                        b=compare_regions(q[at],*args,nc[at],fc[at],packet['scales'][at],actions)
                        for aa,bb in (('selected','frozen_teacher_region'),('frozen_teacher_region','selected')):
                            if a[aa]['observed_known_mass']!=b[bb]['observed_known_mass']:
                                raise ValueError('Cross-estimator regions must share identical observed labels')
                        records.append(dict(context=context,source=name,seed=seed,arm=arm,recording=str(recording),
                            diagnosis=dict(forest=a,neural=b)))
                beat(state='cost_mechanism_progress',context=context,seed=seed,arm=arm,record_views=len(records))
        del packets
    if len(records)!=1332:raise ValueError('All1332fitting record/head/bank views required')
    summary={}
    for arm in cfg['arms']:
        selected=[r for r in records if r['arm']==arm]
        summary[arm]={region:{estimator:conditional_summary(selected,estimator,pool)
            for estimator,pool in mappings.items()} for region,mappings in dict(
                all_legal=dict(forest='all_legal',neural='all_legal'),
                frozen_neural_selection=dict(forest='frozen_teacher_region',neural='selected'),
                frozen_forest_selection=dict(forest='selected',neural='frozen_teacher_region')).items()}
    value=dict(result_source='fresh_frozen_fitting_only_cross_estimator_cost_diagnosis',records=records,summary=summary,
        optimizer_updates=0,independent_roles_opened=False,deployment_changed=False,
        primary_summary_sha256=run.sha(run.PUBLIC/'summary.json'),diagnostic_code_sha256=run.sha(Path(__file__)),
        diagnostic_bindings={str(p.relative_to(ROOT)):run.sha(p) for p in bindings},
        protocol_sha256=run.sha(run.PUBLIC/'mechanism_protocol.md'))
    content=json.dumps(value,indent=2).encode()+b'\n';path=run.PUBLIC/'fitting_cost_diagnosis.json.gz'
    if verify:
        if gzip.decompress(path.read_bytes())!=content:raise ValueError('Exact cross-estimator fitting diagnosis differs')
        run.once(run.PUBLIC/'fitting_cost_verification.json',dict(exact_replay=True,records=1332,archive_sha256=run.sha(path)))
    elif path.exists():raise ValueError('Completed fitting diagnosis immutable')
    else:
        path.write_bytes(gzip.compress(content,mtime=0));run.once(run.PUBLIC/'fitting_cost_summary.json',summary)
    run.json_write(run.PRIVATE/('cost_mechanism_verify_runtime.json' if verify else 'cost_mechanism_runtime.json'),
        dict(pid=os.getpid(),seconds=time.monotonic()-started))
    beat(state='cost_mechanism_verified' if verify else 'cost_mechanism_complete',record_views=1332)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(args.verify)
