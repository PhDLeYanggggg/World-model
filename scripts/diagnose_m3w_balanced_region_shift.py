"""Prespecified fitting-region check after complete-region training, no tuning."""
import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_balanced_region_cost as run
from scripts.report_m3w_selected_cost_audit import median
from src.evaluation.m3w_policy_region_shift import compare_regions


def execute(verify=False):
    cfg,values,controls=run.admission();models=run.models();started,beat,guard=run.runtime(cfg)
    expected=run.parent.aggregates.aggregate_groups(run.NAME)
    teacher_groups=run.parent.aggregates.aggregate_groups(run.parent.old.NAME)
    records=[]
    for context,items in sorted(values[1]['contexts'].items()):
        packets=run.prepare(values,context,items,beat)['fit']
        for seed in cfg['seeds']:
            for arm,actions in run.parent.parent.api.ARMS.items():
                current=run.parent.parent.api.read_checkpoint(ROOT/models[context,seed,arm]['checkpoint']['path'])
                teacher=run.parent.parent.api.read_checkpoint(ROOT/values[5][context,seed,arm]['checkpoint']['path'])
                for name,packet in sorted(packets.items()):
                    kw=(packet['x'],packet['scene'],packet['envelope'],packet['scales'])
                    p=run.parent.old.api.predict(current,*kw);q=run.parent.old.api.predict(teacher,*kw)
                    choice,_=run.parent.parent.api.select_actions(p,packet['allowed'],arm=arm)
                    old_choice,_=run.parent.parent.api.select_actions(q,packet['allowed'],arm=arm)
                    key=f'{context}_{name}_seed{seed}_fit.json'
                    if (run.fingerprint(choice)!=expected[key]['action_sha256'][arm]
                            or run.fingerprint(old_choice)!=teacher_groups[key]['action_sha256'][arm]):
                        raise ValueError('Exact frozen current and original-teacher fitting choices required')
                    for recording in sorted(set(packet['recordings'])):
                        at=packet['recordings']==recording
                        result=compare_regions(p[at],packet['y'][at],packet['allowed'][at],choice[at],old_choice[at],packet['scales'][at],actions)
                        records.append(dict(context=context,source=name,seed=seed,arm=arm,recording=str(recording),diagnosis=result))
                beat(state='fitting_shift_progress',context=context,seed=seed,arm=arm,record_views=len(records))
    if len(records)!=1332:raise ValueError('All1332fitting record/head/bank views required')
    summary={}
    for arm in run.parent.parent.api.ARMS:
        rows=[r for r in records if r['arm']==arm]
        summary[arm]=dict(new_interventions=sum(r['diagnosis']['selected']['action_views'] for r in rows),
            outside_teacher_region=sum(r['diagnosis']['new_interventions_outside_teacher_region'] for r in rows),
            teacher_region_vs_actual_selection={pool:{key:median(rows,pool,key) for key in
                ('harm_prediction_ratio','easy_harm_prediction_ratio','reference_inflation_ratio','easy_reference_inflation_ratio')}
                for pool in ('frozen_teacher_region','selected')})
    value=dict(result_source='fresh_frozen_TRAIN_complete_region_mechanism_diagnosis',records=records,summary=summary,
        primary_summary_sha256=run.sha(run.PUBLIC/'summary.json'),diagnostic_code_sha256=run.sha(Path(__file__)),
        optimizer_updates=0,independent_roles_opened=False,deployment_changed=False)
    content=json.dumps(value,indent=2).encode()+b'\n';path=run.PUBLIC/'fitting_region_shift.json.gz'
    if verify:
        if gzip.decompress(path.read_bytes())!=content:raise ValueError('Exact complete-region fitting diagnosis differs')
        run.once(run.PUBLIC/'fitting_region_shift_verification.json',dict(exact_replay=True,archive_sha256=run.sha(path),records=1332))
    elif path.exists():raise ValueError('Completed fitting diagnosis immutable')
    else:
        path.write_bytes(gzip.compress(content,mtime=0));run.once(run.PUBLIC/'fitting_region_shift_summary.json',summary)
    run.json_write(run.PRIVATE/('fitting_shift_verify_runtime.json' if verify else 'fitting_shift_runtime.json'),
        dict(pid=os.getpid(),seconds=time.monotonic()-started))
    beat(state='fitting_shift_verified' if verify else 'fitting_shift_complete')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',action='store_true');args=p.parse_args()
    run.PRIVATE.mkdir(parents=True,exist_ok=True)
    with (run.PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);execute(args.verify)
