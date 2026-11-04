"""Fitting-only risk reliability before the amplitude study's held readout."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_amplitude_cost as experiment
from src.evaluation.m3w_selected_cost_audit import selected_costs
from src.world_model.m3w_multiaction_cost import select_actions

source,api=experiment.source,experiment.api
OUT=experiment.PUBLIC/'fitting_check'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('evaluate','verify'))
    args=parser.parse_args()
    if args.phase=='evaluate'and(experiment.PUBLIC/'summary.json').exists():
        raise ValueError('Initial fitting check must precede new held policy readout')
    source.PRIVATE=experiment.PRIVATE
    with(experiment.PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg=source.read(experiment.CONFIG)
        start,guard,beat=source.runtime(cfg,'preheld_amplitude_fitting_check')
        data,jobs,refs=experiment.register(cfg,beat)
        freeze=source.read(experiment.PUBLIC/'training_freeze.json')
        if freeze['complete_heads']!=36 or not freeze['all18pairs_exact']:
            raise ValueError('Complete all36matched fits first')
        models={}
        for ref in freeze['fits']:
            receipt=source.read(source.checked(ref))
            models[receipt['job']['key'],receipt['arm']]=receipt['checkpoint']
        if set(models)!={(j['key'],a)for j in jobs for a in api.ARMS}:
            raise ValueError('All18jobs require both frozen scientific endpoints')
        records=[]
        for job in jobs:
            packet=experiment.view(data,job,refs,training=True)
            packet.pop('learning',None)
            c,ids=packet['causal'],packet['ids']
            prediction={};choices={}
            for arm in api.ARMS:
                state=api.base.base.read_checkpoint(source.checked(models[job['key'],arm]))
                prediction[arm]=api.predict(state,c)
                allowed=api.for_arm(dict(causal=c),arm)['causal']['allowed']
                choices[arm+'_independent'],_=select_actions(prediction[arm],allowed,arm='physical_neural')
                del state
            binary=api.for_arm(dict(causal=c),'binary_only')['causal']['allowed']
            choices['amplitude_bank_binary_independent'],_=select_actions(prediction['amplitude_bank'],binary,arm='physical_neural')
            choices['all_legal_full_amplitude']=source.np.where(c['allowed'][:,6],6,-1).astype(source.np.int64)
            # Diagnostic targets follow all causal choices, and never become inputs.
            truth=api.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
            for arm,pred in prediction.items():
                for region,choice in choices.items():
                    records.append(dict(job=job['key'],arm=arm,region=region,
                        scores=selected_costs(pred,truth['moments'],truth['observed_future'],choice,c['groups'])))
            beat(state='amplitude_fitting_regions_checked',job=job['key'],views=len(records))
            del packet,c,prediction,choices,truth
            source.gc.collect();guard()
        if len(records)!=144:
            raise ValueError('All18jobs times2models times4fixed causal action regions required')
        value=source.stable(dict(rows=records,result_source='fresh_pre_new_held_fitting_diagnosis_not_independent_test',
            original_evaluation_before_new_held_readout=True,selection_regions_include_candidate_identity=True,
            registration_sha256=source.sha(experiment.REG),training_sha256=source.sha(experiment.PUBLIC/'training_freeze.json'),
            bindings={str(p.relative_to(ROOT)):source.sha(p)for p in
                (Path(__file__),ROOT/'src/evaluation/m3w_selected_cost_audit.py')},
            new_training=False,new_threshold_or_checkpoint_selection=False,stage5c_executed=False,smc_enabled=False))
        source.once(OUT/'analysis.json',value)
        source.once(OUT/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            all_scalars_exact=args.phase=='verify',new_training=False))
        beat(state='complete',verify=args.phase=='verify')


if __name__=='__main__':
    main()
