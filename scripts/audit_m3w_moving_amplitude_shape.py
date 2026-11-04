"""Inspect frozen fitting-side amplitude curves without new policy evaluation."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_amplitude_cost as experiment
from src.evaluation.m3w_amplitude_shape_audit import diagnose,TOLERANCE
from src.world_model.m3w_multiaction_cost import select_actions

source,api=experiment.source,experiment.api
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_shape_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_amplitude_shape_v1'
FILES=(Path(__file__),ROOT/'src/evaluation/m3w_amplitude_shape_audit.py',
    ROOT/'tests/test_m3w_amplitude_shape_audit.py',PUBLIC/'protocol.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','evaluate','verify'))
    args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    with(PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg=source.read(experiment.CONFIG)
        start,guard,beat=source.runtime(cfg,'fitting_amplitude_shape_'+args.phase)
        data,jobs,refs=experiment.register(cfg,beat)
        freeze=source.read(experiment.PUBLIC/'training_freeze.json')
        verify=source.read(experiment.PUBLIC/'verification.json')
        if (freeze['complete_heads']!=36 or not freeze['all18pairs_exact']
                or verify['summary_sha256']!=source.sha(experiment.PUBLIC/'summary.json')):
            raise ValueError('Complete exact parent experiment required')
        binding=source.stable(dict(bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
            parent_registration=source.artifact(experiment.REG),
            parent_training=source.artifact(experiment.PUBLIC/'training_freeze.json'),
            parent_verification=source.artifact(experiment.PUBLIC/'verification.json'),
            normalized_curve_tolerance=TOLERANCE,expected_fitting_views=36,
            new_training=False,new_held_policy_evaluation=False,new_data_roles=False,
            calibration_opened=False,confirmation_opened=False,stage5c_executed=False,smc_enabled=False))
        source.once(PUBLIC/'registration.json',binding)
        if args.phase=='register':
            return
        for p in (*FILES,PUBLIC/'registration.json'):
            old=source.subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,
                capture_output=True,check=True).stdout
            if source.hashlib.sha256(old).hexdigest()!=source.sha(p):
                raise ValueError('Commit the audit definition before real fitting-side readout')
        receipt=PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed audit phase is immutable')
        fits={}
        for ref in freeze['fits']:
            item=source.read(source.checked(ref));fits[item['job']['key'],item['arm']]=item
        records=[]
        for job in jobs:
            packet=experiment.view(data,job,refs,training=True)
            packet.pop('learning',None)
            c,ids=packet['causal'],packet['ids']
            prediction={};choices={}
            for arm in api.ARMS:
                item=fits[job['key'],arm]
                state=api.base.base.read_checkpoint(source.checked(item['checkpoint']))
                p=api.predict(state,c)
                if source.array_hash(p)!=item['fitting_predictions_sha256']:
                    raise ValueError('Fitting-side predictions changed from the original frozen head')
                allowed=api.for_arm(dict(causal=c),arm)['causal']['allowed']
                choices[arm],_=select_actions(p,allowed,arm='physical_neural')
                prediction[arm]=p
                del state
            labels=api.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],
                data['valid'][ids],job['easy_cut'])
            for arm,p in prediction.items():
                result=diagnose(p,labels['moments'],labels['observed_future'],choices[arm],c['envelope'])
                if any(v for comp in result['bounded_labels'].values()for v in comp.values()):
                    raise ValueError('True fixed-mask harm curves violate the claimed numerical shape')
                records.append(dict(job=job['key'],arm=arm,scores=result))
            beat(state='fitting_shape_checked',job=job['key'],views=len(records),total=36)
            del packet,c,prediction,choices,labels
            source.gc.collect();guard()
        if len(records)!=36:
            raise ValueError('All frozen fitting views required')
        source.once(PUBLIC/'analysis.json',source.stable(dict(rows=records,
            registration_sha256=source.sha(PUBLIC/'registration.json'),
            result_source='fresh_fitting_only_diagnosis_after_parent_readout_not_new_test',
            labels_used_for_selection=False,new_training=False,new_policy=False,
            stage5c_executed=False,smc_enabled=False)))
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            all36fitting_prediction_hashes_exact=True,all_scalars_exact=args.phase=='verify'))
        beat(state='complete')


if __name__=='__main__':
    main()
