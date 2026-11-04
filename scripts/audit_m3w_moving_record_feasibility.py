"""Exact label-aware fixed-proposal risk feasibility, without policy training."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from src.evaluation.m3w_record_risk_feasibility import witness

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_record_feasibility_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_record_feasibility_v1'
PARENT=source.PUBLIC


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('evaluate','verify'))
    args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    source.PRIVATE=PRIVATE
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg=source.read(source.CONFIG)
        start,guard,beat=source.runtime(cfg,'fixed_proposal_feasibility')
        data,jobs,refs=source.admission(cfg,beat)
        rows=[]
        for job in jobs:
            packet=source.view(data,job,refs,training=False)
            ids,paths=packet['ids'],packet['paths']
            target=data['target_eval'][ids]-data['origin'][ids,None]
            valid=data['valid'][ids]
            floor,proposal,cv=paths[:,job['floor_index']],paths[:,6],paths[:,1]
            re,_=source.readout.errors(floor,target,valid)
            pe,_=source.readout.errors(proposal,target,valid)
            ce,_=source.readout.errors(cv,target,valid)
            choices,records=witness(re,pe,ce,packet['causal']['allowed'][:,6],packet['recordings'],
                easy_cut=job['easy_cut'])
            oracle=source.np.where(choices[:,None,None],proposal,floor)
            check=source.readout.policy(oracle,floor,cv,target,valid,packet['recordings'],
                easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])
            for recording,record in records.items():
                actual=check['recordings'][recording]
                if actual['supported'] != record['binary_fixed_proposal_finite_risk_feasible']:
                    raise ValueError('Exact witness criterion disagrees with unchanged evaluator')
                rows.append(dict(job=job['key'],recording=recording,seed=job['seed'],
                    producer_fold=job['producer_fold'],**record,witness_evaluator=actual))
            beat(state='witness_gate_verified',job=job['key'],required_record_views=len(rows))
            del packet,paths,target,valid,floor,proposal,cv,re,pe,ce,choices,oracle
            source.gc.collect()
            guard()
        if len(rows) != 348 or len(set(r['recording'] for r in rows)) != 58:
            raise ValueError('Preserve all348repeated views from58actual-query recordings')
        value=source.stable(dict(rows=rows,required_record_views=len(rows),unique_recordings=58,
            feasible_record_views=sum(r['binary_fixed_proposal_finite_risk_feasible']for r in rows),
            exact_witness_gate_agreement_all_views=True,
            result_source='fresh_label_aware_post_readout_diagnostic_not_deployable_or_independent',
            parent_summary=source.artifact(PARENT/'summary.json'),
            parent_verification=source.artifact(PARENT/'verification.json'),
            bindings={str(p.relative_to(ROOT)):source.sha(p) for p in (Path(__file__),PUBLIC/'protocol.md',
                ROOT/'src/evaluation/m3w_record_risk_feasibility.py',ROOT/'tests/test_m3w_record_risk_feasibility.py')},
            original24group_primary='not_estimable_not_passed',new_training=False,
            labels_used_for_oracle_diagnosis_only=True,oracle_row_choices_exported=False,
            source_roles_changed=False,budget_changed=False,independent_confirmation=False,
            stage5c_executed=False,smc_enabled=False))
        source.once(PUBLIC/'analysis.json',value)
        source.once(PUBLIC/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            all_scalars_exact=args.phase == 'verify',new_training=False))
        beat(state='complete',verify=args.phase == 'verify')


if __name__ == '__main__':
    main()
