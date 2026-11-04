"""Fixed amplitude controls and label-aware feasibility, not selector training."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from src.evaluation.m3w_amplitude_opportunity import FRACTIONS,rollout
from src.evaluation.m3w_record_risk_feasibility import witness

NAME='moving_amplitude_opportunity_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09'/NAME
PRIVATE=ROOT/'data/stage_cvpr2027_experiments'/NAME
PARENT=source.PUBLIC
FEAS=ROOT/'outputs/publication_readiness_2026_09/moving_record_feasibility_v1'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_amplitude_opportunity.py',
    ROOT/'tests/test_m3w_amplitude_opportunity.py',ROOT/'src/evaluation/m3w_record_risk_feasibility.py')


def registration():
    if (not source.read(FEAS/'verify_execution.json')['all_scalars_exact']
            or not source.read(PARENT/'verification.json')['all36cost_predictions_and_policy_scores_exact']):
        raise ValueError('Complete verified binary parent studies required')
    value=source.stable(dict(fractions=list(FRACTIONS),
        bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        parents={str(p.relative_to(ROOT)):source.artifact(p)for p in
            (PARENT/'registration.json',PARENT/'summary.json',FEAS/'analysis.json',FEAS/'verify_execution.json')},
        original_risk_budget=.02,no_fraction_selected_from_held_scores=True,
        labels_for_diagnosis_only=True,new_training=False,new_data_roles=False,
        stage5c_executed=False,smc_enabled=False))
    source.once(REG,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','evaluate','verify'))
    args=parser.parse_args()
    registration()
    if args.phase == 'register':
        return
    for p in (*FILES,REG):
        old=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(old).hexdigest() != source.sha(p):
            raise ValueError('Commit complete amplitude definition before scoring')
    PRIVATE.mkdir(parents=True,exist_ok=True)
    source.PRIVATE=PRIVATE
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg=source.read(source.CONFIG)
        start,guard,beat=source.runtime(cfg,'amplitude_opportunity')
        data,jobs,refs=source.admission(cfg,beat)
        old={(r['job'],r['recording']):r for r in source.read(FEAS/'analysis.json')['rows']}
        policies=[];feasibility=[]
        for job in jobs:
            packet=source.view(data,job,refs,training=False)
            ids,paths=packet['ids'],packet['paths']
            target=data['target_eval'][ids]-data['origin'][ids,None]
            valid=data['valid'][ids]
            floor,proposal,cv=paths[:,job['floor_index']],paths[:,6],paths[:,1]
            re,_=source.readout.errors(floor,target,valid)
            ce,_=source.readout.errors(cv,target,valid)
            parent=source.read(PARENT/'groups'/(job['key']+'.json'))
            old_policies={r['site']:r['policies']['local_history']['uncontrolled_neural']for r in parent['rows']}
            max_fraction={str(r):None for r in sorted(set(packet['recordings']))}
            for alpha in FRACTIONS:
                path=rollout(floor,proposal,alpha)
                pe,_=source.readout.errors(path,target,valid)
                legal=source.np.linalg.norm(path-floor,axis=-1).max(1)>0
                _,records=witness(re,pe,ce,legal,packet['recordings'],easy_cut=job['easy_cut'])
                for rec,row in records.items():
                    if row['binary_fixed_proposal_finite_risk_feasible']:
                        max_fraction[rec]=alpha
                    if alpha == 1.:
                        if row != {k:v for k,v in old[job['key'],rec].items()
                                if k not in ('job','recording','seed','producer_fold','witness_evaluator')}:
                            raise ValueError('Full-amplitude witness must exactly reproduce original binary diagnosis')
                for site in sorted(set(packet['sites'])):
                    take=packet['sites']==site
                    score=source.readout.policy(path[take],floor[take],cv[take],target[take],valid[take],
                        packet['recordings'][take],easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])
                    if alpha == 1.:
                        source.api.base.base.exact(score,old_policies[str(site)])
                    policies.append(dict(job=job['key'],site=str(site),seed=job['seed'],alpha=alpha,score=score))
            for rec,alpha in max_fraction.items():
                original=old[job['key'],rec]
                feasibility.append(dict(job=job['key'],recording=rec,
                    original_binary_feasible=original['binary_fixed_proposal_finite_risk_feasible'],
                    original_reason=original['reason'],any_fraction_feasible=alpha is not None,
                    largest_fraction_with_label_aware_witness=alpha))
            beat(state='all_fixed_amplitudes_scored',job=job['key'],record_views=len(feasibility))
            del packet,paths,target,valid,floor,proposal,cv,re,ce,path,pe
            source.gc.collect();guard()
        if len(feasibility) != 348 or len(policies) != 138*len(FRACTIONS):
            raise ValueError('Complete original view and amplitude matrix required')
        summary={}
        for alpha in FRACTIONS:
            rows=[r for r in policies if r['alpha']==alpha]
            sites=sorted({r['site']for r in rows})
            known=source.group_interval(rows,lambda r:r['score']['known_ADE_gain_floor_percent'],sites,cfg)
            lower=source.group_interval(rows,lambda r:r['score']['completion_lower_gain_floor_percent'],sites,cfg)
            records=[v for r in rows for v in r['score']['recordings'].values()]
            summary[str(alpha)]=dict(alpha=alpha,known_gain=known,completion_lower_gain=lower,
                supported_record_views=sum(r['supported']for r in records),required_record_views=len(records),
                known_all_risk_violations=sum(r['known_all_risk']is not None and r['known_all_risk']>.02 for r in records),
                known_easy_risk_violations=sum(r['known_easy_risk']is not None and r['known_easy_risk']>.02 for r in records),
                undefined_easy_views=sum(r['completion_easy_risk_upper']is None for r in records))
        result=source.stable(dict(policies=policies,feasibility=feasibility,summary=summary,
            registration_sha256=source.sha(REG),all_alpha1_original_policies_and_witnesses_exact=True,
            original24group_primary='not_estimable_not_passed',
            result_source='fresh_fixed_causal_controls_and_label_aware_opportunity_on_verified_forecast_bank',
            no_amplitude_selected=True,new_training=False,independent_confirmation=False,
            new_deployment=False,stage5c_executed=False,smc_enabled=False))
        source.once(PUBLIC/'analysis.json',result)
        source.once(PUBLIC/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            all_scalars_exact=args.phase=='verify',new_training=False))
        beat(state='complete',verify=args.phase=='verify')


if __name__ == '__main__':
    main()
