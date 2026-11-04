"""Producer-selected social motion controls, separate from frozen cost floors."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from src.world_model import m3w_moving_social_causal_baselines as api
from src.evaluation.m3w_native_metrics import native_errors
from src.evaluation.m3w_european_eqmotion import paired_group, interval

np = source.np
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/moving_social_causal_baselines_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/moving_social_causal_baselines_v1'
REG = PUBLIC/'registration.json'
FILES = (Path(__file__), ROOT/'src/world_model/m3w_moving_social_causal_baselines.py',
    ROOT/'tests/test_m3w_moving_social_causal_baselines.py', PUBLIC/'protocol.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register','evaluate','verify'))
    args = parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    source.PRIVATE = PRIVATE
    cfg = source.read(source.CONFIG)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        start, guard, beat = source.runtime(cfg,args.phase)
        data, jobs, refs = source.admission(cfg,beat)
        registration = dict(bindings={str(p.relative_to(ROOT)):source.sha(p) for p in FILES},
            parent_registration=source.artifact(source.REG), candidates=api.CANDIDATES,
            producer_fit_only_coefficient_selection=True, frozen_existing_floors_changed=False,
            new_neural_training=False, independent_roles_opened=False,
            original24group_primary='not_estimable_not_passed', stage5c_executed=False,smc_enabled=False)
        source.once(REG,source.stable(registration))
        if args.phase == 'register':
            return
        for path in (*FILES,REG):
            blob=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if hashlib.sha256(blob).hexdigest()!=source.sha(path):
                raise ValueError('Commit fixed baseline definition before held scoring')
        acceleration=api.accelerations(data['history'],data['geometry'],data['shared_context'])
        selected={};groups={key:[] for key in ('nearest8','all_current')}
        for fold in range(3):
            job=next(j for j in jobs if j['producer_fold']==fold)
            fit=np.flatnonzero(np.isin(data['sites'],job['producer_sites']))
            held=np.flatnonzero(~np.isin(data['sites'],job['producer_sites']))
            costs=np.empty((len(fit),len(api.CANDIDATES)))
            costs[:,:6]=data['baseline_ade'][fit]
            target=data['target_eval'][fit]-data['origin'][fit,None]
            for k,candidate in enumerate(api.CANDIDATES[6:],6):
                path=api.rollout(data['history'][fit],acceleration[candidate['family']][fit],candidate)
                costs[:,k]=native_errors(path,target,data['valid'][fit],np.ones(len(fit)))[0]
            predictions={}
            selected[str(fold)]={}
            for family,indices in (('nearest8',list(range(15))),('all_current',list(range(24)))):
                k,scores=api.fitting_choice(costs,data['sites'][fit],indices)
                candidate=api.CANDIDATES[k]
                a=acceleration.get(candidate['family'],np.zeros((len(data['sites']),2)))
                predictions[family]=api.rollout(data['history'][held],a[held],candidate)
                selected[str(fold)][family]=dict(index=k,candidate=candidate,
                    producer_fit_sites=sorted(set(data['sites'][fit])),fitting_ids_sha256=source.array_hash(fit),
                    fitting_relative_scores=scores,held_ids_sha256=source.array_hash(held))
            for seed in cfg['seeds']:
                ref=refs[f'eqmotion_fold{fold}_seed{seed}']
                with np.load(source.checked(ref),allow_pickle=False) as cache:
                    np.testing.assert_array_equal(cache['ids'],held)
                    neural=cache['relative_prediction'].copy()
                for site in sorted(set(data['sites'][held])):
                    use=data['sites'][held]==site;ids=held[use]
                    target=data['target_eval'][ids]-data['origin'][ids,None]
                    for family,path in predictions.items():
                        score=paired_group(neural[use],path[use],target,data['valid'][ids],data['baseline_ade'][ids,1])
                        score['social_physics']=score.pop('PCT');score['neural']=score.pop('eqmotion')
                        groups[family].append(dict(site=str(site),trial=f'producer_fold{fold}',seed=seed,scores=score))
            beat(state='social_baseline_fold_complete',fold=fold)
            guard()
        sites=sorted(set(data['sites']))
        contrasts={family:{metric:interval(rows,metric,sites,3000,cfg['bootstrap_seed'])
            for metric in ('known_pp','completion_lower_pp','completion_upper_pp')} for family,rows in groups.items()}
        value=source.stable(dict(selected=selected,groups=groups,contrasts=contrasts,
            result_source='fresh_physics_computation_on_hash_verified_population_and_replayed_neural_cache',
            baseline_coefficients_fit_on_P_only=True, data_rows=195286, supported_groups=23,
            original24group_primary='not_estimable_not_passed',
            nearest8_matches_forecaster_neighbor_inventory=True,
            all_current_has_extra_partial_history_context=True,
            frozen_cost_floor_or_policy_changed=False,new_neural_training=False,
            independent_confirmation=False,stage5c_executed=False,smc_enabled=False))
        source.once(PUBLIC/'analysis.json',value)
        source.once(PUBLIC/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            all_scalars_exact=args.phase=='verify',new_neural_training=False))
        beat(state='social_baseline_complete')


if __name__=='__main__':
    main()
