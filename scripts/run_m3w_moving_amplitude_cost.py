"""Matched binary versus amplitude cost learning on fixed past-only forecasts."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_shared_motion_cost as source
from src.world_model import m3w_amplitude_cost as api

np=source.np
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_cost_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_amplitude_cost_v1'
PARENT=source.PUBLIC
CONFIG=ROOT/'configs/m3w_moving_amplitude_cost_v1.json'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_amplitude_cost.py',
    ROOT/'tests/test_m3w_amplitude_cost.py',ROOT/'src/evaluation/m3w_amplitude_opportunity.py')
MATCHED=('initial_model','parameters','budget','sampler','draw_hash','fit_ids','preprocess','torch_rng')
POLICIES=('floor','uncontrolled_neural','independent','frame_uniform','joint_matched','joint_free')


def register(cfg,beat):
    old_cfg=source.read(source.CONFIG)
    if (cfg['arms']!=list(api.ARMS) or cfg['fractions']!=[0.,*api.FRACTIONS]
            or cfg['training']!=old_cfg['training'] or cfg['new_heads']!=36
            or cfg['input_dimension']!=428 or cfg['positive_harm_budget']!=.02
            or cfg['easy_net_degradation_budget']!=.02 or cfg['seeds']!=[17,29,43]
            or cfg['joint_solver_seconds_per_query']!=.25 or cfg['bootstrap_resamples']!=3000
            or cfg['device']!='cpu' or cfg['torch_threads']!=4 or cfg['interop_threads']!=1 or cfg['workers']!=0
            or not cfg['full_first_job_refit_each_arm']
            or cfg['normalization']!='unchanged_original_physical_critic_fitting_only'
            or not cfg['true_CV_event_not_bank_slot1'] or not cfg['actual_floor_disagreement_envelope']
            or any(cfg[k]for k in ('new_forecast_training','threshold_search','new_data_roles',
                'calibration_opened','confirmation_opened','deployment_changed','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed matched amplitude experiment and original data/risk roles required')
    opportunity=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_opportunity_v1'
    if (not source.read(opportunity/'verify_execution.json')['all_scalars_exact']
            or not source.read(PARENT/'verification.json')['all36cost_predictions_and_policy_scores_exact']):
        raise ValueError('Complete verified parent forecast/cost/opportunity studies required')
    data,jobs,refs=source.admission(old_cfg,beat)
    value=source.stable(dict(configuration=cfg,bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        parent_registration=source.artifact(source.REG),parent_verification=source.artifact(PARENT/'verification.json'),
        opportunity_registration=source.artifact(opportunity/'registration.json'),
        opportunity_analysis=source.artifact(opportunity/'analysis.json'),
        opportunity_replay=source.artifact(opportunity/'verify_execution.json'),jobs=jobs,forecasts=refs,
        fresh_heads=36,new_primary_updates=72000,parameters=28229,
        original24group_primary='not_estimable_not_passed',supported_groups=23,source_rows=len(data['sites']),
        fixed_empty_event_views_retained=15,exposed_source_development_not_independent=True,
        stage5c_executed=False,smc_enabled=False))
    source.once(REG,value)
    return data,jobs,refs


def check_committed():
    for p in (*FILES,REG):
        old=subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if hashlib.sha256(old).hexdigest()!=source.sha(p):
            raise ValueError('Commit full experiment definition before fitting')


def view(data,job,refs,*,training):
    old=source.view(data,job,refs,training=training)
    ids=old['ids']
    result=api.causal_bank(old_causal=old['causal'],geometry=data['geometry'][ids],
        shared_context=data['shared_context'][ids],paths=old['paths'],floor_index=job['floor_index'])
    out=dict(result,ids=ids,recordings=old['recordings'],frames=old['frames'],sites=old['sites'])
    if training:
        learning=api.training_data(result,data['target_eval'][ids]-data['origin'][ids,None],
            data['valid'][ids],job['easy_cut'],old['learning']['preprocess'])
        source.api.base.base.exact(learning['truth_by_arm']['bound_aware'][:,6],old['learning']['truth_by_arm']['bound_aware'][:,6])
        source.api.base.base.exact(learning['observed_future'],old['learning']['observed_future'])
        if set(learning['preprocess']['source_names'])!=set(job['critic_sites']):
            raise ValueError('Original critic-fitting-only normalization required')
        out['learning']=learning
    return out


def state_path(job,arm,kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def fit(packet,job,arm,cfg,beat,guard,*,kind='fits',resume=False,stop_at=None):
    return api.fit(packet['learning'],arm=arm,seed=job['seed'],settings=cfg['training'],
        identity=dict(registration_sha256=source.sha(REG),job=job),path=state_path(job,arm,kind),
        resume=resume,stop_at=stop_at,guard=guard,heartbeat=lambda **v:beat(job=job['key'],treatment=arm,**v))


def pilot(data,jobs,refs,cfg,beat,guard,resume):
    cells=[]
    for fold in range(3):
        job=next(j for j in jobs if j['critic_fold']==fold)
        packet=view(data,job,refs,training=True)
        pair=[]
        for arm in api.ARMS:
            path=state_path(job,arm,'pilot_resume')
            if path.exists()and not resume:
                raise ValueError('Explicit same-binding pilot resume required')
            previous=api.base.base.read_checkpoint(path)['step'] if path.exists() else 0
            if previous<50:
                fit(packet,job,arm,cfg,beat,guard,kind='pilot_resume',resume=path.exists(),stop_at=50)
            a=fit(packet,job,arm,cfg,beat,guard,kind='pilot_resume',resume=True,stop_at=100)
            b=fit(packet,job,arm,cfg,beat,guard,kind='pilot_direct',
                resume=resume and state_path(job,arm,'pilot_direct').exists(),stop_at=100)
            source.recovered(a,b)
            pair.append(a)
            cells.append(dict(job=job['key'],arm=arm,step=a['step'],budget=a['budget'],seconds=a['seconds'],
                parameters=a['parameters'],exact_resume=True))
        for key in MATCHED:
            api.base.base.exact(pair[0][key],pair[1][key])
        del packet,pair,a,b
        source.gc.collect();guard()
    source.once(PUBLIC/'pilot.json',source.stable(dict(cells=cells,real_Torch_training=True,
        exact_resume_and_pair_sampling_all6cells=True,held_policy_scores_read=False,
        full36fits_projection_seconds_not_actual=6*sum(r['seconds']/r['step']*r['budget']for r in cells))))


def train(data,jobs,refs,cfg,beat,guard,resume):
    if not source.read(PUBLIC/'pilot.json')['exact_resume_and_pair_sampling_all6cells']:
        raise ValueError('Complete real pilot before primary training')
    fits=[]
    for job in jobs:
        packet=view(data,job,refs,training=True)
        pair=[]
        for arm in api.ARMS:
            state=fit(packet,job,arm,cfg,beat,guard,resume=resume and state_path(job,arm).exists())
            if state['step']!=job['fit_scope']['budget']or state['parameters']!=28229:
                raise ValueError('Complete fixed budget and matched capacity required')
            if job==jobs[0]:
                replay=fit(packet,job,arm,cfg,beat,guard,kind='full_refit',
                    resume=resume and state_path(job,arm,'full_refit').exists())
                source.recovered(state,replay)
                source.once(PUBLIC/('first_full_refit_'+arm+'.json'),dict(step=state['step'],scientific_state_exact=True,
                    checkpoint=source.artifact(state_path(job,arm,'full_refit'))))
                del replay
            prediction=api.predict(state,packet['causal'])
            receipt=dict(job=job,arm=arm,step=state['step'],parameters=state['parameters'],seconds=state['seconds'],
                checkpoint=source.artifact(state_path(job,arm)),draw_hash=state['draw_hash'],
                minimum_query_visits=int(state['sampler']['counts'].min()),
                fitting_predictions_sha256=source.array_hash(prediction),
                initial_trace=state['trace'][0],final_trace=state['trace'][-1],
                full_amplitude_original_features_and_targets_exact=True)
            path=PUBLIC/'fits'/(job['key']+'_'+arm+'.json')
            source.once(path,source.stable(receipt));fits.append(source.artifact(path));pair.append(state)
            beat(state='amplitude_head_complete',job=job['key'],treatment=arm,complete=len(fits),total=36)
            del prediction
        for key in MATCHED:
            api.base.base.exact(pair[0][key],pair[1][key])
        del packet,pair,state
        source.gc.collect();guard()
    source.once(PUBLIC/'training_freeze.json',dict(fits=fits,complete_heads=36,new_primary_updates=72000,
        all18pairs_exact=True,real_Torch_training=True,independent_roles_opened=False))


def evaluate(data,jobs,refs,cfg,beat,guard,*,verify):
    freeze=source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_heads']!=36 or not freeze['all18pairs_exact']:
        raise ValueError('Freeze all36complete matched fits before held readout')
    models={}
    for ref in freeze['fits']:
        receipt=source.read(source.checked(ref))
        models[receipt['job']['key'],receipt['arm']]=receipt['checkpoint']
    records=[];receipts=[]
    for job in jobs:
        packet=view(data,job,refs,training=False)
        c,ids=packet['causal'],packet['ids']
        policies={};choices={};cost_hashes={}
        for arm in api.ARMS:
            state=api.base.base.read_checkpoint(source.checked(models[job['key'],arm]))
            pred=api.predict(state,c)
            cost_hashes[arm]=source.array_hash(pred)
            deploy=(arm,'amplitude_bank_binary_deploy')if arm=='amplitude_bank'else(arm,)
            for name in deploy:
                mask=api.for_arm(dict(causal=c),'binary_only'if name.endswith('binary_deploy')else arm)['causal']
                selected,_=source.allocated(pred,mask,packet,job,name,cfg,beat,verify=verify)
                choices[name]=selected
                policies[name]=source.policy_paths(packet['paths'],packet['floor'],selected)
            del state,pred
        old=source.read(PARENT/'groups'/(job['key']+'.json'))
        legacy={r['site']:r['policies']['local_history']for r in old['rows']}
        rows=[]
        for site in sorted(set(packet['sites'])):
            take=packet['sites']==site
            target=data['target_eval'][ids[take]]-data['origin'][ids[take],None]
            valid=data['valid'][ids[take]]
            scored={arm:{p:source.readout.policy(path[take],packet['floor'][take],packet['cv'][take],target,
                valid,packet['recordings'][take],easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])
                for p,path in variants.items()}for arm,variants in policies.items()}
            scored['legacy_binary_cached']=legacy[str(site)]
            primary={p:source.readout.paired(policies['binary_only'][p][take],policies['amplitude_bank'][p][take],
                packet['floor'][take],target,valid)for p in POLICIES}
            shared={p:source.readout.paired(policies['amplitude_bank_binary_deploy'][p][take],policies['amplitude_bank'][p][take],
                packet['floor'][take],target,valid)for p in POLICIES}
            groups=source.query_groups(packet['recordings'][take],packet['frames'][take],ids[take])
            utility={arm:source.matched_utility(policies[arm]['independent'][take],policies[arm]['joint_matched'][take],
                packet['floor'][take],target,valid,choice['independent'][take],choice['joint_matched'][take],groups)
                for arm,choice in choices.items()}
            amplitudes={}
            grid=np.array([0.,*api.FRACTIONS])
            for arm,select in choices.items():
                amplitudes[arm]={}
                for policy,choice in select.items():
                    at=choice[take];changed=at>=0
                    values=np.where(changed,grid[np.maximum(at,0)],0.)
                    amplitudes[arm][policy]=dict(rows=len(at),selected=int(changed.sum()),
                        action_counts=np.bincount(at+1,minlength=8).tolist(),
                        mean_amplitude_all_rows=float(values.mean()),
                        mean_amplitude_selected=float(values[changed].mean())if changed.any()else None)
            rows.append(dict(job=job['key'],site=str(site),producer_fold=job['producer_fold'],seed=job['seed'],
                policies=scored,paired_amplitude_minus_binary=primary,paired_shared_weight_amplitude_minus_binary=shared,
                matched_utility=utility,amplitudes=amplitudes))
        path=PUBLIC/'groups'/(job['key']+'.json')
        source.once(path,source.stable(dict(job=job,rows=rows,cost_hashes=cost_hashes)))
        receipts.append(source.artifact(path));records.extend(rows)
        beat(state='amplitude_held_replayed'if verify else'amplitude_held_evaluated',job=job['key'],groups=len(records))
        del packet,c,policies,choices,rows
        source.gc.collect();guard()
    if len(records)!=138 or len({(r['site'],r['producer_fold'],r['seed'])for r in records})!=138:
        raise ValueError('Every supported group requires both excluded producers and all3actual seeds')
    sites=sorted(set(data['sites']))
    ci=lambda getter:source.group_interval(records,getter,sites,cfg)
    primary={p:{m:ci(lambda r,p=p,m=m:r['paired_amplitude_minus_binary'][p][m])
        for m in ('known_pp','completion_lower_pp','completion_upper_pp')}for p in POLICIES}
    shared={p:{m:ci(lambda r,p=p,m=m:r['paired_shared_weight_amplitude_minus_binary'][p][m])
        for m in ('known_pp','completion_lower_pp','completion_upper_pp')}for p in POLICIES}
    safety={};known={};lower={}
    for arm in ('binary_only','amplitude_bank','amplitude_bank_binary_deploy','legacy_binary_cached'):
        safety[arm]={};known[arm]={};lower[arm]={}
        for p in POLICIES:
            known[arm][p]=ci(lambda r,a=arm,p=p:r['policies'][a][p]['known_ADE_gain_floor_percent'])
            lower[arm][p]=ci(lambda r,a=arm,p=p:r['policies'][a][p]['completion_lower_gain_floor_percent'])
            rs=[v for r in records for v in r['policies'][arm][p]['recordings'].values()]
            easy=[r['known_easy_net_degradation']for r in rs if r['known_easy_net_degradation']is not None]
            safety[arm][p]=dict(required_record_views=len(rs),supported_record_views=sum(r['supported']for r in rs),
                known_all_risk_violations=sum(r['known_all_risk']is not None and r['known_all_risk']>.02 for r in rs),
                known_easy_risk_violations=sum(r['known_easy_risk']is not None and r['known_easy_risk']>.02 for r in rs),
                undefined_easy_views=sum(r['completion_easy_risk_upper']is None for r in rs),
                worst_known_easy_net_degradation_percent=100*max(easy)if easy else None,
                full_original_risk_pass=bool(rs and all(r['supported']for r in rs)))
    summary=source.stable(dict(primary_amplitude_minus_binary=primary,shared_weight_deployment_contrast=shared,
        known_accuracy=known,completion_lower=lower,safety=safety,group_receipts=receipts,
        original24group_primary='not_estimable_not_passed',supported23group_fixed_secondary=primary['joint_free']['completion_lower_pp'],
        legacy_arm_result_source='cached_verified_original_parent_cost_policy_receipts_not_new_inference',
        actual_seeds=[17,29,43],independent_confirmation=False,new_deployment=False,
        stage5c_executed=False,smc_enabled=False))
    source.once(PUBLIC/'summary.json',summary)
    if verify:
        source.once(PUBLIC/'verification.json',dict(all36new_cost_outputs_and_policy_scalars_exact=True,
            saved_solver_bindings_and_feasibility_checked=True,new_bitwise_MILP_reoptimization_not_claimed=True,
            summary_sha256=source.sha(PUBLIC/'summary.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','evaluate','verify'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    with (PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        cfg=source.read(CONFIG)
        start,guard,beat=source.runtime(cfg,args.phase)
        data,jobs,refs=register(cfg,beat)
        if args.phase=='register':
            return
        check_committed()
        if(PUBLIC/(args.phase+'_execution.json')).exists():
            raise ValueError('Completed experiment phase is immutable')
        if args.phase=='pilot':
            pilot(data,jobs,refs,cfg,beat,guard,args.resume)
        elif args.phase=='train':
            train(data,jobs,refs,cfg,beat,guard,args.resume)
        else:
            evaluate(data,jobs,refs,cfg,beat,guard,verify=args.phase=='verify')
        source.once(PUBLIC/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_Torch_training=args.phase in ('pilot','train'),independent_roles_opened=False,
            stage5c_executed=False,smc_enabled=False))
        beat(state='complete')


if __name__=='__main__':
    main()
