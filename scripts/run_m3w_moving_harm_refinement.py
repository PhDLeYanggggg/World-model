"""Registered matched fit-only harm refiners on frozen causal amplitude heads."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sklearn
from scripts import run_m3w_moving_amplitude_cost as parent
from src.world_model import m3w_harm_refinement as api
from src.evaluation.m3w_selected_cost_audit import selected_costs

source=parent.source
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_harm_refinement_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_harm_refinement_v1'
CONFIG=ROOT/'configs/m3w_moving_harm_refinement_v1.json'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_harm_refinement.py',
    ROOT/'tests/test_m3w_harm_refinement.py')
ARMS=('frozen',*api.ARMS)


def register(beat):
    spec=source.read(CONFIG);cfg=source.read(parent.CONFIG)
    if (spec['arms']!=list(api.ARMS) or spec['new_refiners']!=54 or spec['new_primary_tree_fits']!=1080
            or any(spec[k]for k in ('threshold_search','new_data_roles','calibration_opened','confirmation_opened',
                'deployment_changed','new_neural_training','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed matched three-arm cost refinement required')
    data,jobs,refs=parent.register(cfg,beat);controls={}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        row=source.read(source.checked(ref))
        if row['arm']=='amplitude_bank':
            source.checked(row['checkpoint']);controls[row['job']['key']]=row
    previous=ROOT/'outputs/publication_readiness_2026_09/moving_convex_amplitude_v1'
    if not source.read(previous/'verification.json')['all36cost_outputs_and_policy_scalars_exact']or len(controls)!=18:
        raise ValueError('Complete previous study and all frozen base checkpoints required')
    source.once(REG,source.stable(dict(specification=spec,parent_configuration=cfg,
        bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        parent_registration=source.artifact(parent.REG),previous_summary=source.artifact(previous/'summary.json'),
        controls={k:r['checkpoint']for k,r in controls.items()},sklearn_version=sklearn.__version__,
        original24group_primary='not_estimable_not_passed',independent_confirmation=False,
        stage5c_executed=False,smc_enabled=False)))
    return spec,cfg,data,jobs,refs,controls


def committed():
    for p in (*FILES,REG):
        old=source.subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(old).hexdigest()!=source.sha(p):raise ValueError('Commit definition before fitting')


def state_path(job,arm,kind='fits'):
    return PRIVATE/kind/(job['key']+'_'+arm+'.pt.gz')


def one(packet,prediction,job,arm,spec,beat,guard,*,kind='fits',resume=False,stop_at=None):
    return api.fit(packet['learning'],prediction,arm=arm,seed=job['seed'],settings=spec['training'],
        identity=dict(registration_sha256=source.sha(REG),job=job),path=state_path(job,arm,kind),
        resume=resume,stop_at=stop_at,guard=guard,heartbeat=lambda **v:beat(job=job['key'],**v))


def pilot(spec,cfg,data,jobs,refs,controls,beat,guard,resume):
    cells=[]
    for fold in range(3):
        job=next(j for j in jobs if j['critic_fold']==fold);packet=parent.view(data,job,refs,training=True)
        old=api.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        prediction=api.parent.predict(old,packet['causal'])
        for arm in api.ARMS:
            path=state_path(job,arm,'pilot_resume')
            if path.exists()and not resume:raise ValueError('Explicit original pilot resume required')
            step=api.base.read_checkpoint(path)['round']if path.exists()else 0
            if step<2:one(packet,prediction,job,arm,spec,beat,guard,kind='pilot_resume',resume=path.exists(),stop_at=2)
            a=one(packet,prediction,job,arm,spec,beat,guard,kind='pilot_resume',resume=True,stop_at=4)
            b=one(packet,prediction,job,arm,spec,beat,guard,kind='pilot_direct',
                resume=resume and state_path(job,arm,'pilot_direct').exists(),stop_at=4)
            api.exact_refit(a,b)
            cells.append(dict(job=job['key'],arm=arm,rounds=4,seconds=b['seconds'],exact_resume=True))
        del packet,old,prediction,a,b;source.gc.collect();guard()
    source.once(PUBLIC/'pilot.json',source.stable(dict(cells=cells,all9cells_exact=True,
        projection_pure_tree_fit_seconds_not_completion=30*sum(r['seconds']for r in cells))))


def train(spec,cfg,data,jobs,refs,controls,beat,guard,resume):
    if not source.read(PUBLIC/'pilot.json')['all9cells_exact']:raise ValueError('Complete recovery pilot first')
    fits=[]
    for job in jobs:
        packet=parent.view(data,job,refs,training=True);c=packet['causal']
        old=api.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        frozen=api.parent.predict(old,c)
        if source.array_hash(frozen)!=controls[job['key']]['fitting_predictions_sha256']:
            raise ValueError('Frozen fitting cost prediction changed')
        truth=api.parent.targets(packet,data['target_eval'][packet['ids']]-data['origin'][packet['ids'],None],
            data['valid'][packet['ids']],job['easy_cut'])
        for arm in api.ARMS:
            state=one(packet,frozen,job,arm,spec,beat,guard,resume=resume and state_path(job,arm).exists())
            if job==jobs[0]:
                refit=one(packet,frozen,job,arm,spec,beat,guard,kind='full_refit',
                    resume=resume and state_path(job,arm,'full_refit').exists())
                api.exact_refit(state,refit)
                source.once(PUBLIC/('full_refit_'+arm+'.json'),dict(exact=True,rounds=20,
                    checkpoint=source.artifact(state_path(job,arm,'full_refit'))))
            prediction=api.predict(state,c,frozen)
            if api.digest(prediction)!=state['prediction_sha256']:raise ValueError('Full fitting replay must be exact')
            choice=api.select_actions(prediction,c['allowed'],arm='physical_neural')[0]
            receipt=source.stable(dict(job=job,arm=arm,rounds=state['round'],seconds=state['seconds'],
                checkpoint=source.artifact(state_path(job,arm)),tree_hashes=state['tree_hashes'],
                actual_tree_nodes=sum(t.tree_.node_count for t in state['trees']),
                first_trace=state['trace'][0],last_trace=state['trace'][-1],
                fitting_predictions_sha256=source.array_hash(prediction),
                own_selected_costs=selected_costs(prediction,truth['moments'],truth['observed_future'],choice,c['groups']),
                cached_base_checkpoint=controls[job['key']]['checkpoint'],new_neural_training=False))
            path=PUBLIC/'fits'/(job['key']+'_'+arm+'.json');source.once(path,receipt);fits.append(source.artifact(path))
            beat(state='refiner_complete',job=job['key'],arm=arm,complete=len(fits),total=54)
        del packet,c,old,frozen,truth,state,prediction;source.gc.collect();guard()
    source.once(PUBLIC/'training_freeze.json',dict(fits=fits,complete_refiners=54,primary_tree_fits=1080,
        new_neural_training=False,cached_neural_heads=18,all_fits_before_held_evaluation=True))


def evaluate(spec,cfg,data,jobs,refs,controls,beat,guard,verify):
    freeze=source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_refiners']!=54:raise ValueError('Complete all54refiners before held scoring')
    models={}
    for ref in freeze['fits']:
        r=source.read(source.checked(ref));models[r['job']['key'],r['arm']]=r['checkpoint']
    records=[];receipts=[]
    contrasts=dict(adaptive_minus_fixed=('fixed_region','adaptive_region'),
        **{a+'_minus_frozen':('frozen',a)for a in api.ARMS})
    for job in jobs:
        packet=parent.view(data,job,refs,training=False);c,ids=packet['causal'],packet['ids']
        old=api.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        predictions=dict(frozen=api.parent.predict(old,c))
        original=source.read(parent.PUBLIC/'groups'/(job['key']+'.json'))
        if source.array_hash(predictions['frozen'])!=original['cost_hashes']['amplitude_bank']:
            raise ValueError('Frozen held inference differs from complete original control')
        for arm in api.ARMS:
            state=api.base.read_checkpoint(source.checked(models[job['key'],arm]))
            predictions[arm]=api.predict(state,c,predictions['frozen'])
        source.PRIVATE=parent.PRIVATE
        try:choices_old,_=source.allocated(predictions['frozen'],c,packet,job,'amplitude_bank',cfg,beat,verify=True)
        finally:source.PRIVATE=PRIVATE
        choices=dict(frozen=choices_old)
        for arm in api.ARMS:
            choices[arm],_=source.allocated(predictions[arm],c,packet,job,arm,cfg,beat,verify=verify)
        paths={a:source.policy_paths(packet['paths'],packet['floor'],v)for a,v in choices.items()}
        rows=[]
        for site in sorted(set(packet['sites'])):
            at=packet['sites']==site;target=data['target_eval'][ids[at]]-data['origin'][ids[at],None];valid=data['valid'][ids[at]]
            policies={a:{p:source.readout.policy(x[at],packet['floor'][at],packet['cv'][at],target,valid,
                packet['recordings'][at],easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])for p,x in variants.items()}
                for a,variants in paths.items()}
            legacy=next(r for r in original['rows']if r['site']==str(site))['policies']['amplitude_bank']
            api.base.exact(source.stable(policies['frozen']),source.stable(legacy))
            paired={name:{p:source.readout.paired(paths[a][p][at],paths[b][p][at],packet['floor'][at],target,valid)
                for p in parent.POLICIES}for name,(a,b)in contrasts.items()}
            groups=source.query_groups(packet['recordings'][at],packet['frames'][at],ids[at]);matched={};usage={}
            for arm in ARMS:
                matched[arm]=source.matched_utility(paths[arm]['independent'][at],paths[arm]['joint_matched'][at],
                    packet['floor'][at],target,valid,choices[arm]['independent'][at],choices[arm]['joint_matched'][at],groups)
                take=choices[arm]['joint_free'][at]
                alpha=source.np.where(take>=0,source.np.asarray((0.,*api.parent.FRACTIONS))[source.np.maximum(take,0)],0.)
                usage[arm]=dict(action_counts=source.np.bincount(take+1,minlength=8).tolist(),
                    mean_amplitude_all_rows=float(alpha.mean()),selected=int((take>=0).sum()),rows=len(take))
            rows.append(dict(site=str(site),producer_fold=job['producer_fold'],seed=job['seed'],
                policies=policies,paired=paired,matched_utility=matched,usage=usage))
        path=PUBLIC/'groups'/(job['key']+'.json')
        source.once(path,source.stable(dict(job=job,rows=rows,cost_hashes={a:source.array_hash(v)for a,v in predictions.items()})))
        receipts.append(source.artifact(path));records.extend(rows)
        beat(state='refined_held_replayed'if verify else'refined_held_evaluated',job=job['key'],groups=len(records))
        del packet,c,old,predictions,paths,rows,choices;source.gc.collect();guard()
    if len(records)!=138:raise ValueError('All138supported group views required')
    ci=lambda fn:source.group_interval(records,fn,sorted(set(data['sites'])),cfg)
    paired={name:{p:{m:ci(lambda r,n=name,p=p,m=m:r['paired'][n][p][m])for m in
        ('known_pp','completion_lower_pp','completion_upper_pp')}for p in parent.POLICIES}for name in contrasts}
    safety={};known={};lower={}
    for arm in ARMS:
        safety[arm]={};known[arm]={};lower[arm]={}
        for policy in parent.POLICIES:
            known[arm][policy]=ci(lambda r,a=arm,p=policy:r['policies'][a][p]['known_ADE_gain_floor_percent'])
            lower[arm][policy]=ci(lambda r,a=arm,p=policy:r['policies'][a][p]['completion_lower_gain_floor_percent'])
            views=[v for r in records for v in r['policies'][arm][policy]['recordings'].values()]
            easy=[r['known_easy_net_degradation']for r in views if r['known_easy_net_degradation']is not None]
            safety[arm][policy]=dict(required_record_views=len(views),supported_record_views=sum(r['supported']for r in views),
                known_all_risk_violations=sum(r['known_all_risk']is not None and r['known_all_risk']>.02 for r in views),
                known_easy_risk_violations=sum(r['known_easy_risk']is not None and r['known_easy_risk']>.02 for r in views),
                undefined_easy_views=sum(r['completion_easy_risk_upper']is None for r in views),
                worst_known_easy_net_degradation_percent=100*max(easy)if easy else None,
                full_original_risk_pass=bool(views and all(r['supported']for r in views)))
    source.once(PUBLIC/'summary.json',source.stable(dict(paired=paired,known_accuracy=known,completion_lower=lower,safety=safety,
        group_receipts=receipts,original24group_primary='not_estimable_not_passed',
        primary='adaptive_minus_fixed_joint_free_completion_lower',independent_confirmation=False,
        new_deployment=False,new_neural_training=False,stage5c_executed=False,smc_enabled=False)))
    if verify:source.once(PUBLIC/'verification.json',dict(all72cost_outputs_and_policy_scalars_exact=True,
        cached_control_policies_exact_to_parent=True,saved_solver_bindings_and_feasibility_checked=True,
        bitwise_MILP_reoptimization_claimed=False,summary_sha256=source.sha(PUBLIC/'summary.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','evaluate','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    with(PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime_cfg=source.read(parent.CONFIG)
        spec=source.read(CONFIG)
        runtime_cfg.update({k:spec[k]for k in ('disk_reserve_bytes','private_artifact_cap_bytes')})
        start,guard,beat=source.runtime(runtime_cfg,'harm_refinement_'+args.phase)
        spec,cfg,data,jobs,refs,controls=register(beat)
        if args.phase=='register':return
        committed();receipt=PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():raise ValueError('Complete phase is immutable')
        if args.phase=='pilot':pilot(spec,cfg,data,jobs,refs,controls,beat,guard,args.resume)
        elif args.phase=='train':train(spec,cfg,data,jobs,refs,controls,beat,guard,args.resume)
        else:evaluate(spec,cfg,data,jobs,refs,controls,beat,guard,args.phase=='verify')
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_tree_training=args.phase in ('pilot','train'),new_neural_training=False,
            stage5c_executed=False,smc_enabled=False))
        beat(state='complete')


if __name__=='__main__':main()
