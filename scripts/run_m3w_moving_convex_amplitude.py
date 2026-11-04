"""Matched convex amplitude decoder versus verified frozen unconstrained heads."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_amplitude_cost as parent
from src.world_model import m3w_convex_amplitude_cost as api
from src.evaluation.m3w_amplitude_shape_audit import diagnose
from src.evaluation.m3w_selected_cost_audit import selected_costs
from src.world_model.m3w_multiaction_cost import select_actions

source=parent.source
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_convex_amplitude_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_convex_amplitude_v1'
CONFIG=ROOT/'configs/m3w_moving_convex_amplitude_v1.json'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_convex_amplitude_cost.py',
    ROOT/'tests/test_m3w_convex_amplitude_cost.py',ROOT/'src/world_model/m3w_selection_controller.py',
    ROOT/'src/world_model/m3w_bound_aware_cost.py')
POLICIES=parent.POLICIES


def register(beat):
    spec=source.read(CONFIG);cfg=source.read(parent.CONFIG)
    if (spec['new_heads']!=18 or spec['cached_matched_control_heads']!=18
            or spec['new_primary_updates']!=36000 or spec['shape_schema']!='nested_nonnegative_hinges_v1'
            or any(spec[k]for k in ('threshold_search','new_data_roles','calibration_opened',
                'confirmation_opened','deployment_changed','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed one-factor decoder comparison required')
    shape=ROOT/'outputs/publication_readiness_2026_09/moving_amplitude_shape_v1'
    if not source.read(shape/'verify_execution.json')['all_scalars_exact']:
        raise ValueError('Finish the preceding fitting-only shape diagnosis and replay first')
    data,jobs,refs=parent.register(cfg,beat)
    controls={}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        row=source.read(source.checked(ref))
        if row['arm']=='amplitude_bank':
            source.checked(row['checkpoint']);controls[row['job']['key']]=row
    if len(controls)!=18:
        raise ValueError('All18matched frozen unconstrained controls required')
    binding=source.stable(dict(specification=spec,parent_configuration=cfg,
        bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        parent_registration=source.artifact(parent.REG),parent_training=source.artifact(parent.PUBLIC/'training_freeze.json'),
        shape_analysis=source.artifact(shape/'analysis.json'),shape_replay=source.artifact(shape/'verify_execution.json'),
        controls={k:v['checkpoint']for k,v in controls.items()},
        expected_new_heads=18,new_primary_updates=36000,control_result_source='cached_verified_not_new18fits',
        original24group_primary='not_estimable_not_passed',stage5c_executed=False,smc_enabled=False))
    source.once(REG,binding)
    return cfg,data,jobs,refs,controls


def check_committed():
    for p in (*FILES,REG):
        old=source.subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,
            capture_output=True,check=True).stdout
        if source.hashlib.sha256(old).hexdigest()!=source.sha(p):
            raise ValueError('Commit fixed definition before training')


def path_for(job,kind='fits'):
    return PRIVATE/kind/(job['key']+'.pt.gz')


def fit(packet,job,cfg,beat,guard,*,kind='fits',resume=False,stop_at=None):
    return api.fit(packet['learning'],identity=dict(registration_sha256=source.sha(REG),job=job),
        seed=job['seed'],settings=cfg['training'],path=path_for(job,kind),resume=resume,
        stop_at=stop_at,guard=guard,heartbeat=lambda **v:beat(job=job['key'],**v))


def pilot(cfg,data,jobs,refs,controls,beat,guard,resume):
    cells=[]
    for fold in range(3):
        job=next(j for j in jobs if j['critic_fold']==fold)
        packet=parent.view(data,job,refs,training=True)
        path=path_for(job,'pilot_resume')
        if path.exists()and not resume:
            raise ValueError('Explicit same-binding pilot resume required')
        previous=api.parent.base.base.read_checkpoint(path)['step']if path.exists()else 0
        if previous<50:
            fit(packet,job,cfg,beat,guard,kind='pilot_resume',resume=path.exists(),stop_at=50)
        a=fit(packet,job,cfg,beat,guard,kind='pilot_resume',resume=True,stop_at=100)
        b=fit(packet,job,cfg,beat,guard,kind='pilot_direct',
            resume=resume and path_for(job,'pilot_direct').exists(),stop_at=100)
        source.recovered(a,b)
        old=api.parent.base.base.read_checkpoint(parent.state_path(job,'amplitude_bank','pilot_direct'))
        for key in parent.MATCHED:api.parent.base.base.exact(a[key],old[key])
        cells.append(dict(job=job['key'],step=a['step'],seconds=b['seconds'],
            exact_resume=True,initial_states_and100step_draws_equal_original_control=True))
        del packet,a,b,old
        source.gc.collect();guard()
    source.once(PUBLIC/'pilot.json',source.stable(dict(cells=cells,real_Torch_training=True,
        all3cells_exact=True,projection_seconds_not_completed=6*sum(r['seconds']*20 for r in cells))))


def train(cfg,data,jobs,refs,controls,beat,guard,resume):
    if not source.read(PUBLIC/'pilot.json')['all3cells_exact']:
        raise ValueError('Complete the real three-fold recovery pilot')
    fits=[]
    for job in jobs:
        packet=parent.view(data,job,refs,training=True)
        old=api.parent.base.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        state=fit(packet,job,cfg,beat,guard,resume=resume and path_for(job).exists())
        for key in parent.MATCHED:api.parent.base.base.exact(state[key],old[key])
        if job==jobs[0]:
            refit=fit(packet,job,cfg,beat,guard,kind='full_refit',
                resume=resume and path_for(job,'full_refit').exists())
            source.recovered(state,refit)
            out=PRIVATE/'control_refit.pt.gz'
            control=api.parent.fit(packet['learning'],arm='amplitude_bank',identity=old['identity'],
                seed=job['seed'],settings=cfg['training'],path=out,
                resume=resume and out.exists(),guard=guard,heartbeat=lambda **v:beat(state_kind='control_refit',**v))
            source.recovered(old,control)
            source.once(PUBLIC/'first_full_refits.json',dict(convex_exact=True,unconstrained_exact_to_original=True,
                step=state['step'],convex=source.artifact(path_for(job,'full_refit')),control=source.artifact(out)))
            del refit,control
        c,ids=packet['causal'],packet['ids']
        prediction=api.predict(state,c)
        old_prediction=api.predict(old,c)
        if source.array_hash(old_prediction)!=controls[job['key']]['fitting_predictions_sha256']:
            raise ValueError('Fresh control fitting inference differs from frozen receipt')
        choice,_=select_actions(prediction,c['allowed'],arm='physical_neural')
        truth=api.parent.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
        shape=diagnose(prediction,truth['moments'],truth['observed_future'],choice,c['envelope'])
        if any(v for comp in shape['predicted'].values()for v in comp.values()):
            raise ValueError('Constrained head violates its claimed numerical shape')
        receipt=dict(job=job,step=state['step'],parameters=state['parameters'],checkpoint=source.artifact(path_for(job)),
            seconds=state['seconds'],draw_hash=state['draw_hash'],minimum_query_visits=int(state['sampler']['counts'].min()),
            initial_trace=state['trace'][0],final_trace=state['trace'][-1],
            fitting_predictions_sha256=source.array_hash(prediction),fitting_shape=shape,
            own_selected_costs=selected_costs(prediction,truth['moments'],truth['observed_future'],choice,c['groups']),
            cached_control_checkpoint=controls[job['key']]['checkpoint'],paired_scientific_initial_states_and_draws_exact=True)
        output=PUBLIC/'fits'/(job['key']+'.json');source.once(output,source.stable(receipt));fits.append(source.artifact(output))
        beat(state='convex_head_complete',job=job['key'],complete=len(fits),total=18)
        del packet,old,state,c,prediction,old_prediction,truth,shape,receipt
        source.gc.collect();guard()
    source.once(PUBLIC/'training_freeze.json',dict(fits=fits,complete_new_heads=18,new_primary_updates=36000,
        cached_control_heads=18,all18initial_states_and_draws_exact=True,real_Torch_training=True))


def evaluate(cfg,data,jobs,refs,controls,beat,guard,*,verify):
    freeze=source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_new_heads']!=18 or not freeze['all18initial_states_and_draws_exact']:
        raise ValueError('Freeze complete matched training first')
    models={}
    for ref in freeze['fits']:
        r=source.read(source.checked(ref));models[r['job']['key']]=r['checkpoint']
    records=[];receipts=[]
    for job in jobs:
        packet=parent.view(data,job,refs,training=False);c,ids=packet['causal'],packet['ids']
        old=api.parent.base.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        new=api.parent.base.base.read_checkpoint(source.checked(models[job['key']]))
        predictions=dict(unconstrained=api.predict(old,c),convex=api.predict(new,c))
        parent_group=source.read(parent.PUBLIC/'groups'/(job['key']+'.json'))
        if source.array_hash(predictions['unconstrained'])!=parent_group['cost_hashes']['amplitude_bank']:
            raise ValueError('Original held control inference changed')
        source.PRIVATE=parent.PRIVATE
        try:
            choices_old,_=source.allocated(predictions['unconstrained'],c,packet,job,'amplitude_bank',cfg,beat,verify=True)
        finally:
            source.PRIVATE=PRIVATE
        choices_new,_=source.allocated(predictions['convex'],c,packet,job,'convex',cfg,beat,verify=verify)
        paths={a:source.policy_paths(packet['paths'],packet['floor'],choices)for a,choices in
            (('unconstrained',choices_old),('convex',choices_new))}
        rows=[]
        for site in sorted(set(packet['sites'])):
            at=packet['sites']==site;target=data['target_eval'][ids[at]]-data['origin'][ids[at],None];valid=data['valid'][ids[at]]
            policies={a:{p:source.readout.policy(x[at],packet['floor'][at],packet['cv'][at],target,valid,
                packet['recordings'][at],easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])for p,x in variants.items()}
                for a,variants in paths.items()}
            legacy=next(r for r in parent_group['rows']if r['site']==str(site))['policies']['amplitude_bank']
            api.parent.base.base.exact(source.stable(policies['unconstrained']),source.stable(legacy))
            paired={p:source.readout.paired(paths['unconstrained'][p][at],paths['convex'][p][at],packet['floor'][at],
                target,valid)for p in POLICIES}
            groups=source.query_groups(packet['recordings'][at],packet['frames'][at],ids[at])
            matched={a:source.matched_utility(paths[a]['independent'][at],paths[a]['joint_matched'][at],packet['floor'][at],
                target,valid,choices['independent'][at],choices['joint_matched'][at],groups)
                for a,choices in (('unconstrained',choices_old),('convex',choices_new))}
            amplitudes={};grid=source.np.array(api.ALPHA)
            for arm,choices in (('unconstrained',choices_old),('convex',choices_new)):
                amplitudes[arm]={}
                for policy,choice in choices.items():
                    selected=choice[at];changed=selected>=0
                    sizes=source.np.where(changed,grid[source.np.maximum(selected,0)],0.)
                    amplitudes[arm][policy]=dict(action_counts=source.np.bincount(selected+1,minlength=8).tolist(),
                        mean_amplitude_all_rows=float(sizes.mean()),selected=int(changed.sum()),rows=len(selected))
            rows.append(dict(site=str(site),producer_fold=job['producer_fold'],seed=job['seed'],
                policies=policies,paired_convex_minus_unconstrained=paired,matched_utility=matched,amplitudes=amplitudes))
        output=PUBLIC/'groups'/(job['key']+'.json')
        source.once(output,source.stable(dict(job=job,rows=rows,cost_hashes={a:source.array_hash(v)for a,v in predictions.items()})))
        receipts.append(source.artifact(output));records.extend(rows)
        beat(state='convex_held_replayed'if verify else'convex_held_evaluated',job=job['key'],groups=len(records))
        del packet,c,old,new,predictions,paths,rows
        source.gc.collect();guard()
    if len(records)!=138:raise ValueError('Complete supported source matrix required')
    ci=lambda fn:source.group_interval(records,fn,sorted(set(data['sites'])),cfg)
    primary={p:{m:ci(lambda r,p=p,m=m:r['paired_convex_minus_unconstrained'][p][m])for m in
        ('known_pp','completion_lower_pp','completion_upper_pp')}for p in POLICIES}
    safety={};known={};lower={}
    for arm in ('unconstrained','convex'):
        safety[arm]={};known[arm]={};lower[arm]={}
        for policy in POLICIES:
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
    source.once(PUBLIC/'summary.json',source.stable(dict(primary_convex_minus_unconstrained=primary,known_accuracy=known,
        completion_lower=lower,safety=safety,group_receipts=receipts,original24group_primary='not_estimable_not_passed',
        control_result_source='cached_verified_choices_with_fresh_exact_cost_and_scalar_replay',
        independent_confirmation=False,new_deployment=False,stage5c_executed=False,smc_enabled=False)))
    if verify:source.once(PUBLIC/'verification.json',dict(all36cost_outputs_and_policy_scalars_exact=True,
        cached_control_policies_exact_to_parent=True,saved_solver_bindings_and_feasibility_checked=True,
        bitwise_MILP_reoptimization_claimed=False,summary_sha256=source.sha(PUBLIC/'summary.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','evaluate','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    with(PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX | source.fcntl.LOCK_NB)
        start,guard,beat=source.runtime(source.read(parent.CONFIG),'convex_amplitude_'+args.phase)
        cfg,data,jobs,refs,controls=register(beat)
        if args.phase=='register':return
        check_committed()
        receipt=PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():raise ValueError('Complete experiment phase is immutable')
        if args.phase=='pilot':pilot(cfg,data,jobs,refs,controls,beat,guard,args.resume)
        elif args.phase=='train':train(cfg,data,jobs,refs,controls,beat,guard,args.resume)
        else:evaluate(cfg,data,jobs,refs,controls,beat,guard,verify=args.phase=='verify')
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_Torch_training=args.phase in ('pilot','train'),stage5c_executed=False,smc_enabled=False))
        beat(state='complete')


if __name__=='__main__':main()
