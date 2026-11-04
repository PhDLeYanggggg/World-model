"""Matched shared-context harm refinement on the frozen SOURCE forecast bank."""
import argparse
import os
from pathlib import Path
import platform
import signal
import shutil
import sys
import time

if platform.system()=='Darwin'and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before sklearn/Torch import')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sklearn
from scripts import run_m3w_moving_amplitude_cost as parent
from src.world_model import m3w_shared_harm_refinement as api
from src.evaluation.m3w_selected_cost_audit import selected_costs
from src.evaluation.m3w_verified_choice_cache import verified_choices

source=parent.source
OLD=ROOT/'outputs/publication_readiness_2026_09/moving_harm_refinement_v1'
OLD_PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_harm_refinement_v1'
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_shared_harm_refinement_v1'
CONFIG=ROOT/'configs/m3w_moving_shared_harm_refinement_v1.json'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_shared_harm_refinement.py',
    ROOT/'src/world_model/m3w_harm_refinement.py',ROOT/'src/evaluation/m3w_selected_cost_audit.py',
    ROOT/'src/evaluation/m3w_verified_choice_cache.py',ROOT/'tests/test_m3w_shared_harm_refinement.py',
    ROOT/'tests/test_m3w_verified_choice_cache.py')
FRESH=('frozen',*api.ARMS)
CACHED={f'cached_{a}':a for a in api.previous.ARMS}
ARMS=(*FRESH,*CACHED)


def runtime(cfg,spec,phase):
    source.torch.set_num_threads(cfg['torch_threads']);source.torch.set_num_interop_threads(cfg['interop_threads'])
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume unchanged round')))
    start=time.monotonic()
    def guard():
        if shutil.disk_usage(ROOT).free<spec['disk_reserve_bytes']:
            raise OSError(f"Keep registered {spec['disk_reserve_bytes']} byte disk reserve; preserve checkpoints")
        if time.monotonic()-start>cfg['hard_runtime_limit_seconds']:raise TimeoutError('12h phase boundary; resume')
    def beat(**values):
        guard();total=sum(p.stat().st_size for p in PRIVATE.rglob('*')if p.is_file())
        if total>spec['private_artifact_cap_bytes']:raise OSError('Registered private cap exceeded')
        event=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,
            elapsed_seconds=time.monotonic()-start,private_artifact_bytes=total,**values)
        source.atomic(PRIVATE/'heartbeat.json',event)
        with(PRIVATE/'events.jsonl').open('a')as stream:stream.write(source.json.dumps(event)+'\n')
        print(source.json.dumps(event),flush=True)
    beat(state='runtime_ready',machine=platform.machine(),device='cpu',threads=cfg['torch_threads'],
        interop=cfg['interop_threads'],workers=0,resource_probing=False)
    return start,guard,beat


def register(beat):
    spec=source.read(CONFIG);cfg=source.read(parent.CONFIG)
    if (spec['arms']!=list(api.ARMS)or spec['input_dimension']!=435 or spec['features']!=api.SCHEMA
            or spec['new_refiners']!=36 or spec['new_primary_tree_fits']!=720
            or spec['training']!=source.read(ROOT/'configs/m3w_moving_harm_refinement_v1.json')['training']
            or any(spec[k]for k in ('threshold_search','new_data_roles','calibration_opened','confirmation_opened',
                'deployment_changed','new_neural_training','stage5c_executed','smc_enabled'))):
        raise ValueError('Registered matched information comparison required')
    data,jobs,refs=parent.register(cfg,beat);controls={}
    for ref in source.read(parent.PUBLIC/'training_freeze.json')['fits']:
        row=source.read(source.checked(ref))
        if row['arm']=='amplitude_bank':source.checked(row['checkpoint']);controls[row['job']['key']]=row
    verification=source.read(OLD/'verification.json')
    if (len(controls)!=18 or not verification['all72cost_outputs_and_policy_scalars_exact']
            or verification['summary_sha256']!=source.sha(OLD/'summary.json')):
        raise ValueError('Complete frozen prior comparison and18base heads required')
    old_groups=source.read(OLD/'summary.json')['group_receipts']
    for ref in old_groups:source.checked(ref)
    source.once(REG,source.stable(dict(specification=spec,parent_configuration=cfg,
        bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        parent_registration=source.artifact(parent.REG),
        prior_verified_study={k:source.artifact(OLD/k)for k in
            ('summary.json','verification.json','solver_summary.json')},
        prior_group_receipts=old_groups,controls={k:r['checkpoint']for k,r in controls.items()},
        sklearn_version=sklearn.__version__,original24group_primary='not_estimable_not_passed',
        independent_confirmation=False,stage5c_executed=False,smc_enabled=False)))
    return spec,cfg,data,jobs,refs,controls


def committed():
    for p in (*FILES,REG):
        content=source.subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(content).hexdigest()!=source.sha(p):raise ValueError('Commit unchanged definition first')


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
        prediction=api.previous.parent.predict(old,packet['causal'])
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
    source.once(PUBLIC/'pilot.json',source.stable(dict(cells=cells,all6cells_exact=True,
        verification_tree_fits=48,projection_pure_tree_fit_seconds_not_completion=30*sum(r['seconds']for r in cells))))


def train(spec,cfg,data,jobs,refs,controls,beat,guard,resume):
    if not source.read(PUBLIC/'pilot.json')['all6cells_exact']:raise ValueError('Complete recovery pilot first')
    fits=[];common=[]
    for job in jobs:
        packet=parent.view(data,job,refs,training=True);c,ids=packet['causal'],packet['ids']
        old=api.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        frozen=api.previous.parent.predict(old,c)
        if source.array_hash(frozen)!=controls[job['key']]['fitting_predictions_sha256']:
            raise ValueError('Original fitting prediction changed')
        predictions=dict(frozen=frozen)
        truth=api.previous.parent.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
        for arm in api.ARMS:
            state=one(packet,frozen,job,arm,spec,beat,guard,resume=resume and state_path(job,arm).exists())
            if job==jobs[0]:
                refit=one(packet,frozen,job,arm,spec,beat,guard,kind='full_refit',
                    resume=resume and state_path(job,arm,'full_refit').exists())
                api.exact_refit(state,refit)
                source.once(PUBLIC/('full_refit_'+arm+'.json'),dict(exact=True,rounds=20,
                    checkpoint=source.artifact(state_path(job,arm,'full_refit'))))
            prediction=api.predict(state,c,frozen);predictions[arm]=prediction
            if api.previous.digest(prediction)!=state['prediction_sha256']:raise ValueError('Fitting replay changed')
            source.np.testing.assert_array_equal(prediction[:,:,[2,3]],frozen[:,:,[2,3]])
            choice=api.previous.select_actions(prediction,c['allowed'],arm='physical_neural')[0]
            receipt=source.stable(dict(job=job,arm=arm,rounds=state['round'],seconds=state['seconds'],
                feature_dimension=state['feature_dimension'],feature_schema=state['feature_schema'],
                checkpoint=source.artifact(state_path(job,arm)),tree_hashes=state['tree_hashes'],
                actual_tree_nodes=sum(t.tree_.node_count for t in state['trees']),first_trace=state['trace'][0],last_trace=state['trace'][-1],
                fitting_predictions_sha256=source.array_hash(prediction),
                own_selected_costs=selected_costs(prediction,truth['moments'],truth['observed_future'],choice,c['groups']),
                cached_base_checkpoint=controls[job['key']]['checkpoint'],new_neural_training=False))
            path=PUBLIC/'fits'/(job['key']+'_'+arm+'.json');source.once(path,receipt);fits.append(source.artifact(path))
            beat(state='refiner_complete',job=job['key'],arm=arm,complete=len(fits),total=36)
        choices={a:api.previous.select_actions(p,c['allowed'],arm='physical_neural')[0]for a,p in predictions.items()}
        cells=[dict(job=job['key'],cost_model=arm,region=region,scores=selected_costs(p,truth['moments'],
            truth['observed_future'],choice,c['groups']))for region,choice in choices.items()for arm,p in predictions.items()]
        path=PUBLIC/'fitting_common'/(job['key']+'.json')
        source.once(path,source.stable(dict(cells=cells,prediction_hashes={a:source.array_hash(p)for a,p in predictions.items()},
            reference_arrays_pointwise_unchanged=True,labels_C_only=True)));common.append(source.artifact(path))
        del packet,c,old,frozen,truth,state,prediction,predictions,choices,cells;source.gc.collect();guard()
    source.once(PUBLIC/'training_freeze.json',dict(fits=fits,complete_refiners=36,primary_tree_fits=720,
        common_region_cells=162,common_region_receipts=common,new_neural_training=False,
        cached_neural_heads=18,all_fits_before_held_evaluation=True))


def cost_diagnostics(prediction,choice_bank,c,packet,truth):
    result={}
    for policy,choice in choice_bank.items():
        before=source.array_hash(choice);records={}
        for record in sorted(set(packet['recordings'])):
            at=packet['recordings']==record
            records[str(record)]=selected_costs(prediction[at],truth['moments'][at],truth['observed_future'][at],
                choice[at],[source.np.arange(int(at.sum()))])
        result[policy]=dict(query_scores=selected_costs(prediction,truth['moments'],truth['observed_future'],
            choice,c['groups']),recordings=records,choice_sha256=before)
        if before!=source.array_hash(choice):raise ValueError('Post-decision labels mutated choices')
    return result


def evaluate(spec,cfg,data,jobs,refs,controls,beat,guard,verify):
    freeze=source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_refiners']!=36:raise ValueError('Complete all36fits before H evaluation')
    models={}
    for ref in freeze['fits']:
        r=source.read(source.checked(ref));source.checked(r['checkpoint']);models[r['job']['key'],r['arm']]=r['checkpoint']
    old_solvers={(r['job'],r['arm']):r for r in source.read(OLD/'solver_summary.json')}
    contrasts=dict(shared_minus_local=('local_zero','shared_motion'),shared_minus_frozen=('frozen','shared_motion'),
        local_minus_frozen=('frozen','local_zero'),**{f'shared_minus_{k}':(k,'shared_motion')for k in CACHED})
    records=[];receipts=[];costs=[];solver=[]
    for job in jobs:
        packet=parent.view(data,job,refs,training=False);c,ids=packet['causal'],packet['ids']
        old=api.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        predictions=dict(frozen=api.previous.parent.predict(old,c))
        original=source.read(parent.PUBLIC/'groups'/(job['key']+'.json'))
        older=source.read(OLD/'groups'/(job['key']+'.json'))
        if source.array_hash(predictions['frozen'])!=original['cost_hashes']['amplitude_bank']:
            raise ValueError('Original held cost prediction changed')
        for arm in api.ARMS:
            state=api.base.read_checkpoint(source.checked(models[job['key'],arm]))
            predictions[arm]=api.predict(state,c,predictions['frozen'])
            source.np.testing.assert_array_equal(predictions[arm][:,:,[2,3]],predictions['frozen'][:,:,[2,3]])
        source.PRIVATE=parent.PRIVATE
        try:old_choice,_=source.allocated(predictions['frozen'],c,packet,job,'amplitude_bank',cfg,beat,verify=True)
        finally:source.PRIVATE=PRIVATE
        choices=dict(frozen=old_choice)
        for arm in api.ARMS:choices[arm],_=source.allocated(predictions[arm],c,packet,job,arm,cfg,beat,verify=verify)
        for name,old_arm in CACHED.items():
            path=OLD_PRIVATE/'allocation'/(job['key']+'_'+old_arm+'.npz')
            choices[name],receipt=verified_choices(path,old_solvers[job['key'],old_arm]['sha256'],c,packet,
                cfg['joint_solver_seconds_per_query']);solver.append(dict(job=job['key'],arm=name,**receipt))
        for arm in FRESH:
            path=(parent.PRIVATE if arm=='frozen'else PRIVATE)/'allocation'/(job['key']+'_'+
                ('amplitude_bank'if arm=='frozen'else arm)+'.npz')
            with source.np.load(path,allow_pickle=False)as item:meta=source.json.loads(str(item['metadata_json']))
            solver.append(dict(job=job['key'],arm=arm,sha256=source.sha(path),queries=meta['completed_queries'],
                reasons=meta['reason_counts'],nonoptimal=meta['nonoptimal_queries'],
                result_source='cached_verified_fresh_cost_feasibility_replay'if arm=='frozen'else
                    'fresh_run_created_then_cached_verified_on_exact_replay'))
        paths={a:source.policy_paths(packet['paths'],packet['floor'],v)for a,v in choices.items()}
        hashes={a:source.array_hash(p)for a,p in predictions.items()}
        choice_hashes={a:{p:source.array_hash(v)for p,v in ch.items()}for a,ch in choices.items()}
        truth=api.previous.parent.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
        # All decisions exist before these evaluation-only target diagnostics.
        diag={a:cost_diagnostics(p,choices[a],c,packet,truth)for a,p in predictions.items()}
        if (hashes!={a:source.array_hash(p)for a,p in predictions.items()}or choice_hashes!=
                {a:{p:source.array_hash(v)for p,v in ch.items()}for a,ch in choices.items()}):
            raise ValueError('Labels changed predictions or decisions')
        dpath=PUBLIC/'held_costs'/(job['key']+'.json')
        source.once(dpath,source.stable(dict(job=job['key'],cost_hashes=hashes,choice_hashes=choice_hashes,
            diagnostics=diag,labels_used_after_all_decisions=True)));costs.append(source.artifact(dpath))
        rows=[]
        for site in sorted(set(packet['sites'])):
            at=packet['sites']==site;target=data['target_eval'][ids[at]]-data['origin'][ids[at],None];valid=data['valid'][ids[at]]
            policies={a:{p:source.readout.policy(x[at],packet['floor'][at],packet['cv'][at],target,valid,
                packet['recordings'][at],easy_cut=job['easy_cut'],hard_cut=job['hard_cut'])for p,x in variants.items()}
                for a,variants in paths.items()}
            expected=next(r for r in original['rows']if r['site']==str(site))['policies']['amplitude_bank']
            api.base.exact(source.stable(policies['frozen']),source.stable(expected))
            old_expected=next(r for r in older['rows']if r['site']==str(site))['policies']
            for name,tag in CACHED.items():api.base.exact(source.stable(policies[name]),source.stable(old_expected[tag]))
            paired={name:{p:source.readout.paired(paths[a][p][at],paths[b][p][at],packet['floor'][at],target,valid)
                for p in parent.POLICIES}for name,(a,b)in contrasts.items()}
            groups=source.query_groups(packet['recordings'][at],packet['frames'][at],ids[at]);matched={};usage={}
            for arm in ARMS:
                matched[arm]=source.matched_utility(paths[arm]['independent'][at],paths[arm]['joint_matched'][at],
                    packet['floor'][at],target,valid,choices[arm]['independent'][at],choices[arm]['joint_matched'][at],groups)
                take=choices[arm]['joint_free'][at]
                alpha=source.np.where(take>=0,source.np.asarray((0.,*api.previous.parent.FRACTIONS))[source.np.maximum(take,0)],0.)
                usage[arm]=dict(action_counts=source.np.bincount(take+1,minlength=8).tolist(),
                    mean_amplitude_all_rows=float(alpha.mean()),selected=int((take>=0).sum()),rows=len(take))
            rows.append(dict(site=str(site),producer_fold=job['producer_fold'],seed=job['seed'],
                policies=policies,paired=paired,matched_utility=matched,usage=usage))
        path=PUBLIC/'groups'/(job['key']+'.json')
        source.once(path,source.stable(dict(job=job,rows=rows,cost_hashes=hashes,choice_hashes=choice_hashes,
            prior387_scalar_blocks_exact=True,prior387_costs_freshly_inferred=False)))
        receipts.append(source.artifact(path));records.extend(rows)
        beat(state='shared_held_replayed'if verify else'shared_held_evaluated',job=job['key'],groups=len(records))
        del packet,c,ids,old,predictions,paths,rows,choices,diag,truth;source.gc.collect();guard()
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
    source.once(PUBLIC/'solver_summary.json',source.stable(solver))
    source.once(PUBLIC/'summary.json',source.stable(dict(paired=paired,known_accuracy=known,completion_lower=lower,
        safety=safety,group_receipts=receipts,cost_diagnostic_receipts=costs,
        original24group_primary='not_estimable_not_passed',primary='shared_minus_local_joint_free_completion_lower',
        fresh_cost_arrays=54,cached_prior387_choice_sets=54,independent_confirmation=False,
        new_deployment=False,new_neural_training=False,stage5c_executed=False,smc_enabled=False)))
    if verify:source.once(PUBLIC/'verification.json',dict(all54fresh_cost_outputs_and_all108choice_scalar_blocks_exact=True,
        prior387_scalar_blocks_exact_to_original=True,prior387_cost_prediction_freshly_reinferred=False,
        all54post_decision_diagnostics_exact=True,bitwise_MILP_reoptimization_claimed=False,
        summary_sha256=source.sha(PUBLIC/'summary.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','pilot','train','evaluate','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    with(PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        cfg=source.read(parent.CONFIG);spec=source.read(CONFIG);start,guard,beat=runtime(cfg,spec,args.phase)
        spec,cfg,data,jobs,refs,controls=register(beat)
        if args.phase=='register':return
        committed();receipt=PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():raise ValueError('Complete phase is immutable')
        if args.phase=='pilot':pilot(spec,cfg,data,jobs,refs,controls,beat,guard,args.resume)
        elif args.phase=='train':train(spec,cfg,data,jobs,refs,controls,beat,guard,args.resume)
        else:evaluate(spec,cfg,data,jobs,refs,controls,beat,guard,args.phase=='verify')
        source.once(receipt,dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            real_tree_training=args.phase in ('pilot','train'),new_neural_training=False,
            stage5c_executed=False,smc_enabled=False))
        beat(state='complete')


if __name__=='__main__':main()
