"""Frozen forecast-tail attribution and positive-cost source-held probes."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before numerical imports')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_cost_partition as old
from src.evaluation import m3w_forecast_tail_probe as method
from src.evaluation.m3w_amplitude_opportunity import FRACTIONS

source, rel, np = old.source, old.rel, old.np
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_forecast_tail_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_forecast_tail_v1'
CONFIG=ROOT/'configs/m3w_moving_forecast_tail_v1.json'; REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/evaluation/m3w_forecast_tail_probe.py',
    ROOT/'tests/test_m3w_forecast_tail_probe.py')


def register(beat):
    cfg=source.read(CONFIG); _,values=old.register(beat)
    if (cfg['features']!='summary120' or cfg['fresh_fits']!=8 or cfg['cached_controls']!=8
            or cfg['primary_ranking_fraction']!=.1 or cfg['minimum_capture']!=.5
            or cfg['minimum_gain_over_both_controls']!=.05 or cfg['minimum_passing_contexts']!=2
            or cfg['risk_budget']!=.02 or cfg['easy_net_budget']!=.02
            or any(cfg[k] for k in ('threshold_search','new_H_readout','identity_certification',
                'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed causal tail diagnostic and unchanged original rules required')
    rel.base.exact(source.read(old.PUBLIC/'verification.json')['analysis_sha256'],source.sha(old.PUBLIC/'analysis.json'))
    source.once(REG,source.stable(dict(config=cfg,tree=method.parent.TREE,sklearn_version=old.previous.sklearn.__version__,
        files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        parent_registration=source.artifact(old.REG),parent_analysis=source.artifact(old.PUBLIC/'analysis.json'),
        jobs=values[3],partitions={j['key']:values[5][j['key']] for j in values[3]},
        result_role='exposed_own_B_source_held_information_diagnostic',new_V_or_H=False,
        original_rules_unchanged=True,identity_certification=False,independent_confirmation=False)))
    return cfg,values


def inputs(values,job,guard):
    d=old.previous.inputs(values,job,guard)
    data,_=old.previous.parent.inputs(values,job); c=data['causal']
    rows,actions=np.nonzero(c['allowed']); known=data['observed_future'][rows]
    rel.base.exact(c['ids'][rows][known],d['ids'])
    x=c['x'][rows,actions]; h=x[:,:24].reshape(-1,8,3)[...,:2].astype(float)
    delta=np.diff(h,axis=1); speed=np.linalg.norm(delta,axis=-1); length=speed.sum(-1)
    p=x[:,330:354].reshape(-1,12,2).astype(float); floor=x[:,306:330].reshape(-1,12,2).astype(float)
    masks=x[:,224:288].reshape(-1,8,8)
    geometry=dict(past_mean_speed=speed.mean(1),
        past_mean_acceleration=np.linalg.norm(np.diff(delta,axis=1),axis=-1).mean(1),
        past_straightness=np.divide(np.linalg.norm(h[:,-1]-h[:,0],axis=-1),length,
            out=np.zeros_like(length),where=length>1e-12),
        candidate_floor_envelope=np.linalg.norm(p-floor,axis=-1).max(-1),
        neighbor_count=((masks[:,:,-1]>0)&(masks[:,:,-2]>0)).sum(1).astype(float))
    moments=data['truth_by_arm']['bound_aware'].numpy()[rows,actions][known].astype(float)
    np.testing.assert_allclose(moments[:,[1,4]]-.02*moments[:,[2,3]],d['target'],rtol=1e-5,atol=1e-8)
    d.update(actions=actions[known],moments=moments,geometry={k:v[known] for k,v in geometry.items()},
        all_geometry=geometry,all_sites=c['sites'][rows],all_known=known,
        amplitude=np.asarray((0.,*FRACTIONS))[actions[known]])
    d['audit'].update(geometry_before_label_filter=True,geometry_hashes={k:rel.parent.kin.digest(v) for k,v in geometry.items()},
        action_hash=rel.parent.kin.digest(d['actions']),moments_hash=rel.parent.kin.digest(moments),
        future_inference_input=False,fit_role='own_B_only')
    del data,c,x; source.gc.collect(); guard()
    return d


def attribution(d,site):
    train,held=d['sites']!=site,d['sites']==site
    by_axis={}
    for name,values in d['geometry'].items():
        cut=np.quantile(d['all_geometry'][name][d['all_sites']!=site],[.25,.5,.75])
        bins=np.searchsorted(cut,values[held],side='right')
        by_axis[name]=dict(fitting_causal_cuts=cut.tolist(),
            rows=method.group_masses(d['target'][held],d['weights'][held],bins))
    return dict(by_amplitude=method.group_masses(d['target'][held],d['weights'][held],d['amplitude'][held]),
        causal_slices=by_axis,known_easy_rows=int((d['moments'][held,3]>0).sum()),
        unknown_proposals=int(((d['all_sites']==site)&~d['all_known']).sum()),
        not_policy_risk_ratios=True)


def cell(d,job,site,guard,*,resume=False,verify=False,replay=False,kind='fits'):
    train,held=d['sites']!=site,d['sites']==site; x,y,w=d['xs']['summary120'],d['target'],d['weights']
    binding=dict(registration_sha256=source.sha(REG),job=job['key'],held_source=str(site),
        feature_hash=rel.parent.kin.digest(x),target_hash=rel.parent.kin.digest(y),
        train_ids_hash=rel.parent.kin.digest(d['ids'][train]),held_ids_hash=rel.parent.kin.digest(d['ids'][held]))
    path=PRIVATE/kind/(job['key']+'_'+str(site)+'.pt.gz'); saved=None
    if path.exists():
        if not (resume or verify or replay):
            raise ValueError('Existing fit needs explicit resume or replay')
        saved=rel.base.read_checkpoint(path); rel.base.exact(saved['binding'],binding)
    elif verify or replay:
        raise ValueError('No completed fit to verify')
    if saved is None or verify:
        guard(); start=time.monotonic(); prediction,state=method.fit(x,y,train,w)
        fitted=dict(binding=binding,state=state,prediction=prediction,seconds=time.monotonic()-start)
        if saved is None:
            path.parent.mkdir(parents=True,exist_ok=True); rel.base.save_checkpoint(path,fitted); saved=fitted
        else:
            rel.base.exact(method.parent.fingerprint(state),method.parent.fingerprint(saved['state']))
            rel.base.exact(prediction,saved['prediction'])
    control,control_ref=old.cached_control(d,job,site,refit=verify)
    new=method.parent.predict(saved['state'],x[held]); rel.base.exact(new,saved['prediction'])
    scores=dict(positive_tail=new,signed_risk=method.parent.predict(control,x[held]))
    fit_scores=dict(positive_tail=method.parent.predict(saved['state'],x[train]),
        signed_risk=method.parent.predict(control,x[train]))
    return source.stable(dict(**binding,checkpoint=source.artifact(path),control=control_ref,seconds=saved['seconds'],
        rows=int(held.sum()),queries=len(set(d['query'][held])),train_sources=sorted(set(d['sites'][train])),
        prediction_hash=rel.parent.kin.digest(new),
        positive_tail_result_source='fresh_run',signed_control_result_source='cached_verified',
        ranking=method.score_table(y[held],w[held],scores,d['geometry']['candidate_floor_envelope'][held]),
        fitting_ranking=method.score_table(y[train],w[train],fit_scores,d['geometry']['candidate_floor_envelope'][train]),
        attribution=attribution(d,site)))


def pilot(cfg,values,guard,beat,resume):
    job=values[3][-1]; d=inputs(values,job,guard); site=sorted(set(d['sites']))[0]
    a=cell(d,job,site,guard,resume=resume,kind='pilot')
    b=cell(d,job,site,guard,verify=True,kind='pilot'); rel.base.exact(a,b)
    source.once(PUBLIC/'pilot.json',dict(model=a,full_forest_repeat_exact=True,local_fit_reasonable=True,
        seconds_per_fit=a['seconds'],fresh_fits=2,cached_control_refits=1))
    beat(state='pilot_complete',seconds_per_fit=a['seconds'])


def execute(cfg,values,guard,beat,*,resume=False,verify=False,replay=False):
    if not source.read(PUBLIC/'pilot.json')['full_forest_repeat_exact']:
        raise ValueError('Real-data pilot required')
    contexts=[]; complete=0
    for job in values[3]:
        d=inputs(values,job,guard); folds=[]
        for site in sorted(set(d['sites'])):
            record=cell(d,job,site,guard,resume=resume,verify=verify,replay=replay)
            file=PUBLIC/'fits'/(job['key']+'_'+str(site)+'.json')
            if replay:
                rel.base.exact(source.read(file),record)
            else:
                source.once(file,record)
            folds.append(record); complete+=1
            beat(state='tail_probe_replay' if replay else 'tail_probe_verified' if verify else 'tail_probe_fit',
                complete=complete,total=8,job=job['key'],source=str(site),seconds=record['seconds'])
        contexts.append(source.stable(dict(**d['audit'],folds=folds,summary=method.source_summary(folds),
            source_mass=method.group_masses(d['target'],d['weights'],d['sites']),
            amplitude_mass=method.group_masses(d['target'],d['weights'],d['amplitude']))))
        del d; source.gc.collect(); guard()
    passing=sum(c['summary']['passes_information_screen'] for c in contexts)
    result=source.stable(dict(status='fresh_run_source_held_positive_tail_information_test',contexts=contexts,
        fresh_fits=complete,cached_controls=complete,passing_contexts=passing,
        tail_branch_followup_supported=passing>=cfg['minimum_passing_contexts'],
        new_neural_training=False,new_trajectory_training=False,new_V_or_H=False,
        original24group_primary='not_estimable_not_passed',original_rules_unchanged=True,
        identity_certification=False,independent_confirmation=False,policy_evaluation='not_run_not_a_policy',
        deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay:
        rel.base.exact(source.read(PUBLIC/'analysis.json'),result)
    else:
        source.once(PUBLIC/'analysis.json',result)
    if verify:
        source.once(PUBLIC/'verification.json',dict(all8_fresh_and8_control_refits_exact=True,
            analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','pilot','run','verify','replay'))
    p.add_argument('--resume',action='store_true'); args=p.parse_args()
    PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime=old.previous.old.run.old.old.previous.teacher.lineage; runtime.PRIVATE=PRIVATE
        start,guard,beat=runtime.runtime({},source.read(CONFIG),args.phase)
        before={str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()} if args.phase=='replay' else None
        cfg,values=register(beat)
        if args.phase=='register':
            return
        for file in (*FILES,REG):
            content=source.subprocess.run(['git','show','HEAD:'+str(file.relative_to(ROOT))],
                cwd=ROOT,capture_output=True,check=True).stdout
            if source.hashlib.sha256(content).hexdigest()!=source.sha(file):
                raise ValueError('Commit fixed design before real-data fitting')
        receipt=(PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        if receipt.exists():
            raise ValueError('Completed phase immutable; inspect existing receipt')
        if args.phase=='pilot':
            pilot(cfg,values,guard,beat,args.resume)
        else:
            if args.phase=='replay':
                def forbidden(*args,**kwargs):
                    raise RuntimeError('Model fitting forbidden during replay')
                method.fit=forbidden; method.parent.fit_predict=forbidden
            execute(cfg,values,guard,beat,resume=args.resume,verify=args.phase=='verify',replay=args.phase=='replay')
        unchanged=None
        if before is not None:
            rel.base.exact(before,{str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()})
            unchanged=len(before)
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            unchanged_public_files=unchanged,original_rules_unchanged=True,new_V_or_H=False))
        beat(state='complete')


if __name__=='__main__':
    main()
