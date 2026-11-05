"""Fixed ego-frame representation repair, original signed risk task retained."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_forecast_tail as parent
from src.world_model import m3w_ego_frame_basis as features

source,rel,np,method=parent.source,parent.rel,parent.np,parent.method.parent
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_ego_frame_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_ego_frame_v1'
CONFIG=ROOT/'configs/m3w_moving_ego_frame_v1.json'
BINDING_REG=PUBLIC/'registration.json'; REG=PUBLIC/'registration_v2.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_ego_frame_basis.py',
    ROOT/'tests/test_m3w_ego_frame_basis.py',ROOT/'tests/test_m3w_ego_frame_reporting.py',PUBLIC/'serialization_repair.md')


def register(beat):
    cfg=source.read(CONFIG); _,values=parent.register(beat)
    if (cfg['arms']!=list(features.ARMS) or cfg['primary']!='history121_minus_original120'
            or cfg['minimum_joint_cost_reduction']!=.01 or cfg['minimum_passing_contexts']!=2
            or cfg['risk_budget']!=.02 or cfg['easy_net_budget']!=.02
            or any(cfg[k] for k in ('threshold_search','new_H_readout','identity_certification',
                'deployment_changed','submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed representation contrast, unchanged signed targets and rules required')
    rel.base.exact(source.read(parent.PUBLIC/'verification.json')['analysis_sha256'],source.sha(parent.PUBLIC/'analysis.json'))
    source.once(REG,source.stable(dict(config=cfg,tree=method.TREE,files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        parent=source.artifact(parent.REG),negative_tail_probe=source.artifact(parent.PUBLIC/'analysis.json'),
        unchanged_scientific_binding=source.artifact(BINDING_REG),
        jobs=values[3],partitions={j['key']:values[5][j['key']] for j in values[3]},
        result_role='exposed_own_B_source_held_model_input_repair',fresh_fits=16,cached_controls=8,
        original_rules_unchanged=True,new_V_or_H=False,identity_certification=False,independent_confirmation=False)))
    return cfg,values


def inputs(values,job,guard):
    d=parent.old.previous.inputs(values,job,guard)
    data,_=parent.old.previous.parent.inputs(values,job); c=data['causal']
    generated,ratio=features.chunked(c['x'],c['scales']); row,act=np.nonzero(c['allowed']&data['observed_future'][:,None])
    rel.base.exact(c['ids'][row],d['ids'])
    d['xs']={'original120':d['xs']['summary120'],**{k:v[row,act].astype(float) for k,v in generated.items()}}
    audit=[]
    for site in sorted(set(c['sites'])):
        at=(c['sites']==site)&c['allowed'].any(-1)
        audit.append(dict(source=str(site),proposed_rows=int(at.sum()),
            ego_to_context_scale_quantiles=np.quantile(ratio[at],[0,.25,.5,.75,1]).tolist(),
            history_scale_smaller_fraction=float((ratio[at]<1-1e-5).mean())))
    d['audit'].update(new_feature_hashes={k:rel.parent.kin.digest(v) for k,v in d['xs'].items()},
        causal_scale_ratio_hash=rel.parent.kin.digest(ratio),geometry_before_label_filter=True,scale_audit=audit,
        signed_targets_unchanged=True)
    del data,c,generated; source.gc.collect(); guard(); return d


def cell(d,job,site,arm,guard,*,resume=False,verify=False,replay=False,kind='fits'):
    train,held=d['sites']!=site,d['sites']==site; x=d['xs'][arm]; y,w=d['target'],d['weights']
    identity=dict(registration_sha256=source.sha(BINDING_REG),job=job['key'],source=str(site),arm=arm,
        x_hash=rel.parent.kin.digest(x),target_hash=rel.parent.kin.digest(y),
        train_ids_hash=rel.parent.kin.digest(d['ids'][train]),held_ids_hash=rel.parent.kin.digest(d['ids'][held]))
    common=dict(d,xs={'summary120':d['xs']['original120']})
    if arm=='original120':
        state,ref=parent.old.cached_control(common,job,site,refit=verify); seconds=None
    else:
        path=PRIVATE/kind/(job['key']+'_'+str(site)+'_'+arm+'.pt.gz'); saved=None
        if path.exists():
            if not(resume or verify or replay): raise ValueError('Explicit resume or verification required')
            saved=rel.base.read_checkpoint(path); rel.base.exact(saved['identity'],identity)
        elif verify or replay: raise ValueError('Absent model cannot be replayed')
        if saved is None or verify:
            guard(); start=time.monotonic()
            prediction,state=method.fit_predict(x,y,train,held,w,'extra_trees')
            fitted=dict(identity=identity,state=state,prediction=prediction,seconds=time.monotonic()-start)
            if saved is None:
                path.parent.mkdir(parents=True,exist_ok=True); rel.base.save_checkpoint(path,fitted); saved=fitted
            else:
                rel.base.exact(method.fingerprint(saved['state']),method.fingerprint(state))
                rel.base.exact(saved['prediction'],prediction)
        state,seconds,ref=saved['state'],saved['seconds'],source.artifact(path)
        rel.base.exact(saved['prediction'],method.predict(state,x[held]))
    pred=method.predict(state,x[held])
    return source.stable(dict(**identity,execution_registration_sha256=source.sha(REG),
        checkpoint=ref,seconds=seconds,rows=int(held.sum()),
        queries=len(set(d['query'][held])),train_sources=sorted(set(d['sites'][train])),
        result_source='cached_verified' if arm=='original120' else 'fresh_run',prediction_hash=rel.parent.kin.digest(pred),
        target_scale=state['target_scale'].tolist(),metrics=method.previous.metrics(pred,y[held],w[held],state['target_scale']),
        fitting_metrics=method.previous.metrics(method.predict(state,x[train]),y[train],w[train],state['target_scale'])))


def summarize(folds):
    names=('normalized_MSE','total_joint_cost','false_safe_cost','false_reject_cost')
    means={a:{k:float(np.mean([f['arms'][a]['metrics'][k] for f in folds])) for k in names}
        for a in ('original120',*features.ARMS)}
    old,new=means['original120'],means['history121']
    gain=1-new['total_joint_cost']/old['total_joint_cost'] if old['total_joint_cost']>0 else None
    return dict(source_macro=means,primary_cost_reduction=gain,
        primary_MSE_reduction=1-new['normalized_MSE']/old['normalized_MSE'] if old['normalized_MSE']>0 else None,
        passes_information_screen=bool(gain is not None and gain>=.01 and new['false_safe_cost']<=old['false_safe_cost']),
        safety_certificate=False)


def execute(cfg,values,guard,beat,*,resume=False,verify=False,replay=False):
    if not source.read(PUBLIC/'pilot.json')['exact_repeat']: raise ValueError('Real-data pilot required')
    contexts=[]; count=0
    for job in values[3]:
        d=inputs(values,job,guard); folds=[]
        for site in sorted(set(d['sites'])):
            f=dict(source=str(site),arms={})
            for arm in ('original120',*features.ARMS):
                record=cell(d,job,site,arm,guard,resume=resume,verify=verify,replay=replay)
                f['arms'][arm]=record; count+=1
                if arm!='original120': rel.base.exact(record['target_scale'],f['arms']['original120']['target_scale'])
                file=PUBLIC/'fits'/(job['key']+'_'+str(site)+'_'+arm+'.json')
                if replay: rel.base.exact(source.read(file),record)
                else: source.once(file,record)
                beat(state='frame_replay' if replay else 'frame_verify' if verify else 'frame_fit',complete=count,total=24)
            folds.append(f)
        contexts.append(dict(**d['audit'],folds=folds,summary=summarize(folds)))
        del d; source.gc.collect(); guard()
    passing=sum(c['summary']['passes_information_screen'] for c in contexts)
    result=source.stable(dict(status='fresh_run_source_held_ego_frame_repair',contexts=contexts,fresh_fits=16,cached_controls=8,
        passing_contexts=passing,neural_expansion_supported=passing>=2,new_neural_training=False,new_V_or_H=False,
        original24group_primary='not_estimable_not_passed',original_rules_unchanged=True,identity_certification=False,
        independent_confirmation=False,deployment_changed=False,submission_ready=False,stage5c_executed=False,smc_enabled=False))
    if replay: rel.base.exact(source.read(PUBLIC/'analysis.json'),result)
    else: source.once(PUBLIC/'analysis.json',result)
    if verify: source.once(PUBLIC/'verification.json',dict(all24_refits_exact=True,all_target_scales_identical=True,
        analysis_sha256=source.sha(PUBLIC/'analysis.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('register','pilot','run','verify','replay'))
    p.add_argument('--resume',action='store_true'); args=p.parse_args(); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime=parent.old.previous.old.run.old.old.previous.teacher.lineage; runtime.PRIVATE=PRIVATE
        start,guard,beat=runtime.runtime({},source.read(CONFIG),args.phase)
        before={str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()} if args.phase=='replay' else None
        cfg,values=register(beat)
        if args.phase=='register': return
        for file in (*FILES,REG):
            b=source.subprocess.run(['git','show','HEAD:'+str(file.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if source.hashlib.sha256(b).hexdigest()!=source.sha(file): raise ValueError('Commit fixed design before fitting')
        receipt=(PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        if receipt.exists(): raise ValueError('Completed phase immutable')
        if args.phase=='pilot':
            job=values[3][-1]; d=inputs(values,job,guard); site=sorted(set(d['sites']))[0]
            a=cell(d,job,site,'history121',guard,resume=args.resume,kind='pilot')
            b=cell(d,job,site,'history121',guard,verify=True,kind='pilot'); rel.base.exact(a,b)
            source.once(PUBLIC/'pilot.json',dict(model=a,exact_repeat=True,local_fit_reasonable=True))
        else:
            if args.phase=='replay':
                def forbidden(*args,**kwargs): raise RuntimeError('Fitting forbidden on read-only replay')
                method.fit_predict=forbidden
            execute(cfg,values,guard,beat,resume=args.resume,verify=args.phase=='verify',replay=args.phase=='replay')
        unchanged=None
        if before is not None:
            rel.base.exact(before,{str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()}); unchanged=len(before)
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            unchanged_public_files=unchanged,new_V_or_H=False,original_rules_unchanged=True))
        beat(state='complete')


if __name__=='__main__': main()
