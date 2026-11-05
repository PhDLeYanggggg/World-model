"""Fixed-budget source-held neural risk transport, with exact recovery."""
import argparse
from pathlib import Path
import platform
import sys
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch before numerical imports')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_track_analogue as old
from src.world_model import m3w_neural_risk_transport as method
from src.evaluation.m3w_boundary_audit import summarize as boundary

source,rel,np=old.source,old.rel,old.np
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_neural_transport_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_neural_transport_v1'
CONFIG=ROOT/'configs/m3w_moving_neural_transport_v1.json'; REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_neural_risk_transport.py',
    ROOT/'tests/test_m3w_neural_risk_transport.py')
ARMS=('shared64','track32','ridge_plain','ridge_transport','neural_plain','neural_transport')


def register():
    cfg=source.read(CONFIG)
    if ({k:cfg[k] for k in method.SETTINGS}!=method.SETTINGS or cfg['risk_budget']!=.02
            or cfg['easy_net_budget']!=.02 or cfg['minimum_joint_cost_reduction']!=.01
            or cfg['minimum_passing_contexts']!=2 or cfg['primary']!='neural_transport_minus_shared64_and_neural_plain'
            or any(cfg[k] for k in ('threshold_search','identity_certification','new_H_readout','deployment_changed',
                'submission_ready','stage5c_executed','smc_enabled'))):
        raise ValueError('Fixed source-only model comparison and original rules required')
    a=source.read(old.PUBLIC/'analysis.json')
    rel.base.exact(source.read(old.PUBLIC/'verification.json')['analysis_sha256'],source.sha(old.PUBLIC/'analysis.json'))
    source.once(REG,source.stable(dict(config=cfg,files={str(f.relative_to(ROOT)):source.sha(f) for f in FILES},
        parent_registration=source.artifact(old.REG),parent_result=source.artifact(old.PUBLIC/'analysis.json'),
        attribution=source.artifact(ROOT/'outputs/publication_readiness_2026_09/moving_match_attribution_v1/analysis.json'),
        metadata=source.artifact(old.PUBLIC/'metadata.json'),
        role='exposed_own_B_source_held_neural_risk_information',original_rules_unchanged=True,
        new_V_or_H=False,identity_certification=False,independent_confirmation=False)))
    return cfg,a


def load(context):
    reference=source.read(old.previous.audit.PUBLIC/'analysis.json')
    ref=next(c for c in reference['contexts'] if c['job']==context['job'])
    return old.inputs(ref)[0],ref


def identity(d,job,site):
    fit=d['sites']!=site
    return dict(registration_sha256=source.sha(REG),job=job,source=site,
        train_ids=method.digest(d['ids'][fit]),held_ids=method.digest(d['ids'][~fit]),
        input_hash=method.digest(d['x']),known_supervision_hash=method.digest(d['target']))


def exact_fit(a,b):
    for k in ('settings','identity','preprocess','initial_model','model','optimizer','generator','torch_rng',
            'step','epoch','cursor','order','visits','trace','complete','parameters','fit_rows'):
        rel.base.exact(a[k],b[k])


def fold(d,context,f,guard,beat,*,resume=False,verify=False,replay=False):
    site=f['source']; stem=context['job']+'_'+site; path=PRIVATE/'fits'/(stem+'.pt.gz')
    binding=identity(d,context['job'],site)
    if replay or verify:
        saved=rel.base.read_checkpoint(path); rel.base.exact(saved['identity'],binding)
        if not saved['complete']: raise ValueError('Incomplete neural fit')
    else:
        saved=method.fit(d,site,method.SETTINGS,binding,path,resume=resume,heartbeat=beat,guard=guard)
    if verify:
        repeated=method.fit(d,site,method.SETTINGS,binding,PRIVATE/'verify'/(stem+'.pt.gz'),
            resume=resume,heartbeat=beat,guard=guard)
        exact_fit(saved,repeated)
    ridgepath=PRIVATE/'fits'/(stem+'_ridge.pt.gz')
    if ridgepath.exists():
        ridge=rel.base.read_checkpoint(ridgepath); rel.base.exact(ridge['identity'],binding)
        if verify: rel.base.exact(ridge['model'],method.fit_ridge(d,site))
    elif replay or verify: raise ValueError('Missing ridge fit')
    else:
        ridge=dict(identity=binding,model=method.fit_ridge(d,site)); rel.base.save_checkpoint(ridgepath,ridge)
    oldsaved=rel.base.read_checkpoint(source.checked(f['checkpoint']))
    r=oldsaved['result']; at=d['sites']==site; pos=r['donor_positions']
    if (d['sites'][pos]==site).any(): raise ValueError('Query source in reference bank')
    rel.base.exact(d['ids'][pos],r['donor_ids'])
    rel.base.exact(saved['preprocess']['target_scale'],d['target_scale'][at][0])
    outputs=dict(shared64=d['old_prediction'][at],track32=r['prediction'])
    train_metrics={}
    for family,state in (('ridge',ridge['model']),('neural',saved)):
        pred=method.predict(state,d['x'])
        outputs[family+'_plain']=pred[at]
        outputs[family+'_transport']=method.transport(pred[at],pred[pos],d['target'][pos],d['weights'][pos])
        train_metrics[family]=old.method.base.metrics(pred[~at],d['target'][~at],d['weights'][~at],saved['preprocess']['target_scale'])
    metrics={k:old.method.base.metrics(p,d['target'][at],d['weights'][at],saved['preprocess']['target_scale']) for k,p in outputs.items()}
    record=source.stable(dict(source=site,rows=int(at.sum()),fit_rows=int((~at).sum()),
        neural_checkpoint=source.artifact(path),ridge_checkpoint=source.artifact(ridgepath),
        reference_checkpoint=f['checkpoint'],parameters=saved['parameters'],epochs=saved['epoch'],steps=saved['step'],
        minimum_fitting_visits=int(saved['visits'].min()),seconds=saved['seconds'],loss_trace=saved['trace'],
        metrics=metrics,fitting_metrics=train_metrics,
        prediction_hashes={k:method.digest(p) for k,p in outputs.items()},
        boundary=boundary(d['target'][at],outputs['shared64'],outputs['neural_transport'],d['moments'][at],
            d['weights'][at],saved['preprocess']['target_scale']),
        result_source='fresh_run_neural_and_ridge_fitting',controls_source='cached_verified',
        original_rules_unchanged=True,new_V_or_H=False))
    file=PUBLIC/'fits'/(stem+'.json')
    if replay: rel.base.exact(source.read(file),record)
    else: source.once(file,record)
    return record


def summary(folds):
    keys=('normalized_MSE','total_joint_cost','false_safe_cost','false_reject_cost')
    means={a:{k:float(np.mean([f['metrics'][a][k] for f in folds])) for k in keys} for a in ARMS}
    oldm,new=means['shared64'],means['neural_transport']
    gain=1-new['total_joint_cost']/oldm['total_joint_cost'] if oldm['total_joint_cost']>0 else None
    return dict(source_macro=means,cost_reduction=gain,passes_information_screen=bool(gain is not None
        and gain>=.01 and new['false_safe_cost']<=oldm['false_safe_cost']
        and new['total_joint_cost']<means['neural_plain']['total_joint_cost']),safety_certificate=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('register','pilot','run','verify','replay')); p.add_argument('--resume',action='store_true')
    args=p.parse_args(); PRIVATE.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'process.lock').open('a') as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        runtime=old.previous.audit.old.parent.old.previous.old.run.old.old.previous.teacher.lineage
        runtime.PRIVATE=PRIVATE; start,guard,beat=runtime.runtime({},source.read(CONFIG),args.phase)
        before={str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()} if args.phase=='replay' else None
        cfg,previous=register()
        if args.phase=='register': return
        for file in (*FILES,REG):
            b=source.subprocess.run(['git','show','HEAD:'+str(file.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
            if source.hashlib.sha256(b).hexdigest()!=source.sha(file): raise ValueError('Commit fixed design before fitting')
        receipt=(PRIVATE if args.phase=='replay' else PUBLIC)/(args.phase+'_execution.json')
        if receipt.exists(): raise ValueError('Completed phase immutable')
        if args.phase=='pilot':
            c=previous['contexts'][-1]; d,_=load(c); site=c['folds'][0]['source']; ident=identity(d,c['job'],site)
            steps=(int((d['sites']!=site).sum())+511)//512; path=PRIVATE/'pilot/resumed.pt.gz'
            if not path.exists(): method.fit(d,site,method.SETTINGS,ident,path,stop_steps=steps,heartbeat=beat,guard=guard)
            b=method.fit(d,site,method.SETTINGS,ident,path,resume=True,stop_steps=steps*2,heartbeat=beat,guard=guard)
            a=method.fit(d,site,method.SETTINGS,ident,PRIVATE/'pilot/full.pt.gz',resume=args.resume,
                stop_steps=steps*2,heartbeat=beat,guard=guard)
            exact_fit(a,b)
            source.once(PUBLIC/'pilot.json',dict(two_epoch_resume_exact=True,fit_rows=a['fit_rows'],
                steps=a['step'],parameters=a['parameters'],seconds=a['seconds'],
                rough_32epoch_fold_seconds=a['seconds']*16,checkpoint=source.artifact(path)))
        else:
            if not source.read(PUBLIC/'pilot.json')['two_epoch_resume_exact']: raise ValueError('Real pilot required')
            if args.phase=='replay':
                def forbidden(*a,**k): raise RuntimeError('Fitting forbidden during replay')
                method.fit=forbidden; method.fit_ridge=forbidden
            contexts=[]
            for c in previous['contexts']:
                d,_=load(c); folds=[]
                for f in c['folds']:
                    folds.append(fold(d,c,f,guard,beat,resume=args.resume,verify=args.phase=='verify',replay=args.phase=='replay'))
                    beat(state='model_fold_complete',job=c['job'],source=f['source'])
                contexts.append(dict(job=c['job'],folds=folds,summary=summary(folds)))
            passing=sum(c['summary']['passes_information_screen'] for c in contexts)
            result=source.stable(dict(contexts=contexts,fresh_neural_fits=8,fresh_ridge_fits=8,
                passing_contexts=passing,followup_supported=passing>=2,new_neural_risk_training=True,
                new_neural_trajectory_training=False,new_V_or_H=False,original_rules_unchanged=True,
                identity_certification=False,original24group_primary='not_estimable_not_passed',
                independent_confirmation=False,deployment_changed=False,submission_ready=False,
                stage5c_executed=False,smc_enabled=False))
            if args.phase=='replay': rel.base.exact(source.read(PUBLIC/'analysis.json'),result)
            else: source.once(PUBLIC/'analysis.json',result)
            if args.phase=='verify': source.once(PUBLIC/'verification.json',dict(all8_neural_refits_exact=True,
                all8_ridge_refits_exact=True,all_predictions_exact=True,analysis_sha256=source.sha(PUBLIC/'analysis.json')))
        unchanged=None
        if before is not None:
            rel.base.exact(before,{str(f):source.sha(f) for f in PUBLIC.rglob('*') if f.is_file()}); unchanged=len(before)
        source.once(receipt,dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,unchanged_public_files=unchanged))
        beat(state='complete')


if __name__=='__main__': main()
