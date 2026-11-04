"""Internal source-OOF reference prediction over the frozen causal SOURCE bank."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

if platform.system()=='Darwin'and platform.machine()!='arm64':raise RuntimeError('Native arm64 required before training imports')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ.setdefault(key,'4')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_shared_harm_refinement as lineage
from src.world_model import m3w_hurdle_reference as api

source=lineage.source
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/moving_hurdle_reference_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/moving_hurdle_reference_v1'
CONFIG=ROOT/'configs/m3w_moving_hurdle_reference_v1.json';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),CONFIG,PUBLIC/'protocol.md',ROOT/'src/world_model/m3w_hurdle_reference.py',ROOT/'tests/test_m3w_hurdle_reference.py')


def runtime(cfg,spec,phase):
    source.torch.set_num_threads(4);source.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Resume saved forest block')))
    start=time.monotonic()
    def guard():
        if shutil.disk_usage(ROOT).free<spec['disk_reserve_bytes']:raise OSError('Keep registered5GiB reserve; checkpoints preserved')
        if time.monotonic()-start>43200:raise TimeoutError('12h boundary; resume unchanged budget')
    def beat(**values):
        guard();total=sum(p.stat().st_size for p in PRIVATE.rglob('*')if p.is_file())
        if total>spec['private_artifact_cap_bytes']:raise OSError('Registered private artifact cap exceeded')
        event=dict(pid=os.getpid(),phase=phase,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            elapsed_seconds=time.monotonic()-start,private_artifact_bytes=total,**values)
        source.atomic(PRIVATE/'heartbeat.json',event)
        with(PRIVATE/'events.jsonl').open('a')as stream:stream.write(source.json.dumps(event)+'\n')
        print(source.json.dumps(event),flush=True)
    beat(state='runtime_ready',machine=platform.machine(),device='cpu',torch_threads=4,interop=1,workers=0,
        forest_threads=1,resource_probing=False)
    return start,guard,beat


def register(beat):
    spec=source.read(CONFIG)
    if (spec['feature_dimension']!=api.DIMENSION or spec['tasks']!=list(api.TASKS)
            or spec['primary_forests']!=216 or spec['primary_trees']!=9216
            or any(spec[k]for k in ('new_neural_training','new_official_data_roles','new_decisions','threshold_search',
                'confirmation_opened','deployment_changed','stage5c_executed','smc_enabled'))):raise ValueError('Fixed reference-only study required')
    _,cfg,data,jobs,refs,_=lineage.register(beat)
    v=source.read(lineage.PUBLIC/'verification.json')
    if not v['all54fresh_cost_outputs_and_all108choice_scalar_blocks_exact']:raise ValueError('Complete parent before new head study')
    once=dict(bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},configuration=spec,
        parent_registration=source.artifact(lineage.REG),parent_summary=source.artifact(lineage.PUBLIC/'summary.json'),
        factor_audit=source.artifact(ROOT/'outputs/publication_readiness_2026_09/moving_record_risk_factors_v1/analysis.json'),
        inner_source_folds={j['key']:api.inner_folds(j['critic_sites'])for j in jobs},
        original24group_primary='not_estimable_not_passed',new_official_data_roles=False,
        independent_confirmation=False,new_decisions=False,stage5c_executed=False,smc_enabled=False)
    source.once(REG,source.stable(once));return spec,cfg,data,jobs,refs


def committed():
    for p in (*FILES,REG):
        b=source.subprocess.run(['git','show','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(b).hexdigest()!=source.sha(p):raise ValueError('Commit unchanged definition before real fitting')


def packet(data,job,refs,training):
    p=lineage.parent.view(data,job,refs,training=training)
    if training:p.pop('learning')
    return p


def labels(data,p,job):
    ids=p['ids'];t=lineage.parent.api.targets(p,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
    return t['moments'][:,6,3]/p['causal']['scales'],t['observed_future']


def checkpoint(job,partition,task,kind='fits'):return PRIVATE/kind/(job['key']+'_'+partition+'_'+task+'.pt.gz')


def one(p,y,k,job,sites,partition,task,spec,beat,guard,*,kind='fits',resume=False,stop_at=None):
    return api.fit(p['causal'],y,k,sites,task=task,settings=spec['training'],seed=job['seed'],
        identity=dict(registration_sha256=source.sha(REG),job=job,partition=partition),
        path=checkpoint(job,partition,task,kind),resume=resume,stop_at=stop_at,checkpoint_every=spec['checkpoint_every'],
        beat=lambda **v:beat(job=job['key'],partition=partition,**v),guard=guard)


def pilot(spec,cfg,data,jobs,refs,beat,guard,resume):
    rows=[]
    for fold in range(3):
        job=next(j for j in jobs if j['critic_fold']==fold);p=packet(data,job,refs,True);y,k=labels(data,p,job)
        mapping=api.inner_folds(job['critic_sites']);sites=[s for s in job['critic_sites']if mapping[s]!=0]
        for task in api.TASKS:
            path=checkpoint(job,'inner0',task,'pilot_resume');step=api.base.read_checkpoint(path)['trees']if path.exists()else 0
            if path.exists()and not resume:raise ValueError('Explicit pilot resume required')
            if step<8:one(p,y,k,job,sites,'inner0',task,spec,beat,guard,kind='pilot_resume',resume=path.exists(),stop_at=8)
            a=one(p,y,k,job,sites,'inner0',task,spec,beat,guard,kind='pilot_resume',resume=True,stop_at=16)
            b=one(p,y,k,job,sites,'inner0',task,spec,beat,guard,kind='pilot_direct',
                resume=resume and checkpoint(job,'inner0',task,'pilot_direct').exists(),stop_at=16)
            api.exact_refit(a,b);rows.append(dict(job=job['key'],task=task,rows=b['rows'],seconds=b['seconds'],exact=True))
        del p,y,k,a,b;source.gc.collect();guard()
    projection=24*sum(spec['training']['trees'][r['task']]/16*r['seconds']for r in rows)
    source.once(PUBLIC/'pilot.json',source.stable(dict(cells=rows,all9resume_cells_exact=True,
        verification_trees=288,projected_pure_fit_seconds_not_completion=projection)))


def prior(c,y,k,sites):
    use=source.np.isin(c['sites'],sites)&k&c['allowed'][:,6];w=api.weights(c,use)
    return float(source.np.sum(w*y)),float(source.np.sum(w*(y>0)))


def train(spec,cfg,data,jobs,refs,beat,guard,resume):
    if not source.read(PUBLIC/'pilot.json')['all9resume_cells_exact']:raise ValueError('Complete real resume pilot first')
    receipts=[]
    for job in jobs:
        p=packet(data,job,refs,True);c=p['causal'];y,k=labels(data,p,job);mapping=api.inner_folds(job['critic_sites'])
        oof={name:source.np.full(len(y),source.np.nan)for name in ('prior','prior_event','direct','event','positive','hurdle')};models=[]
        for fold in range(3):
            sites=[s for s in job['critic_sites']if mapping[s]!=fold];at=source.np.isin(c['sites'],[s for s in mapping if mapping[s]==fold])
            states={}
            for task in api.TASKS:
                path=checkpoint(job,'inner'+str(fold),task)
                state=one(p,y,k,job,sites,'inner'+str(fold),task,spec,beat,guard,resume=resume and path.exists())
                states[task]=state;models.append(dict(partition='inner'+str(fold),task=task,checkpoint=source.artifact(path),
                    trees=state['trees'],rows=state['rows'],seconds=state['seconds'],last_trace=state['trace'][-1]))
            predictions=api.predictions(states,c);mu,event=prior(c,y,k,sites)
            oof['prior'][at]=mu;oof['prior_event'][at]=event
            for name,value in predictions.items():oof[name][at]=value[at]
            del states,predictions,state;source.gc.collect();guard()
        if not all(source.np.isfinite(x).all()for x in oof.values()):raise ValueError('Every C row needs exactly one source-excluded prediction')
        scores=api.metrics(y,k,c,oof);order=('prior','direct','hurdle')
        selected=min(order,key=lambda name:(scores[name]['MSE'],order.index(name)))
        states={}
        for task in api.TASKS:
            path=checkpoint(job,'full',task)
            state=one(p,y,k,job,job['critic_sites'],'full',task,spec,beat,guard,resume=resume and path.exists());states[task]=state
            models.append(dict(partition='full',task=task,checkpoint=source.artifact(path),trees=state['trees'],
                rows=state['rows'],seconds=state['seconds'],last_trace=state['trace'][-1]))
            if job==jobs[0]:
                ref=checkpoint(job,'full',task,'full_refit')
                replica=one(p,y,k,job,job['critic_sites'],'full',task,spec,beat,guard,kind='full_refit',resume=resume and ref.exists())
                api.exact_refit(state,replica)
                source.once(PUBLIC/('full_refit_'+task+'.json'),dict(exact=True,checkpoint=source.artifact(ref)))
        mu,event=prior(c,y,k,job['critic_sites']);full_prior=dict(reference=mu,event=event)
        cache=PRIVATE/'oof'/(job['key']+'.npz');cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            if not resume:raise ValueError('Explicit unchanged OOF resume required')
            with source.np.load(cache,allow_pickle=False)as old:
                for name,value in oof.items():source.np.testing.assert_array_equal(old[name],value)
        else:source.np.savez_compressed(cache,**oof)
        path=PUBLIC/'fits'/(job['key']+'.json')
        source.once(path,source.stable(dict(job=job,inner_folds=mapping,models=models,OOF_metrics=scores,
            OOF_predictions=source.artifact(cache),selected_by_C_OOF=selected,full_prior=full_prior,
            C_ids_sha256=source.array_hash(p['ids']),C_target_sha256=source.array_hash(y),
            all_inner_models_exclude_their_validation_sources=True,no_critic_or_global_C_normalizer_in_inputs=True,
            new_held_predictions_before_this_freeze=False)));receipts.append(source.artifact(path))
        beat(state='reference_job_frozen',job=job['key'],complete=len(receipts),total=18,selected_by_C_OOF=selected)
        del p,c,y,k,oof,states,state;source.gc.collect();guard()
    source.once(PUBLIC/'training_freeze.json',dict(fits=receipts,complete_jobs=18,primary_forests=216,primary_trees=9216,
        all_primary_fits_before_new_H_prediction=True,new_decisions=False,new_neural_training=False))


def evaluate(spec,cfg,data,jobs,refs,beat,guard,verify):
    freeze=source.read(PUBLIC/'training_freeze.json')
    if freeze['complete_jobs']!=18:raise ValueError('Freeze all18source jobs first')
    models={}
    for ref in freeze['fits']:
        row=source.read(source.checked(ref));models[row['job']['key']]=row
        for item in row['models']:source.checked(item['checkpoint'])
    receipts=[]
    for job in jobs:
        row=models[job['key']];p=packet(data,job,refs,False);c=p['causal']
        states={m['task']:api.base.read_checkpoint(source.checked(m['checkpoint']))for m in row['models']if m['partition']=='full'}
        prediction=api.predictions(states,c);prediction['prior']=source.np.full(len(p['ids']),row['full_prior']['reference'])
        prediction['prior_event']=source.np.full(len(p['ids']),row['full_prior']['event'])
        hashes={name:source.array_hash(value)for name,value in prediction.items()}
        y,k=labels(data,p,job);scores=api.metrics(y,k,c,prediction)
        by_source={site:api.metrics(y,k,c,prediction,sources=[site])for site in sorted(set(p['sites']))}
        if hashes!={name:source.array_hash(value)for name,value in prediction.items()}:raise ValueError('Labels changed reference predictions')
        path=PUBLIC/'held'/(job['key']+'.json')
        source.once(path,source.stable(dict(job=job,metrics=scores,by_source=by_source,prediction_hashes=hashes,
            selected_by_C_OOF=row['selected_by_C_OOF'],prediction_labels_used_only_after_inference=True,
            H_ids_sha256=source.array_hash(p['ids']))));receipts.append(source.artifact(path))
        beat(state='reference_H_replayed'if verify else'reference_H_predicted',job=job['key'],complete=len(receipts),total=18)
        del p,c,states,prediction,y,k;source.gc.collect();guard()
    source.once(PUBLIC/'evaluation.json',dict(receipts=receipts,complete_jobs=18,new_decisions=False,
        independent_confirmation=False,original24group_primary='not_estimable_not_passed',stage5c_executed=False,smc_enabled=False))
    if verify:source.once(PUBLIC/'verification.json',dict(all18held_reference_prediction_arrays_and_metrics_exact=True,
        evaluation_sha256=source.sha(PUBLIC/'evaluation.json'),new_training=False,new_decisions=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','pilot','train','evaluate','verify'))
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    with(PRIVATE/'process.lock').open('a')as lock:
        source.fcntl.flock(lock,source.fcntl.LOCK_EX|source.fcntl.LOCK_NB)
        spec=source.read(CONFIG);cfg=source.read(lineage.parent.CONFIG);start,guard,beat=runtime(cfg,spec,args.phase)
        spec,cfg,data,jobs,refs=register(beat)
        if args.phase=='register':return
        committed();receipt=PUBLIC/(args.phase+'_execution.json')
        if receipt.exists():raise ValueError('Complete phase is immutable')
        if args.phase=='pilot':pilot(spec,cfg,data,jobs,refs,beat,guard,args.resume)
        elif args.phase=='train':train(spec,cfg,data,jobs,refs,beat,guard,args.resume)
        else:evaluate(spec,cfg,data,jobs,refs,beat,guard,args.phase=='verify')
        source.once(receipt,dict(pid=os.getpid(),seconds=time.monotonic()-start,
            peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
            new_reference_training=args.phase in ('pilot','train'),new_neural_training=False,new_decisions=False))
        beat(state='complete')


if __name__=='__main__':main()
