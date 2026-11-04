"""Replay the learned neural reference control on exactly the new head's H rows."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_hurdle_reference as run

source=run.source;api=run.api
PUBLIC=run.PUBLIC/'strong_control';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),PUBLIC/'protocol.md')


def register():
    v=source.read(run.PUBLIC/'verification.json')
    if not v['all18held_reference_prediction_arrays_and_metrics_exact']:raise ValueError('Complete main prediction/replay first')
    source.once(REG,source.stable(dict(bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        training_freeze=source.artifact(run.PUBLIC/'training_freeze.json'),
        head_evaluation=source.artifact(run.PUBLIC/'evaluation.json'),head_verification=source.artifact(run.PUBLIC/'verification.json'),
        neural_control_registration=source.artifact(run.lineage.REG),
        input_scope='same_H_population_two_explicit_estimands',post_readout_diagnostic=True,
        new_training=False,new_decisions=False,model_reselection=False,independent_confirmation=False)))


def scores(y,k,c,prediction,sites=None,*,bounded=False):
    use=c['allowed'][:,6].copy()
    if sites is not None:use&=source.np.isin(c['sites'],sites)
    if not bounded:use&=k
    w=api.weights(c,use);target=source.np.where(k,y,0.)
    return {name:dict(MSE=float(source.np.sum(w*(value-target)**2)),
        MAE=float(source.np.sum(w*source.np.abs(value-target))))for name,value in prediction.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','run','verify'))
    args=parser.parse_args();register()
    if args.phase=='register':return
    for path in (*FILES,REG):
        b=source.subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(b).hexdigest()!=source.sha(path):raise ValueError('Commit comparison definition before computing it')
    if(PUBLIC/(args.phase+'_execution.json')).exists():raise ValueError('Completed comparison is immutable')
    if args.phase=='verify'and not(PUBLIC/'run_execution.json').exists():raise ValueError('Complete comparison before replay')
    source.PRIVATE=run.PRIVATE;spec=source.read(run.CONFIG);cfg=source.read(run.lineage.parent.CONFIG)
    start,guard,beat=run.runtime(cfg,spec,'strong_reference_'+args.phase)
    spec,cfg,data,jobs,refs=run.register(beat)
    fits={source.read(source.checked(r))['job']['key']:source.read(source.checked(r))for r in
        source.read(run.PUBLIC/'training_freeze.json')['fits']}
    old_controls=source.read(run.lineage.REG)['controls'];records=[];receipts=[]
    for job in jobs:
        row=fits[job['key']];p=run.packet(data,job,refs,False);c=p['causal']
        states={m['task']:api.base.read_checkpoint(source.checked(m['checkpoint']))for m in row['models']if m['partition']=='full'}
        pred=api.predictions(states,c)
        pred['prior']=source.np.full(len(p['ids']),row['full_prior']['reference']);pred['prior_event']=source.np.full(len(p['ids']),row['full_prior']['event'])
        original=source.read(run.PUBLIC/'held'/(job['key']+'.json'))
        if {k:source.array_hash(v)for k,v in pred.items()}!=original['prediction_hashes']:
            raise ValueError('All108new reference arrays must match the completed evaluation')
        old=api.base.read_checkpoint(source.checked(old_controls[job['key']]))
        control=run.lineage.api.previous.parent.predict(old,c)
        parent=source.read(run.lineage.PUBLIC/'groups'/(job['key']+'.json'))
        if source.array_hash(control)!=parent['cost_hashes']['frozen']:raise ValueError('Frozen learned neural control changed')
        # This is an existing invariant, not a new reference-sharing repair.
        if not source.np.array_equal(control[:,:,3],source.np.broadcast_to(control[:,6,3,None],control[:,:,3].shape)):
            raise ValueError('The inherited reference is expected to be candidate invariant already')
        predictions={k:pred[k]for k in ('prior','direct','hurdle')}
        predictions['C_selected']=pred[row['selected_by_C_OOF']]
        predictions['old_neural']=control[:,6,3]/c['scales']
        y,k=run.labels(data,p,job)
        by_source={str(site):{name:scores(y,k,c,predictions,[site],bounded=bounded)for name,bounded in
            (('conditional_known',False),('original_bounded_label',True))}for site in sorted(set(p['sites']))}
        path=PUBLIC/'groups'/(job['key']+'.json')
        source.once(path,source.stable(dict(job=job,by_source=by_source,reference_candidate_invariance_already_exact=True,
            new_head_prediction_hashes=original['prediction_hashes'],old_neural_cost_sha256=source.array_hash(control),
            H_ids_sha256=source.array_hash(p['ids']),model_reselection=False)))
        receipts.append(source.artifact(path))
        records.extend(dict(site=site,seed=job['seed'],producer_fold=job['producer_fold'],scores=value)for site,value in by_source.items())
        beat(state='strong_reference_replayed'if args.phase=='verify'else'strong_reference_evaluated',job=job['key'],complete=len(receipts),total=18)
        del p,c,states,pred,predictions,control,old,y,k;source.gc.collect();guard()
    if len(records)!=138:raise ValueError('All supported source-group views required')
    effects={estimand:{model:source.group_interval(records,
        lambda r,e=estimand,m=model:r['scores'][e]['old_neural']['MSE']-r['scores'][e][m]['MSE'],
        sorted({r['site']for r in records}),cfg)for model in ('prior','direct','hurdle','C_selected')}
        for estimand in ('conditional_known','original_bounded_label')}
    source.once(PUBLIC/'analysis.json',source.stable(dict(effects=effects,receipts=receipts,
        old18neural_cost_arrays_and_new108reference_arrays_match_original_hashes=True,
        original_reference_was_already_candidate_invariant=True,new_training=False,new_decisions=False,
        post_readout_diagnostic=True,independent_confirmation=False,submission_ready=False,stage5c_executed=False,smc_enabled=False)))
    source.once(PUBLIC/(args.phase+'_execution.json'),dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,new_training=False,new_decisions=False))
    if args.phase=='verify':source.once(PUBLIC/'verification.json',dict(all_arrays_and_two_estimand_metrics_exact=True,
        analysis_sha256=source.sha(PUBLIC/'analysis.json'),new_training=False,new_decisions=False))
    beat(state='complete')


if __name__=='__main__':main()
