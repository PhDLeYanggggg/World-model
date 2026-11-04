"""Replay source-excluded training predictions without refitting or using H."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_hurdle_reference as run

source=run.source;api=run.api
PUBLIC=run.PUBLIC/'oof_verification';REG=PUBLIC/'registration.json'
FILES=(Path(__file__),PUBLIC/'protocol.md')


def register():
    freeze=source.read(run.PUBLIC/'training_freeze.json')
    if freeze['complete_jobs']!=18:raise ValueError('Complete all216forests first')
    for ref in freeze['fits']:
        row=source.read(source.checked(ref))
        for model in row['models']:source.checked(model['checkpoint'])
    source.once(REG,source.stable(dict(bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        training_registration=source.artifact(run.REG),training_freeze=source.artifact(run.PUBLIC/'training_freeze.json'),
        expected_jobs=18,expected_model_bindings=216,expected_OOF_arrays=108,
        new_training=False,new_H_prediction=False,new_decisions=False,changed_selection_rule=False)))
    return freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('register','verify'))
    args=parser.parse_args();freeze=register()
    if args.phase=='register':return
    source.PRIVATE=run.PRIVATE
    for path in (*FILES,REG):
        b=source.subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(b).hexdigest()!=source.sha(path):raise ValueError('Commit verification before replay')
    spec=source.read(run.CONFIG);cfg=source.read(run.lineage.parent.CONFIG)
    start,guard,beat=run.runtime(cfg,spec,'source_OOF_verification')
    spec,cfg,data,jobs,refs=run.register(beat)
    receipts={source.read(source.checked(ref))['job']['key']:source.read(source.checked(ref))for ref in freeze['fits']}
    checked_models=0;checked_arrays=0;results=[]
    for job in jobs:
        receipt=receipts[job['key']];p=run.packet(data,job,refs,True);c=p['causal'];y,k=run.labels(data,p,job)
        x=api.inputs(c);mapping=api.inner_folds(job['critic_sites'])
        rebuilt={name:source.np.full(len(y),source.np.nan)for name in ('prior','prior_event','direct','event','positive','hurdle')}
        model_states={}
        for model in receipt['models']:
            state=api.base.read_checkpoint(source.checked(model['checkpoint']));partition=model['partition'];task=model['task']
            expected=job['critic_sites']if partition=='full'else[s for s in job['critic_sites']if mapping[s]!=int(partition[-1])]
            api.base.exact(state['fit_sources'],sorted(expected))
            use=source.np.isin(c['sites'],expected)&k&c['allowed'][:,6];w=api.weights(c,use)
            if task=='positive':use&=y>0;w[~use]=0;w/=w.sum()
            at=source.np.flatnonzero(use);target=(y[at]>0).astype(int)if task=='event'else y[at]
            for key,value in dict(fit_ids_sha256=api.old.digest(c['ids'][at]),fit_feature_sha256=api.old.digest(x[at]),
                    target_sha256=api.old.digest(target),weight_sha256=api.old.digest(w[at])).items():
                if state[key]!=value:raise ValueError('Model was fitted on different rows/features/labels/weights')
            if api.model_hash(state['model'])!=state['model_sha256']:raise ValueError('Scientific forest state changed')
            if api.old.digest(api.predict_task(state,x[at]))!=state['trace'][-1]['prediction_sha256']:
                raise ValueError('Complete fitting prediction changed')
            if state['trees']!=spec['training']['trees'][task]:raise ValueError('Complete original forest budget required')
            model_states[partition,task]=state;checked_models+=1
        for fold in range(3):
            at=source.np.isin(c['sites'],[s for s in mapping if mapping[s]==fold])
            fit=[s for s in mapping if mapping[s]!=fold]
            states={t:model_states['inner'+str(fold),t]for t in api.TASKS}
            pred=api.predictions(states,c);mu,event=run.prior(c,y,k,fit)
            rebuilt['prior'][at]=mu;rebuilt['prior_event'][at]=event
            for name,value in pred.items():rebuilt[name][at]=value[at]
        with source.np.load(source.checked(receipt['OOF_predictions']),allow_pickle=False)as cache:
            for name,value in rebuilt.items():source.np.testing.assert_array_equal(value,cache[name]);checked_arrays+=1
        scores=api.metrics(y,k,c,rebuilt);api.base.exact(source.stable(scores),receipt['OOF_metrics'])
        selected=min(('prior','direct','hurdle'),key=lambda name:(scores[name]['MSE'],('prior','direct','hurdle').index(name)))
        if selected!=receipt['selected_by_C_OOF']:raise ValueError('Selection must depend only on source-excluded C scores')
        results.append(dict(job=job['key'],models=12,arrays=6,selected_by_C_OOF=selected,all_exact=True))
        beat(state='source_OOF_replayed',job=job['key'],complete=len(results),total=18)
        del p,c,x,y,k,rebuilt,model_states,state,pred;source.gc.collect();guard()
    if checked_models!=216 or checked_arrays!=108:raise ValueError('Complete all model bindings and OOF arrays')
    source.once(PUBLIC/'verification.json',source.stable(dict(results=results,checked_models=checked_models,
        exact_OOF_arrays=checked_arrays,all18OOF_metrics_and_model_choices_exact=True,
        all216fit_source_feature_target_weight_bindings_exact=True,new_training=False,new_H_prediction=False,
        independent_confirmation=False,new_decisions=False)))
    source.once(PUBLIC/'execution.json',dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,new_training=False,new_H_prediction=False))
    beat(state='complete')


if __name__=='__main__':main()
