"""All18fitting-job common-region reliability audit of the frozen54refiners."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_m3w_moving_harm_refinement as run
from scripts.report_m3w_moving_harm_refinement import factorization,text_once
from src.evaluation.m3w_selected_cost_audit import selected_costs

source=run.source
PUBLIC=run.PUBLIC/'fitting_check'
PRIVATE=run.PRIVATE/'fitting_check'
REG=PUBLIC/'registration.json'
FILES=(Path(__file__),PUBLIC/'protocol.md')


def register():
    freeze=source.read(run.PUBLIC/'training_freeze.json')
    if freeze['complete_refiners']!=54:raise ValueError('Freeze all54refiners first')
    models={}
    for ref in freeze['fits']:
        row=source.read(source.checked(ref));source.checked(row['checkpoint'])
        models[row['job']['key'],row['arm']]=row
    source.once(REG,source.stable(dict(bindings={str(p.relative_to(ROOT)):source.sha(p)for p in FILES},
        scientific_registration=source.artifact(run.REG),training_freeze=source.artifact(run.PUBLIC/'training_freeze.json'),
        jobs=18,predictors=list(run.ARMS),regions=list(run.ARMS),cells_per_job=16,
        new_training=False,held_outcome_selection=False,independent_confirmation=False)))
    return models


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','audit'));args=parser.parse_args();models=register()
    if args.phase=='register':return
    for path in (*FILES,REG):
        old=source.subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
        if source.hashlib.sha256(old).hexdigest()!=source.sha(path):raise ValueError('Commit diagnostic first')
    PRIVATE.mkdir(parents=True,exist_ok=True);source.PRIVATE=PRIVATE
    cfg=source.read(run.parent.CONFIG);cfg.update(disk_reserve_bytes=5368709120,private_artifact_cap_bytes=20971520)
    start,guard,beat=source.runtime(cfg,'harm_fixed_region_fitting_diagnosis')
    spec,old_cfg,data,jobs,refs,controls=run.register(beat)
    legacy=source.read(run.parent.PUBLIC/'fitting_check/analysis.json');rows=[];hashes=[]
    for job in jobs:
        packet=run.parent.view(data,job,refs,training=True);c,ids=packet['causal'],packet['ids']
        state=run.api.base.read_checkpoint(source.checked(controls[job['key']]['checkpoint']))
        predictions=dict(frozen=run.api.parent.predict(state,c))
        if source.array_hash(predictions['frozen'])!=controls[job['key']]['fitting_predictions_sha256']:
            raise ValueError('Original fitting prediction changed')
        for arm in run.api.ARMS:
            state=run.api.base.read_checkpoint(source.checked(models[job['key'],arm]['checkpoint']))
            predictions[arm]=run.api.predict(state,c,predictions['frozen'])
            if source.array_hash(predictions[arm])!=models[job['key'],arm]['fitting_predictions_sha256']:
                raise ValueError('Refined fitting prediction changed')
            source.np.testing.assert_array_equal(predictions[arm][:,:,[2,3]],predictions['frozen'][:,:,[2,3]])
        choices={a:run.api.select_actions(p,c['allowed'],arm='physical_neural')[0]for a,p in predictions.items()}
        truth=run.api.parent.targets(packet,data['target_eval'][ids]-data['origin'][ids,None],data['valid'][ids],job['easy_cut'])
        for region,choice in choices.items():
            for arm,prediction in predictions.items():
                scores=selected_costs(prediction,truth['moments'],truth['observed_future'],choice,c['groups'])
                if arm==region=='frozen':
                    expected=next(r['scores']for r in legacy['rows']if r['job']==job['key']
                        and r['arm']=='amplitude_bank'and r['region']=='amplitude_bank_independent')
                    run.api.base.exact(source.stable(scores),source.stable(expected))
                rows.append(dict(job=job['key'],cost_model=arm,region=region,scores=scores))
        hashes.append(dict(job=job['key'],prediction_hashes={a:source.array_hash(p)for a,p in predictions.items()}))
        beat(state='common_fitting_regions_verified',job=job['key'],complete=len(hashes),total=18)
        del packet,c,state,predictions,choices,truth;source.gc.collect();guard()
    if len(rows)!=288:raise ValueError('All18x16cells required')
    aggregate=[]
    for region in run.ARMS:
        for arm in run.ARMS:
            values=[r['scores']for r in rows if r['region']==region and r['cost_model']==arm]
            actual=source.np.sum([r['realized_selected_known_moments']for r in values],0)
            predicted=source.np.sum([r['predicted_on_selected_known_moments']for r in values],0)
            aggregate.append(dict(region=region,cost_model=arm,selected=sum(r['selected']for r in values),
                selected_known=sum(r['selected_known']for r in values),**factorization(actual,predicted)))
    source.once(PUBLIC/'analysis.json',source.stable(dict(rows=rows,aggregate=aggregate,hashes=hashes,
        all72fitting_prediction_hashes_exact=True,all18frozen_scalar_blocks_exact=True,
        reference_arrays_pointwise_unchanged=True,result_source='fresh_run_C_fitting_inference_cached_verified_weights_no_training',
        independent_confirmation=False,new_model_or_threshold_selected=False)))
    table=['# Common-region fitting reliability','',
        'Fresh C-fitting inference, cached-verified weights. All72prediction hashes and18old scalar blocks replay exactly. Not held performance or independent risk.',
        '', '|Choice region|Cost predictor|Selected repeated rows|Known selected|H actual/predicted|EH actual/predicted|',
        '|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'undefined'if x is None else f'{x:.6f}'
    for r in aggregate:
        table.append(f"|{r['region']}|{r['cost_model']}|{r['selected']}|{r['selected_known']}|{fmt(r['moments']['H']['ratio'])}|{fmt(r['moments']['EH']['ratio'])}|")
    table+=['','Compare predictors within the same choice-region block. Own-region estimates alone cannot establish calibration. References are pointwise unchanged; pooled fitting masses can hide per-query/recording violations. No new model selection, deployment, metric/seconds, Stage5C or SMC.']
    text_once(PUBLIC/'readout.md','\n'.join(table)+'\n')
    source.once(PUBLIC/'execution.json',dict(pid=source.os.getpid(),seconds=time.monotonic()-start,
        peak_RSS_bytes=source.resource.getrusage(source.resource.RUSAGE_SELF).ru_maxrss,
        cells=288,all72prediction_hashes_exact=True,new_training=False))
    beat(state='complete')


if __name__=='__main__':main()
