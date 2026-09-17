"""Numerically refine flagged point decisions and summarize every frozen setting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest,ExperimentContract
from src.evaluation.m3w_stationary_scene_context import forecast_metrics
from src.evaluation.m3w_geometric_median_refinement import refine_geometric_median
from scripts.run_m3w_stationary_start_probe import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    registration=json.loads(args.registration.read_text())
    for name,sha in registration['bindings'].items():
        if file_digest(ROOT/name)!=sha:
            raise ValueError('Refinement binding changed: '+name)
    parent=json.loads((ROOT/registration['source_registration']).read_text())
    for name,sha in parent['bindings'].items():
        if file_digest(ROOT/name)!=sha:
            raise ValueError('Parent binding changed: '+name)
    protocol=ExperimentContract(json.loads((ROOT/parent['parent_protocol']).read_text()),ROOT)
    if protocol.digest!=parent['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    original=json.loads((ROOT/registration['source_report']).read_text())
    source=ROOT/registration['source_directory']
    cache=ROOT/parent['source_directory']/'scene_rows.npz'
    if file_digest(cache)!=original['identity']['source_cache_sha256'] or original['status']!='complete':
        raise ValueError('Incomplete or changed source study')
    with np.load(cache,allow_pickle=False) as a:
        targets,basis,scale,parent_scale=[a[k].copy() for k in ('targets','basis','scale','parent_scale')]
        rows=json.loads(str(a['rows_json']))
    if any(r['data_role']!='fit' or protocol.protocol['assignments'][r['recording_id']]!='fit' for r in rows):
        raise ValueError('Only fit rows allowed')
    output,reports=args.output.resolve(),args.report_dir.resolve()
    if not output.is_relative_to(ROOT) or not reports.is_relative_to(ROOT):
        raise ValueError('Local artifacts required')
    if output.exists() or reports.exists():
        raise ValueError('Never overwrite numerical evidence')
    output.mkdir(parents=True)
    trials=[]
    for t in original['trials']:
        name=f"fold{t['fold']}_seed{t['seed']}_{t['feature']}_extra_trees"
        path=source/(name+'.npz')
        if file_digest(path)!=t['predictions_sha256']:
            raise ValueError('Original predictions changed')
        with np.load(path,allow_pickle=False) as a:
            held,train,weights,old,mean,probability=[a[k].copy() for k in ('held','train','weights','median','mean','probability')]
            solver=json.loads(str(a['solver_info_json']))
        if {rows[i]['physical_scene'] for i in train}&{rows[i]['physical_scene'] for i in held}:
            raise ValueError('Scene leakage in saved weights')
        prediction=old.copy()
        details=[]
        for q,steps in enumerate(solver):
            for step,info in enumerate(steps):
                if info['converged']:
                    continue
                initial=old[q,step]@basis[held[q]]/scale[held[q]]
                value,new=refine_geometric_median(targets[train,step],weights[q],initial,
                                                tolerance=parent['solver_tolerance'])
                prediction[q,step]=(value@basis[held[q]].T)*scale[held[q]]
                details.append({'query_index':q,'step':step,'original_gap':info['gap_bound'],**new})
        # Explain the zero decision using only the training conditional distribution.
        lengths=np.linalg.norm(targets[train],axis=-1)
        unit=np.divide(targets[train],lengths[...,None],out=np.zeros_like(targets[train]),where=lengths[...,None]>0)
        mass=weights@(lengths==0)
        directional=np.linalg.norm(np.einsum('qn,ntd->qtd',weights,unit),axis=-1)
        zero_optimal=directional<=mass
        flags=probability>=parent['fixed_probability_gate']
        arms={'mean':mean,'median':prediction,'mean_fixed_gate':mean*flags[:,None,None],
              'median_fixed_gate':prediction*flags[:,None,None]}
        # Held labels are accessed only after weights, refinement and all decisions.
        with np.load(cache,allow_pickle=False) as a:
            native=a['native'][held].copy()
        groups=[(rows[i]['recording_id'],rows[i]['agent_id'],rows[i]['first_row']) for i in held]
        metrics={}
        for arm,p in arms.items():
            m=forecast_metrics(p,native,parent_scale[held],protocol.protocol['development_evaluation']['easy_threshold'],groups)
            still=~np.any(native!=0,axis=(1,2))
            m['native_easy_absolute_harm']=float(np.linalg.norm(p[still],axis=-1).mean()) if still.any() else None
            m['exact_nonzero_forecast_rate']=float(np.any(p!=0,axis=(1,2)).mean())
            metrics[arm]=m
        restored=np.einsum('qtd,qdk->qtk',prediction,basis[held])/scale[held,None,None]
        risk=np.einsum('qnt,qn->q',np.linalg.norm(restored[:,None]-targets[train][None],axis=-1),weights)/12
        if risk.mean()>t['expected_training_risk']['median']+1e-10:
            raise ValueError('Conditional training risk increased')
        np.savez(output/(name+'.npz'),held=held,median=prediction,refinement_json=np.array(json.dumps(details)))
        trials.append({'fold':t['fold'],'seed':t['seed'],'feature':t['feature'],
            'original_prediction_sha256':file_digest(path),'refined_prediction_sha256':file_digest(output/(name+'.npz')),
            'refined_steps':len(details),'certified_refinements':sum(int(d['converged']) for d in details),
            'atom_refinements':sum(d['method']=='certified_support_atom' for d in details),
            'optimizer_status_counts':{str(k):sum(d.get('optimizer_status')==k for d in details) for k in (0,2)},
            'max_refined_gap_bound':max([d['gap_bound'] for d in details],default=0.),
            'max_native_prediction_change':float(np.max(np.abs(prediction-old))),
            'conditional_zero_optimal_steps':int(zero_optimal.sum()),'total_steps':zero_optimal.size,
            'zero_optimal_despite_zero_mass_below_half':int((zero_optimal&(mass<.5)).sum()),
            'zero_optimal_whole_forecasts':int(zero_optimal.all(1).sum()),
            'expected_training_risk_refined':float(risk.mean()),'metrics':metrics})
    report={'result_source':'fresh_run_training_only_numerical_refinement_and_fixed_held_scoring',
        'registration_sha256':file_digest(args.registration),'trials':trials,'trials_count':len(trials),
        'original_unconverged_steps':sum(t['refined_steps'] for t in trials),
        'remaining_unconverged_steps':sum(t['refined_steps']-t['certified_refinements'] for t in trials),
        'atom_refinements':sum(t['atom_refinements'] for t in trials),'elapsed_seconds':time.monotonic()-started,
        'new_models_fitted':0,'new_threshold_or_model_selection':False,'parent_protocol_changed':False,
        'development_calibration_confirmation_labels_opened':False,'deployment':False,
        'stage5c_executed':False,'smc_enabled':False}
    reports.mkdir(parents=True)
    atomic_json(reports/'metrics.json',report)
    atomic_json(output/'completion.json',{'report_sha256':file_digest(reports/'metrics.json'),
                                        'registration_sha256':file_digest(args.registration)})
    lines=['# Conditional Mean Versus Conditional ADE Decision','',
        'Same frozen forests, fit-scene folds, features, training labels and fixed0.9 gate. No new model fit.',
        'All18 settings retained. Metrics below are seed means, not independent scene estimates.', '',
        '| Held source | Features | Mean gain | Median gain | Mean gated gain | Median gated gain | Median native still harm |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for fold in (0,1):
        for feature in ('pooled','scene','scene_neighbor'):
            part=[t for t in trials if (t['fold'],t['feature'])==(fold,feature)]
            values=[np.mean([t['metrics'][arm]['gain_vs_cv_pct'] for t in part]) for arm in ('mean','median','mean_fixed_gate','median_fixed_gate')]
            harm=np.mean([t['metrics']['median']['native_easy_absolute_harm'] for t in part])
            lines.append('| '+' | '.join(['ETH' if fold==0 else 'Hotel',feature]+[f'{v:+.6f}%' for v in values]+[f'{harm:.9f}'])+' |')
    lines+=['','## Every Setting','',
        '| Held | Seed | Features | Median gain | Gated gain | Median intervention | Gated intervention | Zero-optimal steps | Zero optimum with <half zero mass |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for t in trials:
        m,g=t['metrics']['median'],t['metrics']['median_fixed_gate']
        lines.append(f"| {t['fold']} | {t['seed']} | {t['feature']} | {m['gain_vs_cv_pct']:+.6f}% | {g['gain_vs_cv_pct']:+.6f}% | {m['exact_nonzero_forecast_rate']:.4%} | {g['exact_nonzero_forecast_rate']:.4%} | {t['conditional_zero_optimal_steps']}/{t['total_steps']} | {t['zero_optimal_despite_zero_mass_below_half']} |")
    lines+=['',f"Numerical refinement: {report['original_unconverged_steps']} flagged steps, {report['remaining_unconverged_steps']} still uncertified, {report['atom_refinements']} exact support-atom solutions.",
        'The original approximate results are preserved separately. Refinement uses training labels only.',
        'Easy percentage ratios remain undefined because still-row CV error is zero; absolute native harm is shown.',
        'No independent-site inference, new deployment, metric/seconds claim, Stage5C or SMC.', '']
    (reports/'results.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2),flush=True)


if __name__=='__main__':
    main()
