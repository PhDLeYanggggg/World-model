"""One-factor fit-only point-decision experiment under frozen tree partitions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-trials', type=int)
    args = parser.parse_args()
    if platform.system()=='Darwin' and platform.machine()!='arm64':
        raise SystemExit('Use the arm64 environment')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    import numpy as np
    import joblib
    import sklearn
    from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
    from src.evaluation.m3w_conditional_geometric_median import forest_training_weights, conditional_ade_prediction
    from src.evaluation.m3w_stationary_scene_context import forecast_metrics, FEATURE_SETS
    from scripts.run_m3w_stationary_start_probe import atomic_json, digest
    registration = json.loads(args.registration.read_text())
    if registration['role']!='fit_only_frozen_partition_point_decision':
        raise ValueError('Unexpected scientific role')
    for path, sha in registration['bindings'].items():
        if file_digest(ROOT/path)!=sha:
            raise ValueError('Registered binding changed: '+path)
    contract = ExperimentContract(json.loads((ROOT/registration['parent_protocol']).read_text()),ROOT)
    if contract.digest!=registration['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    source = ROOT/registration['source_directory']
    cache = source/'scene_rows.npz'
    source_report = json.loads((ROOT/registration['source_report']).read_text())
    if file_digest(cache)!=source_report['cache_sha256']:
        raise ValueError('Frozen feature/label cache changed')
    with np.load(cache, allow_pickle=False) as a:
        x, target, basis, scale, parent_scale = [a[k].copy() for k in ('features','targets','basis','scale','parent_scale')]
        rows = json.loads(str(a['rows_json']))
        valid = a['valid'].copy()
    if (not valid.all() or any(r['data_role']!='fit' or contract.protocol['assignments'][r['recording_id']]!='fit' for r in rows)):
        raise ValueError('All unchanged fit-only reference frames must be valid')
    folds = np.array([r['fit_fold'] for r in rows])
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT) or not reports.is_relative_to(ROOT):
        raise ValueError('Keep experiment outputs local')
    identity = {'registration_sha256':file_digest(args.registration),'parent_protocol_sha256':contract.digest,
                'source_cache_sha256':file_digest(cache),'numpy':np.__version__,'sklearn':sklearn.__version__}
    if args.resume:
        if json.loads((output/'identity.json').read_text())!=identity:
            raise ValueError('Resume identity mismatch')
    else:
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'identity.json',identity)
    if (reports/'metrics.json').exists():
        raise ValueError('Preserve original evidence; use a different report directory on resume')
    began = time.monotonic()
    def heartbeat(state, **extra):
        atomic_json(output/'heartbeat.json',{'pid':os.getpid(),'state':state,'elapsed_seconds':time.monotonic()-began,**extra})
    trials = [t for t in source_report['trials'] if t['family']=='extra_trees']
    if len(trials)!=registration['planned_trials']:
        raise ValueError('Unexpected frozen model grid')
    completed, fresh, reused = [], 0, 0
    for trial in trials:
        if args.max_trials is not None and fresh>=args.max_trials:
            break
        fold,seed,feature = trial['fold'],trial['seed'],trial['feature_name']
        name = f'fold{fold}_seed{seed}_{feature}_extra_trees'
        fitted_path = source/(name+'.joblib')
        if file_digest(fitted_path)!=trial['checkpoint_sha256']:
            raise ValueError('Frozen model changed')
        trial_file, predictions_file = output/(name+'.json'), output/(name+'.npz')
        if trial_file.exists():
            old = json.loads(trial_file.read_text())
            if (old['identity_sha256']!=digest(identity) or old['source_checkpoint_sha256']!=trial['checkpoint_sha256']
                    or file_digest(predictions_file)!=old['predictions_sha256']):
                raise ValueError('Completed decision checkpoint changed')
            completed.append({**old,'result_source':'cached_verified'}); reused+=1
            continue
        train,held = np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        if {rows[i]['physical_scene'] for i in train}&{rows[i]['physical_scene'] for i in held}:
            raise ValueError('Crossfit scene leakage')
        models = joblib.load(fitted_path)
        columns = FEATURE_SETS[feature]
        heartbeat('conditional_distribution_and_point_solver',trial=name,completed_trials=len(completed))
        started = time.monotonic()
        weights = forest_training_weights(models['regressor'],x[train][:,columns],x[held][:,columns])
        mean = np.einsum('qn,ntd->qtd',weights,target[train])
        original_mean = models['regressor'].predict(x[held][:,columns]).reshape(-1,12,2)
        np.testing.assert_allclose(mean,original_mean,atol=1e-12,rtol=0)
        median, solver = conditional_ade_prediction(target[train],weights,
            tolerance=registration['solver_tolerance'],max_iterations=registration['solver_max_iterations'])
        probability = models['classifier'].predict_proba(x[held][:,columns])[:,1]
        switched = probability>=registration['fixed_probability_gate']
        def restore(pred):
            return np.einsum('nki,nji->nkj',pred,basis[held])*scale[held,None,None]
        mean_native,median_native = restore(mean),restore(median)
        with np.load(source/(name+'.predictions.npz'),allow_pickle=False) as a:
            np.testing.assert_array_equal(a['held'],held)
            np.testing.assert_allclose(mean_native,a['prediction'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(probability,a['probability'],atol=1e-12,rtol=0)
        # No held targets have entered any of the four frozen predictions above.
        with np.load(cache,allow_pickle=False) as a:
            native = a['native'][held].copy()
        groups = [(rows[i]['recording_id'],rows[i]['agent_id'],rows[i]['first_row']) for i in held]
        arms = {'mean':mean_native,'median':median_native,
                'mean_fixed_gate':mean_native*switched[:,None,None],
                'median_fixed_gate':median_native*switched[:,None,None]}
        metrics = {}
        for arm,pred in arms.items():
            m = forecast_metrics(pred,native,parent_scale[held],contract.protocol['development_evaluation']['easy_threshold'],groups)
            m['exact_nonzero_forecast_rate'] = float(np.any(pred!=0,axis=(1,2)).mean())
            m['native_easy_absolute_harm'] = float(np.linalg.norm(pred[~np.any(native!=0,axis=(1,2))],axis=-1).mean())
            metrics[arm]=m
        expected_risk = {}
        for arm,pred in [('mean',mean),('median',median),('zero',np.zeros_like(mean))]:
            expected_risk[arm] = np.einsum('qnt,qn->q',np.linalg.norm(pred[:,None]-target[train][None],axis=-1),weights)/12
        if np.any(expected_risk['median']>np.minimum(expected_risk['mean'],expected_risk['zero'])+1e-8):
            raise ValueError('Conditional ADE solver did not preserve expected training risk')
        info = [d for path in solver for d in path]
        payload = {'result_source':'fresh_run_point_decision_under_cached_verified_forests',
            'identity_sha256':digest(identity),'source_checkpoint_sha256':trial['checkpoint_sha256'],
            'fold':fold,'seed':seed,'feature':feature,'train_rows':len(train),'held_rows':len(held),
            'source_mean_replay_max_abs_error':float(np.max(np.abs(mean-original_mean))),
            'solver_converged_steps':sum(int(d['converged']) for d in info),'solver_steps':len(info),
            'solver_max_iterations':max(d['iterations'] for d in info),'solver_max_gap_bound':max(d['gap_bound'] for d in info),
            'zero_optimal_steps':sum(int(d['zero_optimal']) for d in info),
            'expected_training_risk':{k:float(v.mean()) for k,v in expected_risk.items()},
            'metrics':metrics,'compute_seconds':time.monotonic()-started,'new_models_fitted':0,
            'new_deployment':False}
        tmp = predictions_file.with_suffix('.tmp.npz')
        np.savez(tmp,held=held,train=train,weights=weights,mean=mean_native,median=median_native,
                 probability=probability,solver_info_json=np.array(json.dumps(solver)),
                 **{'expected_risk_'+key:value for key,value in expected_risk.items()})
        os.replace(tmp,predictions_file)
        payload['predictions_sha256']=file_digest(predictions_file)
        atomic_json(trial_file,payload)
        completed.append(payload);fresh+=1
        print(json.dumps({'trial':name,'seconds':payload['compute_seconds'],
            'mean_gain_pct':metrics['mean']['gain_vs_cv_pct'],'median_gain_pct':metrics['median']['gain_vs_cv_pct'],
            'median_guarded_gain_pct':metrics['median_fixed_gate']['gain_vs_cv_pct'],
            'unconverged_steps':len(info)-payload['solver_converged_steps']}),flush=True)
    complete = len(completed)==registration['planned_trials']
    report = {'identity':identity,'status':'complete' if complete else 'partial',
        'fresh_point_decision_trials':fresh,'cached_verified_trials':reused,'planned_trials':registration['planned_trials'],
        'trials':completed,'elapsed_seconds':time.monotonic()-began,'new_models_fitted':0,
        'parent_metric_or_target_changed':False,'development_calibration_confirmation_labels_opened':False,
        'independent_confirmation':False,'deployment':False,'stage5c_executed':False,'smc_enabled':False}
    reports.mkdir(parents=True,exist_ok=True)
    atomic_json(reports/'metrics.json',report)
    atomic_json(output/('completion.json' if complete else 'partial.json'),
                {'report_sha256':file_digest(reports/'metrics.json'),'identity_sha256':digest(identity)})
    heartbeat(report['status'],completed_trials=len(completed))
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','identity')}),flush=True)


if __name__=='__main__':
    main()
