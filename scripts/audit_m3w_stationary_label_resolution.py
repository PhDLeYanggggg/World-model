"""Recompute fit-only label precision and fixed-forecast error decomposition."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_stationary_label_resolution import (
    printed_interval, constant_compatible, project_supplied_h, resolution_counts,
    slice_forecast_metrics,
)
from src.evaluation.m3w_stationary_scene_context import FEATURE_SETS


def parse_source(path):
    source, precision = {}, Counter()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields)!=8:
            raise ValueError(f'Unexpected obsmat row at {number}')
        frame, agent = float(fields[0]), float(fields[1])
        if not frame.is_integer() or not agent.is_integer():
            raise ValueError('Noninteger source identity')
        key = (int(frame), int(agent))
        if key in source:
            raise ValueError('Duplicate canonical identity')
        tokens = (fields[2], fields[4])
        for token in tokens:
            precision[str(printed_interval(token)[2])] += 1
        source[key] = {'tokens': tokens, 'xy': np.array([float(v) for v in tokens]), 'source_line': number}
    return source, dict(sorted(precision.items()))


def quantiles(values):
    return np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist() if len(values) else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    import joblib
    registration = json.loads(args.registration.read_text())
    if registration['role']!='fit_only_label_side_diagnostic':
        raise ValueError('Unexpected audit role')
    for name, sha in registration['bindings'].items():
        if file_digest(ROOT/name)!=sha:
            raise ValueError('Frozen audit binding changed: '+name)
    contract = ExperimentContract(json.loads((ROOT/registration['parent_protocol']).read_text()), ROOT)
    if contract.digest!=registration['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    scene_report = json.loads((ROOT/registration['scene_report']).read_text())
    cache = args.source/'scene_rows.npz'
    if file_digest(cache)!=scene_report['cache_sha256']:
        raise ValueError('Original scene rows changed')
    with np.load(cache, allow_pickle=False) as a:
        x, native, basis, scale, changed = [a[k].copy() for k in ('features','native','basis','scale','start')]
        rows = json.loads(str(a['rows_json']))
    changed = changed.astype(bool)
    if any(contract.protocol['assignments'][r['recording_id']]!='fit' or r['data_role']!='fit' for r in rows):
        raise ValueError('Non-fit row in label diagnostic')
    sources, matrices, audit = {}, {}, {}
    for rid in sorted({r['recording_id'] for r in rows}):
        reader, _ = contract.open_recording(rid, purpose='fit')
        path = ROOT/reader.metadata['files'][0]['path']
        if file_digest(path)!=reader.metadata['files'][0]['sha256']:
            raise ValueError('Canonical coordinate source changed')
        source, precision = parse_source(path)
        for point in reader.points:
            np.testing.assert_array_equal(source[(int(point[0]), int(point[1]))]['xy'], point[2:4])
        h_path = path.parent/'H.txt'
        matrices[rid] = np.loadtxt(h_path); sources[rid] = source
        projected = project_supplied_h(reader.points[:, 2:4], matrices[rid])
        integer_residual = np.linalg.norm(projected-np.round(projected), axis=1)
        half_residual = np.linalg.norm(projected-np.round(projected*2)/2, axis=1)
        audit[rid] = {'source_sha256': file_digest(path), 'H_sha256': file_digest(h_path),
            'source_rows': len(source), 'source_cache_coordinates_exact': True,
            'printed_quantum_counts': precision,
            'inferred_pixel_integer_residual_quantiles': quantiles(integer_residual),
            'inferred_pixel_half_grid_residual_quantiles': quantiles(half_residual),
            'within_0_001_integer_pixel_fraction': float((integer_residual<=.001).mean()),
            'within_0_001_half_pixel_fraction': float((half_residual<=.001).mean()),
            'projection_or_measurement_uncertainty_verified': False}
    maximum, endpoint, compatible = [], [], []
    for i, row in enumerate(rows):
        source = sources[row['recording_id']]
        frame, agent, step = row['frame_id'], row['agent_id'], row['native_frame_step']
        history = [source[(frame+j*step, agent)] for j in range(-7, 1)]
        future = [source[(frame+j*step, agent)] for j in range(1, 13)]
        center = history[-1]['xy']
        if any(np.any(p['xy']!=center) for p in history):
            raise ValueError('Frozen stationary history changed')
        future_xy = np.array([p['xy'] for p in future])
        np.testing.assert_array_equal(future_xy-center, native[i])
        projected = project_supplied_h(np.vstack([center, future_xy]), matrices[row['recording_id']])
        delta = np.linalg.norm(projected[1:]-projected[0], axis=1)
        maximum.append(float(delta.max())); endpoint.append(float(delta[-1]))
        compatible.append(constant_compatible([history[-1]['tokens']]+[p['tokens'] for p in future]))
    maximum, endpoint, compatible = np.array(maximum), np.array(endpoint), np.array(compatible)
    floor = np.linalg.norm(native, axis=-1).mean(1)
    folds = np.array([r['fit_fold'] for r in rows])
    masks = {'still': ~changed,
             'changed_at_most_half_pixel': changed&(maximum<=.501),
             'changed_half_to_one_and_half_pixels': changed&(maximum>.501)&(maximum<=1.501),
             'changed_one_and_half_to_five_pixels': changed&(maximum>1.501)&(maximum<=5.001),
             'changed_above_five_pixels': changed&(maximum>5.001)}
    if not np.all(sum(m.astype(int) for m in masks.values())==1):
        raise ValueError('Magnitude partition does not retain every row')
    masks['changed_returned_to_origin'] = changed&(np.linalg.norm(native[:, -1], axis=1)==0)
    for rid, record in audit.items():
        part = np.array([r['recording_id']==rid for r in rows])
        indices = np.flatnonzero(part)
        agent_keys = [(rows[i]['recording_id'], rows[i]['agent_id']) for i in indices]
        run_keys = [(rows[i]['recording_id'], rows[i]['agent_id'], rows[i]['first_row']) for i in indices]
        returned = masks['changed_returned_to_origin']&part
        record.update(rows=int(part.sum()), changed_rows=int(changed[part].sum()),
            agents=len(set(agent_keys)), runs=len(set(run_keys)),
            changed_compatible_with_text_rounding=int((part&changed&compatible).sum()),
            unchanged_compatible_with_text_rounding=int((part&~changed&compatible).sum()),
            max_inferred_pixel_displacement_quantiles_all=quantiles(maximum[part]),
            max_inferred_pixel_displacement_quantiles_changed=quantiles(maximum[part&changed]),
            returned_rows=int(returned.sum()), returned_max_inferred_pixel_quantiles=quantiles(maximum[returned]),
            returned_native_cv_error_share=float(floor[returned].sum()/floor[part].sum()),
            thresholds=resolution_counts(maximum[part], changed[part], floor[part], agent_keys, run_keys,
                thresholds=registration['thresholds'], tolerance=registration['tolerance']),
            slices={name: {'rows': int((mask&part).sum()),
                'native_cv_ade': float(floor[mask&part].mean()) if (mask&part).any() else None,
                'cv_error_share': float(floor[mask&part].sum()/floor[part].sum())} for name, mask in masks.items()})
    expected = {(f,s,feature,model) for f in (0,1) for s in (17,29,43)
                for feature in FEATURE_SETS for model in ('linear','extra_trees')}
    actual = [(t['fold'],t['seed'],t['feature_name'],t['family']) for t in scene_report['trials']]
    if len(actual)!=len(expected) or set(actual)!=expected:
        raise ValueError('Missing/duplicate frozen model setting')
    decompositions, prediction_hashes = [], {}
    replay_max = 0.
    for t in scene_report['trials']:
        name = f"fold{t['fold']}_seed{t['seed']}_{t['feature_name']}_{t['family']}"
        checkpoint = args.source/(name+'.joblib')
        if file_digest(checkpoint)!=t['checkpoint_sha256']:
            raise ValueError('Frozen model changed')
        models = joblib.load(checkpoint)
        held = np.flatnonzero(folds==t['fold'])
        v = x[held][:, FEATURE_SETS[t['feature_name']]]
        probability = models['classifier'].predict_proba(v)[:, 1]
        prediction = models['regressor'].predict(v).reshape(-1,12,2)
        prediction = np.einsum('nki,nji->nkj', prediction, basis[held])*scale[held,None,None]
        guarded = prediction*(probability>=.9)[:,None,None]
        saved_path = args.source/(name+'.predictions.npz')
        with np.load(saved_path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['held'], held)
            for value, key in ((probability,'probability'),(prediction,'prediction'),(guarded,'guarded')):
                difference = float(np.max(np.abs(value-saved[key])))
                if difference>1e-12:
                    raise ValueError('Frozen prediction replay mismatch')
                replay_max = max(replay_max, difference)
        prediction_hashes[name] = file_digest(saved_path)
        for arm, p in [('unrestricted',prediction),('fixed_gate',guarded)]:
            decompositions.append({'fold':t['fold'],'seed':t['seed'],'features':t['feature_name'],
                'family':t['family'],'arm':arm,
                'slices':{key:slice_forecast_metrics(p,native[held],mask[held]) for key,mask in masks.items()}})
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Workspace-local row artifacts only')
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'label_resolution.npz',maximum_inferred_pixels=maximum,endpoint_inferred_pixels=endpoint,
             text_rounding_compatible=compatible,native_changed=changed)
    report = {'result_source':'fresh_run_source_decimal_audit_H_projection_and_frozen_prediction_replay',
        'registration_sha256':file_digest(args.registration),'scene_cache_sha256':file_digest(cache),
        'audit':audit,'frozen_prediction_hashes':prediction_hashes,'decompositions':decompositions,
        'model_replays':2*len(scene_report['trials']),'max_prediction_replay_difference':replay_max,
        'new_models_fitted':0,'input_features_changed':False,'labels_or_primary_metric_changed':False,
        'development_calibration_confirmation_labels_opened':False,'threshold_selected':False,
        'physical_motion_or_annotation_error_certified':False,'new_deployment':False,
        'stage5c_executed':False,'smc_enabled':False}
    args.report_dir.mkdir(parents=True,exist_ok=True)
    path = args.report_dir/'audit.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'completion.json').write_text(json.dumps({'report_sha256':file_digest(path),
        'registration_sha256':file_digest(args.registration),'local_rows_sha256':file_digest(output/'label_resolution.npz')},indent=2)+'\n')
    lines = ['# Stationary Label Resolution Audit','',
        'Fresh fit-only source/projection audit and frozen-forecast replay; no models refitted or labels changed.',
        'Inferred pixels are computed under supplied H, not verified image/annotation synchronization.', '',
        '## Source Precision and Support','',
        '| Fit source | Stationary windows | Changed | Agents | Runs | Changed compatible with text rounding | Returned to origin |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for rid,a in audit.items():
        lines.append(f"| {rid} | {a['rows']} | {a['changed_rows']} | {a['agents']} | {a['runs']} | {a['changed_compatible_with_text_rounding']} | {a['returned_rows']} |")
    lines += ['', 'Closed decimal intervals represent printed precision only, not measurement uncertainty.',
        'A non-overlap rules out a constant value explainable by that serialization precision alone.',
        'It does not prove physical movement, annotation accuracy or identifiable human intent.', '',
        '## Fixed Inferred-Image Sensitivity','',
        'Above means maximum displacement > threshold +0.001 inferred pixels. Boundary counts expose arithmetic sensitivity.',
        'Cuts are label-side descriptions, not selected targets, physical thresholds or allowed deployment filters.', '',
        '| Source | Threshold | Rows above | Agents | Runs | CV error share above | Boundary rows |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for rid,a in audit.items():
        for t in a['thresholds']:
            lines.append(f"| {rid} | {t['inferred_pixel_threshold']} | {t['rows_above']} | {t['agents_above']} | {t['runs_above']} | {t['cv_error_share_above']:.4%} | {t['rows_within_numerical_boundary']} |")
    lines += ['', '## Every Frozen Regressor on Fixed Slices','',
        'Seed-mean native-ADE gain versus CV, computed separately by recording. Empty/zero-CV slices are null, not passes.',
        'Still-row absolute native harm is retained. Returned-origin rows overlap the magnitude bins.', '',
        '| Held | Features | Family | Arm | Still harm | <=0.5 px gain | 0.5--1.5 px gain | 1.5--5 px gain | >5 px gain |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    names=list(masks)[:5]
    for fold in (0,1):
        for feature in FEATURE_SETS:
            for family in ('linear','extra_trees'):
                for arm in ('unrestricted','fixed_gate'):
                    part=[t for t in decompositions if (t['fold'],t['features'],t['family'],t['arm'])==(fold,feature,family,arm)]
                    values=[]
                    for name in names:
                        key='native_harm' if name=='still' else 'gain_vs_cv_pct'
                        vals=[t['slices'][name].get(key) for t in part]
                        values.append(f'{np.mean(vals):+.6f}' if all(v is not None for v in vals) else 'null')
                    lines.append('| '+' | '.join(['ETH' if fold==0 else 'Hotel',feature,family,arm]+values)+' |')
    lines += ['', 'All72 corrected models were hash-verified and their predictions replayed within1e-12.',
        'No settings or labels were selected. No independent-scene significance claim is made.',
        'Detailed decimal precision, inferred lattice residuals, quantiles and all seed/slice metrics are in audit.json.',
        'Raw source rows, per-row projections and checkpoints stay local. No metric/seconds, Stage5C or SMC claim.', '']
    (args.report_dir/'results.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'complete','audit':audit,'model_replays':report['model_replays'],
                      'max_replay_difference':replay_max},indent=2))


if __name__=='__main__':
    main()
