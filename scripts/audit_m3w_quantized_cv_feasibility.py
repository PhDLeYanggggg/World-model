"""Test the fit-only hidden-CV hypothesis without fitting or selecting a model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.audit_m3w_stationary_label_resolution import parse_source
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_quantized_motion_feasibility import quantized_cv_feasibility


def summarize(rows, results, mask, floor):
    indices = np.flatnonzero(mask)
    denominator = float(floor[mask].sum())
    categories = {}
    for name, flag in [('feasible', True), ('infeasible', False), ('not_run', None)]:
        selected = [i for i in indices if results[i]['feasible'] is flag]
        categories[name] = {
            'rows': len(selected),
            'agents': len({(rows[i]['recording_id'], rows[i]['agent_id']) for i in selected}),
            'runs': len({(rows[i]['recording_id'], rows[i]['agent_id'], rows[i]['first_row']) for i in selected}),
            'native_cv_error_sum': float(floor[selected].sum()),
            'native_cv_error_share': float(floor[selected].sum()/denominator) if denominator>0 else None,
        }
    return {'rows': len(indices), 'status_counts': dict(Counter(results[i]['status'] for i in indices)),
            'categories': categories, 'error_share_denominator': denominator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    registration = json.loads(args.registration.read_text())
    if registration['role']!='fit_only_label_side_diagnostic':
        raise ValueError('Only the registered label-side audit is authorized')
    for path, digest in registration['bindings'].items():
        if file_digest(ROOT/path)!=digest:
            raise ValueError('Frozen audit binding changed: '+path)
    contract = ExperimentContract(json.loads((ROOT/registration['parent_protocol']).read_text()), ROOT)
    if contract.digest!=registration['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    scene_report = json.loads((ROOT/registration['scene_report']).read_text())
    cache = ROOT/registration['scene_cache']
    if file_digest(cache)!=scene_report['cache_sha256']:
        raise ValueError('Frozen row cache changed')
    with np.load(cache, allow_pickle=False) as a:
        rows = json.loads(str(a['rows_json']))
        native, changed = a['native'].copy(), a['start'].astype(bool)
    if any(r['data_role']!='fit' or contract.protocol['assignments'][r['recording_id']]!='fit' for r in rows):
        raise ValueError('Non-fit row in diagnostic')
    output, reports = args.output.resolve(), args.report_dir.resolve()
    if not output.is_relative_to(ROOT) or not reports.is_relative_to(ROOT):
        raise ValueError('Keep artifacts in this workspace')
    if output.exists() or reports.exists():
        raise ValueError('Preserve prior outputs; use a fresh registered destination')
    sources, matrices, lineage = {}, {}, {}
    for rid in sorted({r['recording_id'] for r in rows}):
        reader, _ = contract.open_recording(rid, purpose='fit')
        path = ROOT/reader.metadata['files'][0]['path']
        if file_digest(path)!=reader.metadata['files'][0]['sha256']:
            raise ValueError('Canonical source changed')
        source, _ = parse_source(path)
        for point in reader.points:
            np.testing.assert_array_equal(source[(int(point[0]), int(point[1]))]['xy'], point[2:4])
        sources[rid], matrices[rid] = source, np.loadtxt(path.parent/'H.txt')
        lineage[rid] = {'source_sha256': file_digest(path), 'H_sha256': file_digest(path.parent/'H.txt'),
                        'source_cache_coordinates_exact': True}
    results = []
    for i, row in enumerate(rows):
        source = sources[row['recording_id']]
        frame, step, agent = row['frame_id'], row['native_frame_step'], row['agent_id']
        xy = np.array([source[(frame+j*step, agent)]['xy'] for j in range(-7, 13)])
        np.testing.assert_array_equal(xy[8:]-xy[7], native[i])
        if not np.all(xy[:8]==xy[7]):
            raise ValueError('Stationary history no longer matches')
        result = quantized_cv_feasibility(xy, np.arange(-7, 13), matrices[row['recording_id']],
                                         half_width=registration['pixel_half_width'])
        results.append(result)
        if (i+1)%50==0:
            print(json.dumps({'progress': i+1, 'total': len(rows), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    floor = np.linalg.norm(native, axis=-1).mean(1)
    returned = changed&(np.linalg.norm(native[:, -1], axis=1)==0)
    per_source = {}
    for rid in sources:
        part = np.array([r['recording_id']==rid for r in rows])
        per_source[rid] = {name: summarize(rows, results, part&mask, floor) for name, mask in
                          [('all_stationary', np.ones(len(rows), bool)), ('changed', changed),
                           ('unchanged', ~changed), ('changed_returned_origin', returned)]}
    output.mkdir(parents=True)
    local = output/'row_statuses.json'
    local.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    report = {
        'result_source': 'fresh_run_fit_only_conditional_linear_program_feasibility',
        'registration_sha256': file_digest(args.registration), 'scene_cache_sha256': file_digest(cache),
        'source_lineage': lineage, 'pixel_half_width': registration['pixel_half_width'],
        'observed_steps': 8, 'future_steps_diagnostic_only': 12, 'time_unit': 'native_annotation_step_not_seconds',
        'per_source': per_source, 'rows': len(rows), 'elapsed_seconds': time.monotonic()-started,
        'max_feasible_reprojection_excess': max([r.get('max_pixel_constraint_excess', 0.) for r in results if r['feasible'] is True], default=None),
        'new_models_trained': 0, 'future_labels_enter_inference': False, 'selection_performed': False,
        'development_calibration_confirmation_labels_opened': False, 'labels_or_primary_metric_changed': False,
        'measurement_uncertainty_model_verified': False, 'physical_motion_proven': False,
        'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False,
    }
    reports.mkdir(parents=True)
    path = reports/'audit.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'completion.json').write_text(json.dumps({'report_sha256': file_digest(path),
        'registration_sha256': file_digest(args.registration), 'local_rows_sha256': file_digest(local)}, indent=2)+'\n')
    lines = ['# Conditional Hidden-CV Feasibility', '',
        'Fresh fit-only linear programs; no predictor training or model selection.',
        'Each recorded inferred integer pixel is allowed a closed +/-0.501 pixel box under supplied H.',
        'The fitted diagnostic line has constant velocity in dataset-local coordinates, not image coordinates.',
        'All eight past and twelve future labels constrain this diagnostic oracle; none enters model inputs.', '',
        '| Source | Label-side subset | Rows | Feasible | Infeasible | Not run | Infeasible CV-error share |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for rid, subsets in per_source.items():
        for subset, record in subsets.items():
            c = record['categories']
            share = c['infeasible']['native_cv_error_share']
            value = 'null' if share is None else f'{share:.4%}'
            lines.append(f"| {rid} | {subset} | {record['rows']} | {c['feasible']['rows']} | {c['infeasible']['rows']} | {c['not_run']['rows']} | {value} |")
    lines += ['', 'Rows overlap and are not independent observations. Agent/run counts and solver statuses are in audit.json.',
        'Infeasibility rules out only this hidden-CV plus rounding explanation. It does not prove annotation error,',
        'human intent, true acceleration, or that a different model cannot learn a predictor.',
        'Feasibility is also not proof of physical motion or an available inference-time velocity.',
        'No exclusions, new target labels, threshold tuning, development/calibration/confirmation access,',
        'metric/seconds claims, deployment, Stage5C or SMC. Original failed predictors remain failed.', '']
    (reports/'results.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__=='__main__':
    main()
