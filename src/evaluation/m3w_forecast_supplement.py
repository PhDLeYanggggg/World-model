"""Fixed-forecast diagnostics, not model selection or new horizon conditioning."""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.evaluation.m3w_development_evaluation import (
    score_scene, summarize_rows, _slice_metrics, paired_control_errors,
)


def attach_count_controls(decision):
    comparison = decision['coverage_match']
    return {**decision, 'arms': {**decision['arms'],
        'independent_count_reference': comparison['reference'],
        'joint_exact_count': comparison['joint_exact']}}


def score_raw_prefix(scene, decision, labels, *, raw_horizon):
    """Score an exact requested point within the already-made full prediction."""
    if type(raw_horizon) is not int or raw_horizon <= 0:
        raise ValueError('Explicit positive integer raw horizon required')
    offsets = scene['agents'][0]['inputs']['prediction_frame_offsets']
    if any(not np.array_equal(a['inputs']['prediction_frame_offsets'], offsets) for a in scene['agents']):
        raise ValueError('A common native forecast grid is required')
    hits = np.flatnonzero(offsets == raw_horizon)
    if len(hits) != 1:
        return None
    length = int(hits[0]) + 1
    sliced_scene = {**scene, 'horizon_raw': raw_horizon, 'agents': [
        {**a, 'inputs': {**a['inputs'], 'prediction_frame_offsets': offsets[:length]}}
        for a in scene['agents']]}
    sliced_decision = {**decision, 'baseline': decision['baseline'][:, :length],
        'arms': {name: {**a, 'prediction': a['prediction'][:, :length]} for name, a in decision['arms'].items()}}
    sliced_labels = [{**label, **{key: label[key][:length] for key in
        ('future_frame_ids', 'future_xy_dataset_local', 'future_label_mask')}} for label in labels]
    rows = score_scene(sliced_scene, sliced_decision, sliced_labels, label_policy='complete_requested_path')
    for row in rows:
        row['full_inference_horizon_raw'] = scene['horizon_raw']
        row['scale_source'] = 'unchanged_full_forecast_past_transform'
    return rows


def summarize_supplement(full_rows, prefix_rows, query_counts, contract):
    rules = contract.protocol['development_evaluation']
    fixed = {'aggregation': contract.protocol['task']['aggregation'],
             'n_bootstrap': contract.protocol['bootstrap_resamples'], 'seed': rules['bootstrap_seed']}
    full = summarize_rows(full_rows, metric=contract.protocol['task']['primary_metric'],
        error_unit='past_normalized', easy_threshold=rules['easy_threshold'], hard_threshold=rules['hard_threshold'],
        bootstrap_resamples=fixed['n_bootstrap'], bootstrap_seed=fixed['seed'], aggregation=fixed['aggregation'])
    counts = Counter(r['status'] for r in query_counts)
    matched_ids = {(r['recording_id'], r['frame_id'], r['horizon_raw']) for r in query_counts if r['matched']}
    nonzero_ids = {(r['recording_id'], r['frame_id'], r['horizon_raw']) for r in query_counts if r['nonzero_matched']}
    comparison = {}
    for name, ids in (('matched_including_zero', matched_ids), ('matched_nonzero_only', nonzero_ids)):
        selected = [r for r in full_rows if (r['recording_id'], r['frame_id'], r['horizon_raw']) in ids]
        comparison[name] = {m: paired_control_errors(selected, 'joint_exact_count', 'independent_count_reference',
                                                     metric=m, **fixed) for m in ('ade', 'fde')}
    prefix = {'status': 'not_run_raw_horizon_not_on_native_prediction_grid'}
    if prefix_rows:
        # Easy/hard membership is inherited from the registered full-path baseline,
        # not newly defined using a favorable shorter-prefix error threshold.
        lookup = {(r['recording_id'], r['frame_id'], r['agent_id']): r['baseline_ade'] for r in full_rows}
        easy, hard = [], []
        unknown = 0
        for r in prefix_rows:
            value = lookup[(r['recording_id'], r['frame_id'], r['agent_id'])]
            if value is None:
                unknown += 1
            elif value <= rules['easy_threshold']:
                easy.append(r)
            elif value >= rules['hard_threshold']:
                hard.append(r)
        subsets = {'all': prefix_rows, 'full_path_defined_easy': easy, 'full_path_defined_hard': hard}
        prefix = {'status': 'scored_exact_prefix_not_reconditioned_short_horizon',
            'pair_proxy_scope': 'full_forecast_decision_not_prefix_physical_safety',
            'past_supported_agent_queries': len(prefix_rows), 'unknown_full_path_slice_agent_queries': unknown,
            'normalization_scale': 'inherited_from_full_12_step_forecast_not_refitted_for_raw50',
            'arms': {arm: {name: {m: _slice_metrics(rows, arm, metric=m, **fixed) for m in ('ade', 'fde')}
                           for name, rows in subsets.items()} for arm in sorted(prefix_rows[0]['arms'])},
            'per_recording_native_units': {}}
        for recording in sorted({r['recording_id'] for r in prefix_rows}):
            part = [r for r in prefix_rows if r['recording_id'] == recording]
            native = {}
            for metric in ('ade', 'fde'):
                valid = [r for r in part if r[f'baseline_{metric}'] is not None]
                native[metric] = {'count': len(valid), 'baseline': float(np.mean([r[f'baseline_{metric}']*r['scale'] for r in valid])) if valid else None,
                    'arms': {arm: float(np.mean([r['arms'][arm][metric]*r['scale'] for r in valid])) if valid else None
                             for arm in part[0]['arms']}}
            prefix['per_recording_native_units'][recording] = native
    return {'full_forecast_controls': full, 'raw50_prefix': prefix,
        'exact_count_status_counts_scene_queries': dict(counts), 'matched_control_comparisons': comparison,
        'solver_unmatched_queries_retained_in_full_summary': True,
        'zero_count_matches_are_not_coupling_evidence': True, 'eligible_for_model_selection': False,
        'independent_confirmation': False, 'metric_or_seconds_claim': False}
