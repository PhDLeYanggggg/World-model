"""Fixed-forecast mechanism summaries; no model selection or metric amendment."""
from __future__ import annotations

import numpy as np

from src.evaluation.m3w_development_evaluation import summarize_rows, paired_control_errors

CONTROLS = ('independent_coupling_reference','unary_geometry_exact_count','joint_coupling_exact_count')


def verify_original_rows(original, current):
    if len(original) != len(current):
        raise ValueError('Frozen query population changed')
    mismatches = 0
    for before,after in zip(original,current):
        if {k:v for k,v in before.items() if k != 'arms'} != {k:v for k,v in after.items() if k != 'arms'}:
            raise ValueError('Original query, scale or label eligibility changed')
        for arm in ('floor','uncontrolled'):
            if before['arms'][arm] != after['arms'][arm]:
                raise ValueError('Frozen forecast scoring changed')
        mismatches += int(any(before['arms'][arm] != after['arms'][arm]
                              for arm in ('independent','scene_uniform','joint')))
    return dict(agent_rows_verified=len(current),fixed_forecast_error_rows_different=0,
                legacy_control_rows_different=mismatches)


def compact_query(scene, decision):
    r = decision['interaction_controls']
    result = {k:scene[k] for k in ('recording_id','physical_scene','frame_id','horizon_raw')}
    fields = ('reference_count','matched','nonzero_matched','potential_nonadditive_edges_at_count',
              'joint_minus_unary_switch_identities','predicted_full_objective_advantage',
              'absolute_product_sum_bound','candidate_nonidentical_forecasts','selected_nonidentical_forecasts')
    result.update({k:r[k] for k in fields})
    result['agent_count'] = len(scene['agents'])
    result['arms'] = {name:{k:arm[k] for k in ('solver_optimal','reason','full_objective',
        'unary_objective','product_objective','mean_predicted_gain','mean_predicted_harm','mean_pair_proxy',
        'numerical','predicted_constraints_satisfied')} for name,arm in r['controls'].items()}
    result['new_reference_vs_legacy_switch_disagreements'] = int(np.count_nonzero(
        decision['arms'][CONTROLS[0]]['switch'] != decision['arms']['independent']['switch']))
    return result


def summarize_controls(rows, queries, protocol):
    rules = protocol['development_evaluation']
    fixed = dict(aggregation=protocol['task']['aggregation'],n_bootstrap=protocol['bootstrap_resamples'],
                 seed=rules['bootstrap_seed'])
    full = summarize_rows(rows,metric=protocol['task']['primary_metric'],aggregation=fixed['aggregation'],
        error_unit=rules['error_unit'],easy_threshold=rules['easy_threshold'],hard_threshold=rules['hard_threshold'],
        bootstrap_resamples=fixed['n_bootstrap'],bootstrap_seed=fixed['seed'])
    keys = lambda r:(r['recording_id'],r['frame_id'],r['horizon_raw'])
    contrasts = {}
    for name,selected_queries in [('all',queries),('matched',[r for r in queries if r['matched']]),
                                  ('matched_nonzero',[r for r in queries if r['nonzero_matched']])]:
        selected_ids = {keys(r) for r in selected_queries}
        subset = [r for r in rows if keys(r) in selected_ids]
        contrasts[name] = {}
        for left,right in [(CONTROLS[2],CONTROLS[1]),(CONTROLS[2],CONTROLS[0]),(CONTROLS[1],CONTROLS[0])]:
            label = left+'_minus_'+right
            contrasts[name][label] = {}
            for metric in ('ade','fde'):
                value = paired_control_errors(subset,left,right,metric=metric,**fixed)
                value.pop('coverage_matched',None)
                value.update(selected_past_agent_count_matched=name != 'all',
                    scored_agent_count_matched=None,realized_risk_matched=False)
                contrasts[name][label][metric] = value
    summary = dict(scene_queries=len(queries),agent_queries=len(rows),
        matched_queries=sum(r['matched'] for r in queries),nonzero_matches=sum(r['nonzero_matched'] for r in queries),
        unmatched_queries=sum(not r['matched'] for r in queries),
        possible_product_queries=sum(r['potential_nonadditive_edges_at_count']>0 for r in queries),
        joint_unary_switch_disagreements=sum(r['joint_minus_unary_switch_identities'] for r in queries),
        new_reference_legacy_switch_disagreements=sum(r['new_reference_vs_legacy_switch_disagreements'] for r in queries),
        proxy_advantage_queries=sum(r['predicted_full_objective_advantage'] is not None
            and r['predicted_full_objective_advantage']>1e-10 for r in queries),
        changed_forecast_count_mismatch_queries=sum(len(set(r['selected_nonidentical_forecasts'].values()))>1 for r in queries))
    return dict(full=full,contrasts=contrasts,query_summary=summary,eligible_for_selection=False,
                independent_confirmation=False,primary_metric_changed=False)
