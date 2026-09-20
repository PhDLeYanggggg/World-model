"""Descriptive risk accounting on frozen decisions, never a deployment gate.

Missing selected-agent costs remain unknown. An observed nonnegative-harm sum
over the original past population is a lower bound, not a zero imputation.
"""
from __future__ import annotations

import math

import numpy as np

STATUSES = ('proven_exceeds', 'known_within', 'indeterminate')
IDENTITY = ('recording_id', 'physical_scene', 'frame_id', 'horizon_raw')


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Finite nonnegative {name} required')
    return float(value)


def audit_query(rows, query, *, control, query_control, budget, easy_threshold, tolerance=1e-10):
    budget = _finite_nonnegative(budget, 'budget')
    easy_threshold = _finite_nonnegative(easy_threshold, 'easy threshold')
    tolerance = _finite_nonnegative(tolerance, 'tolerance')
    if budget == 0 or not rows or len(rows) != query['agent_count']:
        raise ValueError('Positive budget and complete original query population required')
    identity = tuple(query[k] for k in IDENTITY)
    seen = set()
    counts = dict(agents=len(rows), labeled_agents=0, missing_ade_agents=0,
                  switched_agents=0, unknown_selected_agents=0, easy_agents=0)
    sums = dict(observed_positive_harm_sum=0., observed_net_excess_sum=0.,
                easy_baseline_sum=0., easy_selected_sum=0., easy_positive_harm_sum=0.,
                easy_net_excess_sum=0.)
    for row in rows:
        if tuple(row[k] for k in IDENTITY) != identity or row['agent_id'] in seen:
            raise ValueError('Query/agent identity changed or duplicated')
        seen.add(row['agent_id'])
        arm = row['arms'][control]
        if type(arm['switch']) is not bool:
            raise ValueError('Explicit Boolean frozen choice required')
        switched = arm['switch']
        counts['switched_agents'] += switched
        baseline, selected = row['baseline_ade'], arm['ade']
        if (baseline is None) != (selected is None):
            raise ValueError('Baseline/candidate label mask mismatch')
        if baseline is None:
            counts['missing_ade_agents'] += 1
            counts['unknown_selected_agents'] += switched
            continue
        baseline = _finite_nonnegative(baseline, 'baseline ADE')
        selected = _finite_nonnegative(selected, 'selected ADE')
        if not switched and selected != baseline:
            raise ValueError('Unswitched fallback error differs from baseline')
        counts['labeled_agents'] += 1
        excess = selected - baseline
        harm = max(excess, 0.)
        sums['observed_positive_harm_sum'] += harm
        sums['observed_net_excess_sum'] += excess
        if baseline <= easy_threshold:
            counts['easy_agents'] += 1
            sums['easy_baseline_sum'] += baseline
            sums['easy_selected_sum'] += selected
            sums['easy_positive_harm_sum'] += harm
            sums['easy_net_excess_sum'] += excess
    arm = query['arms'][query_control]
    predicted = _finite_nonnegative(arm['mean_predicted_harm'], 'predicted mean harm')
    predicted_pass = predicted <= budget + tolerance
    if arm['predicted_constraints_satisfied'] and not predicted_pass:
        raise ValueError('Stored predicted-budget pass contradicts numeric score')
    lower = sums['observed_positive_harm_sum']/len(rows)
    exact = lower if counts['unknown_selected_agents'] == 0 else None
    status = 'proven_exceeds' if lower > budget + tolerance else (
        'known_within' if exact is not None else 'indeterminate')
    return dict(zip(IDENTITY, identity)) | dict(
        **counts, **sums, predicted_mean_harm=predicted, budget=budget,
        predicted_budget_fraction=predicted/budget, predicted_budget_pass=predicted_pass,
        harm_lower_bound=lower, realized_harm_exact=exact, budget_status=status,
        lower_bound_above_predicted=lower > predicted + tolerance,
        easy_label_is_inference_input=False)


def _mean(values):
    return float(np.mean(values)) if values else None


def summarize_queries(records, bins):
    if not records:
        raise ValueError('Nonempty frozen query records required')
    identities = [tuple(r[k] for k in IDENTITY) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Query identity duplicated')
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0):
        raise ValueError('Increasing finite fixed bin boundaries required')
    if edges[0] > min(r['predicted_budget_fraction'] for r in records) or edges[-1] < max(r['predicted_budget_fraction'] for r in records):
        raise ValueError('Predicted scores outside fixed bins; do not silently discard')
    sums = {k:sum(r[k] for r in records) for k in (
        'agents','labeled_agents','missing_ade_agents','switched_agents','unknown_selected_agents',
        'easy_agents','observed_positive_harm_sum','observed_net_excess_sum','easy_baseline_sum',
        'easy_selected_sum','easy_positive_harm_sum','easy_net_excess_sum')}
    groups = {}
    for status in STATUSES:
        subset = [r for r in records if r['budget_status'] == status]
        groups[status] = dict(queries=len(subset), switched_queries=sum(r['switched_agents'] > 0 for r in subset),
            easy_positive_harm_sum=sum(r['easy_positive_harm_sum'] for r in subset),
            easy_net_excess_sum=sum(r['easy_net_excess_sum'] for r in subset),
            easy_baseline_sum=sum(r['easy_baseline_sum'] for r in subset),
            positive_harm_sum=sum(r['observed_positive_harm_sum'] for r in subset))
    reliability = []
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        subset = [r for r in records if lo <= r['predicted_budget_fraction'] and
                  (r['predicted_budget_fraction'] < hi or (i == len(edges)-2 and r['predicted_budget_fraction'] == hi))]
        exact = [r for r in subset if r['realized_harm_exact'] is not None]
        reliability.append(dict(lower_fraction=float(lo), upper_fraction=float(hi), queries=len(subset),
            predicted_harm_mean=_mean([r['predicted_mean_harm'] for r in subset]),
            observed_harm_lower_bound_mean=_mean([r['harm_lower_bound'] for r in subset]),
            fully_scored_selected_queries=len(exact),
            exact_harm_mean=_mean([r['realized_harm_exact'] for r in exact]),
            predicted_harm_mean_on_exact_subset=_mean([r['predicted_mean_harm'] for r in exact])))
    baseline = sums['easy_baseline_sum']
    easy_pct = 100*sums['easy_net_excess_sum']/baseline if baseline > 0 else None
    return dict(**sums, queries=len(records), switched_queries=sum(r['switched_agents'] > 0 for r in records),
        predicted_budget_failure_queries=sum(not r['predicted_budget_pass'] for r in records),
        lower_bound_above_predicted_queries=sum(r['lower_bound_above_predicted'] for r in records),
        mean_query_predicted_harm=_mean([r['predicted_mean_harm'] for r in records]),
        mean_query_harm_lower_bound=_mean([r['harm_lower_bound'] for r in records]),
        mean_past_agent_harm_lower_bound=sums['observed_positive_harm_sum']/sums['agents'],
        exact_selected_cost_queries=sum(r['realized_harm_exact'] is not None for r in records),
        easy_degradation_percent=easy_pct, budget_status=groups,
        easy_positive_harm_fraction={k:g['easy_positive_harm_sum']/sums['easy_positive_harm_sum']
            if sums['easy_positive_harm_sum'] > 0 else None for k,g in groups.items()},
        reliability_bins=reliability,
        risk_diagnostic_is_not_primary_metric=True, independent_calibration=False,
        missing_selected_costs_imputed=False, deployment_rule_changed=False)
