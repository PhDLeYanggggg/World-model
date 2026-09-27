"""Explicit matched-flat contrast; retain the sealed parent's numeric estimator."""
import numpy as np
from src.evaluation import m3w_partial_neighbor_refit as previous


def slice_metrics(new, flat, reference, cv, subset):
    result = previous.slice_metrics(new, flat, reference, cv, subset)
    return {k.replace('legacy', 'flat'): v for k, v in result.items()}


def paired_localities(rows, sites, key, draws=3000, seed=71431):
    result = previous.paired_localities(rows, sites, key, draws, seed)
    # Do not inherit the generic minimum-as-worst naming for harm metrics.
    result.pop('worst_locality', None)
    values = list(result['by_site'].values())
    if all(v is not None for v in values):
        result['minimum_locality'] = min(values)
        result['maximum_locality'] = max(values)
    return result


def causal_slices(geometry):
    mask = geometry[:, 230:294].reshape(-1, 8, 8).astype(bool)
    count = mask[:, :, -1].sum(1)
    partial = (mask.any(2) & ~mask.all(2)).any(1)
    return dict(no_neighbors=count == 0, one_neighbor=count == 1,
                multiple_neighbors=count >= 2, partial_history=partial,
                no_partial_history=~partial)


def gates(doc, easy_guard):
    s = doc['summaries']
    p = s['ADE_all']['gain_vs_flat_percent']
    e = s['ADE_positive_easy']['gain_vs_flat_percent']
    h = s['ADE_hard']['gain_vs_flat_percent']
    result = dict(primary_positive_lower_CI=p['ci95'] is not None and p['ci95'][0] > 0,
        easy_vs_flat_within_two_percent=e['point'] is not None and e['point'] >= -easy_guard,
        hard_vs_flat_nonnegative=h['point'] is not None and h['point'] >= 0)
    result['exploratory_forecaster_benefit'] = all(result.values())
    result.update(independent_confirmation=False, policy_safety_established=False,
                  deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    return result
