"""Matched output-unit contrast, using the already frozen locality estimator."""
from src.evaluation import m3w_agent_track_refit as previous

paired_localities = previous.paired_localities
causal_slices = previous.causal_slices


def slice_metrics(new, grouped, reference, cv, subset):
    values = previous.slice_metrics(new, grouped, reference, cv, subset)
    return {k.replace('flat', 'grouped'): v for k, v in values.items()}


def gates(doc, easy_guard):
    s = doc['summaries']
    p = s['ADE_all']['gain_vs_grouped_percent']
    e = s['ADE_positive_easy']['gain_vs_grouped_percent']
    h = s['ADE_hard']['gain_vs_grouped_percent']
    out = dict(primary_positive_lower_CI=p['ci95'] is not None and p['ci95'][0] > 0,
        easy_vs_grouped_within_two_percent=e['point'] is not None and e['point'] >= -easy_guard,
        hard_vs_grouped_nonnegative=h['point'] is not None and h['point'] >= 0)
    out['exploratory_forecaster_benefit'] = all(out.values())
    out.update(independent_confirmation=False, policy_safety_established=False,
               deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    return out
