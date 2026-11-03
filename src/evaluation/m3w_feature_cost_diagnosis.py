"""Frozen feature-head diagnosis on identical selected cost sets."""
import numpy as np

from src.evaluation.m3w_cost_mass_decomposition import MOMENTS

ARMS = ('zero64', 'interaction64')
ROLES = ('critic_fit', 'double_excluded')
SCOPES = ('full_legal_neural', 'selected_full_proposal',
          'selected_scaled_proposal', 'selected_actual_interpolation')


def packet_for_role(job, role):
    if role not in ROLES:
        raise ValueError('Only registered fitting and excluded SOURCE roles')
    if role == 'critic_fit':
        return dict(job, evaluation_sites=job['critic_sites'],
                    evaluation_scope=job['fit_scope'])
    return job


def describe_records(records):
    out = dict(record_views=len(records), selected_rows=sum(r['selected'] for r in records))
    for scope in SCOPES:
        out[scope] = {}
        for moment in MOMENTS:
            values = [r[scope][moment]['predicted_over_target'] for r in records]
            defined = [v for v in values if v is not None]
            out[scope][moment] = dict(defined_record_views=len(defined),
                undefined_record_views=len(values)-len(defined),
                quantiles_0_25_50_75_100=np.quantile(defined, [0, .25, .5, .75, 1]).tolist() if defined else None)
    for channel, numerator, denominator in (('all', 'H', 'R'), ('easy', 'EH', 'ER')):
        pairs = [(r['selected_actual_risks']['predicted_'+channel],
                  r['selected_actual_risks']['target_'+channel]) for r in records]
        failed = [r for r, (p, t) in zip(records, pairs)
                  if p is not None and t is not None and p <= .02+1e-12 and t > .02]
        ratio = lambda r, k: r['selected_actual_interpolation'][k]['predicted_over_target']
        out[channel+'_risk'] = dict(defined_views=sum(t is not None for _, t in pairs),
            predicted_safe_actual_violations=len(failed),
            actual_violations=sum(t is not None and t > .02 for _, t in pairs),
            false_safe_numerator_underpredicted=sum(ratio(r, numerator) is not None and ratio(r, numerator) < 1 for r in failed),
            false_safe_denominator_overpredicted=sum(ratio(r, denominator) is not None and ratio(r, denominator) > 1 for r in failed))
    full = lambda r, scope: r[scope]['EH']['predicted_over_target']
    out['selected_full_EH_below_full_legal_views'] = sum(
        full(r, 'selected_full_proposal') is not None and full(r, 'full_legal_neural') is not None
        and full(r, 'selected_full_proposal') < full(r, 'full_legal_neural') for r in records)
    no_unknown = [r for r in records if r['selected'] > 0 and r['unknown_selected'] == 0]
    values = [full(r, 'selected_actual_interpolation') for r in no_unknown]
    values = [v for v in values if v is not None]
    out['posthoc_no_unknown_selected'] = dict(record_views=len(no_unknown), defined_EH_views=len(values),
        median_EH_prediction_target_ratio=float(np.median(values)) if values else None,
        easy_violations=sum(r['selected_actual_risks']['target_easy'] is not None
                            and r['selected_actual_risks']['target_easy'] > .02 for r in no_unknown))
    return out


def summarize(groups, jobs):
    expected = {(j['key'], role) for j in jobs for role in ROLES}
    if len(jobs) != 18 or {j['seed'] for j in jobs} != {17, 29, 43}:
        raise ValueError('All18 jobs and three actual parent seeds required')
    if len(groups) != 36 or {(g['job'], g['role']) for g in groups} != expected:
        raise ValueError('Every fitting/excluded group required, without duplicates')
    result = {}
    for role in ROLES:
        result[role] = {}
        for arm in ARMS:
            records = [r for g in groups if g['role'] == role for r in g['arms'][arm]['records'].values()]
            if len(records) != 444:
                raise ValueError('Keep all444 repeated recording/pipeline views in each role/arm')
            result[role][arm] = describe_records(records)
    return dict(result_source='fresh_zero_update_frozen_feature_heads_SOURCE_same_selected_set_diagnosis',
        jobs=18, models=36, job_role_groups=36, inference_role_passes=72,
        held36cost_and_amplitude_arrays_exact=True, held_completion_risks_match_parent=True,
        descriptive_record_views_not_independent_scenes=True, new_optimizer_updates=0,
        independent_confirmation=False, thresholds_changed=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False, results=result)
