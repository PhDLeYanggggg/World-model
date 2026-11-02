"""Scalar SOURCE cost summaries; not a validation or deployment decision."""
import statistics
import numpy as np

from src.evaluation.m3w_selected_cost_diagnosis import MOMENTS

RATIOS = ('harm_prediction_ratio', 'easy_harm_prediction_ratio',
          'reference_inflation_ratio', 'easy_reference_inflation_ratio')
SCOPES = ('all_legal', 'gate_passing', 'selected', 'neural_legal')
FAMILIES = ('weighted_centered', 'mean_absolute')


def aggregate(rows, trials):
    expected = {(t, f) for t in trials for f in FAMILIES}
    identities = [(r['trial'], r['family']) for r in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError('Every frozen producer and both cost families required')
    if any(len(r['groups']) != 8 or len({g['site'] for g in r['groups']}) != 8 for r in rows):
        raise ValueError('All eight producer-excluded SOURCE localities required')
    for trial in trials:
        paired = [r for r in rows if r['trial'] == trial]
        scopes = [{g['site']:g['target_cost_sha256'] for g in r['groups']} for r in paired]
        if paired[0]['rows'] != paired[1]['rows'] or scopes[0] != scopes[1]:
            raise ValueError('Both families require exactly matched SOURCE rows and native targets')
    output = {}
    fraction = lambda a, b: float(a/b) if b > 0 else None
    for family in FAMILIES:
        groups = [g for r in rows if r['family'] == family for g in r['groups']]
        pooled = {}
        for scope in SCOPES:
            cells = [g['diagnosis'][scope] for g in groups]
            p = np.array([[c['predicted_known_mass'][k] for k in MOMENTS] for c in cells]).sum(0)
            y = np.array([[c['observed_known_mass'][k] for k in MOMENTS] for c in cells]).sum(0)
            pooled[scope] = dict(known_action_views=sum(c['known_action_views'] for c in cells),
                unknown_action_views=sum(c['unknown_action_views'] for c in cells),
                predicted_known_mass=dict(zip(MOMENTS, p.tolist())),
                observed_known_mass=dict(zip(MOMENTS, y.tolist())),
                harm_prediction_ratio=fraction(p[1], y[1]), easy_harm_prediction_ratio=fraction(p[4], y[4]),
                reference_inflation_ratio=fraction(p[2], y[2]), easy_reference_inflation_ratio=fraction(p[3], y[3]),
                predicted_all_risk=fraction(p[1], p[2]), observed_all_risk=fraction(y[1], y[2]),
                predicted_easy_risk=fraction(p[4], p[3]), observed_easy_risk=fraction(y[4], y[3]))
        records = [v for g in groups for v in g['selected_recordings'].values()]
        medians = {}
        for key in RATIOS:
            values = [r[key] for r in records if r[key] is not None]
            medians[key] = dict(median=statistics.median(values) if values else None, defined_views=len(values))
        output[family] = dict(site_groups=len(groups), record_views=len(records), pooled=pooled,
            selected_record_ratio_summaries=medians,
            known_all_violating_views=sum(r['observed_all_risk_on_known'] is not None
                and r['observed_all_risk_on_known'] > .02 for r in records),
            known_easy_violating_views=sum(r['observed_easy_risk_on_known'] is not None
                and r['observed_easy_risk_on_known'] > .02 for r in records),
            undefined_selected_easy_views=sum(r['observed_easy_risk_on_known'] is None for r in records))
    return dict(families=output, model_states=len(rows), site_groups=sum(len(r['groups']) for r in rows),
        source_resubstitution_not_validation=True, repeated_views_not_independent=True,
        policy_changed=False, confirmation_access=False, deployment_changed=False)
