import copy
import pytest
from src.evaluation import m3w_calibration_source_fit as api


def cases():
    output = []
    for family in api.FAMILIES:
        cells = []
        for j in range(8):
            mass = dict(benefit=1., harm=2., reference=4., easy_reference=1., easy_harm=.1)
            pred = dict(benefit=1., harm=1., reference=5., easy_reference=2., easy_harm=.01)
            scope = dict(known_action_views=10, unknown_action_views=2,
                         predicted_known_mass=pred, observed_known_mass=mass)
            rec = dict(harm_prediction_ratio=.5, easy_harm_prediction_ratio=.1,
                reference_inflation_ratio=1.25, easy_reference_inflation_ratio=2.,
                observed_all_risk_on_known=.5, observed_easy_risk_on_known=.1)
            if j == 0:
                rec = {key:None for key in rec}
            cells.append(dict(site=str(j), target_cost_sha256=str(j),
                diagnosis={s:copy.deepcopy(scope) for s in api.SCOPES}, selected_recordings={str(j):rec}))
        output.append(dict(trial='producer', family=family, rows=100, groups=cells))
    return output


def test_matching_groups_retain_unknowns_and_condition_record_medians():
    out = api.aggregate(cases(), ['producer'])
    a = out['families']['mean_absolute']
    assert out['model_states'] == 2 and out['site_groups'] == 16
    assert a['pooled']['selected']['observed_all_risk'] == .5
    assert a['pooled']['selected']['unknown_action_views'] == 16
    assert a['selected_record_ratio_summaries']['harm_prediction_ratio'] == dict(median=.5, defined_views=7)
    assert a['undefined_selected_easy_views'] == 1
    assert a['known_easy_violating_views'] == 7
    assert out['source_resubstitution_not_validation'] and not out['deployment_changed']


@pytest.mark.parametrize('change', ['missing_family', 'duplicate_family', 'missing_site', 'different_targets'])
def test_incomplete_or_mismatched_comparisons_rejected(change):
    rows = cases()
    if change == 'missing_family':
        rows.pop()
    elif change == 'duplicate_family':
        rows[-1]['family'] = rows[0]['family']
    elif change == 'missing_site':
        rows[0]['groups'].pop()
    else:
        rows[-1]['groups'][0]['target_cost_sha256'] = 'changed'
    with pytest.raises(ValueError):
        api.aggregate(rows, ['producer'])
