import copy
import pytest
from scripts.run_m3w_selection_harm_deviance import summarize
from scripts.report_m3w_selection_harm_deviance import pool_fixed_costs


def rows():
    output = []
    for fold in range(3):
        for seed in (17, 29, 43):
            for site in range(6):
                policies = {}
                for arm in ('mean_absolute', 'easy_deviance', 'dual_deviance'):
                    view = dict(supported=False, known_all_risk=.03, known_easy_risk=.05,
                                completion_easy_risk_upper=.06, known_easy_net_degradation=.08)
                    policies[arm] = dict(known_ADE_gain_floor_percent=10. if arm == 'dual_deviance' else 1.,
                                         recordings={'r':view})
                effects = {a:dict(known_pp=9., completion_lower_pp=5., completion_upper_pp=12.)
                           for a in ('easy_deviance', 'dual_deviance')}
                output.append(dict(trial=f'single{fold}_seed{seed}', site=str(site), policies=policies, effects=effects))
    return output


CONFIG = dict(bootstrap_resamples=3000, bootstrap_seed=20261102)


def test_positive_mean_and_primary_cannot_override_original_risk_failure():
    out = summarize(rows(), CONFIG)
    assert out['primary_pass'] and not out['original_record_risk_pass'] and not out['candidate_pass']
    assert out['safety']['dual_deviance']['worst_known_easy_net_degradation_percent'] == 8.
    assert out['selection_development_not_confirmation'] and not out['deployment_changed']


def test_missing_primary_bound_is_not_a_success_and_all_roster_is_retained():
    data = rows(); data[0]['effects']['dual_deviance']['completion_lower_pp'] = None
    out = summarize(data, CONFIG)
    assert out['effects']['dual_deviance']['completion_lower_pp']['point'] is None
    assert not out['primary_pass'] and out['groups'] == 54


def test_missing_or_duplicate_producer_views_rejected():
    data = rows(); data.pop()
    with pytest.raises(ValueError): summarize(data, CONFIG)
    data = rows(); data[0] = copy.deepcopy(data[1])
    with pytest.raises(ValueError): summarize(data, CONFIG)


def fixed_costs():
    cell = dict(known_action_views=2,
                normalized_MSE={k:.25 for k in ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')},
                observed_known_mass={k:4. for k in ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')})
    return {a:[copy.deepcopy(cell)] for a in ('mean_absolute', 'easy_deviance', 'dual_deviance')}


def test_fixed_bank_aggregation_rejects_changed_support_or_labels():
    data = fixed_costs()
    assert pool_fixed_costs(data)['dual_deviance']['normalized_MSE']['harm'] == .25
    data['dual_deviance'][0]['known_action_views'] = 1
    with pytest.raises(ValueError, match='support and target masses'): pool_fixed_costs(data)
    data = fixed_costs(); data['easy_deviance'][0]['observed_known_mass']['easy_harm'] = 5.
    with pytest.raises(ValueError, match='support and target masses'): pool_fixed_costs(data)


def test_fixed_bank_empty_or_nonfinite_costs_are_not_valid():
    data = fixed_costs(); data['easy_deviance'] = []
    with pytest.raises(ValueError): pool_fixed_costs(data)
    data = fixed_costs(); data['dual_deviance'][0]['normalized_MSE']['harm'] = float('nan')
    with pytest.raises(ValueError): pool_fixed_costs(data)
