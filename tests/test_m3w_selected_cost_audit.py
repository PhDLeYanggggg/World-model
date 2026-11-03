import numpy as np

from src.evaluation.m3w_selected_cost_audit import selected_costs


def example():
    predicted = np.zeros((2, 7, 5))
    predicted[:, 6] = [2, .01, 10, 5, .001]
    truth = np.zeros_like(predicted)
    truth[0, 6] = [0, 4, 10, 5, 3]
    truth[1, 6] = [0, 2, 0, 0, 2]
    return dict(predicted=predicted, truth=truth, known=np.array([True, False]),
        choice=np.array([6, 6]), groups=[np.array([0, 1])])


def test_predicted_feasibility_does_not_imply_realized_safety():
    r = selected_costs(**example())
    assert r['predicted_full_all_risk'] == .001
    assert r['known_realized_all_risk'] == .4
    assert r['known_realized_easy_risk'] == .6
    assert r['completion_all_risk_upper'] == .6
    assert r['known_harm_mass_over_predicted_same_rows'] == 400
    assert r['violations']['all']['predicted_budget_violating_queries'] == 0
    assert r['violations']['all']['known_realized_budget_violating_queries'] == 1
    assert not r['labels_used_for_selection']


def test_empty_selection_is_undefined_not_a_measured_safe_zero():
    data = example()
    data['choice'][:] = -1
    r = selected_costs(**data)
    assert r['selected'] == 0
    assert r['known_realized_all_risk'] is None
    assert r['known_harm_mass_over_predicted_same_rows'] is None
    assert r['violations']['easy']['known_reference_unsupported_queries'] == 1
