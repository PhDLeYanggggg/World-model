import numpy as np

from scripts import report_m3w_moving_readout_refit as report
from scripts import run_m3w_moving_readout_refit as run


def test_flow_keeps_benefit_harm_and_missing_outcomes_separate():
    moments = np.zeros((4, 2, 5))
    moments[0, 0] = [3, 0, 20, 0, 0]
    moments[1, 0] = [0, 2, 5, 5, 2]
    moments[2, 0] = [900, 900, 900, 900, 900]
    truth = dict(moments=moments, observed_future=np.array([True, True, False, True]))
    f = run.flow(truth, np.zeros(4, int), np.array([0, 0, -1, -1]), np.array([0, -1, 0, -1]))
    pooled = report.pool_flow([dict(action_flow=f), dict(action_flow=f)])
    assert pooled['retained']['known_moments'] == [6., 0., 40., 0., 0.]
    assert pooled['newly_rejected']['known_net_benefit'] == -4.
    assert pooled['newly_rejected']['known_easy_budget_excess'] == 3.8
    assert pooled['newly_admitted']['unknown_rows'] == 2
    assert pooled['newly_admitted']['known_moments'] == [0.]*5
    assert all(v['repeated_rows_not_independent'] for v in pooled.values())


def test_reporting_keeps_empty_and_failed_rows_visible():
    p = dict(known=dict(mean=None), hard=dict(mean=-1.), switch_percent=dict(mean=0.),
        safety=dict(supported=0, required=17, known_all_violations=2,
            known_easy_violations=5, undefined_selected_easy=15))
    text = report.policy_table({n: p for n in ('public_router', 'conditioned_veto',
        'mean_veto', 'quantile90_veto', 'readout_veto')})
    assert text.count('NA') == 5 and text.count('0/17') == 5
    assert 'readout_veto' in text and '-1.000000' in text


def test_least_squares_optimality_does_not_calibrate_selected_subset():
    # Mathematical control: exact weighted linear optimum, nonlinear selection bias.
    x = np.array([-2., -1., 0., 1., 2.])
    design = np.column_stack([np.ones(5), x])
    target = np.column_stack([x*x-.5+.5*x]*2)
    beta, stats = run.api.solve(design, target, np.ones(5)/5, np.zeros((2, 2)))
    prediction = design@beta
    residual = target-prediction
    assert np.max(np.abs(design.T@residual)) < 1e-12
    selected = prediction[:, 0] < .75
    assert selected.sum() == 1 and residual[selected, 0].mean() > 1.
    assert stats['relative_normal_residual'] < 1e-12
