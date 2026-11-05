from scripts.report_m3w_moving_full_objective_risk import comparison


def test_lower_MSE_does_not_hide_increased_sign_cost():
    r = comparison(dict(job='x', stop_reason='budget_exhausted_not_proven_converged',
        initial_fitting=dict(mean_MSE=2., task_MSE=[1., 3.], observed_sign_regret=[1., 0.], false_safe_channels=[2, 0]),
        final_fitting=dict(mean_MSE=1., task_MSE=[.5, 1.5], observed_sign_regret=[2., 0.],
            false_safe_channels=[1, 0], gradient_inf=.1)))
    assert r['MSE_reduction_percent'] == 50
    assert r['sign_regret_reduction_percent'] == [-100., None]
    assert r['false_safe_after'] == [1, 0]
    assert r['stop_reason'] == 'budget_exhausted_not_proven_converged'
