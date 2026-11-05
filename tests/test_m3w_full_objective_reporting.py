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


def test_failed_fitting_screen_does_not_read_held_packets(tmp_path, monkeypatch):
    from scripts import run_m3w_moving_full_objective_risk as run
    monkeypatch.setattr(run, 'PUBLIC', tmp_path)
    run.source.once(tmp_path/'training_freeze.json', dict(complete_heads=3, all_before_validation=True))
    run.source.once(tmp_path/'fitting_screen.json', dict(followup_readout_allowed=False))
    def forbidden(*args, **kwargs):
        raise AssertionError('No held packets after failed fitting screen')
    monkeypatch.setattr(run.old, 'packet', forbidden)
    run.evaluate({}, None, forbidden, forbidden)
    result = run.source.read(tmp_path/'analysis.json')
    assert result['status'] == 'fitting_complete_held_readout_not_run'
    assert not result['deployment_changed'] and not result['identity_certification']


def test_fixed_rules_and_choices_before_future_readout():
    import ast
    from pathlib import Path
    from scripts import run_m3w_moving_full_objective_risk as run
    cfg = run.source.read(run.CONFIG)
    assert cfg['risk_budget'] == cfg['easy_net_budget'] == .02
    assert cfg['new_heads'] == 3 and cfg['training']['outer_steps'] == 200
    assert not any(cfg[k] for k in ('threshold_search', 'new_H_readout', 'identity_certification', 'deployment_changed'))
    tree = ast.parse(Path(run.__file__).read_text())
    evaluate = next(ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    assert evaluate.index('api.parent.parent.choose(') < evaluate.index("data['target_eval']")
    assert "freeze['complete_heads'] != 3" in evaluate


def test_attribution_keeps_unknown_cost_and_new_errors_distinct():
    import numpy as np
    from scripts.diagnose_m3w_full_objective_errors import decompose
    y = np.array([[[2., -1.]], [[4., 3.]]])
    before = np.array([[[1., 1.]], [[0., 0.]]])
    after = np.array([[[-1., -1.]], [[1., 1.]]])
    r = decompose(before, after, y, np.array([.5, .5]), np.ones((2, 1), bool),
        np.array([True, False]), np.ones(2, bool))
    assert r['newly_wrong_cost'] == [1., 0.]
    assert r['repaired_wrong_cost'] == [0., .5]
    assert r['initial']['unknown_bound_MSE'] == [8., 4.5]
    assert r['final']['false_safe_cost'] == [1., 0.]
