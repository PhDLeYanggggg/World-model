from scripts.report_m3w_moving_risk_convergence import summarize, reduction
from scripts.run_m3w_moving_risk_convergence import summarize as screen, ARMS


def test_fit_loss_cannot_override_failed_source_held_result():
    metrics = {a: dict(normalized_MSE=1., total_joint_cost=1., false_safe_cost=0.,
                      false_reject_cost=1., accepted_known_proposals=0, false_safe_count=0) for a in ARMS}
    metrics['long_transport'] = dict(metrics['long_transport'], total_joint_cost=2., false_safe_cost=1.,
                                    accepted_known_proposals=3, false_safe_count=2)
    f = dict(source='held', fit_rows=100, extra_updates=4096, epochs=100,
             metrics=metrics, fitting_diagnosis=dict(start_fit_loss=1., end_fit_loss=.01,
                last_quarter_relative_reduction=.001, operational_plateau=True,
                mathematical_convergence_proved=False, selected_by_held_outcomes=False))
    c = dict(job='mock', folds=[f], summary=screen([f]))
    result = summarize(dict(contexts=[c], passing_contexts=0))
    assert result['fitting_loss_improved_folds'] == 1
    assert result['passing_contexts'] == 0
    assert not c['summary']['passes_information_screen']
    assert not result['deployment_changed'] and not result['mathematical_convergence_proved']
    assert result['contexts'][0]['arms']['long_transport']['harmful'] == 2
    assert reduction(0., 0.) is None
