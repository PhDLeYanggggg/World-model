from scripts.report_m3w_moving_neural_transport import reduction, summary
from scripts.run_m3w_moving_neural_transport import summary as model_summary, ARMS


def fixture():
    metrics={a:dict(normalized_MSE=1.,total_joint_cost=1.,false_safe_cost=0.,false_reject_cost=1.,
        accepted_known_proposals=1,false_safe_count=0) for a in ARMS}
    metrics['neural_transport']=dict(metrics['neural_transport'],normalized_MSE=.2,total_joint_cost=.5,
        false_safe_cost=.1,false_reject_cost=.4,false_safe_count=1)
    return dict(metrics=metrics,steps=100,rows=10,minimum_fitting_visits=32)


def test_better_loss_does_not_erase_harm_or_promote_model():
    f=fixture(); s=model_summary([f]); assert not s['passes_information_screen']
    a=dict(contexts=[dict(job='mock',folds=[f],summary=s)],passing_contexts=0)
    r=summary(a)
    assert r['primary_updates']==100 and r['evaluated_known_rows']==10
    assert r['contexts'][0]['arms']['neural_transport']['harmful']==1
    assert not r['safety_certificate'] and not r['deployment_changed']
    assert r['new_neural_risk_training'] and not r['new_neural_trajectory_training']


def test_zero_reference_reduction_is_unestimable():
    assert reduction(0.,0.) is None
