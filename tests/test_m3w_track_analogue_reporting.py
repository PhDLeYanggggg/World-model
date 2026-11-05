from scripts.report_m3w_moving_track_analogue import reduction, summary, fmt
from src.evaluation.m3w_track_analogue import context_summary


def fold():
    arms = {}
    for name, mse, cost, harm in [('shared64', 1., 1., 0.), ('prior', 2., 2., 0.),
                                ('track32', .5, .9, .1), ('misaligned32', 1., 1.2, .2)]:
        arms[name] = dict(normalized_MSE=mse, total_joint_cost=cost,
            false_safe_cost=harm, false_reject_cost=cost-harm,
            accepted_known_proposals=3, false_safe_count=int(harm > 0))
    return dict(metrics=arms)


def test_lower_loss_and_cost_cannot_hide_more_harm():
    result = context_summary([fold()])
    assert result['primary_cost_reduction'] > 0
    assert not result['passes_information_screen']
    assert not result['safety_certificate']


def test_summary_retains_failure_and_exact_counts():
    f = fold()
    a = dict(contexts=[dict(job='test', folds=[f, f], summary=context_summary([f, f]))], passing_contexts=0)
    result = summary(a)
    row = result['contexts'][0]['arms']['track32']
    assert row['MSE_reduction_percent'] == 50.
    assert row['admissions'] == 6 and row['harmful_admissions'] == 2
    assert not result['safety_certificate'] and not result['deployment_changed']
    assert not result['independent_confirmation']


def test_zero_reference_is_not_a_perfect_score():
    assert reduction(0., 0.) is None
    assert fmt(None) == 'NA'
