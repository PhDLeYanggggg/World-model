import pytest
from scripts import report_m3w_moving_source_robust as report
from scripts import replay_m3w_moving_source_robust as replay


def rows():
    out = []
    for i in range(3):
        for arm in report.api.ARMS:
            cost = {k:2. if arm == 'mean_continue' else 3. for k in
                ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')}
            v = 2. if arm == 'mean_continue' else 1.
            out.append(dict(job=str(i), arm=arm, step=1000, parameters=68356,
                final_fitting={'mean_MSE':v}, joint_attribution=cost,
                final_sources=dict(worst_source_objective=v, by_source={'a':dict(total=v, queries=10, joint=cost)})))
    return out


def test_better_worst_loss_does_not_hide_higher_harm():
    out = report.comparisons(rows(), [{'key':str(i)} for i in range(3)])
    assert all(r['worst_source_objective_reduction_percent'] == 50. for r in out)
    assert all(r['false_safe_cost_reduction_percent'] == -50. for r in out)
    assert out[0]['per_source'][0]['total_joint_cost_reduction_percent'] == -50.
    assert report.reduction(0, 1) is None


def test_partial_or_wrong_budget_rejected():
    jobs = [{'key':str(i)} for i in range(3)]
    wrong = rows(); wrong[0]['step'] = 999
    for bad in (rows()[:-1], rows()[:-1]+[rows()[0]], wrong):
        with pytest.raises(ValueError, match='six unique'):
            report.comparisons(bad, jobs)


def test_not_run_is_not_a_safety_pass():
    analysis = dict(status='fitting_complete_held_readout_not_run', reason='screen failed')
    screen = dict(followup_readout_allowed=False)
    out = report.evidence_status(analysis, screen)
    assert out['held_readout'] == out['original_risk_pass'] == 'not_run'
    assert not out['deployment_changed'] and not out['submission_ready']
    with pytest.raises(ValueError, match='No held metrics'):
        report.evidence_status(dict(analysis, held_policy={}), screen)
    with pytest.raises(ValueError, match='Fixed fitting screen'):
        report.evidence_status(analysis, dict(followup_readout_allowed=True))


def test_incomplete_replay_never_writes(tmp_path):
    with pytest.raises(ValueError, match='Complete original run required'):
        replay.require_completed(tmp_path)
    assert not list(tmp_path.iterdir())
