import pytest
from scripts import report_m3w_moving_rollout_history as report
from scripts import replay_m3w_moving_rollout_history as replay


def rows():
    return [dict(job=str(i), arm=arm, step=2000, parameters=68356,
        final_fitting={'mean_MSE': 2. if arm == 'masked' else 1.},
        joint_attribution={k: 2. if arm == 'masked' else 3. for k in
            ('false_safe_cost', 'false_reject_cost', 'total_joint_cost')})
        for i in range(3) for arm in report.api.ARMS]


def test_lower_mse_does_not_hide_worse_decision_cost():
    r = report.comparisons(rows(), [{'key': str(i)} for i in range(3)])
    assert all(x['mean_MSE_reduction_percent'] == 50. for x in r)
    assert all(x['total_joint_cost_reduction_percent'] == -50. for x in r)
    assert report.reduction(0., 1.) is None


def test_partial_or_duplicate_model_roster_rejected():
    jobs = [{'key': str(i)} for i in range(3)]
    for bad in (rows()[:-1], rows()[:-1]+[rows()[0]]):
        with pytest.raises(ValueError, match='six unique'):
            report.comparisons(bad, jobs)


def test_incomplete_replay_never_creates_receipts(tmp_path):
    with pytest.raises(ValueError, match='Complete original run required'):
        replay.require_completed(tmp_path)
    assert not list(tmp_path.iterdir())


def test_source_influence_keeps_negative_and_unknown_results():
    out = report.source_consistency({'known_pp': {'by_source': {'a': 3., 'b': -1., 'c': 0.}}})
    assert (out['positive'], out['negative'], out['unchanged']) == (1, 1, 1)
    assert out['leave_one_source_out_mean_pp']['a'] == -.5
    assert out['full_primary_unchanged'] and not out['sampling_CI']
    out = report.source_consistency({'known_pp': {'by_source': {'a': 3., 'b': None}}})
    assert out['status'] == 'not_estimable'
