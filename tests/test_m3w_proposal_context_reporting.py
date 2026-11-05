import copy
import pytest
from scripts import report_m3w_moving_proposal_context as report
from scripts.replay_m3w_moving_proposal_context import require_completed


def rows():
    jobs = [dict(key=str(i)) for i in range(3)]
    data = [dict(job=j['key'], arm=arm, step=1000, parameters=76548,
        final_fitting=dict(mean_MSE=1 if arm == 'same_action' else .9),
        joint_attribution=dict(total_joint_cost=1, false_safe_cost=0, false_reject_cost=1))
        for j in jobs for arm in report.api.ARMS]
    return data, jobs


def test_complete_matched_report_and_zero_denominator():
    data, jobs = rows(); value = report.comparisons(data, jobs)
    assert value[0]['mean_MSE_reduction_percent'] == pytest.approx(10)
    assert value[0]['false_safe_cost_reduction_percent'] is None
    for bad in (data[:-1], data[:-1]+[data[0]]):
        with pytest.raises(ValueError, match='six unique'):
            report.comparisons(bad, jobs)
    bad = copy.deepcopy(data); bad[0]['step'] = 999
    with pytest.raises(ValueError):
        report.comparisons(bad, jobs)


def test_not_run_cannot_be_reported_as_pass():
    analysis = dict(status='fitting_complete_held_readout_not_run', reason='failed screen')
    screen = dict(followup_readout_allowed=False)
    result = report.evidence_status(analysis, screen)
    assert result['original_risk_pass'] == result['held_readout'] == 'not_run'
    assert not result['submission_ready'] and not result['deployment_changed']
    with pytest.raises(ValueError):
        report.evidence_status(dict(analysis, held_policy={}), screen)
    with pytest.raises(ValueError):
        report.evidence_status(analysis, dict(followup_readout_allowed=True))


def test_completed_replay_refuses_incomplete_run(tmp_path):
    with pytest.raises(ValueError, match='Complete original run'):
        require_completed(tmp_path)


def test_bridge_removal_keeps_frozen_model_unchanged(tmp_path):
    import torch
    from scripts.diagnose_m3w_moving_proposal_context import remove_bridge
    original = dict(model={'risk.bridge.weight':torch.ones(4, 4),
        'decision.bridge.weight':torch.ones(4, 4), 'risk.encoder.weight':torch.ones(4, 3)})
    changed = remove_bridge(original)
    assert original['model']['risk.bridge.weight'].sum() == 16
    assert changed['model']['risk.bridge.weight'].sum() == 0
    assert changed['model']['decision.bridge.weight'].sum() == 0
    assert torch.equal(original['model']['risk.encoder.weight'], changed['model']['risk.encoder.weight'])
