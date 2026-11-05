import copy
import pytest
from scripts import report_m3w_moving_peer_participation as report
from scripts.replay_m3w_moving_peer_participation import require_completed


def rows():
    jobs = [dict(key=str(i)) for i in range(3)]
    data = [dict(job=j['key'], arm=arm, step=1000, parameters=76548,
        result_source='fresh_run' if arm in report.api.NEW_ARMS else 'cached_verified',
        final_fitting=dict(mean_MSE=1 if arm == 'same_action' else .9),
        joint_attribution=dict(total_joint_cost=1, false_safe_cost=0, false_reject_cost=1))
        for j in jobs for arm in report.api.ARMS]
    return data, jobs


def test_all_factor_contrasts_and_cache_provenance():
    data, jobs = rows(); values = report.comparisons(data, jobs)
    assert len(values) == 12
    assert values[0]['contrast'] == 'primary_identity_all_peers'
    assert values[0]['mean_MSE_reduction_percent'] == pytest.approx(10)
    assert values[0]['false_safe_cost_reduction_percent'] is None
    for bad in (data[:-1], data[:-1]+[data[0]]):
        with pytest.raises(ValueError, match='Twelve unique'):
            report.comparisons(bad, jobs)
    bad = copy.deepcopy(data); bad[0]['result_source'] = 'fresh_run'
    with pytest.raises(ValueError, match='cached'):
        report.comparisons(bad, jobs)


def test_not_run_never_counted_as_scientific_pass():
    analysis = dict(status='factorial_fitting_complete_held_readout_not_run')
    screen = dict(followup_readout_allowed=False)
    result = report.evidence_status(analysis, screen)
    assert result['original_risk_pass'] == result['held_readout'] == 'not_run'
    assert not result['submission_ready'] and not result['deployment_changed']
    with pytest.raises(ValueError):
        report.evidence_status(dict(analysis, held_policy={}), screen)
    with pytest.raises(ValueError):
        report.evidence_status(analysis, dict(followup_readout_allowed=True))


def test_replay_requires_completed_evidence(tmp_path):
    with pytest.raises(ValueError, match='Complete original run'):
        require_completed(tmp_path)


def test_report_and_readonly_run_share_lock(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(report.run, 'PRIVATE', tmp_path)
    calls = []; monkeypatch.setattr(report, 'render', lambda:calls.append(True))
    with (tmp_path/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            report.main()
    assert calls == []
    report.main(); assert calls == [True]


def test_context_separation_is_finite_and_does_not_change_parent(tmp_path):
    from tests.test_m3w_proposal_context_risk import fixture
    from scripts.diagnose_m3w_moving_peer_participation import context_separation
    d, warm, _ = fixture(tmp_path); before = copy.deepcopy(warm)
    result = context_separation(d, warm, lambda:None)
    report.api.base.exact(before, warm)
    assert set(result) == {'same_active', 'proposed_action', 'proposed_all'}
    for value in result.values():
        assert value['proposals'] == 3
        assert value['relative_RMS_difference'] >= 0
        assert -1 <= value['cosine_mean'] <= 1
