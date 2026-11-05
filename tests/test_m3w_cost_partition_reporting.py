import copy
import pytest
from scripts import report_m3w_moving_cost_partition as report


def fixture():
    contexts = []
    for i,n in enumerate((2,3,3)):
        metrics = dict(normalized_MSE=1., total_joint_cost=0., false_safe_cost=0., false_reject_cost=0.)
        arms = {a:dict(metrics=dict(metrics, known_proposals=4, true_joint_safe_proposals=2,
            accepted_known_proposals=0, false_safe_count=0, false_reject_count=2),
            decomposition=dict(same_partition_factorization_max_abs_difference=0.,
                original_leaf_mean_reconstruction_max_abs_difference=0., direct_vs_factorized_joint_sign_changes=0,
                original_vs_reconstructed_joint_sign_changes=0)) for a in report.run.method.ARMS}
        contexts.append(dict(job=str(i), own_B_rows=20, known_proposed_rows=16, unknown_proposed_rows=2,
            folds=[dict(held_source=str(k), arms=copy.deepcopy(arms)) for k in range(n)],
            summary=dict(source_macro={a:dict(metrics) for a in arms})))
    return dict(cells=24, contexts=contexts, fresh_primary_fits=16, cached_controls=8,
        primary_passing_contexts=0, neural_mechanism_followup_supported=False)


def test_summary_retains_controls_and_undefined_zero_cost():
    result = fixture(); s = report.summarize(result)
    assert len(s['rows']) == 9 and len(s['decisions']) == 9
    assert s['rows'][0]['joint_cost_reduction_percent'] is None
    assert not s['deployment_changed'] and not s['independent_confirmation']
    result['contexts'][0]['folds'][0]['held_source'] = '1'
    with pytest.raises(ValueError):
        report.summarize(result)


def test_report_lock_and_replay_refuse_incomplete(tmp_path, monkeypatch):
    import fcntl
    from scripts import replay_m3w_moving_cost_partition as replay
    monkeypatch.setattr(report.run, 'PRIVATE', tmp_path); monkeypatch.setattr(report.run, 'PUBLIC', tmp_path)
    with (tmp_path/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            report.main()
    with pytest.raises(ValueError, match='Complete study'):
        replay.require_complete()
    with pytest.raises(RuntimeError, match='cannot fit'):
        replay.forbidden_fit()
