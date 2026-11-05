import copy
import pytest
from scripts import report_m3w_moving_peer_path_probe as report


def fixture():
    contexts = []
    for i, n in enumerate((2, 3, 3)):
        metrics = dict(normalized_MSE=1., total_joint_cost=0., false_safe_cost=0., false_reject_cost=0.)
        families = {f:dict(summary={a:dict(metrics, normalized_MSE=.8 if a == 'aligned180' else 1.)
            for a in report.run.method.ARMS}, passes_internal_screen=True) for f in report.run.method.FAMILIES}
        contexts.append(dict(job=str(i), own_B_rows=20, known_proposed_rows=16, unknown_proposed_rows=2,
            sources=list(range(n)), folds=[dict(held_source=str(k)) for k in range(n)], families=families))
    return dict(fits=48, primary_family='extra_trees', contexts=contexts, passing_contexts=3, followup_neural_supported=True)


def test_report_keeps_all_families_and_na_denominators():
    result = fixture(); summary = report.summarize(result)
    assert len(summary['rows']) == 6 and summary['source_folds'] == 8
    assert summary['rows'][0]['MSE_reduction_percent'] == pytest.approx(20.)
    assert summary['rows'][0]['joint_cost_reduction_percent'] is None
    assert not summary['neural_training'] and not summary['deployment_changed']
    for key, value in (('fits', 47), ('primary_family', 'ridge')):
        with pytest.raises(ValueError):
            report.summarize(dict(result, **{key:value}))
    bad = copy.deepcopy(result); bad['contexts'][0]['folds'][0]['held_source'] = '1'
    with pytest.raises(ValueError):
        report.summarize(bad)


def test_reporting_uses_shared_lock(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(report.run, 'PRIVATE', tmp_path)
    called = []; monkeypatch.setattr(report, 'render', lambda:called.append(True))
    with (tmp_path/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            report.main()
    assert not called


def test_readonly_replay_cannot_fit_or_run_incomplete(tmp_path, monkeypatch):
    from scripts import replay_m3w_moving_peer_path_probe as replay
    monkeypatch.setattr(replay.run, 'PUBLIC', tmp_path)
    with pytest.raises(ValueError, match='Complete study'):
        replay.require_complete()
    with pytest.raises(RuntimeError, match='cannot train'):
        replay.forbidden_fit()


def test_decision_counts_do_not_treat_no_switch_as_success():
    result = fixture()
    for c in result['contexts']:
        for fold in c['folds']:
            fold['families'] = {'extra_trees':{a:dict(accepted_known_proposals=0,
                true_joint_safe_proposals=3, false_safe_count=0, false_reject_count=3)
                for a in report.run.method.ARMS}}
    counts = report.decision_counts(result)
    assert counts[0]['arms']['aligned180']['accepted_known_proposals'] == 0
    assert counts[0]['arms']['aligned180']['false_reject_count'] == 6
    assert counts[0]['source_macro_false_safe_cost']['aligned180'] == 0
    assert report.reduction(0, 0) is None
