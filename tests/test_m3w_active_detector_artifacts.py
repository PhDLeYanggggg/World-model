import json
import pytest
from scripts import report_m3w_active_detector_forecaster as report


def fixtures(root, reference):
    refs = []
    for i in range(9):
        for arm in report.run.api.REPRESENTATIONS:
            metrics = dict(all=dict(ADE=dict(indexed=4, known=3), FDE=dict(known=2)),
                           complete_future=dict(ADE=dict(indexed=2)))
            record = dict(reference_ADE=dict(mean=reference, p95=reference),
                          native_ADE=dict(history=dict(mean=1.03, p95=1.05)))
            value = dict(trial='single' + str(i), representation=arm, kind='single',
                         result=dict(rows=4, complete_future_rows=2, partial_future_rows=1,
                                     missing_future_rows=1, stationary_observed_rows=0,
                                     sites={'site0': dict(metrics=metrics, records={'record0': record})}))
            path = root / (str(i) + '_' + arm + '.json')
            path.write_text(json.dumps(value))
            refs.append(dict(path=path.name, sha256=report.run.sha(path)))
    return dict(groups=refs)


@pytest.mark.parametrize('reference', [0., 1.])
def test_active_artifact_keeps_unique_support_repeated_views_and_undefined_tails(tmp_path, monkeypatch, reference):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    value = fixtures(tmp_path, reference)
    result = report.support_profile(value)
    assert result['producer_views'] == 9 and result['indexed_row_views'] == 36
    assert result['unique_queries'] == dict(indexed=4, complete_future=2, partial_future=1, missing_future=1)
    assert result['unique_recordings'] == 1
    tails = result['known_record_mean_tail_diagnostic']
    assert tails['mean_worse_by_over2percent'] == (9 if reference else 0)
    assert tails['p95_worse_by_over2percent'] == (9 if reference else 0)
    assert tails['undefined_mean_views'] == (0 if reference else 9)
    assert tails['descriptive_not_approved_positive_harm_policy_gate']
    with pytest.raises(ValueError):
        report.support_profile(dict(groups=value['groups'][:-1]))
    with pytest.raises(ValueError):
        report.support_profile(dict(groups=value['groups'][:-1] + value['groups'][:1]))


def test_active_artifact_rejects_support_drift_or_aggregate_hash_change(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    complete = fixtures(tmp_path, 1.)
    ref = next(v for v in complete['groups'] if v['path'] == '1_history.json')
    path = tmp_path / ref['path']
    value = json.loads(path.read_text())
    value['result']['sites']['site0']['metrics']['all']['ADE']['known'] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='changed'):
        report.support_profile(complete)
    ref['sha256'] = report.run.sha(path)
    with pytest.raises(ValueError, match='support'):
        report.support_profile(complete)


def test_active_report_retains_first_exact_replay_runtime_but_not_scientific_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(report.run, 'PUBLIC', tmp_path)
    first = dict(summary_sha256='original', replay=dict(pid=1, seconds=2.))
    assert report.freeze_evidence(first) == first
    later = dict(summary_sha256='original', replay=dict(pid=3, seconds=4.))
    assert report.freeze_evidence(later) == first
    with pytest.raises(ValueError):
        report.freeze_evidence(dict(later, summary_sha256='changed'))
