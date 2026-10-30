import copy
import json

import pytest

from scripts import report_m3w_active_bank_cost as report


def aggregate_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    refs = []
    for context in range(6):
        for j in range(4):
            source = f's{4 * (context % 3) + j}'
            for seed in (17, 29, 43):
                value = dict(context=f'c{context}', source=source, seed=seed,
                    ids_sha256=f'ids-{source}', result=dict(rows=10, known_rows=8,
                        unknown_rows=2, recordings={f'{source}-r0': {}, f'{source}-r1': {}}))
                path = tmp_path / f'c{context}_{source}_{seed}.json'
                path.write_text(json.dumps(value))
                refs.append(dict(path=path.name, sha256=report.run.sha(path)))
    return dict(groups=refs)


def test_support_counts_unique_queries_not_seeds_contexts_or_candidate_banks(tmp_path, monkeypatch):
    complete = aggregate_fixture(tmp_path, monkeypatch)
    result = report.support_profile(complete)
    assert result['unique_queries'] == 120
    assert result['unique_known_queries'] == 96
    assert result['unique_unknown_queries'] == 24
    assert result['unique_recordings'] == 24
    assert result['repeated_row_views'] == 720
    assert result['repeated_known_views'] == 576
    assert result['repeated_unknown_views'] == 144
    assert result['repeated_views_not_independent_samples']


def test_support_rejects_duplicate_group_changed_ids_and_incomplete_matrix(tmp_path, monkeypatch):
    complete = aggregate_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='All72'):
        report.support_profile(dict(groups=complete['groups'][:-1]))
    with pytest.raises(ValueError, match='Repeated source'):
        report.support_profile(dict(groups=complete['groups'][:-1] + complete['groups'][:1]))
    changed = copy.deepcopy(complete)
    path = tmp_path / changed['groups'][-1]['path']
    value = json.loads(path.read_text())
    value['ids_sha256'] = 'changed-support'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='aggregate changed'):
        report.support_profile(changed)
    changed['groups'][-1]['sha256'] = report.run.sha(path)
    with pytest.raises(ValueError, match='changed query support'):
        report.support_profile(changed)


def test_evidence_keeps_first_exact_replay_runtime_but_not_scientific_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(report.run, 'PUBLIC', tmp_path)
    first = dict(replay=dict(pid=1, seconds=10), scientific=dict(gain=-.1), bindings='frozen')
    assert report.freeze(first) == first
    later = dict(first, replay=dict(pid=2, seconds=9))
    assert report.freeze(later) == first
    with pytest.raises(ValueError, match='artifact differs'):
        report.freeze(dict(later, scientific=dict(gain=.1)))
    with pytest.raises(ValueError, match='artifact differs'):
        report.freeze(dict(later, bindings='changed-code'))


def test_empty_interval_is_not_a_safe_zero():
    assert report.interval(dict(point=None, CI95=None)) == 'not_estimable'
    assert report.interval(dict(point=.1, CI95=[-.2, .3])) == '+0.100000 [-0.200000, +0.300000]'


def test_selected_moment_aggregation_weights_support_and_keeps_missing_undefined():
    def row(n, value):
        d = dict(known_selected_rows=n, known_legal_action_rows=2 * n,
            selected_harm_underprediction_rows=n, selected_easy_harm_underprediction_rows=0,
            selected_normalized_bias=[value] * 5 if n else None,
            normalized_moment_MSE=[value] * 5 if n else None)
        return dict(result=dict(calibration=dict(new_neural=d),
            policies=dict(new_neural=dict(action_counts={str(i): 0 for i in range(-1, 7)}))))
    result = report.selected_moment_summary([row(1, 1.), row(3, 3.), row(0, None)], 'new_neural')
    assert result['known_selected_repeated_views'] == 4
    assert result['weighted_selected_bias'] == [2.5] * 5
    assert result['weighted_legal_moment_MSE'] == [2.5] * 5
    assert not result['probability_calibration_or_safety_certificate']
    empty = report.selected_moment_summary([row(0, None)], 'new_neural')
    assert empty['weighted_selected_bias'] is None
    assert empty['weighted_legal_moment_MSE'] is None
