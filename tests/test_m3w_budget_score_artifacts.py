import copy
import json

import pytest

from scripts import report_m3w_budget_score_cost as report


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    monkeypatch.setattr(report.prior, 'ROOT', tmp_path)
    refs = []
    for objective in report.run.api.OBJECTIVES:
        for context in range(6):
            for j in range(4):
                source = f's{4 * (context % 3) + j}'
                for seed in (17, 29, 43):
                    value = dict(context=f'c{context}', source=source, seed=seed, objective_arm=objective,
                        ids_sha256=f'ids-{source}', result=dict(rows=10, known_rows=8, unknown_rows=2,
                            recordings={f'{source}-r0': {}, f'{source}-r1': {}}))
                    path = tmp_path / f'{objective}_c{context}_{source}_{seed}.json'
                    path.write_text(json.dumps(value))
                    refs.append(dict(path=path.name, sha256=report.run.sha(path)))
    return dict(groups=refs)


def test_counts_never_multiply_unique_queries_by_objectives_banks_or_seeds(tmp_path, monkeypatch):
    complete = fixture(tmp_path, monkeypatch)
    profile = report.support_profile(complete)
    assert profile['unique_queries'] == 120
    assert profile['unique_recordings'] == 24
    assert profile['unique_known_queries'] == 96
    assert profile['repeated_row_views_per_objective'] == 720
    assert profile['repeated_row_views_all_objectives'] == 2160
    assert profile['objective_banks_and_seeds_not_independent_samples']


def test_all216_unique_groups_and_same_support_between_objectives_required(tmp_path, monkeypatch):
    complete = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='All216'):
        report.support_profile(dict(groups=complete['groups'][:-1]))
    with pytest.raises(ValueError, match='Repeated'):
        report.support_profile(dict(groups=complete['groups'][:-1] + complete['groups'][:1]))
    changed = copy.deepcopy(complete)
    for ref in changed['groups']:
        path = tmp_path / ref['path']
        value = json.loads(path.read_text())
        if value['source'] == 's0' and value['objective_arm'] == 'upper_score':
            value['ids_sha256'] = 'coherent-but-different-upper-support'
            path.write_text(json.dumps(value))
            ref['sha256'] = report.run.sha(path)
    with pytest.raises(ValueError, match='identical query support'):
        report.support_profile(changed)


def test_evidence_does_not_replace_initial_exact_replay_or_allow_scientific_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(report.run, 'PUBLIC', tmp_path)
    first = dict(replay=dict(pid=1, seconds=10), summary='frozen', bindings='original')
    assert report.freeze(first) == first
    assert report.freeze(dict(first, replay=dict(pid=2, seconds=9))) == first
    with pytest.raises(ValueError):
        report.freeze(dict(first, summary='different-result'))
    with pytest.raises(ValueError):
        report.freeze(dict(first, bindings='changed-code'))


def test_figure_uses_all18_new_bank_curves_per_objective(tmp_path, monkeypatch):
    from src.evaluation import m3w_budget_score_readout as score
    from tests.test_m3w_budget_score_readout import records
    rows, contexts, cfg = records()
    summary = score.summarize(rows, contexts, cfg, dict(count=162, exact_original_controls=54))
    public, private = tmp_path / 'public', tmp_path / 'private'
    public.mkdir()
    private.mkdir()
    monkeypatch.setattr(report.run, 'PUBLIC', public)
    monkeypatch.setattr(report.run, 'PRIVATE', private)
    (public / 'summary.json').write_text(json.dumps(summary))
    (public / 'evidence.json').write_text(json.dumps(dict(summary_sha256=report.run.sha(public / 'summary.json'))))
    traces = [dict(objective_arm=o, variant='new_neural', trace=[dict(step=1, loss=2.), dict(step=2000, loss=.5)])
        for o in report.run.api.OBJECTIVES for _ in range(18)]
    (public / 'training_traces.json').write_text(json.dumps(dict(rows=traces)))
    report.figures()
    assert (public / 'figures/summary.svg').stat().st_size > 1000
    assert (private / 'summary.png').stat().st_size > 1000
    (public / 'training_traces.json').write_text(json.dumps(dict(rows=traces[:-1])))
    with pytest.raises(ValueError, match='All18'):
        report.figures()


def test_full_reference_support_is_not_selected_support_and_is_seed_bank_invariant():
    rows = [dict(context='c0', source='s0', seed=s, objective_arm='original',
        result=dict(recordings=dict(r0={v: dict(full_known_easy_reference_mass=2., selected_count=0)
            for v in report.run.api.active.VARIANTS}))) for s in (17, 29, 43)]
    support = report.full_reference_support(rows)
    assert support['seed_deduplicated_context_record_views'] == 1
    assert support['zero_full_known_easy_reference_views'] == 0
    assert support['positivity_is_not_selected_policy_safety']
    changed = copy.deepcopy(rows)
    changed[-1]['result']['recordings']['r0']['new_neural']['full_known_easy_reference_mass'] = 0
    with pytest.raises(ValueError, match='selected bank'):
        report.full_reference_support(changed)
