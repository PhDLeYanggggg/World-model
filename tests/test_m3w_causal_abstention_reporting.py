import numpy as np

from scripts.report_m3w_european_causal_abstention import removal_summary, POLICIES


def test_all_policy_names_are_retained():
    assert len(POLICIES) == 10 and len(set(POLICIES)) == 10
    for guard in ('stop', 'support', 'combined'):
        assert all(guard+s in POLICIES for s in ('', '_risk', '_random'))


def test_removal_summary_uses_common_denominator():
    records = {'a': dict(floor_error_sum=20, avoided_harm=2, lost_benefit=1),
               'b': dict(floor_error_sum=100, avoided_harm=1, lost_benefit=3)}
    out = removal_summary([dict(removal={s: records for s in ('all', 'easy', 'hard', 'complete')})])['all']
    np.testing.assert_allclose(out['avoided_harm_pp'], [5.5, 5.5])
    np.testing.assert_allclose(out['lost_benefit_pp'], [4, 4])
    np.testing.assert_allclose(out['change_pp'], [1.5, 1.5])


def test_nontrivial_count_matched_query_accounting(tmp_path, monkeypatch):
    import json
    from scripts import report_m3w_european_causal_abstention as report
    monkeypatch.setattr(report.run, 'PRIVATE', tmp_path/'private')
    monkeypatch.setattr(report.run, 'PUBLIC', tmp_path/'public')
    p = report.run.PRIVATE/'batch/decisions/example.npz'; p.parent.mkdir(parents=True)
    arrays = dict(query=np.array([0, 0, 1, 1]))
    for parent in ('cv_targets', 'floor_both'):
        arrays[parent+'__original'] = np.array([1, 1, 1, 0], bool)
        for guard in ('stop', 'support', 'combined'):
            arrays[parent+'__'+guard] = np.array([1, 0, 0, 0], bool)
            for control in ('risk', 'random'):
                arrays[parent+'__'+guard+'_'+control] = np.array([0, 1, 0, 0], bool)
    np.savez(p, **arrays)
    p.with_suffix('.json').write_text(json.dumps(dict(fitted_support=dict(boxes=[]))))
    out = report.decision_context({'batch': {'example': {}}})['batch']['example']['policies']['cv_targets__stop']
    assert out['free_choice_queries'] == 1 and out['fully_rejected_queries'] == 1
    assert out['risk_different_rows'] == 2 and out['risk_different_queries'] == 1
