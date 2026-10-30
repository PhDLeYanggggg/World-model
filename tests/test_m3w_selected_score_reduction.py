import copy
import numpy as np
import pytest
from scripts import probe_m3w_selected_scores as report
from tests.test_m3w_selected_score_audit import fixture


def rows(monkeypatch):
    contexts = {f'c{i}': [f's{4*(i%3)+j}' for j in range(4)] for i in range(6)}
    class Registration:
        def read_text(self):
            import json
            return json.dumps(dict(contexts=contexts))
    monkeypatch.setattr(report.unit, 'REG', Registration())
    p, y, scale, allowed, groups = fixture(); original = np.array([0, 0, -1, -1])
    decoded = np.array([1, 0, 1, -1]); empty = np.full(4, -1)
    values = {mode+'_'+arm+'_'+scope: report.audit(p, y, scale, allowed, groups, original,
        original if scope == 'original' else decoded)
        for mode in ('old', 'new') for arm in ('row', 'set') for scope in ('original', 'decoded')}
    out = [dict(context=c, source=s, seed=seed, scores=copy.deepcopy(values), legacy_MSE_controls_close=True)
        for c, ss in contexts.items() for s in ss for seed in (17, 29, 43)]
    for r in out:
        if r['source'] == 's0': r['scores']['old_row_original'] = report.audit(p, y, scale, allowed, groups, original, empty)
    return out


def test_conditional_support_does_not_impute_missing_sources_or_claim_full_effect(monkeypatch):
    s = report.summarize(rows(monkeypatch)); v = s['variants']['old_row_original']
    mse = v['moments']['mean_score_squared_error']
    assert mse['supported_views'] == 66 and mse['required_views'] == 72
    assert mse['supported_localities'] == 11 and mse['required_localities'] == 12
    assert mse['conditional_support_only_not_full_cohort_effect']
    assert v['counts']['rows'] == 288 and v['counts']['queries'] == 144
    assert len(s['variants']) == 8 and s['all288_original_selected_MSE_controls_close']
    assert not s['independent_confirmation'] and not s['deployment_changed']


def test_reduction_requires_all_original_context_source_seed_groups(monkeypatch):
    values = rows(monkeypatch)
    with pytest.raises(ValueError): report.summarize(values[:-1])
    with pytest.raises(ValueError): report.summarize(values[:-1]+[values[0]])
