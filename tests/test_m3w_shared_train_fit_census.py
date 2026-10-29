import copy

import pytest

from scripts import diagnose_m3w_shared_train_fit as api


def rows():
    result = dict(rows=20, unknown_rows=1, policies={arm: dict(
        known_easy_positive_risk=.01, finite_completion_supported=False)
        for arm in api.evaluation.ARMS}, scores={arm: 1. for arm in api.evaluation.ARMS})
    values = [dict(context='c'+str(i//12), source='s'+str(i%12), seed=17,
                   result=copy.deepcopy(result)) for i in range(72)]
    return values, {(r['context'], r['source'], r['seed']): copy.deepcopy(r) for r in values}


def test_fit_pass_held_failure_is_not_scientific_success():
    fit, held = rows()
    for row in held.values():
        row['result']['policies']['shared_source']['known_easy_positive_risk'] = .1
        row['result']['scores']['shared_source'] = 3.
    result = api.summarize(fit, held)
    assert result['arms']['shared_source']['fit_known_pass_held_fail'] == 72
    assert result['arms']['shared_source']['median_held_fit_MSE_ratio'] == 3
    assert not result['scientific_success'] and result['optimizer_steps'] == 0


def test_missing_or_duplicate_fit_view_rejected():
    fit, held = rows()
    with pytest.raises(ValueError):
        api.summarize(fit[:-1], held)
    with pytest.raises(ValueError):
        api.summarize(fit[:-1]+[fit[0]], held)


def test_undefined_easy_is_not_a_known_pass():
    fit, held = rows()
    for row in fit:
        row['result']['policies']['local']['known_easy_positive_risk'] = None
    for row in held.values():
        row['result']['policies']['local']['known_easy_positive_risk'] = .1
    result = api.summarize(fit, held)['arms']['local']
    assert result['fit_known_easy_undefined'] == 72
    assert result['fit_known_pass_held_fail'] == 0
