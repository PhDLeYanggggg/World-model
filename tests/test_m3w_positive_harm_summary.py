import copy
import numpy as np
from src.world_model import m3w_positive_harm as api
from scripts.verify_m3w_positive_harm import aggregate, compare
from test_m3w_past_quality_auxiliary import fixture

CFG = dict(bootstrap_resamples=3000, bootstrap_seed=20261002)


def rows():
    state, x, e, y, q, s, r, f = fixture()
    fitted = api.fit(state, x, e, y, q, s, r, f, settings={})
    original, positive, support, _ = api.predict(state, fitted, x, e, q)
    result, _ = api.evaluate(state, dict(original=original, positive=positive, additive=original),
        y, e, np.ones(len(x), bool), support, s, r, f//4, np.arange(len(x)))
    return [dict(source='source'+str(i), result=copy.deepcopy(result)) for i in range(12)]


def test_independent_scalar_bootstrap_summary_matches():
    r = rows()
    assert compare(api.summarize(r, CFG), aggregate(r, CFG)) > 50


def test_undefined_risk_does_not_become_safety_success():
    r = rows()
    for row in r:
        for p in row['result']['policies'].values():
            p['easy_selected_risk_upper'] = None; p['finite_completion_supported'] = False
    out = api.summarize(r, CFG)
    assert out['positive']['defined_easy_risk'] == 0
    assert not out['advance_to_transfer'] and 'worst_risk_not_preserved' in out['gate_failure_reasons']


def test_zero_predictive_contrasts_do_not_advance():
    r = rows()
    for row in r:
        for key in row['result']['contrasts']: row['result']['contrasts'][key] = 0.
    out = api.summarize(r, CFG)
    assert not out['advance_to_transfer']
    assert 'MSE_not_supported_vs_original' in out['gate_failure_reasons']
    assert 'matched_utility_not_supported_vs_additive' in out['gate_failure_reasons']
