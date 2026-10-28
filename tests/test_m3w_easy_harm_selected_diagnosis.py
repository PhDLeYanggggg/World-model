import pytest

from scripts import diagnose_m3w_easy_harm_selected_v3 as api


def test_signed_selected_risk_identity():
    d = dict(known_selected=4, predicted_moments=[4, 4, 100, 80, 1],
             observed_moments=[4, 8, 100, 100, 8], unknown_outcomes_imputed=False)
    r = api.decompose(d, .02)
    assert r['predicted_slack_pp'] == pytest.approx(-.6)
    assert r['harm_underprediction_pp'] == pytest.approx(7.)
    assert r['reference_overprediction_pp'] == pytest.approx(-.4)
    assert r['realized_excess_pp'] == pytest.approx(6.)
    assert r['predicted_actual_harm_ratio'] == pytest.approx(.125)


def test_zero_denominator_is_undefined_not_safe():
    d = dict(known_selected=0, predicted_moments=[0]*5,
             observed_moments=[0]*5, unknown_outcomes_imputed=False)
    r = api.decompose(d, .02)
    assert r['realized_excess_pp'] is None
    assert r['predicted_actual_harm_ratio'] is None


@pytest.mark.parametrize('change', ['negative', 'nonfinite', 'imputed', 'budget'])
def test_malformed_inputs_fail_closed(change):
    d = dict(known_selected=1, predicted_moments=[1]*5,
             observed_moments=[1]*5, unknown_outcomes_imputed=False)
    budget = .02
    if change == 'negative': d['observed_moments'][4] = -1
    elif change == 'nonfinite': d['predicted_moments'][2] = float('nan')
    elif change == 'imputed': d['unknown_outcomes_imputed'] = True
    else: budget = .05
    with pytest.raises(ValueError): api.decompose(d, budget)


def test_missing_views_not_silently_averaged():
    assert api.equal_locality_mean([('a', 1.), ('a', 3.), ('b', 9.)], 2) == 5.5
    assert api.equal_locality_mean([('a', 1.), ('a', None), ('b', 9.)], 2) is None
    assert api.equal_locality_mean([('a', 1.)], 2) is None
