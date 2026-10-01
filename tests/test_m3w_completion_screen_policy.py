import copy
import pytest
from src.world_model.m3w_completion_screen_policy import choose


def candidates():
    return {k: dict(bound=dict(finite_completion_supported=True, reasons=[],
        all_selected_risk_upper=.01, easy_selected_risk_upper=.01,
        easy_degradation_upper=0., selected_net_gain_lower_mass=1.),
        old=dict(all_gain_fraction=.1, selected_count=10)) for k in ('mse', 'final', 'initial')}


def test_same_ranking_and_tie_order():
    rows = candidates()
    assert choose(rows)['selected'] == 'mse'
    rows['initial']['old']['selected_count'] = 9
    assert choose(rows)['selected'] == 'initial'
    rows['final']['old']['all_gain_fraction'] = .2
    assert choose(rows)['selected'] == 'final'


def test_support_not_raw_gain_decides_eligibility():
    rows = candidates()
    for name in rows:
        rows[name]['bound']['finite_completion_supported'] = False
    assert choose(rows)['selected'] == 'fallback'
    rows['initial']['bound']['finite_completion_supported'] = True
    rows['final']['old']['all_gain_fraction'] = 100
    before = copy.deepcopy(rows)
    result = choose(rows)
    assert result['selected'] == 'initial' and rows == before
    assert not result['per_row_label_inference_gate']


@pytest.mark.parametrize('key,value', [('all_selected_risk_upper', None),
    ('easy_selected_risk_upper', .02001), ('selected_net_gain_lower_mass', 0),
    ('all_selected_risk_upper', float('nan'))])
def test_unknown_or_invalid_support_cannot_pass(key, value):
    rows = candidates(); rows['mse']['bound'][key] = value
    with pytest.raises(ValueError):
        choose(rows)


def test_missing_candidate_rejected():
    with pytest.raises(ValueError):
        choose({})
