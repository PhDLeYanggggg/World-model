import pytest

from src.world_model.m3w_stream_spline_calibration import require_roles, select_recording_rule


def test_roles_require_exact_exclusions_and_seed():
    identity = dict(fit=['c', 'd'], omitted=['a', 'b'], seed=17)
    require_roles(identity, ['a', 'b', 'c', 'd'], ('a', 'b'), 17)
    with pytest.raises(ValueError):
        require_roles(dict(identity, fit=['b', 'd']), ['a', 'b', 'c', 'd'], ('a', 'b'), 17)
    with pytest.raises(ValueError):
        require_roles(identity, ['a', 'b', 'c', 'd'], ('a', 'b'), 29)


def test_recording_selection_cannot_trade_donor_risk_for_another_donor():
    def cell(risk, dual, gain):
        return dict(risk_pass=risk, dual_positive=dual, vs_cv=gain, views=10)

    data = {None: {'a': [cell(3, 2, 1.)], 'b': [cell(3, 1, 1.)]},
            (0., 1.): {'a': [cell(2, 1, 10.)], 'b': [cell(9, 8, 10.)]},
            (.005, 1.): {'a': [cell(3, 3, 2.)], 'b': [cell(3, 1, 1.)]},
            (.015, 1.): {'a': [cell(3, 2, .5)], 'b': [cell(3, 1, .5)]}}
    rule, table = select_recording_rule(data)
    assert rule == (.005, 1.)
    assert [r['admissible'] for r in table] == [True, False, True, False]
