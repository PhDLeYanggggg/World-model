import pytest

from src.world_model import m3w_dronecrowd_calibration as m


def test_confirmation_is_denied_even_if_index_is_wrong():
    spec = dict(recording_role_index={'a': 'calibration', 'b': 'calibration'},
                assignments=dict(calibration={'recordings': ['a', 'b']},
                                 confirmation={'recordings': ['b']}))
    m.require_calibration(spec, 'a')
    with pytest.raises(PermissionError):
        m.require_calibration(spec, 'b')
    with pytest.raises(PermissionError):
        m.require_calibration(spec, 'unknown')


def test_undefined_or_zero_switches_never_win_calibration():
    s = dict(risk={k: dict(supported_views=119, required_views=120, switches=20) for k in m.GATES},
             effects={f'{k}_vs_cv': dict(completion_lower_pp=dict(interval95=[.1, .3], point=.2))
                      for k in m.GATES})
    assert m.winner(s) is None
    s['risk']['gate_0'].update(supported_views=120, switches=0)
    assert m.winner(s) is None
    s['risk']['gate_1'].update(supported_views=120)
    assert m.winner(s) == 'gate_1'
