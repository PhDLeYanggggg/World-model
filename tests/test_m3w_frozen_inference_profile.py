import numpy as np

from scripts.profile_m3w_frozen_inference import compare


def sample(value=1., choice=0):
    prediction = np.array([[[value, 0.]]], dtype=np.float32)
    return {17: {'self': prediction}}, [np.array([[choice]])], [prediction], {}


def test_profile_rejects_float_drift_even_when_choices_stay_the_same():
    a, b = sample(), sample(np.nextafter(np.float32(1.), np.float32(2.)))
    assert compare(a, a)['exact']
    result = compare(a, b)
    assert not result['exact']
    assert result['network_parity']['self']['changed_values'] == 1
    assert result['changed_actions'] == 0
    assert result['changed_selected_path_values'] == 1


def test_profile_detects_choice_changes_and_signed_zero():
    assert compare(sample(), sample(choice=-1))['changed_actions'] == 1
    result = compare(sample(0.), sample(-0.))
    assert result['network_parity']['self']['changed_values'] == 0
    assert not result['network_parity']['self']['bitwise']
    assert not result['exact']
