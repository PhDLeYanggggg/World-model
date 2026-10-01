import numpy as np
import pytest
from src.world_model import m3w_selected_pool_accounting as api


def inputs():
    # The retained row has 3% harm; removing a safe reference row raises 1.5% to 3%.
    y = np.array([[0., 3., 100., 100., 3.], [1., 0., 100., 100., 0.], [np.nan]*5])
    p = np.array([[4., 1., 100., 100., 1.]]*3)
    return y, p, p.copy(), np.array([5., 5., 5.]), np.array([True]*3), np.array([True, False, True])


def test_removing_safe_mass_can_raise_risk():
    result = api.account(*inputs(), audit=True)
    c = result['easy_contrast']
    assert c['risk_delta'] == pytest.approx(.015)
    assert c['harm_retained'] == 1 and c['reference_retained'] == .5
    assert c['new_known_violation']
    assert result['scalar_and_identity_checks'] == 76


def test_unknown_is_not_zero_or_inference_exclusion():
    result = api.account(*inputs())
    assert result['kept']['rows'] == 2 and result['kept']['unknown'] == 1
    assert result['kept']['unknown_envelope'] == 5
    assert result['kept']['truth'][2] == 100


def test_abstention_remains_undefined():
    args = list(inputs()); args[-1] = np.zeros(3, bool)
    result = api.account(*args)
    assert result['kept']['easy']['observed_risk'] is None
    assert result['easy_contrast']['risk_delta'] is None
    assert not result['easy_contrast']['new_known_violation']


def test_label_change_never_changes_action_counts():
    args = list(inputs()); before = api.account(*args)
    args[0] = np.full((3, 5), np.nan)
    after = api.account(*args)
    assert before['kept']['rows'] == after['kept']['rows']
    assert after['kept']['known'] == 0 and after['kept']['unknown'] == 2


def test_reject_non_nested_sets():
    args = list(inputs()); args[-2] = np.zeros(3, bool)
    with pytest.raises(ValueError): api.account(*args)


def test_reject_partial_moment_missingness():
    args = list(inputs()); args[0][0, 0] = np.nan
    with pytest.raises(ValueError): api.account(*args)


def test_partition_and_scale_invariance():
    args = list(inputs()); original = api.account(*args, audit=True)
    scaled = [a*7 if i < 4 else a for i, a in enumerate(args)]
    other = api.account(*scaled, audit=True)
    assert original['easy_contrast'] == pytest.approx(other['easy_contrast'])


def test_known_zero_reference_is_not_pass():
    args = list(inputs()); args[0][:2] = 0
    result = api.account(*args, audit=True)
    assert result['raw']['easy']['observed_risk'] is None
    assert result['easy_contrast']['mass_identity'] is None
