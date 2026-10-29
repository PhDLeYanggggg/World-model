import numpy as np
import pytest

from scripts import evaluate_m3w_decoder_risk_factorial as api
from tests.test_m3w_shared_train_risk import packet


def predictions():
    return {arm: np.tile([.1, 0., 1., 1., 0.], (32, 1)) for arm in api.ARMS}


def test_fixed_identical_policies_cancel_unknowns_and_interaction():
    result, _ = api.evaluate(packet(), predictions())
    assert result['unknown_rows'] == 1
    assert all(abs(v) < 1e-12 for v in result['contrasts'].values())
    api.check_result(result)


def test_no_imputed_empty_risk_or_missing_comparator():
    p = predictions()
    p['local'][:, 0] = 0
    result, _ = api.evaluate(packet(), p)
    policy = result['matched']['source_risk_minus_local']['source_risk']
    assert policy['selected_count'] == 0 and policy['easy_selected_risk_upper'] is None
    assert not policy['finite_completion_supported']
    with pytest.raises(ValueError):
        api.evaluate(packet(), {})


def test_corrupt_risk_rejected():
    result, _ = api.evaluate(packet(), predictions())
    result['policies']['source_utility']['selected_known_harm_mass'] += 1
    with pytest.raises(ValueError):
        api.check_result(result)


def test_missing_fixed_views_rejected():
    with pytest.raises(ValueError):
        api.summarize([], {'contexts': {}, 'seeds': [17]}, {})
