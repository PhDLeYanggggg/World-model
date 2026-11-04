import json
import numpy as np
import pytest
from scripts.run_m3w_moving_joint_harm_runtime import serializable_prior


def test_only_event_prior_is_converted_without_mutating_input():
    prior = np.full((7, 4), .25)
    original = lambda v: json.loads(json.dumps(v, sort_keys=True, allow_nan=False))
    value = dict(event_prior=prior, checkpoint='unchanged')
    result = serializable_prior(value, original)
    np.testing.assert_array_equal(result['event_prior'], prior)
    assert value['event_prior'] is prior and result['checkpoint'] == 'unchanged'
    assert serializable_prior({'a': 3}, original) == {'a': 3}
    with pytest.raises(TypeError): serializable_prior({'other': prior}, original)
