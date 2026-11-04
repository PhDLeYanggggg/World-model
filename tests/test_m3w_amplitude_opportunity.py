import numpy as np
import pytest
from src.evaluation.m3w_amplitude_opportunity import FRACTIONS,rollout


def test_endpoints_are_exact_and_all_fractions_are_frozen():
    b=np.arange(48,dtype=float).reshape(2,12,2)
    p=-b
    assert np.array_equal(rollout(b,p,0.),b)
    assert np.array_equal(rollout(b,p,1.),p)
    assert FRACTIONS == (1/32,1/16,1/8,1/4,1/2,1.)
    for alpha in FRACTIONS:
        np.testing.assert_allclose(rollout(b,p,alpha)-b,alpha*(p-b),atol=1e-14)


def test_causal_path_builder_rejects_future_label_parameters_and_invalid_inputs():
    b=np.zeros((2,12,2));p=np.ones_like(b)
    with pytest.raises(TypeError):
        rollout(b,p,.5,future_endpoint=np.ones((2,2)))
    with pytest.raises(ValueError):
        rollout(b,p,.37)
    p[0,0,0]=np.nan
    with pytest.raises(ValueError):
        rollout(b,p,.5)
