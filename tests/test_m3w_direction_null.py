import numpy as np
import pytest

from src.evaluation.m3w_direction_null import rotation_nulls, rotated_costs


def test_nulls_preserve_per_step_magnitude_and_zero_target_harm():
    p = np.random.default_rng(17).normal(size=(9,12,2))
    for value in rotation_nulls(p).values():
        np.testing.assert_allclose(np.linalg.norm(value,axis=-1),np.linalg.norm(p,axis=-1),rtol=1e-14)
    errors = rotated_costs(p,np.zeros_like(p))
    for value in errors.values(): np.testing.assert_allclose(value,errors['original'],rtol=1e-14)


def test_correct_direction_is_better_than_fixed_nulls_and_input_unchanged():
    p = np.zeros((2,12,2)); p[:,:,0] = 2
    before = p.copy(); costs = rotated_costs(p,p)
    assert not costs['original'].any()
    np.testing.assert_allclose(costs['plus90'],np.sqrt(8))
    np.testing.assert_allclose(costs['minus90'],np.sqrt(8))
    np.testing.assert_allclose(costs['reverse'],4)
    np.testing.assert_array_equal(p,before)
    with pytest.raises(ValueError): rotation_nulls(np.zeros((2,2)))
    with pytest.raises(ValueError): rotated_costs(p,np.full_like(p,np.nan))
