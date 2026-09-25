import numpy as np
import pytest
from src.world_model.m3w_producer_conditioned import augment, placebo_tag, producer_roles, safe_choice, independent_choice


def test_producer_exclusion_and_held_source_boundary():
    sites = np.array(['a', 'b', 'c', 'd', 'e'])
    np.testing.assert_array_equal(producer_roles(sites, np.arange(4), np.array([4]), [['a', 'b'], ['c', 'd']]), [1, 1, 0, 0])
    with pytest.raises(ValueError): producer_roles(sites, np.arange(4), np.array([0, 4]), [['a', 'b'], ['c', 'd']])
    with pytest.raises(ValueError): producer_roles(sites, np.arange(4), np.array([4]), [['a', 'b'], ['b', 'c']])


def test_equal_dimension_and_causal_features_unchanged():
    x = np.arange(4*380, dtype=np.float32).reshape(4, 380); ids = np.arange(4); tags = np.array([1, 1, 0, 0])
    for arm in ('global', 'producer', 'placebo'):
        out = augment(x, tags, ids, arm)
        assert out.shape == (4, 382)
        np.testing.assert_array_equal(out[:, :380], x)
    np.testing.assert_array_equal(augment(x, tags, ids, 'global')[:, -2:], 0)
    np.testing.assert_array_equal(augment(x, tags, ids, 'producer')[:, -2:], [[0, 1], [0, 1], [1, 0], [1, 0]])


def test_placebo_is_order_invariant_and_nonconstant():
    ids = np.arange(10000); out = placebo_tag(ids)
    assert .45 < out.mean() < .55
    np.testing.assert_array_equal(placebo_tag(ids[::-1]), out[::-1])
    np.testing.assert_array_equal(placebo_tag(ids[:10]), out[:10])


def test_safe_rule_replays_and_never_undoes_stop():
    u = np.array([[2., 1.], [2., 1.], [2., 1.], [0., 1.]])
    r = np.array([[10., .1], [10., .3], [10., .1], [10., 0.]])
    move = np.array([True, True, False, True])
    np.testing.assert_array_equal(safe_choice(u, r, move), [True, False, False, False])
    np.testing.assert_array_equal(safe_choice(u, r, move), independent_choice(u, r, move))
    with pytest.raises(ValueError): safe_choice(u, r, move, .03)


def test_invalid_identity_or_future_width_rejected():
    with pytest.raises(ValueError): augment(np.zeros((2, 382)), np.array([0, 1]), np.arange(2), 'producer')
    with pytest.raises(ValueError): augment(np.zeros((2, 380)), np.array([0, 2]), np.arange(2), 'producer')
