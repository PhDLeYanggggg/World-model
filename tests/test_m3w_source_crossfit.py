import numpy as np
import pytest

from src.world_model.m3w_source_crossfit import (
    nested_partition, fold_normalizer, validate_query, assemble_oof, cost_labels,
)


def test_outer_and_inner_sites_are_excluded_and_roles_cover_source():
    sites = np.array(['outer', 'a', 'b', 'c', 'a'])
    train, held, outer = nested_partition(sites, np.arange(5), np.arange(5), 'outer', 'a')
    np.testing.assert_array_equal(train, [2, 3])
    np.testing.assert_array_equal(held, [1, 4])
    np.testing.assert_array_equal(outer, [0])


@pytest.mark.parametrize('identity', ['track', 'recording'])
def test_alias_leakage_is_rejected(identity):
    tracks, records = np.arange(3), np.arange(3)
    (tracks if identity == 'track' else records)[2] = 1
    with pytest.raises(ValueError, match='leakage'):
        nested_partition(np.array(['outer', 'a', 'b']), tracks, records, 'outer', 'a')


def test_outer_role_cannot_be_inner_role():
    with pytest.raises(ValueError, match='Distinct'):
        nested_partition(np.array(['outer', 'a']), np.arange(2), np.arange(2), 'outer', 'outer')


def test_normalizer_ignores_held_and_outer_statistics():
    x = np.arange(24, dtype=np.float32).reshape(6, 4)
    a, _, _ = fold_normalizer(x, [2, 3, 4])
    x[[0, 1, 5]] = 1e7
    b, _, _ = fold_normalizer(x, [2, 3, 4])
    for key in ('mean', 'std', 'constant'):
        np.testing.assert_array_equal(a[key], b[key])


def test_queries_fail_closed_before_negative_indexing():
    allowed = np.array([False, False, True, False])
    for ids in ([-1], [4], [0], [3], []):
        with pytest.raises(ValueError, match='role'):
            validate_query(ids, 2, 4, allowed)
    np.testing.assert_array_equal(validate_query([2, 2], 2, 4, allowed), [2, 2])


def test_oof_merge_requires_exact_once_coverage_and_alignment():
    pieces = [dict(ids=np.array([5, 7]), prediction=np.ones((2, 12, 2)), cost_scale=np.ones(2)),
              dict(ids=np.array([3]), prediction=np.zeros((1, 12, 2)), cost_scale=np.ones(1)*2)]
    prediction, scale = assemble_oof(np.array([3, 5, 7]), pieces)
    assert not prediction[0].any() and scale.tolist() == [2, 1, 1]
    with pytest.raises(ValueError, match='exactly one'):
        assemble_oof(np.array([3, 5, 7]), pieces + pieces[:1])
    with pytest.raises(ValueError, match='coverage'):
        assemble_oof(np.array([3, 5, 6]), pieces)


def test_cost_labels_are_supervision_not_inputs():
    p = np.ones((2, 12, 2)); target = np.zeros_like(p); target[1] = 2
    result = cost_labels(p, target, np.array([2, 4]))
    assert result['harmful'].tolist() == [True, False]
    assert result['beneficial'].tolist() == [False, True]
    np.testing.assert_allclose(result['signed_gain'], [-np.sqrt(2)/2, np.sqrt(2)/4])
