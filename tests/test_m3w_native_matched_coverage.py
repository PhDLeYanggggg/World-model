import inspect
import numpy as np
import pytest
from src.evaluation.m3w_native_matched_coverage import (
    top_count, ratio_score, matched_policies, hash_priority, paired_scene_contrast,
)


def test_exact_count_and_query_id_ties():
    ids = np.array([30, 10, 20, 40])
    mask = top_count(np.array([1., 1., 1., 9.]), np.array([1, 1, 1, 0], bool), ids, 2)
    np.testing.assert_array_equal(mask, [False, True, True, False])
    assert not top_count(np.ones(4), np.ones(4, bool), ids, 0).any()


@pytest.mark.parametrize('count', [-1, 5, 1.5, True])
def test_invalid_budget_rejected(count):
    with pytest.raises(ValueError):
        top_count(np.ones(4), np.ones(4, bool), np.arange(4), count)


def test_scores_and_alignment_rejected():
    with pytest.raises(ValueError):
        top_count(np.array([np.nan, 1.]), np.ones(2, bool), np.arange(2), 1)
    with pytest.raises(ValueError):
        top_count(np.ones(2), np.ones(2, bool), np.array([1, 1]), 1)
    with pytest.raises(ValueError):
        ratio_score(np.array([[1., -.1]]))


def test_ratio_zero_preserved_not_epsilon_gain():
    np.testing.assert_allclose(ratio_score(np.array([[0, 0], [2, 1], [1, 2]])), [0, 1/3, -1/3])


def test_hash_order_invariant():
    ids = np.array([7, 3, 9, 1])
    perm = np.array([2, 0, 3, 1])
    p = hash_priority(ids, 'site', 17)
    np.testing.assert_array_equal(hash_priority(ids[perm], 'site', 17), p[perm])


@pytest.mark.parametrize('budget', ['positive_gain', 'harm_fraction_0p1'])
def test_matched_counts_without_future_or_support(budget):
    a = np.array([[10., .5], [2, 1], [1, 3], [0, 0], [3, .2]])
    scores = dict(underharm4=a, mse=a[::-1].copy(), ridge_raw=a-1)
    same = np.array([False, False, False, True, False])
    choices = matched_policies(scores, same, np.arange(5), site='site', seed=17, budget=budget)
    n = int(choices['asym_rule'].sum())
    assert all(v.sum() == n and not v[3] for v in choices.values())
    assert not {'target', 'future', 'valid', 'label', 'known'} & set(inspect.signature(matched_policies).parameters)


def test_paired_contrast_uses_scenes_not_rows():
    r = paired_scene_contrast([2., 5., 4., 8.], [1., 4., 3., 7.])
    assert r['mean_gain_difference_pp'] == 1
    assert r['ci95_pp'] == [1, 1]
    assert len(r['scene_differences_pp']) == 4
