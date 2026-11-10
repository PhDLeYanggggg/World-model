import numpy as np

from src.world_model.m3w_exchange_direction_bank import (
    bridge_fraction, direction_bank, match_direction, select, suffix_scores,
)
from src.world_model.m3w_ols_replacement import rollout


def test_direction_replacement_preserves_every_radius():
    cv = np.zeros((1, 12, 2))
    a = cv.copy()
    a[0, :, 0] = np.arange(12)
    b = cv.copy()
    b[0, 1:, 1] = -1.
    b[0, -1] = 0.
    p = match_direction(cv, a, b)
    np.testing.assert_allclose(np.linalg.norm(p-cv, axis=-1), np.linalg.norm(a-cv, axis=-1))
    assert p[0, 2, 1] == -2.
    np.testing.assert_array_equal(p[0, -1], a[0, -1])


def test_history_only_bank_and_suffix_are_translation_invariant():
    h = np.random.default_rng(8).normal(size=(8, 8, 2)).cumsum(1)
    cv, ols, _, _ = rollout(h, 12)
    a = np.linspace(0., 1., len(h))
    anchor = cv+a[:, None, None]*(ols-cv)
    np.testing.assert_allclose(bridge_fraction(cv, ols, anchor), a)
    q = direction_bank(h, cv, anchor)
    np.testing.assert_allclose(direction_bank(h+7., cv, anchor), q, atol=1e-12)
    np.testing.assert_allclose(suffix_scores(h+7., a), suffix_scores(h, a), atol=1e-12)


def test_consensus_requires_each_observed_suffix_to_improve():
    s = np.array([[[1., .2, 2.]], [[1., .2, 2.]], [[1., 1.1, 2.]]])
    mean, unanimous = select(s)
    assert mean.tolist() == [1]
    assert unanimous.tolist() == [0]
