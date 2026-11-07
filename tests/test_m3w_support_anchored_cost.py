from itertools import combinations

import numpy as np

from src.world_model import m3w_support_anchored_cost as model


def test_top_k_maximizes_the_anchored_surrogate():
    value = np.array([2., 1., 4., 3.]); radius = np.array([3., 10., 20., 1.])
    prob = np.array([.05, .2, .5, .02]); ref = np.array([True, True, False, False])
    score = model.anchored_score(value, radius, prob, ref)
    def objective(a):
        return ((a-ref.astype(float))*(1-prob)*value-abs(a-ref.astype(float))*prob*radius).sum()
    masks = [np.isin(np.arange(4), ix) for ix in combinations(range(4), 2)]
    choice = np.isin(np.arange(4), np.argsort(-score)[:2])
    np.testing.assert_allclose(objective(choice), max(map(objective, masks)))


def test_no_missingness_recovers_dispersion_and_preserves_count():
    query = np.array([0, 0, 0, 1, 1, 2]); ids = np.arange(len(query))
    scores = np.array([[9, 8, 7, 2, 1, 4], [8, 6, 9, 1, 2, 4]], dtype=float)
    pred = np.stack((scores, np.zeros_like(scores)), axis=-1)
    chosen = model.choices(query, ids, pred, np.full(len(query), 9.), np.zeros(len(query)), 0.)
    for arm in chosen[2:]:
        np.testing.assert_array_equal(chosen[1], arm)
    np.testing.assert_array_equal(chosen.sum(1), np.full(len(model.ARMS), 3))


def test_high_missingness_penalizes_changing_not_selecting_per_se():
    result = model.anchored_score(np.ones(2), np.full(2, 10.), np.ones(2), [True, False])
    np.testing.assert_array_equal(result, [10., -10.])
