import numpy as np

from scripts.run_m3w_vehicle_opportunity import rank_metrics
from src.world_model.m3w_vehicle_opportunity import WIDTH, nominate, summarize, top_mask


def test_screen_inputs_and_count_are_invariant_to_actor_order():
    x = np.random.default_rng(17).normal(size=(2, 12, 50))
    v = summarize(x, 4)
    assert v.shape == (WIDTH,)
    np.testing.assert_allclose(v, summarize(x[::-1], 4))
    np.testing.assert_array_equal(top_mask(np.array([1., 1., 2., 1.]), 2), [True, False, True, False])


def test_nomination_is_valid_even_when_ego_policy_abstains():
    x = np.zeros((2, 12, 3))
    x[..., 1] = 0.8
    x[1, 3, 0] = 0.4
    pick, score = nominate(x)
    assert pick == (1, 4) and score < 0


def test_tied_scores_do_not_get_artificial_rank_credit():
    labels = np.array([False, True, True, False])
    assert rank_metrics(np.ones(4), labels) == {'auc': 0.5, 'ap': 0.5}
    assert rank_metrics(labels.astype(float), labels) == {'auc': 1., 'ap': 1.}
    assert rank_metrics(np.ones(4), np.zeros(4, dtype=bool)) is None
