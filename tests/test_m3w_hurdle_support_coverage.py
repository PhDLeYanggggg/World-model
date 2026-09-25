import numpy as np
import pytest
from src.evaluation.m3w_hurdle_support_coverage import support_choices, support_decomposition, ARMS


def test_support_difference_keeps_full_anchors_and_matches_common_counts():
    p = np.array([[0., 0.], [1., .01], [1., .04], [1., .01]])
    h = np.array([[1., .01], [1., .04], [1., .01], [1., .01]])
    choices, counts = support_choices(np.ones(4), p, h, np.ones(4, bool),
                                     np.array(['a', 'a', 'a', 'a']), np.arange(4), budget=.02)
    assert choices['hurdle_original'][0] and not choices['hurdle_common'][0]
    assert counts['a']['hurdle_outside_common'] == 1
    assert choices['hurdle_at_product'].sum() == choices['product_common'].sum() == 2
    assert choices['product_at_hurdle'].sum() == choices['hurdle_common'].sum() == 2
    assert not choices['hurdle_at_product'][0]


def test_support_effect_is_retained_in_full_total():
    cv = np.full(4, 2.)
    errors = {name: cv.copy() for name in ARMS}
    errors['hurdle_original'] = cv * .7
    errors['hurdle_common'] = cv * .8
    errors['hurdle_at_product'] = cv * .9
    errors['product_at_hurdle'] = cv * .95
    result = support_decomposition(errors, cv, np.array(['a', 'a', 'b', 'b']),
                                   mask=np.ones(4, bool), resamples=3000, seed=1)
    assert result['full_total']['mean_gain_difference_pp'] == pytest.approx(30)
    assert result['total']['mean_gain_difference_pp'] == pytest.approx(20)
    assert result['support_difference']['mean_gain_difference_pp'] == pytest.approx(10)
    assert result['ranking_at_product_count']['mean_gain_difference_pp'] == pytest.approx(10)


def test_separate_scalar_sorting_matches_vectorized_support_paths():
    from scripts.verify_m3w_european_hurdle_coverage import manual_choices
    rng = np.random.default_rng(18)
    p = np.column_stack((rng.uniform(.5, 1.5, 100), rng.uniform(0, .04, 100))).astype(np.float32)
    h = np.column_stack((rng.uniform(.5, 1.5, 100), rng.uniform(0, .04, 100))).astype(np.float32)
    p[:10, 0] = 0
    u = rng.normal(size=100).astype(np.float32)
    moving = rng.random(100) > .2
    sites = np.array(['a', 'b'] * 50)
    ids = rng.permutation(100)
    actual, _ = support_choices(u, p, h, moving, sites, ids, budget=.02)
    manual = manual_choices(u, dict(product_mse=p, hurdle=h), moving, sites, ids, .02)
    for key in ARMS:
        np.testing.assert_array_equal(actual[key], manual[key])
