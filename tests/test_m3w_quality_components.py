import inspect
import numpy as np
import pytest
from src.world_model import m3w_quality_component_diagnostic as api


def test_all_zero_corrections_are_original_except_explicit_preprojection():
    raw = np.array([[1., .1, 10., 5., .01], [2., .2, 20., 10., .02]])
    result = api.variants(raw, np.zeros_like(raw), np.ones(2)*4)
    assert len(result) == 13
    for p in result.values():
        np.testing.assert_array_equal(p, raw)


def test_one_component_and_complement_hold_others_fixed():
    raw = np.array([[1., .1, 10., 5., .01]])
    d = np.array([[.1, .01, 1., .5, .001]])
    p = api.variants(raw, d, np.array([4.]))
    for j, name in enumerate(api.COMPONENTS):
        one = np.zeros_like(d); one[:, j] = d[:, j]
        np.testing.assert_array_equal(p['only_'+name], raw+one)
        complement = d.copy(); complement[:, j] = 0
        np.testing.assert_array_equal(p['without_'+name], raw+complement)


def test_reference_inflation_can_open_action_without_changing_harm():
    raw = np.array([[1., .3, 10., 10., .3]])
    delta = np.array([[0., 0., 10., 10., 0.]])
    p = api.variants(raw, delta, np.array([4.]))
    masks = np.ones(1, bool)
    assert not api.quality.eligible(p['original'], masks, masks)[0]
    assert api.quality.eligible(p['quality'], masks, masks)[0]
    assert not api.quality.eligible(p['only_reference'], masks, masks)[0]


def test_component_projection_is_not_assumed_additive():
    raw = np.array([[1., 1., 10., 5., .5]])
    d = np.array([[2., 0., 0., 0., 0.]])
    p = api.variants(raw, d, np.array([2.]))
    assert p['only_benefit'][0, 1] < raw[0, 1]
    assert p['quality_preprojection'][0, :2].sum() > 2
    assert p['quality'][0, :2].sum() == 2


def test_raw_inference_signature_has_no_labels_or_future():
    assert list(inspect.signature(api.raw_predictions).parameters) == [
        'state', 'fit', 'x', 'envelope', 'past_quality']


def test_slack_identity_and_unknown_are_separate():
    p = np.array([[.2, .1, 5., 5., .1], [.1, .1, 3., 3., .1]])
    y = np.array([[0., 1., 2., 2., 1.], [np.nan]*5])
    d = api.slack_decomposition(p, y, np.ones(2, bool), np.array([2., 4.]))
    assert d['unknown'] == 1 and d['unknown_envelope_mass'] == 4
    assert d['easy']['observed_excess_mass'] == .96
    assert d['easy']['harm_underestimate_mass'] == .9
    assert d['easy']['reference_inflation_budget_mass'] == .06


def test_evaluation_retains_unknown_and_exact_query_counts():
    raw = np.tile([.5, .01, 3., 3., .01], (5, 1))
    y = np.tile([.2, 0., 2., 2., 0.], (5, 1)); y[-1] = np.nan
    env = np.ones(5)
    p = api.variants(raw, np.zeros_like(raw), env)
    r = api.evaluate(p, y, env, np.ones(5, bool), np.ones(5, bool),
                     np.array([0, 0, 1, 1, 1]), np.zeros(5), np.arange(5))
    for v in r['arms'].values():
        assert v['full']['selected_unknown'] == 1
        assert v['cohorts']['retained']['unknown'] == 1
        assert v['full_utility_difference_percent'] == 0
        assert v['matched_utility_difference_percent'] == 0


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        api.variants(np.ones((4, 6)), np.ones((4, 6)), np.ones(4))
