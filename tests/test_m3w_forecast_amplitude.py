import inspect
import numpy as np
import pytest

from src.evaluation import m3w_forecast_amplitude as api
from src.world_model.m3w_fixed_floor_probe import targets
from src.evaluation.m3w_multiaction_cost_readout import compact_score


def fixture():
    floor = np.zeros((3, 12, 2))
    paths = np.full((3, 7, 12, 2), 4.)
    costs = np.broadcast_to([1., .1, 10., 10., .1], (3, 7, 5)).copy()
    return floor, paths, {k: costs.copy() for k in api.ARMS}, np.ones((3, 7), bool)


def test_neutral_and_zero_are_exact_and_do_not_claim_empty_safety():
    floor, paths, prediction, allowed = fixture()
    np.testing.assert_array_equal(api.blend(floor, paths, 1), paths)
    _, _, neutral, _ = api.decisions(prediction, allowed, floor, paths, 1)
    for name, p in prediction.items():
        expected, _ = api.select_actions(p, allowed, arm=name)
        np.testing.assert_array_equal(neutral[name], expected)
        np.testing.assert_array_equal(neutral[name + '_same_set'], expected)
    out, legal, choice, _ = api.decisions(prediction, allowed, floor, paths, 0)
    np.testing.assert_array_equal(out, np.broadcast_to(floor[:, None], paths.shape))
    assert not legal.any()
    assert all(np.all(c == -1) for c in choice.values())
    y = np.broadcast_to([1., 0., 10., 10., 0.], (3, 7, 5)).copy()
    score = compact_score(y, np.ones((3, 7)), choice['physical_neural'])
    assert score['easy_selected_risk_upper'] is None
    assert not score['budget_status']['supported']


def test_cost_transport_keeps_reference_and_is_not_exact_fractional_error():
    _, _, prediction, _ = fixture()
    p = prediction['physical_neural']
    out = api.transported_bounds(p, .25)
    np.testing.assert_array_equal(out[..., (2, 3)], p[..., (2, 3)])
    np.testing.assert_array_equal(out[..., (0, 1, 4)], .25 * p[..., (0, 1, 4)])
    # A tied endpoint can improve at an intermediate point; alpha*B is only a lower bound.
    endpoint = targets(np.array([1.]), np.array([1.]), np.array([1.]), 2.)
    midway = targets(np.array([1.]), np.array([1.]), np.array([0.]), 2.)
    assert midway[0, 0] > .5 * endpoint[0, 0]


def test_random_fixed_mask_convexity_with_unknown_rows():
    rng = np.random.default_rng(10)
    floor = rng.normal(size=(23, 12, 2))
    paths = rng.normal(size=(23, 7, 12, 2))
    truth = rng.normal(size=floor.shape)
    mask = rng.random((23, 12)) > .4
    mask[0] = False
    def error(value):
        loss = np.linalg.norm(value - truth, axis=-1)
        count = mask.sum(1)
        return np.divide(np.where(mask, loss, 0).sum(1), count,
                         out=np.full(len(mask), np.nan), where=count > 0)
    reference = error(floor)
    def labels(candidate):
        return np.stack([targets(reference, reference, error(p), 5.) for p in candidate.swapaxes(0, 1)], 1)
    original = labels(paths)
    for alpha in (0., .0625, .125, .25, .5, 1.):
        check = api.check_convex_labels(original, labels(api.blend(floor, paths, alpha)), alpha)
        assert check['known_rows'] == 22 and check['unknown_rows'] == 1
        assert not check['learned_risk_certificate']


def test_reselection_and_same_set_are_distinct_causal_controls():
    floor, paths, prediction, allowed = fixture()
    for p in prediction.values():
        p[..., 1] = 1.
        p[..., 0] = 2.
        p[..., 4] = .5
    _, _, full, _ = api.decisions(prediction, allowed, floor, paths, 1.)
    _, _, small, _ = api.decisions(prediction, allowed, floor, paths, .125)
    assert np.all(full['physical_neural'] == -1)
    assert np.all(small['physical_neural'] >= 0)
    assert np.all(small['physical_neural_same_set'] == -1)
    assert 'future' not in inspect.signature(api.decisions).parameters
    with pytest.raises(TypeError):
        api.decisions(prediction, allowed, floor, paths, .5, future_endpoint=np.ones((3, 2)))


@pytest.mark.parametrize('alpha', [-.1, 1.1, np.nan, np.inf, True])
def test_invalid_amplitude_rejected(alpha):
    floor, paths, _, _ = fixture()
    with pytest.raises(ValueError):
        api.blend(floor, paths, alpha)


def test_changed_unknown_support_or_convex_violation_is_rejected():
    original = np.broadcast_to([1., 0., 10., 10., 0.], (3, 7, 5)).copy()
    changed = original.copy()
    changed[0] = np.nan
    with pytest.raises(AssertionError):
        api.check_convex_labels(original, changed, .5)
    changed = original.copy()
    changed[..., 0] = 0.
    with pytest.raises(ValueError):
        api.check_convex_labels(original, changed, .5)


def test_native_partial_label_scoring_and_cross_amplitude_contrast():
    from scripts.diagnose_m3w_forecast_amplitude import evaluation_packet, paired_outputs
    from src.evaluation.m3w_native_metrics import native_errors
    floor, paths, _, allowed = fixture()
    truth = np.full_like(floor, 2.)
    valid = np.ones((3, 12), bool)
    valid[0, 6:] = False
    valid[2] = False
    truth[~valid] = np.nan
    source = dict(origin=np.full((3, 2), 3.), target_eval=truth + 3., valid=valid)
    reference, fde = native_errors(floor + 3., source['target_eval'], valid, np.ones(3))
    packet = dict(ids=np.arange(3), cv_ade=reference, floor_ade=reference, floor_fde=fde,
                  floor_rollout=floor, rollouts=paths)
    endpoint = evaluation_packet(packet, paths, allowed, source, dict(easy_cut=10.))
    fractional = evaluation_packet(packet, api.blend(floor, paths, .5), allowed, source, dict(easy_cut=10.))
    check = api.check_convex_labels(endpoint['y'], fractional['y'], .5)
    assert check['known_rows'] == 2 and check['unknown_rows'] == 1
    assert np.isnan(fractional['fde'][0]).all() and np.isnan(fractional['ade'][2]).all()
    contrast = paired_outputs(endpoint, np.zeros(3, int), fractional, np.zeros(3, int))
    assert contrast['known_pp'] == pytest.approx(100.)
    assert contrast['unknown_disagreement_mass'] == pytest.approx(np.sqrt(8.))
    assert contrast['lower_pp'] == pytest.approx(50.)


def test_incomplete_roster_cannot_produce_a_summary():
    from scripts.diagnose_m3w_forecast_amplitude import summarize
    with pytest.raises(ValueError, match='Full original fitting roster'):
        summarize([], dict(families=['forest', 'neural']), {})
