import numpy as np
import pytest

from src.evaluation.m3w_native_metrics import native_errors
from src.world_model.m3w_geometric_cost_head import rollout_envelope
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds


def test_known_rows_recover_native_accounting():
    y = np.array([[2., 0., 3., 3., 0.], [0., .01, 3., 3., .01]])
    z = completion_bounds(y, np.ones(2, bool), np.array([2., .01]))
    assert z['all_selected_risk_upper'] == pytest.approx(.01 / 6)
    assert z['easy_selected_risk_upper'] == pytest.approx(.01 / 6)
    assert z['easy_degradation_upper'] == pytest.approx(-1.99 / 6)
    assert z['selected_net_gain_lower_mass'] == pytest.approx(1.99)
    assert z['finite_completion_supported']
    assert not z['population_safety_guarantee']


def test_unknown_harm_not_zero_and_actions_not_changed():
    y = np.array([[1., 0., 100., 100., 0.], [np.nan] * 5])
    take = np.ones(2, bool)
    z = completion_bounds(y, take, np.array([1., .5]))
    assert z['selected_unknown_envelope_mass'] == .5
    assert z['selected_net_gain_lower_mass'] == .5
    assert z['all_selected_risk_upper'] == .005
    assert z['easy_degradation_upper'] == 0.
    assert z['finite_completion_supported']
    np.testing.assert_array_equal(take, [True, True])


def test_benefit_does_not_cancel_positive_harm_budget():
    y = np.array([[2., 0., 2., 2., 0.], [0., .2, 2., 2., .2]])
    z = completion_bounds(y, np.ones(2, bool), np.array([2., .2]))
    assert z['selected_net_gain_lower_mass'] > 0
    assert z['all_selected_risk_upper'] == .05
    assert not z['finite_completion_supported']


def test_zero_envelope_and_zero_denominator_are_different():
    z = completion_bounds(np.full((1, 5), np.nan), np.ones(1, bool), np.zeros(1))
    assert z['selected_unknown_envelope_mass'] == 0
    assert z['all_selected_risk_upper'] is None
    assert not z['finite_completion_supported']
    z = completion_bounds(np.array([[1., 0., 2., 0., 0.]]), np.ones(1, bool), np.ones(1))
    assert z['easy_selected_risk_upper'] is None
    assert not z['finite_completion_supported']


def test_unknown_unselected_reference_can_dilute_negative_easy_degradation():
    y = np.array([[1., 0., 2., 2., 0.], [np.nan] * 5])
    z = completion_bounds(y, np.array([True, False]), np.ones(2))
    assert z['selected_unknown_envelope_mass'] == 0
    assert z['easy_degradation_upper'] == 0
    actual = -1 / (2 + 1e6)
    assert actual > -.5 and actual <= z['easy_degradation_upper']


def test_zero_floor_easy_row_still_contributes_harm():
    y = np.array([[1., 0., 100., 100., 0.], [0., 3., 0., 0., 3.]])
    z = completion_bounds(y, np.ones(2, bool), np.array([1., 3.]))
    assert z['easy_selected_risk_upper'] == .03
    assert z['easy_degradation_upper'] == .02
    assert not z['finite_completion_supported']


def test_max_not_mean_required_for_arbitrary_observed_steps():
    floor = np.zeros((1, 12, 2))
    neural = floor.copy(); neural[0, -1, 0] = 12
    target = floor.copy(); valid = np.zeros((1, 12), bool); valid[0, -1] = True
    f, _ = native_errors(floor, target, valid, np.ones(1))
    n, _ = native_errors(neural, target, valid, np.ones(1))
    assert (n-f)[0] == rollout_envelope(floor, neural)[0] == 12
    assert (n-f)[0] > np.linalg.norm(neural-floor, axis=-1).mean()


def test_geometric_bounds_cover_random_partial_label_completions():
    rng = np.random.default_rng(82173)
    for _ in range(100):
        f = rng.normal(size=(30, 12, 2))
        n = f + rng.normal(size=f.shape)
        target = rng.normal(size=f.shape) * rng.uniform(.1, 100)
        valid = rng.random((30, 12)) < .4; valid[:, 0] = True
        fe, _ = native_errors(f, target, valid, np.ones(30))
        ne, _ = native_errors(n, target, valid, np.ones(30))
        env = rollout_envelope(f, n)
        assert np.all(np.abs(fe-ne) <= env + 1e-10)
        b, h = np.maximum(fe-ne, 0), np.maximum(ne-fe, 0)
        easy = rng.random(30) < .5
        y = np.column_stack((b, h, fe, fe*easy, h*easy))
        unknown = rng.random(30) < .35
        take = rng.random(30) < .7
        hidden = y.copy(); hidden[unknown] = np.nan
        z = completion_bounds(hidden, take, env)
        assert (b[take]-h[take]).sum() >= z['selected_net_gain_lower_mass'] - 1e-10
        for key, harm, ref in [('all_selected_risk_upper', h, fe),
                               ('easy_selected_risk_upper', h*easy, fe*easy)]:
            if z[key] is not None:
                assert harm[take].sum()/ref[take].sum() <= z[key] + 1e-10
        if z['easy_degradation_upper'] is not None:
            actual = ((h-b)*easy)[take].sum()/(fe*easy).sum()
            assert actual <= z['easy_degradation_upper'] + 1e-10


@pytest.mark.parametrize('y,take,env', [
    ([[np.nan, 0, 1, 1, 0]], [True], [1]),
    ([[1, 0, 1, 1, 0]], [1], [1]),
    ([[1, 0, 1, 1, 0]], [True], [-1]),
    ([[1, 0, 1, 1, 0]], [True], [.5]),
    ([[1, 0, 1, .5, 0]], [True], [1]),
    ([[1, .5, 1, 1, .5]], [True], [2]),
    ([[2, 0, 1, 1, 0]], [True], [2]),
])
def test_invalid_contract_rejected(y, take, env):
    with pytest.raises(ValueError):
        completion_bounds(np.asarray(y), np.asarray(take), np.asarray(env))
