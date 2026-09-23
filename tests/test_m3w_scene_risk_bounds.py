"""Numerical/theoretical fixtures, not independent real-world calibration."""
import math

import numpy as np
import pytest

from src.evaluation.m3w_scene_risk_bounds import (
    hb_p_value, hb_upper_mean, joint_monotonicity_witness,
    screen_hb_diagnostic, zero_loss_required_scenes,
)


def test_zero_loss_matches_closed_form_not_analytic_baseline_exemption():
    for n in (1, 4, 6, 30, 70, 112, 149, 300):
        expected = -math.expm1(math.log(.05) / n)
        assert hb_upper_mean(0., n, .05) == pytest.approx(expected, abs=2e-14)
        assert hb_upper_mean(0., n, .05) > 0
    assert zero_loss_required_scenes(.02, .05, family_size=1, risk_count=1) == 149


def test_fractional_scene_loss_uses_ceiling_in_bentkus_term():
    mean, n, null = .045, 100, .1
    kl = mean * math.log(mean / null) + (1-mean) * math.log((1-mean)/(1-null))
    binomial = sum(math.comb(n, k) * null**k * (1-null)**(n-k)
                   for k in range(math.ceil(n*mean)+1))
    correct = min(1., math.exp(-n*kl), math.e*binomial)
    incorrect = min(1., math.exp(-n*kl), math.e*sum(
        math.comb(n, k)*null**k*(1-null)**(n-k) for k in range(math.floor(n*mean)+1)))
    assert hb_p_value(mean, n, null) == pytest.approx(correct)
    assert hb_p_value(mean, n, null) > incorrect


def test_null_not_above_observation_is_not_rejected():
    assert hb_p_value(.3, 70, .2) == 1
    assert hb_p_value(.3, 70, .3) == 1
    assert hb_upper_mean(1., 70, .05) == 1
    assert hb_p_value(0., 70, .1) == pytest.approx(.9**70)


@pytest.mark.parametrize("mean,n,level", [(-.1, 10, .05), (1.1, 10, .05),
    (float("nan"), 10, .05), (.1, 0, .05), (.1, True, .05), (.1, 10.5, .05),
    (.1, 10, 0.), (.1, 10, 1.), (.1, 10, float("nan"))])
def test_invalid_inputs_refused(mean, n, level):
    with pytest.raises(ValueError):
        hb_upper_mean(mean, n, level)


def test_upper_endpoint_is_conservative_side_of_numerical_root():
    for mean in (0., .001, .045, .1, .7, .99):
        upper = hb_upper_mean(mean, 100, .005)
        assert mean < upper <= 1
        if upper < 1:
            assert hb_p_value(mean, 100, upper) <= .005 * (1+1e-12)
            assert hb_p_value(mean, 100, upper-1e-8) > .005


def screen(losses, cluster_ids=None, fitted=(), policies=None):
    n, m, k = losses.shape
    return screen_hb_diagnostic(losses=losses,
        cluster_ids=cluster_ids or [f"scene{i}" for i in range(n)],
        fitted_cluster_ids=fitted, policy_ids=policies or [f"p{i}" for i in range(m)],
        lower=[0]*k, upper=[1]*k, tolerance=[.1]*k, delta=.05)


def test_multiplicity_and_scene_ids_are_preserved():
    small = screen(np.zeros((70, 1, 1)))
    large = screen(np.zeros((70, 5, 2)))
    assert small["upper_risk_bound"][0, 0] < large["upper_risk_bound"][0, 0]
    assert large["per_hypothesis_error_level"] == .005
    assert not large["independence_verified"]
    assert not large["fixed_family_verified"]
    assert not large["real_calibration_executed"]
    assert not large["physical_safety_certified"]
    with pytest.raises(ValueError, match="unique"):
        screen(np.zeros((2, 1, 1)), cluster_ids=["same", "same"])
    with pytest.raises(ValueError, match="overlap"):
        screen(np.zeros((2, 1, 1)), fitted=["scene0"])
    with pytest.raises(ValueError, match="identities"):
        screen(np.zeros((2, 2, 1)), policies=["same", "same"])


def test_loss_bounds_and_missing_values_are_not_silently_repaired():
    for value in (np.nan, np.inf, -1, 2):
        with pytest.raises(ValueError):
            screen(np.full((3, 1, 1), value))


def test_hb_no_looser_than_existing_hoeffding_bound_on_grid():
    for n in (4, 30, 70):
        for value in (0., .05, .25, .8, 1.):
            result = screen(np.full((n, 2, 2), value))
            assert np.all(result["upper_risk_bound"] <= result["hoeffding_reference_bound"] + 1e-13)


def test_declared_nonunit_loss_scale_and_offset_are_respected():
    result = screen_hb_diagnostic(losses=np.full((70, 1, 1), -2.),
        cluster_ids=[f"s{i}" for i in range(70)], fitted_cluster_ids=[], policy_ids=["p"],
        lower=[-2.], upper=[3.], tolerance=[-1.], delta=.05)
    assert result["upper_risk_bound"][0, 0] == pytest.approx(-2+5*hb_upper_mean(0., 70, .05))


def test_exact_binomial_null_size_checks_bounded_not_only_binary_losses():
    for n in (5, 20, 80):
        for low, high, p in ((0., 1., .1), (.02, .4, .25), (.1, .9, .65)):
            null_mean = low+(high-low)*p
            for level in (.01, .05, .1):
                size = sum(math.comb(n, k)*p**k*(1-p)**(n-k)
                    for k in range(n+1)
                    if hb_p_value(low+(high-low)*k/n, n, null_mean) <= level)
                assert size <= level + 1e-13


def test_fewer_switches_need_not_make_joint_risk_monotone():
    witness = joint_monotonicity_witness()
    assert witness["switch_counts"] == [2, 1, 0]
    assert witness["proximity_events"] == [0, 1, 0]
    assert witness["monotone_in_decreasing_intervention"] is False
    assert witness["physical_collision_claim"] is False
