from fractions import Fraction as F

import pytest

from scripts.verify_m3w_conditional_risk_identities import (
    aggregation_witness, exact_contract_checks, expectation, global_fit_witness,
    joint_easy_target_witness, ratio, report, score_calibration_witness,
)


def test_low_mse_and_correct_mean_do_not_protect_selected_group():
    r = global_fit_witness()
    assert r["global_mean_harm"] == r["predicted_global_mean_harm"] == F(501, 100)
    assert r["harm_mse"] == F(1, 98)
    assert r["constant_mean_harm_mse"] == F(249099, 10000)
    assert r["selected_fraction"] == F(1, 100)
    assert r["selected_harm"] == 1
    assert r["selected_predicted_harm"] == 0
    assert r["selected_actual_net_gain"] == -1


def test_calibration_on_score_does_not_cover_extra_scene_context():
    r = score_calibration_witness()
    assert r["actual_selected_population_harm"] == F(1, 2)
    assert r["predicted_selected_population_harm"] == F(1, 4)


def test_easy_harm_joint_moment_not_product_of_marginals():
    r = joint_easy_target_witness()
    assert r["joint_easy_positive_harm"] == F(1, 200)
    assert r["product_of_marginals"] == r["conditional_covariance"] == F(1, 400)
    assert r["two_percent_easy_budget"] == F(1, 10000)


def test_scene_aggregation_changes_the_claim():
    r = aggregation_witness()
    assert r["pooled_relative_degradation"] == F(1, 101)
    assert r["equal_scene_relative_degradation"] == F(1, 2)


def test_contracts_include_all_binary_policies_and_nonvacuous_budget_cases():
    r = exact_contract_checks()
    assert r["binary_policies_checked"] == 8
    assert r["sufficient_easy_budget_cases"] == 4
    assert r["full_information_tower_rules_checked"] == 4
    assert r["zero_denominator_rejected"]


@pytest.mark.parametrize("weights,values", [([], []), ([F(1)], []),
    ([F(1, 2)], [F(1)]), ([F(-1), F(2)], [F(1), F(1)])])
def test_invalid_finite_probability_law_rejected(weights, values):
    with pytest.raises(ValueError):
        expectation(weights, values)


@pytest.mark.parametrize("denominator", [F(0), F(-1)])
def test_nonpositive_reference_denominator_is_not_imputed(denominator):
    with pytest.raises(ValueError, match="undefined"):
        ratio(F(0), denominator)


def test_report_cannot_be_mistaken_for_real_training_or_calibration():
    r = report()
    assert r["status"] == "fresh_run_exact_synthetic_identity_checks_not_real_data_evidence"
    for key in ("source_data_read", "training", "policy_selection",
                "evaluation_protocol_changed", "independent_calibration",
                "novelty_or_safety_theorem_claim"):
        assert r[key] is False
    assert r["examples"]["global_fit"]["harm_mse"]["exact"] == "1/98"
    assert len(r["implementation_sha256"]) == 64
