"""Exact finite-population witnesses, not a model fit or calibration experiment."""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def expectation(weights, values):
    if len(weights) != len(values) or not weights or sum(weights) != 1:
        raise ValueError("Use matching nonempty probability weights summing to one")
    if any(p < 0 for p in weights):
        raise ValueError("Negative probability")
    return sum((p * v for p, v in zip(weights, values)), F(0))


def ratio(numerator, denominator):
    if denominator <= 0:
        raise ValueError("Nonpositive reference denominator: relative risk undefined")
    return numerator / denominator


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def global_fit_witness():
    p = [F(1, 100), F(49, 100), F(1, 2)]
    harm = [F(1), F(0), F(10)]
    predicted_harm = [F(0), F(1, 49), F(10)]
    predicted_benefit = [F(1, 10), F(0), F(0)]
    # Membership is fixed by predicted scores, never by observed harm.
    action = [F(int(b - h >= F(1, 50) and h <= F(1, 20)))
              for b, h in zip(predicted_benefit, predicted_harm)]
    mean = expectation(p, harm)
    predicted_mean = expectation(p, predicted_harm)
    mse = expectation(p, [(h - q) ** 2 for h, q in zip(harm, predicted_harm)])
    constant_mse = expectation(p, [(h - mean) ** 2 for h in harm])
    coverage = expectation(p, action)
    observed_selected = ratio(expectation(p, [a * h for a, h in zip(action, harm)]), coverage)
    predicted_selected = ratio(expectation(p, [a * h for a, h in zip(action, predicted_harm)]), coverage)
    check(mean == predicted_mean, "Global mean must match exactly")
    check(mse < constant_mse, "Witness must improve global MSE")
    check(observed_selected > predicted_selected, "Selected cost must be underestimated")
    return dict(global_mean_harm=mean, predicted_global_mean_harm=predicted_mean,
                harm_mse=mse, constant_mean_harm_mse=constant_mse,
                selected_fraction=coverage, selected_harm=observed_selected,
                selected_predicted_harm=predicted_selected,
                selected_actual_net_gain=-observed_selected)


def score_calibration_witness():
    p = [F(1, 4)] * 4
    harm = [F(1), F(1), F(0), F(0)]
    score = [F(1, 2)] * 4
    past_scene_context = [F(1), F(1), F(0), F(0)]
    observed = expectation(p, [a * h for a, h in zip(past_scene_context, harm)])
    predicted = expectation(p, [a * h for a, h in zip(past_scene_context, score)])
    check(expectation(p, harm) == score[0], "Single score bin must be calibrated")
    check(observed > predicted, "Extra context can invalidate score-only calibration")
    return dict(calibrated_score=score[0], actual_selected_population_harm=observed,
                predicted_selected_population_harm=predicted,
                selected_fraction=expectation(p, past_scene_context))


def joint_easy_target_witness():
    p = [F(1, 2), F(1, 2)]
    baseline_loss = [F(1, 100), F(1)]
    harm = [F(1, 100), F(0)]
    easy = [F(int(b <= F(1, 10))) for b in baseline_loss]
    joint = expectation(p, [w * h for w, h in zip(easy, harm)])
    product = expectation(p, easy) * expectation(p, harm)
    budget = F(1, 50) * expectation(p, [w * b for w, b in zip(easy, baseline_loss)])
    check(joint > product, "Product of marginal moments is not the joint moment")
    return dict(joint_easy_positive_harm=joint, product_of_marginals=product,
                conditional_covariance=joint - product, two_percent_easy_budget=budget)


def aggregation_witness():
    # Two equally weighted scenes; all agents are easy under this toy threshold.
    baseline_loss = [F(1, 100), F(1)]
    excess_loss = [F(1, 100), F(0)]
    pooled = ratio(sum(excess_loss), sum(baseline_loss))
    equal_scene = expectation([F(1, 2)] * 2,
                              [ratio(d, b) for d, b in zip(excess_loss, baseline_loss)])
    check(pooled < F(1, 50) < equal_scene, "Aggregation must reverse the 2% decision")
    return dict(pooled_relative_degradation=pooled,
                equal_scene_relative_degradation=equal_scene,
                illustrative_tolerance=F(1, 50))


def exact_contract_checks():
    """Exhaust all binary rules on one finite law; no sampled data or fitting."""
    p = [F(1, 10), F(3, 10), F(3, 5)]
    baseline = [F(1, 10), F(1), F(2)]
    candidate = [F(1, 5), F(4, 5), F(3)]
    predicted = [F(1, 20), F(1, 10), F(4, 5)]
    easy = [F(int(b <= 1)) for b in baseline]
    delta = [n - b for n, b in zip(candidate, baseline)]
    harm = [max(d, F(0)) for d in delta]
    residual = [h - q for h, q in zip(harm, predicted)]
    mse = expectation(p, [r ** 2 for r in residual])
    budget = F(1, 50) * expectation(p, [w * b for w, b in zip(easy, baseline)])
    sufficient_passes = 0
    for bits in range(8):
        a = [F((bits >> i) & 1) for i in range(3)]
        coverage = expectation(p, a)
        error = expectation(p, [s * r for s, r in zip(a, residual)])
        check(error ** 2 <= coverage * mse, "Cauchy-Schwarz inequality failed")
        signed_easy = expectation(p, [w * s * d for w, s, d in zip(easy, a, delta)])
        positive_easy = expectation(p, [w * s * h for w, s, h in zip(easy, a, harm)])
        check(signed_easy <= positive_easy, "Positive harm must bound signed excess")
        if positive_easy <= budget:
            check(signed_easy <= budget, "Sufficient easy-risk condition failed")
            sufficient_passes += 1
    check(sufficient_passes > 0, "Sufficient implication must be exercised")
    # Four outcomes share one observed state per pair. Actions may use the state.
    p4 = [F(1, 4)] * 4
    h4 = [F(0), F(2), F(1), F(3)]
    mu4 = [F(1), F(1), F(2), F(2)]
    tower_rules = 0
    for bits in range(4):
        a4 = [F(bits & 1)] * 2 + [F((bits >> 1) & 1)] * 2
        check(expectation(p4, [a * h for a, h in zip(a4, h4)]) ==
              expectation(p4, [a * mu for a, mu in zip(a4, mu4)]),
              "Full-information conditional mean identity failed")
        tower_rules += 1
    try:
        ratio(F(0), F(0))
    except ValueError:
        zero_denominator_rejected = True
    else:
        raise AssertionError("Zero denominator cannot silently become zero risk")
    return dict(binary_policies_checked=8, sufficient_easy_budget_cases=sufficient_passes,
                full_information_tower_rules_checked=tower_rules,
                zero_denominator_rejected=zero_denominator_rejected)


def encode(value):
    if isinstance(value, F):
        return dict(exact=str(value), decimal=float(value))
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def report():
    return dict(
        status="fresh_run_exact_synthetic_identity_checks_not_real_data_evidence",
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_data_read=False, training=False, policy_selection=False,
        evaluation_protocol_changed=False, independent_calibration=False,
        novelty_or_safety_theorem_claim=False,
        examples=encode(dict(global_fit=global_fit_witness(),
                             score_calibration=score_calibration_witness(),
                             joint_easy_target=joint_easy_target_witness(),
                             scene_aggregation=aggregation_witness())),
        contract_checks=exact_contract_checks())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(report(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")


if __name__ == "__main__":
    main()
