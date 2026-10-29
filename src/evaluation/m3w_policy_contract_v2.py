"""Prospective v2 accepts genuine negative net easy degradation.

The registered v1 implementation and its reports remain immutable. Positive
harm ratios must still be nonnegative. Only a finite signed net degradation
below zero is a valid improvement, rather than malformed risk.
"""
import math

from src.evaluation import m3w_policy_contract_preflight as legacy


def prospective_advancement(full_policies, utility_intervals, *, expected_views,
                            expected_contrasts):
    budget_inputs = {}
    original = {}
    for identity, policy in full_policies.items():
        value = policy['easy_degradation_upper']
        if value is not None and (isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError('Finite signed easy degradation or undefined required')
        original[identity] = value
        # max(0, value) <= .02 is equivalent to value <= .02 for signed values.
        # Keep original evidence in the result; never alter the caller's report.
        budget_inputs[identity] = {**policy, 'easy_degradation_upper':
                                  max(0., value) if value is not None else None}
    result = legacy.prospective_advancement(
        budget_inputs, utility_intervals, expected_views=expected_views,
        expected_contrasts=expected_contrasts)
    return {**result, 'contract_version': 'signed_easy_degradation_v2',
            'original_easy_degradation_upper': original,
            'legacy_results_rescored': False}
