"""Detect impossible count-matched risk requirements before training admission."""

import math


def assess_count_contract(anchor_counts, *, require_matched_finite_risk):
    if not isinstance(require_matched_finite_risk, bool) or not anchor_counts:
        raise ValueError('Explicit risk contract and nonempty fixed roster required')
    keys = set(); empty = []
    for row in anchor_counts:
        key = (row['group'], row['source'], row['seed'], row['anchor'])
        count = row['selected_count']
        if key in keys or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('Unique roster identities and nonnegative integer counts required')
        keys.add(key)
        if count == 0: empty.append(dict(group=key[0],source=key[1],seed=key[2],anchor=key[3]))
    impossible = bool(empty) and require_matched_finite_risk
    return dict(anchor_views=len(keys), forced_empty_comparisons=empty,
        matched_finite_risk_required=require_matched_finite_risk,
        structural_obstruction_detected=impossible,
        admission='blocked_impossible_contract' if impossible else 'no_zero_anchor_obstruction_other_checks_required',
        empty_risk_is_zero=False, scientific_success=False, full_policy_safety_unchanged=True)


def require_no_structural_obstruction(anchor_counts, *, require_matched_finite_risk):
    report = assess_count_contract(anchor_counts, require_matched_finite_risk=require_matched_finite_risk)
    if report['structural_obstruction_detected']:
        raise ValueError('Zero-count anchors force undefined matched risk; revise the prospective contract before new training')
    return report


def prospective_advancement(full_policies, utility_intervals, *, expected_views,
                            expected_contrasts):
    """Separate complete deployment risk from same-count utility comparisons.

    The roster and contrasts must be fixed before evaluation. This cannot
    certify population safety or change a deployment, even when it passes.
    """
    if (not expected_views or not expected_contrasts
            or len(set(expected_views)) != len(expected_views)
            or len(set(expected_contrasts)) != len(expected_contrasts)
            or set(full_policies) != set(expected_views)
            or set(utility_intervals) != set(expected_contrasts)):
        raise ValueError('Complete prespecified policy and contrast rosters required')
    failures = []
    for identity in expected_views:
        policy = full_policies[identity]
        count = policy['selected_count']
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('Invalid selected count')
        if count == 0:
            failures.append(dict(view=identity, reason='empty_full_policy_not_supported'))
        for name in ('all_selected_risk_upper', 'easy_selected_risk_upper', 'easy_degradation_upper'):
            value = policy[name]
            if value is None:
                failures.append(dict(view=identity, reason=name+'_undefined'))
            elif (isinstance(value, bool) or not isinstance(value, (int,float))
                  or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid risk value')
            elif value > .02 + 1e-12:
                failures.append(dict(view=identity, reason=name+'_over_unchanged_budget'))
        if policy['finite_completion_supported'] is not True:
            failures.append(dict(view=identity, reason='full_completion_not_supported'))
    for name in expected_contrasts:
        interval = utility_intervals[name]
        if interval is None:
            failures.append(dict(contrast=name,reason='utility_not_estimable'))
            continue
        if (len(interval) != 2 or any(isinstance(x,bool) or not isinstance(x,(int,float))
                                    or not math.isfinite(x) for x in interval)
                or interval[0] > interval[1]):
            raise ValueError('Invalid fixed utility interval')
        if interval[0] <= 0:
            failures.append(dict(contrast=name,reason='utility_lower_not_positive'))
    return dict(development_advancement=not failures, failures=failures,
                risk_budget=.02, matched_risk_required_for_advancement=False,
                independent_confirmation=False, deployment_changed=False,
                population_safety_proven=False, scientific_success=False)
