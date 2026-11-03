"""Causal nonempty same-count utility controls, never a safety certificate."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import paired


def matched_utility(independent_path, joint_path, floor, target, valid,
                    independent_choice, joint_choice, groups):
    first, second, baseline = map(np.asarray, (independent_path, joint_path, floor))
    original, matched = np.asarray(independent_choice), np.asarray(joint_choice)
    n = len(first)
    if (first.shape != (n, 12, 2) or second.shape != first.shape or baseline.shape != first.shape
            or original.shape != (n,) or matched.shape != (n,)
            or original.dtype.kind not in 'iu' or matched.dtype.kind not in 'iu'
            or (original < -1).any() or (matched < -1).any()
            or (original >= 7).any() or (matched >= 7).any()
            or not all(np.isfinite(v).all() for v in (first, second, baseline))):
        raise ValueError('Finite fixed rollouts and aligned causal choices required')
    covered = np.concatenate(groups) if groups else np.empty(0, int)
    if not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError('Every causal row must belong to exactly one query')
    if (not np.array_equal(first[original < 0], baseline[original < 0])
            or not np.array_equal(second[matched < 0], baseline[matched < 0])):
        raise ValueError('No-intervention rows must retain their original floor')
    take = np.zeros(n, bool)
    nonempty, total_count = 0, 0
    for group in groups:
        count = int((original[group] >= 0).sum())
        if count != int((matched[group] >= 0).sum()):
            raise ValueError('Intervention count must match in every query')
        if count:
            take[group] = True
            nonempty += 1
            total_count += count
    if nonempty:
        result = paired(first[take], second[take], baseline[take],
            np.asarray(target)[take], np.asarray(valid)[take])
        status = 'utility_only_causal_nonempty_query_population'
        if result['known_pp'] is None:
            status = 'not_estimable_without_known_reference_mass'
    else:
        result = dict(known_pp=None, completion_lower_pp=None, completion_upper_pp=None,
            unknown_path_disagreement=0., required_rows=0, known_rows=0)
        status = 'not_estimable_empty_matched_control'
    return dict(status=status, contrast=result, nonempty_queries=nonempty,
        empty_queries=len(groups)-nonempty, matched_interventions_each_policy=total_count,
        eligible_rows=int(take.sum()), eligibility_from_causal_choices_not_future_labels=True,
        comparison='joint_matched_minus_same_arm_independent',
        safety_gate_usage=False, full_strategy_risk_population_unchanged=True)
