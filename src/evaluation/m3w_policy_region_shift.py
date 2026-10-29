"""Separate fixed-teacher calibration from drift under the new learned policy."""
import numpy as np
from src.evaluation.m3w_selected_cost_diagnosis import audit_record, subset_summary


def compare_regions(prediction, target, allowed, current, teacher, scales, actions):
    p, teacher = np.asarray(prediction), np.asarray(teacher)
    if teacher.shape != (len(p),):
        raise ValueError('Aligned frozen teacher choices required')
    mask = np.zeros_like(allowed); ix = np.flatnonzero(teacher >= 0)
    if not np.isin(teacher[ix], actions).all() or not allowed[ix, teacher[ix]].all():
        raise ValueError('Teacher region contains illegal or no-op actions')
    mask[ix, teacher[ix]] = True
    result = audit_record(p, target, allowed, current, scales, actions)
    result['frozen_teacher_region'] = subset_summary(p, target, mask, scales)
    result['new_interventions_outside_teacher_region'] = int(((current >= 0) & (current != teacher)).sum())
    result['same_teacher_action_kept'] = int(((current >= 0) & (current == teacher)).sum())
    result['diagnostic_role'] = 'fitting_only_stale_region_check_not_independent_calibration'
    return result
