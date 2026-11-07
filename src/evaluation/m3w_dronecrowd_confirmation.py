"""Readout helpers for the locked, independently grouped confirmation role."""
import numpy as np

from src.world_model.m3w_dronecrowd_calibration import GATES, MARGINS, winner


def selected_gate(doc):
    """Refuse a stale, empty, or retrospectively changed calibration choice."""
    if (doc['calibration_pass'] is not True or doc['confirmation_outcome_access']
            or doc['independent_confirmation'] or not doc['original_rules_unchanged']):
        raise ValueError('Calibration did not qualify for confirmation')
    summary = doc['summary']
    if (summary['recordings'], summary['source_groups'], summary['views']) != (40, 15, 120):
        raise ValueError('Calibration result is not on the registered roster')
    chosen = doc['selected_arm']
    if chosen not in GATES or chosen != winner(summary):
        raise ValueError('Calibration winner differs from the fixed selection rule')
    if doc['selected_normalized_gain_margin'] != MARGINS[GATES.index(chosen)]:
        raise ValueError('Calibration threshold changed')
    return chosen


def require_confirmation(manifest, sequence):
    if (manifest['recording_role_index'].get(sequence) != 'confirmation'
            or sequence not in manifest['assignments']['confirmation']['recordings']
            or sequence in manifest['assignments']['calibration']['recordings']):
        raise PermissionError('Confirmation runner cannot access this recording')


def matched_forecast(cv, slots, values, selected_slots):
    """Apply a common query schedule; this is a utility-only control."""
    active = (selected_slots >= 0).any(1)
    if not np.array_equal((slots[active] >= 0).sum(1), (selected_slots[active] >= 0).sum(1)):
        raise ValueError('Matched action count differs at an active query')
    i, p = slots[active], values[active]
    out = cv.copy()
    out[i[i >= 0]] = p[i >= 0]
    return out


def verdict(summary, pair_stats):
    selected = summary['risk']['selected']
    risk = selected['supported_views'] == selected['required_views'] and selected['switches'] > 0

    def positive(row):
        return row is not None and row['interval95'] is not None and row['interval95'][0] > 0.

    effects = summary['effects']
    against_cv = positive(effects['selected_vs_cv']['completion_lower_pp'])
    against_ols = positive(effects['selected_vs_ols4']['completion_lower_pp'])
    joint = all(positive(effects[f'selected_vs_{arm}']['completion_lower_pp'])
                for arm in ('matched_independent', 'matched_scene', 'matched_pair_additive'))
    relative = positive(pair_stats['selected_vs_matched_pair_additive']['grid_ade_lower'])
    return dict(original_recording_risk_pass=risk, conservative_utility_over_cv=against_cv,
        conservative_utility_over_ols4=against_ols, matched_joint_utility_pass=joint,
        relative_geometry_pass=relative, independent_forecast_claim_supported=risk and against_cv and against_ols,
        independent_joint_claim_supported=risk and against_cv and against_ols and joint and relative)
