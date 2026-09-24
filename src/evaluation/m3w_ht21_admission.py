"""Fail-closed intake boundary, not approval of a new forecasting protocol."""


def admission(recording, requested_role):
    if requested_role not in {
        'source_audit', 'supervised_training', 'representation_pretraining',
        'risk_calibration', 'official_eval', 'confirmation',
    }:
        raise ValueError('Unknown HT21 data role')
    blockers = []
    if not recording['ground_truth_present']:
        blockers.append('released_ground_truth_missing_detections_are_not_targets')
    motion = recording['metadata']['camera_motion']
    if motion is not False:
        blockers.append('camera_motion_present_or_unknown_no_compensation')
    blockers.extend([
        'offline_interpolation_not_verified_online_observation',
        'physical_site_independence_unverified',
        'forecast_clock_population_and_scientific_roles_not_registered',
        'dataset_use_terms_not_resolved_for_scientific_admission',
    ])
    return dict(sequence=recording['sequence'], requested_role=requested_role,
                allowed=requested_role == 'source_audit',
                supervised_forecasting=False, calibration=False, confirmation=False,
                scientific_use_blockers=blockers)


def require_role(recording, requested_role):
    result = admission(recording, requested_role)
    if not result['allowed']:
        raise ValueError('HT21 remains source-audit only: ' + '; '.join(result['scientific_use_blockers']))
    return result
