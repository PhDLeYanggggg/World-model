"""Produce label-blind alternatives, not an approved scientific role manifest."""
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_m3w_dronecrowd_cross_archive import HOME, OLD, PUBLIC, digest, once, read

SALT = 'm3w-dronecrowd-role-proposal-v1:'


def proposed_groups(recording_group, quarantine):
    groups = set(recording_group.values())
    if not set(quarantine) <= groups:
        raise ValueError('Unknown quarantined group')
    ordered = sorted(groups-set(quarantine), key=lambda g: hashlib.sha256((SALT+g).encode()).hexdigest())

    def role(group_ids):
        return dict(groups=sorted(group_ids), recordings=sorted(k for k, v in recording_group.items() if v in group_ids))

    cut = max(1, round(len(ordered)/3))
    return dict(
        excluded=role(quarantine),
        split_calibration_confirmation=(dict(calibration=role(ordered[:cut]), confirmation=role(ordered[cut:]))
            if len(ordered) > 1 else None),
        frozen_zero_shot=dict(calibration=role([]), confirmation=role(ordered)) if ordered else None)


def main():
    path = PUBLIC/'analysis.json'; analysis = read(path); reg = PUBLIC/'registration.json'
    if (analysis['clip_pairs'] != analysis['exposed_recordings']*analysis['external_recordings']
            or analysis['registration_sha256'] != digest(reg) or analysis['outcome_labels_opened']):
        raise ValueError('Only the complete label-blind source screen can inform this proposal')
    constraints = OLD/'exclusion_constraints.json'; groups = read(constraints)['recording_group']
    value = dict(kind='prospective_alternatives_not_a_role_assignment',
        cross_archive_analysis_sha256=digest(path), exclusion_constraints_sha256=digest(constraints),
        source_order_salt=SALT, alternatives=proposed_groups(groups, analysis['quarantined_groups']),
        training_and_model_selection='existing_development_data_only',
        observation_steps=8, prediction_steps=12, raw_annotation_stride=1, coordinate_unit='image_pixel',
        outcome_or_support_counts_used_for_partition=False,
        original_rules_unchanged=True, risk_budget=.02, independent_physical_sites_verified=False,
        scientific_roles_assigned=False, predictive_admission=False, outcome_readout_allowed=False,
        stage5c_executed=False, smc_enabled=False, submission_ready=False)
    dest = HOME/'dronecrowd_external_role_proposal_v1/alternatives.json'
    once(dest, value)
    for key, alternative in value['alternatives'].items():
        if key != 'excluded' and alternative:
            print(key, {k: (len(v['groups']), len(v['recordings'])) for k, v in alternative.items()})
    print('No roles assigned:', digest(dest))


if __name__ == '__main__':
    main()
