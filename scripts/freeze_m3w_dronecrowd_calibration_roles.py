"""Record the approved 15/30 role split without touching annotation files."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT/'outputs/publication_readiness_2026_09'
PARENT = ROOT/'configs/m3w_external_source_reservations_v1.json'
DEST = ROOT/'configs/m3w_dronecrowd_calibration_confirmation_v2.json'
GROUPS = HOME/'dronecrowd_grouping_v2/exclusion_constraints.json'
PROPOSAL = HOME/'dronecrowd_external_role_proposal_v1/alternatives.json'
PREFLIGHT = HOME/'dronecrowd_blind_preflight_v1/analysis.json'
SCREEN = HOME/'dronecrowd_cross_archive_v1/analysis.json'
BANK = HOME/'dronecrowd_exact_bank_v2/analysis.json'
MODE = 'split_calibration_confirmation'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def partition(recording_group, assignment):
    seen, assigned, records = set(), set(), {}
    if set(assignment) != {'calibration', 'confirmation'}:
        raise ValueError('Both and only the approved roles are required')
    for role, spec in assignment.items():
        groups, ids = set(spec['groups']), set(spec['recordings'])
        if len(groups) != len(spec['groups']) or len(ids) != len(spec['recordings']):
            raise ValueError('Duplicate group or recording in role')
        if groups & seen or ids & assigned or ids != {r for r, g in recording_group.items() if g in groups}:
            raise ValueError('Split overlaps or divides a complete source group')
        seen |= groups
        assigned |= ids
        records.update({r: role for r in ids})
    if assigned != set(recording_group) or seen != set(recording_group.values()):
        raise ValueError('Approved split must account for all reserved records')
    return records


def build():
    old, group, proposal, pre, bank = map(read, (PARENT, GROUPS, PROPOSAL, PREFLIGHT, BANK))
    groups = group['recording_group']
    reserved = {k.removeprefix('dronecrowd_'): r for k, r in old['records'].items()
                if r['dataset'] == 'DroneCrowd'}
    if (set(reserved) != set(groups) or any(r['reserved_role'] != 'confirmation'
            or r['exclusion_group'] != groups[k] or r['predictive_admission'] for k, r in reserved.items())):
        raise ValueError('The original reservation or exclusion group identity changed')
    if (proposal['exclusion_constraints_sha256'] != digest(GROUPS)
            or proposal['cross_archive_analysis_sha256'] != digest(SCREEN)
            or proposal['outcome_or_support_counts_used_for_partition']
            or proposal['alternatives']['excluded']['recordings']
            or pre['future_label_method_called'] or pre['easy_reference_measured']
            or bank['external_outcomes_opened']):
        raise ValueError('Require intact pre-outcome provenance')
    if read(PREFLIGHT.with_name('replay_execution.json'))['analysis_sha256'] != digest(PREFLIGHT):
        raise ValueError('Input preflight did not replay')
    for p in (PARENT, GROUPS, PROPOSAL):
        if bank['bindings'][str(p.relative_to(ROOT))] != digest(p):
            raise ValueError('Mismatch against verified bank lineage')
    assignment = proposal['alternatives'][MODE]
    index = partition(groups, assignment)
    counts = pre['prospective_alternatives'][MODE]
    for role, pair in (('calibration', (15, 40)), ('confirmation', (30, 72))):
        if ((len(assignment[role]['groups']), len(assignment[role]['recordings'])) != pair
                or (counts[role]['exclusion_groups'], counts[role]['recordings']) != pair):
            raise ValueError('Approved roster changed')
    return dict(kind='approved_dronecrowd_role_overlay_v2', approval_date='2026-11-07',
        approval='user_selected_15_calibration_30_confirmation', selected_alternative=MODE,
        supersedes_dronecrowd_roles_only=str(PARENT.relative_to(ROOT)),
        bindings={str(p.relative_to(ROOT)): digest(p) for p in (PARENT, GROUPS, PROPOSAL, PREFLIGHT, SCREEN, BANK)},
        assignments=assignment, recording_role_index=index, input_only_counts=counts,
        observation_steps=8, prediction_steps=12, annotation_stride=1, coordinate_unit='image_pixel',
        risk_budget=.02, original_rules_unchanged=True, independence_is_exclusion_group_level=True,
        independent_physical_sites_verified=False, predictive_admission=False,
        calibration_outcome_readout_allowed=False, confirmation_outcome_readout_allowed=False,
        outcome_labels_read_by_this_step=False, stage5c_executed=False, smc_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'verify'))
    phase = parser.parse_args().phase
    doc = json.dumps(build(), indent=2, sort_keys=True, allow_nan=False)+'\n'
    if DEST.exists() or phase == 'verify':
        if DEST.read_text() != doc:
            raise ValueError('Frozen role manifest does not match the approved allocation')
    else:
        with DEST.open('x') as f:
            f.write(doc)
    print(digest(DEST))


if __name__ == '__main__':
    main()
