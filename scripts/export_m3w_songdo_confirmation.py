"""Publish the approved Songdo result from score receipts, without identifiers."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import report_m3w_songdo_secondary as secondary

RUN = ROOT/'data/stage_cvpr2027_experiments/songdo_confirmation_v1'
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/songdo_calibrated_20261110'
ARMS = secondary.primary.ARMS


def metric(row, *, groups=False):
    result = {k: row[k] for k in ('point', 'ci95', 'status', 'defined_seed_views', 'required_seed_views')}
    result['estimable_components'] = sum(x is not None for x in row['by_component'].values())
    if groups:
        result['component_points'] = [v for _, v in sorted(row['by_component'].items())]
    if row['worst_estimable_source_value'] is not None:
        result['worst_estimable_source_value'] = row['worst_estimable_source_value']
    return result


def section(row, *, matched):
    data = dict(metrics={k: metric(v) for k, v in row['metrics'].items()},
                worst_flight_seed_available_ade_native=row['worst_flight_seed_available_ade_native'])
    for k in ('vs_cv', 'vs_ols4', 'edited_only'):
        if k in row:
            data[k] = {name: metric(r) for name, r in row[k].items()}
    if 'original_risk' in row or 'risk' in row:
        data['risk'] = row['original_risk' if matched else 'risk']
    return data


def export(run, public):
    c, c_hash = secondary.load(public/'calibration_receipt.json')
    registration, reg_hash = secondary.load(run/'registration.json')
    freeze, freeze_hash = secondary.load(run/'freeze.json')
    cut, cut_hash = secondary.load(run/'calibration_cutoffs.json')
    if (c['registration_sha256'] != reg_hash or c['freeze_sha256'] != freeze_hash
            or c['calibration_cutoffs_sha256'] != cut_hash or c['confirmation_outcomes_opened']
            or not c['original_2pct_unchanged'] or freeze['thresholds'] != c['thresholds']
            or cut['support_rows'] != c['calibration_support_rows']):
        raise ValueError('Confirmation must use the pre-outcome calibration freeze')
    roles = registration['flight_roles']
    by_role = {k: [g for role, g in roles.values() if role == k] for k in ('calibration', 'confirmation')}
    if [(len(g), len(set(g))) for g in by_role.values()] != [(40, 3), (160, 14)] or (
            set(by_role['calibration']) & set(by_role['confirmation'])):
        raise ValueError('The registered 40/160 split changed')
    report = secondary.read_finished(run)
    if report['matched_only'] or tuple(report[k] for k in ('flights', 'components', 'flight_seed_views', 'matched_seed_views')) != (160, 14, 480, 480):
        raise ValueError('Expected the complete registered confirmation')
    b = report['binding']
    # One hash binds the local per-flight receipts without publishing flight names.
    root = sha256(''.join(sorted(b['flight_score_sha256'].values())).encode()).hexdigest()
    return dict(result_source='fresh_source_disjoint_confirmation_replayed_from_score_receipts',
                flights=160, source_components=14, flight_seed_views=480, matched_seed_views=480,
                seeds=[17, 29, 43], calibration_flights=40, calibration_components=3,
                risk_rule='original_three_2pct_checks', original_2pct_unchanged=True,
                matching='exactly_two_edits_same_query_and_peak_radius',
                matched={a: section(r, matched=True) for a, r in report['matched'].items()},
                matched_joint_minus={a: {k: metric(v, groups=True) for k, v in r.items()}
                                     for a, r in report['matched_joint_minus'].items()},
                released={a: section(report['released'][a], matched=False) for a in ARMS},
                binding=dict(registration_sha256=reg_hash, published_calibration_receipt_sha256=c_hash,
                             calibration_cutoffs_sha256=cut_hash, release_freeze_sha256=freeze_hash,
                             primary_score_sha256=b['primary_score_sha256'], flight_score_hash_root=root),
                no_fitting_on_confirmation=True, export_read_labels=False, submission_ready=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=RUN)
    p.add_argument('--output', type=Path, default=PUBLIC/'confirmation_receipt.json')
    args = p.parse_args()
    receipt = export(args.run, PUBLIC)
    secondary.write(args.output, receipt)
    print(json.dumps({k: receipt[k] for k in ('flights', 'source_components', 'flight_seed_views')}, indent=2))
