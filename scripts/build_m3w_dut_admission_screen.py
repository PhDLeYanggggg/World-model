"""Bind DUT diagnostic intake evidence for fail-closed admission, not approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_unification.m3w_causal_recordings import RecordingWindows
from src.evaluation.m3w_recording_lineage import sha256
from src.evaluation.m3w_intake_admission import validate_admission


def reference(root, path):
    return {'path': str(path.resolve().relative_to(root.resolve())), 'sha256': sha256(path)}


def build(root, source_manifest, conversion_report, quality_audit, cache, output):
    root = Path(root).resolve()
    if output.exists():
        raise ValueError('Preserve existing admission evidence; choose a new screen path')
    source, conversion, audit = (json.loads(p.read_text()) for p in (source_manifest, conversion_report, quality_audit))
    if (source['status'] != 'verified' or audit['source_manifest_sha256'] != sha256(source_manifest)
            or audit['build_report_sha256'] != sha256(conversion_report)
            or sha256(cache / 'run_identity.json') != conversion['run_identity_sha256']):
        raise ValueError('Changed or unverified DUT intake evidence')
    declared = {r['id'] for r in conversion['recordings']}
    if declared != {r['recording'] for r in audit['clips']} or not set(audit['quarantine_recordings']) <= declared:
        raise ValueError('Incomplete quality audit coverage')
    screen = {'schema_version': 1, 'kind': 'm3w_diagnostic_intake_screen',
        'result_source': 'fresh_run_binding_cached_verified_conversion_and_audit',
        'source_manifest': reference(root, source_manifest), 'conversion_report': reference(root, conversion_report),
        'quality_audit': reference(root, quality_audit), 'recordings': {},
        'source_use_approved': False, 'scientific_roles_assigned': False,
        'independent_confirmation_approved': False, 'future_label_api_calls': 0,
        'prediction_accuracy_evaluated': False, 'training_run': False,
        'script_sha256': sha256(Path(__file__))}
    for row in conversion['recordings']:
        directory = cache / row['id']
        receipt = json.loads((directory / 'completion.json').read_text())
        if receipt['metadata_sha256'] != sha256(directory / 'metadata.json'):
            raise ValueError('Changed converted metadata')
        reader = RecordingWindows(directory)
        metadata = reader.metadata
        if metadata['id'] != row['id'] or metadata['physical_scene'] != row['physical_scene']:
            raise ValueError('Recording identity mismatch')
        findings = [f for f in audit['annotation_flags'] if f['recording'] == row['id']]
        blocked = row['id'] in audit['quarantine_recordings'] or bool(findings)
        screen['recordings'][row['id']] = {
            'dataset': metadata['dataset'], 'physical_scene': metadata['physical_scene'],
            'cache_path': str(directory.resolve().relative_to(root)),
            'metadata_sha256': sha256(directory / 'metadata.json'),
            'source_file_hashes': {f['path']: f['sha256'] for f in metadata['files']},
            'quality_disposition': 'quarantine' if blocked else 'no_flags_in_limited_screen',
            'blocking_findings': findings,
            'source_use_decision': 'pending_not_issued_by_this_script',
            'historical_predictive_use': 'unknown_not_proven_untouched',
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(screen, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return screen


def check_refusals(root, screen, screen_path):
    checks = []
    for name, entry in screen['recordings'].items():
        metadata = json.loads((root / entry['cache_path'] / 'metadata.json').read_text())
        record = {**entry, 'intake_screen': reference(root, screen_path)}
        for role in ('fit', 'development', 'calibration', 'confirmation'):
            try:
                validate_admission(root, name, record, metadata, role)
            except ValueError as exc:
                checks.append({'recording': name, 'attempted_role_for_check_only': role,
                               'admitted': False, 'reason': str(exc)})
            else:
                raise ValueError('Pending diagnostic source incorrectly admitted')
    return {'result_source': 'fresh_run_real_metadata_admission_refusal_checks', 'checks': checks,
        'screen': reference(root, screen_path), 'refusals': len(checks), 'admitted_recordings': 0,
        'actual_roles_assigned': False, 'formal_protocol_changed': False, 'source_use_approval_issued': False,
        'future_label_api_calls': 0, 'training_run': False, 'prediction_accuracy_evaluated': False,
        'approval_authenticated': False, 'independence_proven': False,
        'code_sha256': {str(p.relative_to(root)): sha256(p) for p in
            (root / 'src/evaluation/m3w_intake_admission.py', Path(__file__))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = ROOT / 'outputs/publication_readiness_2026_09/dut_causal_intake'
    parser.add_argument('--source-manifest', type=Path, default=directory / 'source_manifest.json')
    parser.add_argument('--conversion-report', type=Path, default=directory / 'build_report.json')
    parser.add_argument('--quality-audit', type=Path, default=directory / 'independent_recount.json')
    parser.add_argument('--cache', type=Path, default=ROOT / 'data/stage_cvpr2027_causal/dut_diagnostic')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/intake_admission/dut_screen.json')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/intake_admission/dut_refusals.json')
    args = parser.parse_args()
    if not all(p.resolve().is_relative_to(ROOT) for p in (args.output, args.report)):
        raise SystemExit('Outputs must stay within workspace')
    screen = build(ROOT, args.source_manifest, args.conversion_report, args.quality_audit, args.cache, args.output)
    checks = check_refusals(ROOT, screen, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(checks, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:v for k,v in checks.items() if k not in ('checks', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    main()
