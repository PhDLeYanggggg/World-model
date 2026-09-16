"""Snapshot reviewed development assets or check an explicitly approved protocol.

This script never assigns a split, approves a protocol, fits a model, or reads
future labels through the dataset API. New draft choices remain unresolved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_unification.m3w_causal_recordings import RecordingWindows, validate_catalog
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest


def snapshot_draft(root: Path):
    config_path = root / 'configs/m3w_publication_recordings.json'
    config = json.loads(config_path.read_text())
    audit_path = root / config['source_identity_audit']
    audit = json.loads(audit_path.read_text())
    catalog = validate_catalog(config, root, audit)
    cache = root / 'data/stage_cvpr2027_causal'
    manifest_path = cache / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['config_sha256'] != file_digest(config_path):
        raise ValueError('Cache/catalog identity changed')
    records, evidence = {}, []
    for item in catalog:
        if not item['enabled']:
            continue
        reader = RecordingWindows(cache / item['id'])
        if any(reader.metadata[key] != item[key] for key in ('id', 'physical_scene', 'canonical', 'files')):
            raise ValueError('Cached source lineage changed')
        sources = {f['path'] for f in item['files']}
        prior = [r for r in audit['sources'] if r['source'] in sources and r['cache_rows']]
        history = 'development_exposed' if prior else 'unknown'
        records[item['id']] = {
            'physical_scene': item['physical_scene'],
            'cache_path': str((cache / item['id']).relative_to(root)),
            'metadata_sha256': file_digest(cache / item['id'] / 'metadata.json'),
            'historical_use': history,
            'review_evidence': [str(audit_path.relative_to(root)),
                                'outputs/publication_readiness_2026_09/recording_lineage_audit.md'],
        }
        evidence.append({'recording': item['id'], 'physical_scene': item['physical_scene'],
                         'history': history, 'legacy_cache_splits': sorted({s for r in prior for s in r['cache_rows']}),
                         'indexed_views': len(reader), 'metadata_sha256': records[item['id']]['metadata_sha256'],
                         'array_hashes_verified': True})
    bindings = [config_path, audit_path, manifest_path,
                root / 'src/data_unification/m3w_causal_recordings.py',
                root / 'src/evaluation/m3w_experiment_contract.py',
                root / 'src/world_model/m3w_joint_intervention.py']
    protocol = {
        'schema_version': 1, 'status': 'draft', 'scope': 'confirmatory',
        'calibration_receipt': 'data/stage_cvpr2027_causal/experiment_control/calibration.json',
        'confirmation_receipt': 'data/stage_cvpr2027_causal/experiment_control/confirmation.json',
        'bindings': {str(path.relative_to(root)): file_digest(path) for path in bindings},
        'records': records, 'assignments': {name: 'unassigned' for name in records}, 'fit_folds': {},
        'task': {'history_steps': None, 'prediction_unit': None, 'horizon': None,
                 'coordinate_claim': 'dataset_local_unverified', 'primary_metric': None, 'aggregation': None},
        'risk': {'delta': None, 'risks': [], 'easy_definition': None, 'easy_degradation_max': .02},
        'seeds': [], 'bootstrap_unit': None, 'bootstrap_resamples': None, 'approval': None,
        'unresolved': ['Primary temporal protocol and evaluation aggregation',
                       'Fit/development/calibration/confirmation roles and crossfit folds',
                       'Independent unexposed calibration and confirmation assets',
                       'Bounded risk definition, tolerances, error probability and easy definition',
                       'Three or more seeds, cluster bootstrap specification and explicit user approval'],
    }
    return protocol, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'configs/m3w_independent_experiment.draft.json')
    parser.add_argument('--snapshot-draft', action='store_true')
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'outputs/publication_readiness_2026_09/experiment_contract')
    args = parser.parse_args()
    evidence = []
    if args.snapshot_draft:
        if args.protocol.exists():
            raise SystemExit('Refusing to overwrite existing scientific choices; use a new explicit protocol path')
        protocol, evidence = snapshot_draft(ROOT)
        args.protocol.parent.mkdir(parents=True, exist_ok=True)
        with args.protocol.open('x') as stream:
            json.dump(protocol, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        protocol = json.loads(args.protocol.read_text())
    artifacts = json.loads(args.artifacts.read_text()) if args.artifacts else []
    try:
        contract = ExperimentContract(protocol, ROOT, artifacts)
        allowed, refusal = True, None
    except (ValueError, KeyError, TypeError, OSError) as exc:
        allowed, refusal = False, f'{type(exc).__name__}: {exc}'
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'result_source': 'fresh_run',
        'scope': 'experiment_contract_preflight_not_model_performance',
        'protocol_path': str(args.protocol), 'protocol_sha256': protocol_digest(protocol),
        'snapshot_source': 'cached_verified' if evidence else 'not_run_no_snapshot_requested',
        'records_checked': evidence, 'contract_ready': allowed, 'refusal': refusal,
        'fit_or_evaluation_executed': False, 'future_label_api_called': False,
        'approval_authenticated': False, 'independence_proven': False,
        'artifact_manifests_supplied': len(artifacts),
        'deployment_promoted': False, 'stage5c_executed': False, 'smc_enabled': False,
        'source_hashes': {p: file_digest(ROOT / p) for p in
                          ['src/evaluation/m3w_experiment_contract.py', 'scripts/check_m3w_experiment_contract.py']},
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / 'preflight.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Independent Experiment Contract Preflight', '',
             'Fresh execution of a provenance/role check, not a forecast experiment.', '',
             f'- Contract ready: {allowed}', f'- Refusal: {refusal}',
             f'- Protocol digest: `{result["protocol_sha256"]}`',
             '- No split assigned or approval issued by this script.',
             '- Historical development exposure cannot be cleared by renaming data.',
             '- A passed metadata check cannot authenticate approval, prove IID, or certify safety.', '',
             '| recording | physical scene | historical use | legacy cache roles | indexed views |',
             '| --- | --- | --- | --- | ---: |']
    for row in evidence:
        lines.append(f'| {row["recording"]} | {row["physical_scene"]} | {row["history"]} | '
                     f'{", ".join(row["legacy_cache_splits"])} | {row["indexed_views"]} |')
    lines += ['', 'The role-bound interface and receipt apply only to callers that use them. Existing '
              'legacy scripts are not silently redirected. This is not a system-wide access-control boundary.', '',
              'A confirmatory protocol needs reviewed independent calibration/confirmation sources. '
              'These assets remain development material; zero new independent holdout evidence is created.', '',
              'Stage5C and SMC remain disabled. No metric, seconds-level or submission-ready claim.']
    (args.report_dir / 'preflight.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'contract_ready': allowed, 'refusal': refusal,
                      'recordings_checked': len(evidence), 'training_or_eval_run': False}))
    # A draft snapshot is allowed to complete; a run preflight must fail closed.
    if not allowed and not args.snapshot_draft:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
