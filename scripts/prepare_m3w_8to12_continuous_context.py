"""Repair a packaged development source without changing or blessing old caches."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.data_unification.m3w_causal_recordings import write_recording
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest, protocol_digest
from scripts.audit_m3w_students01_packaging import inspect_packaging


def main():
    destination = ROOT / 'configs/m3w_8to12_continuous_context_v5.json'
    cache = ROOT / 'data/stage_cvpr2027_experiments/continuous_context_v5/ucy_students01'
    if destination.exists() or cache.exists():
        raise SystemExit('Versioned protocol/cache exists; do not overwrite')
    parent = 'configs/m3w_8to12_public_predictors_v4.json'
    p = json.loads((ROOT / parent).read_text())
    parent_digest = protocol_digest(p)
    original = 'external_data/OpenTraj/datasets/UCY/students01/students001.txt'
    packaged = 'external_data/OpenTraj/datasets/UCY/students01/students001-trajnet.txt'
    points = np.loadtxt(ROOT / original)
    audit = inspect_packaging(points, np.loadtxt(ROOT / packaged))
    if not audit['exact_full_track_prefix_floor_length_over_20_reconstruction']:
        raise SystemExit('Source repair premise failed')
    metadata = write_recording(cache, points, {'id': 'ucy_students01', 'physical_scene': 'ucy_university',
        'family': 'UCY', 'canonical': 'UCY/students01/students001.txt',
        'files': [{'path': original, 'sha256': file_digest(ROOT / original)}],
        'exposure_status': 'development_exposed',
        'source_representation': 'continuous_annotation_ids_no_20_point_tail_filter',
        'annotation_generation_causality': 'upstream_interpolation_not_verified',
        'packaging_audit': 'outputs/publication_readiness_2026_09/students01_packaging_audit/audit.json'})
    if metadata['observation_step_windows'] != audit['support']['continuous']['complete_8to12_windows']:
        raise SystemExit('Continuous window count mismatch')
    record = p['records']['ucy_students01']
    record['cache_path'] = str(cache.relative_to(ROOT))
    record['metadata_sha256'] = file_digest(cache / 'metadata.json')
    record['review_evidence'] += ['outputs/publication_readiness_2026_09/students01_packaging_audit/audit.json']
    decision = 'outputs/publication_readiness_2026_09/continuous_context_v5_decision.md'
    p['experimental_change'] = {'parent_protocol_sha256': parent_digest,
        'factor': 'continuous_students01_identity_and_context_repair', 'decision': decision,
        'fit_recordings_changed': False, 'development_support_changed': True,
        'development_adaptive_not_confirmatory': True}
    for stage in ('calibration', 'confirmation'):
        p[f'{stage}_receipt'] = f'data/stage_cvpr2027_experiments/continuous_context_v5/{stage}_forbidden.json'
    bindings = set(p['bindings']) | {parent, decision, original, packaged,
        'scripts/prepare_m3w_8to12_continuous_context.py', 'scripts/audit_m3w_students01_packaging.py',
        'scripts/run_m3w_continuous_predictor_pair.py',
        'outputs/publication_readiness_2026_09/students01_packaging_audit/audit.json'}
    p['bindings'] = {name: file_digest(ROOT / name) for name in sorted(bindings)}
    p['approval'] = {'approved_by': 'research_agent_under_explicit_user_delegation_2026-09-16',
        'decision_reference': decision, 'protocol_sha256': protocol_digest(p)}
    contract = ExperimentContract(p, ROOT)
    with destination.open('x') as stream:
        json.dump(p, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'protocol': str(destination), 'sha256': contract.digest,
        'students01_complete_8to12_windows': metadata['observation_step_windows'],
        'raw_t50_available_not_evaluated': metadata['raw_exact_windows']['50'],
        'new_independent_scene': False}))


if __name__ == '__main__':
    main()
