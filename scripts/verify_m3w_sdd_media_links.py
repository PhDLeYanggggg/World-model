"""Check published source links, evidence bindings and deterministic export."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def main():
    reports = ROOT/'outputs/publication_readiness_2026_09/sdd_media_alignment'
    manifest_path = reports/'diagnostic_media_links.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest['data_role'] != 'source_audit_only' or manifest['new_training_source_admitted']
            or manifest['official_eval_allowed'] or len(manifest['records']) != 60):
        raise ValueError('Wrong role, admission or source count')
    for path, digest in manifest['report_bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Changed bound report')
    for name in ('audit.json', 'resize_mapping.json', 'nexus_identity.json'):
        evidence = json.loads((reports/name).read_text())
        codes = evidence.get('code_sha256', evidence.get('identity', {}).get('code_sha256', {}))
        for path, digest in codes.items():
            if file_digest(ROOT/path) != digest:
                raise ValueError('Changed audit/geometry implementation')
    paths = set()
    for entry in manifest['records']:
        for name in ('annotations', 'reference', 'video'):
            if file_digest(ROOT/entry[name+'_path']) != entry[name+'_sha256']:
                raise ValueError('Changed source file')
        if entry['video_path'] in paths or not entry['coverage']['decoded_range_covers_annotations']:
            raise ValueError('Invalid correspondence or index support')
        paths.add(entry['video_path'])
    before = file_digest(manifest_path)
    subprocess.run([sys.executable, 'scripts/build_m3w_sdd_diagnostic_media_links.py'], cwd=ROOT, check=True)
    if before != file_digest(manifest_path):
        raise ValueError('Deterministic source export changed')
    result = dict(result_source='cached_verified_sources_code_and_deterministic_export',
        manifest_sha256=before, distinct_annotation_records=60, distinct_video_files=len(paths),
        source_file_hash_checks=180, deterministic_manifest_rebuild_exact=True,
        no_new_decode_or_model_training=True, physical_time_or_metric_claim=False,
        training_admission_or_split_changed=False, semantic_all_rows_certification=False)
    json_write(reports/'link_verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
