"""Read fixed author source and demonstrate the annotation-time boundary.

Third-party source is never imported or executed. The numerical witness is
synthetic and is not an audit of actual DroneCrowd annotation values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_dronecrowd_metadata import plain_path
from scripts.fetch_m3w_dut_annotations import blob_hash, fetch
from src.evaluation.m3w_dronecrowd_intake import inspect_xml
from src.evaluation.m3w_zara_media_lineage import interpolate_controls

COMMIT = '7de990ac0f7882dc0420b0f529b08951ae0f1230'
SOURCE_FILES = {
    'README.md': (19491, '8529153ebd8073ca85e1e780a279389b2b39fbf9'),
    'cli.py': (45587, '4e5cba7867818d25b3c8febd0cc796d10bb54439'),
    'models.py': (11044, '752060732bba4ae6c88a352b7b4808976045e0c3'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_bytes(directory, download=False, fetcher=fetch):
    directory = plain_path(directory)
    files, manifest = {}, []
    for name, (size, expected_blob) in SOURCE_FILES.items():
        path = plain_path(directory / name)
        url = f'https://raw.githubusercontent.com/cvondrick/vatic/{COMMIT}/{name}'
        if path.exists():
            if path.stat().st_size != size:
                raise ValueError('Author source size changed')
            data = path.read_bytes()
        elif download:
            data = fetcher(url, size)
        else:
            raise ValueError('Pinned source absent; explicit --download-source required')
        if len(data) != size or blob_hash(data) != expected_blob:
            raise ValueError('Author source differs from pinned Git blob')
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
        files[name] = data.decode('utf-8')
        manifest.append({'name': name, 'bytes': size, 'git_blob': expected_blob,
                         'sha256': sha(data), 'source_url': url})
    return files, manifest


def reviewed_source_findings(files):
    source = files['cli.py']
    begin = source.index('    def dumpxml(self, file, data):')
    end = source.index('    def dumpjson(self, file, data):', begin)
    xml_method = source[begin:end]
    signatures = {
        'dump_calls_linear_fill': 'path = vision.track.interpolation.LinearFill(track.boxes)' in source,
        'xml_writes_frame': 'box.frame' in xml_method,
        'xml_writes_visibility': 'box.lost' in xml_method and 'box.occluded' in xml_method,
        'xml_reads_generated_flag': 'box.generated' in xml_method,
        'xml_writes_keyframe_flag': 'keyframe' in xml_method,
        'other_export_paths_preserve_generated': 'file.write(str(box.generated))' in source
            and "data['generated'] = box.generated" in source,
        'model_optional_interpolation': 'def getboxes(self, interpolate = False' in files['models.py']
            and 'self.interpolatecache = LinearFill(result)' in files['models.py'],
    }
    expected = {key: key not in {'xml_reads_generated_flag', 'xml_writes_keyframe_flag'} for key in signatures}
    if signatures != expected:
        raise ValueError('Source signatures differ from the manually reviewed fixed version')
    return {
        'checks_are_pinned_source_patterns_not_third_party_execution': signatures,
        'xml_export_method_lines': [source[:begin].count('\n') + 1, source[:end].count('\n')],
        'dronecrowd_used_this_exact_version': 'unknown',
        'dronecrowd_actual_interpolation': 'not_run_archive_unread',
        'inference': 'original_XML_may_already_omit_interpolation_provenance',
    }


def unflagged_xml(frames, xy):
    """Synthetic serialization witness, not a reimplementation of VATIC."""
    root = ET.Element('annotations', count='1')
    track = ET.SubElement(root, 'track', id='0', label='synthetic_person')
    for frame, (x, y) in zip(frames, xy, strict=True):
        ET.SubElement(track, 'box', frame=str(int(frame)), xtl=str(float(x)),
                      ytl=str(float(y)), xbr=str(float(x + 2)), ybr=str(float(y + 2)),
                      outside='0', occluded='0')
    return ET.tostring(root, encoding='utf-8')


def interpolation_witness():
    frames = np.arange(11, dtype=np.int64)
    controls = np.asarray([[0., 0., 0., 0.], [10., 0., 10., 0.]])
    dense, latest_source, _ = interpolate_controls(controls, frames)
    direct = np.column_stack([frames, np.zeros_like(frames)]).astype(float)
    sparse_export, direct_export = unflagged_xml(frames, dense), unflagged_xml(frames, direct)
    altered = controls.copy()
    altered[-1, 0] = 20.
    perturbed, _, _ = interpolate_controls(altered, frames)
    query_frame = 8
    history = np.arange(1, 9)
    screen = inspect_xml(sparse_export, '00001')
    return {
        'result_source': 'synthetic_constructive_witness_not_dataset_measurement',
        'observed_frames': history.tolist(), 'query_frame': query_frame,
        'all_nominal_input_frames_not_after_query': bool((history <= query_frame).all()),
        'source_control_after_query': int(latest_source[history].max()),
        'history_rows_changed_by_altering_future_control': int(np.any(dense[history] != perturbed[history], axis=1).sum()),
        'same_unflagged_xml_from_direct_or_interpolated_rows': sparse_export == direct_export,
        'unflagged_xml_sha256': sha(sparse_export),
        'structural_screen_status': screen['status'],
        'structural_screen_interpolation_provenance': screen['interpolation_provenance'],
        'causal_rows_exported_by_screen': screen['causal_feature_rows_exported'],
        'last_causal_fd_velocity_before': (dense[8] - dense[7]).tolist(),
        'last_causal_fd_velocity_after': (perturbed[8] - perturbed[7]).tolist(),
        'interpretation': 'causal_finite_difference_cannot_remove_future_dependency_in_supplied_coordinates',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--download-source', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/VATIC_source_review')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs/publication_readiness_2026_09/annotation_export_provenance_v1')
    args = parser.parse_args()
    source, output = plain_path(args.source_root), plain_path(args.output_dir)
    if not source.is_relative_to(ROOT / 'external_data') or not output.is_relative_to(ROOT / 'outputs'):
        raise SystemExit('Keep third-party source ignored under external_data, reports under outputs')
    files, manifest = source_bytes(source, args.download_source)
    result = {
        'result_source': 'fresh_run', 'source_review_scope': 'fixed_VATIC_source_and_synthetic_witness',
        'vatic_repository': 'https://github.com/cvondrick/vatic', 'vatic_commit': COMMIT,
        'source_files': manifest, 'source_findings': reviewed_source_findings(files),
        'witness': interpolation_witness(),
        'read_paper_scope': 'DroneCrowd_arXiv_2105.02440v1_sections_3.1_and_3.2',
        'paper_source': 'https://arxiv.org/pdf/2105.02440',
        'actual_dronecrowd_xml': 'not_run_download_confirmation_pending',
        'source_paper_reports_vatic_use': True,
        'physical_sites_and_camera_motion_verified': False,
        'protocol_changed': False, 'scientific_roles_assigned': False,
        'real_training': False, 'new_model_scores': False,
        'new_admitted_recordings': 0, 'stage5c_executed': False, 'smc_enabled': False,
        'implementation_sha256': {p: sha((ROOT / p).read_bytes()) for p in (
            'scripts/audit_m3w_annotation_export_provenance.py',
            'src/evaluation/m3w_dronecrowd_intake.py',
            'src/evaluation/m3w_zara_media_lineage.py',
            'tests/test_m3w_annotation_export_provenance.py')},
    }
    path = output / 'analysis.json'
    if args.verify:
        if result != json.loads(path.read_text()):
            raise SystemExit('Source review or constructive witness differs from saved evidence')
    else:
        output.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result_source': 'cached_verified' if args.verify else 'fresh_run',
                      'analysis_sha256': sha(path.read_bytes()),
                      'source_files_verified': len(manifest),
                      'synthetic_witness': result['witness'], 'new_admitted_recordings': 0}, indent=2))


if __name__ == '__main__':
    main()
