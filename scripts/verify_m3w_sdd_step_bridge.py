"""Independent full-index checks using observed contiguous runs, not window scans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_sdd_state_support import load_source
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write


def expected_current_rows(source, labels, stride):
    keep = (source[:, 6] == 0) & (source[:, 5] % stride == 0)
    points, types = source[keep], labels[keep]
    if not len(points):
        return points, np.empty(0, dtype=np.int64)
    boundary = np.r_[True, (np.diff(points[:, 0]) != 0) | (np.diff(points[:, 5]) != stride)]
    position = np.arange(len(points))
    run_start = np.maximum.accumulate(np.where(boundary, position, 0))
    eligible = (position-run_start >= 7) & (types == 'Pedestrian')
    return points, position[eligible]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.registration.read_text())
    for name, digest in config['bindings'].items():
        if file_digest(ROOT/name) != digest:
            raise ValueError('Bound input changed: '+name)
    report_path = ROOT/config['reports']/'report.json'
    report = json.loads(report_path.read_text())
    if report['identity']['registration_sha256'] != file_digest(args.registration):
        raise ValueError('Registration mismatch')
    links = json.loads((ROOT/config['source_manifest']).read_text())
    source_by_id = {r['annotation_key']: r for r in links['records']}
    split = json.loads((ROOT/config['split_manifest']).read_text())
    training_ids = {r['scene_id']+'/'+r['video_id'] for r in split['video_reports'] if r['split_id'] == 'train'}
    if {r['recording'] for r in report['recordings']} != training_ids:
        raise ValueError('Train roster mismatch')
    checks = []
    for receipt in report['recordings']:
        key = receipt['recording']
        entry = source_by_id[key]
        path = ROOT/entry['annotations_path']
        if file_digest(path) != entry['annotations_sha256']:
            raise ValueError('Annotation changed')
        rows, labels = load_source(path)
        for stride in config['raw_frame_strides']:
            part = receipt['strides'][str(stride)]
            index_path = ROOT/config['output']/key/f'index_stride{stride}.npy'
            if file_digest(index_path) != part['index_sha256']:
                raise ValueError('Saved index changed')
            index = np.load(index_path, mmap_mode='r', allow_pickle=False)
            points, expected = expected_current_rows(rows, labels, stride)
            np.testing.assert_array_equal(index['current_row'], expected)
            np.testing.assert_array_equal(index['history_start'], expected-7)
            assert np.all(index['future_end'] == -1)
            assert np.all(index['future_steps'] == 12)
            assert np.all(index['horizon_raw'] == 12*stride)
            assert len(index) == part['past_only_queries']
            # Track tails must remain eligible even with no next-step label.
            current = points[expected]
            last_by_agent = {int(a): f for a, f in rows[:, [0, 5]]}
            tail = np.asarray([f+stride > last_by_agent[int(a)] for a, f in current[:, [0, 5]]])
            checks.append(dict(recording=key, stride=stride, all_index_rows_verified=len(index),
                               rows_beyond_source_track_future_support=int(tail.sum())))
    result = dict(result_source='fresh_run', method='contiguous_run_prefix_index_enumeration',
        verifier_sha256=file_digest(Path(__file__)), registration_sha256=file_digest(args.registration),
        bridge_report_sha256=file_digest(report_path), complete=True,
        total_index_rows_verified=sum(r['all_index_rows_verified'] for r in checks),
        checks=checks, original_val_test_opened=False, new_training=False)
    json_write(ROOT/config['reports']/'independent_index_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)


if __name__ == '__main__':
    main()
