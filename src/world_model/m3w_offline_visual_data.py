"""Versioned offline-observation adapter over immutable diagnostic media caches."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from src.data_unification.m3w_causal_recordings import BASELINES, causal_coordinate_transform
from src.evaluation.m3w_experiment_contract import ExperimentContract, file_digest
from src.evaluation.m3w_past_video_alignment import native_to_image_xy
from src.world_model.m3w_masked_history_images import MaskedHistoryImageStore, masked_center_patch
from src.world_model.m3w_offline_visual_forecast import geometry_features, normalized_targets


def verify_registration(root, path):
    reg = json.loads(path.read_text())
    if reg['observation_mode'] != 'offline_annotated' or reg['role'] != 'fit_only_exploratory':
        raise ValueError('Explicit offline-observation and fit-only authorization required')
    for name, digest in reg['bindings'].items():
        if file_digest(root / name) != digest:
            raise ValueError('Changed registered input: ' + name)
    parent = ExperimentContract(json.loads((root / reg['parent_protocol']).read_text()), root)
    if parent.digest != reg['parent_protocol_sha256']:
        raise ValueError('Scientific parent changed')
    if reg['recordings'] != [r for r, role in parent.protocol['assignments'].items() if role == 'fit']:
        raise ValueError('The approved full fit cohort must be retained')
    return reg, parent


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def build_inputs(root, reg, parent, output, identity, heartbeat):
    """Decode each historical image once; targets are stored in a separate label array."""
    sys.path.insert(0, str(root / reg['decoder_path']))
    import av
    if av.__version__ != reg['decoder_version']:
        raise ValueError('Decoder version changed')
    manifest = output / 'data_manifest.json'
    if manifest.exists():
        report = json.loads(manifest.read_text())
        if report['identity'] != identity:
            raise ValueError('Changed data identity')
        for name, digest in report['arrays'].items():
            if file_digest(output / name) != digest:
                raise ValueError('Changed data array ' + name)
        return report
    geometry, target, candidates, scale, image_rows, meta = [], [], [], [], [], []
    images, coverages = [np.zeros((1, 3, 32, 32), np.uint8)], [np.zeros((1, 32, 32), np.uint8)]
    offset, summaries = 1, {}
    for rid in reg['recordings']:
        reader, indices = parent.open_recording(rid, purpose='fit')
        points = reader.points
        source_rows = np.zeros(len(points), np.int64)
        if rid in ('ucy_zara01', 'ucy_zara02'):
            store = MaskedHistoryImageStore(root / reg['zara_cache'] / rid,
                                           observation_mode='offline_annotation_diagnostic')
            arrays = store.arrays
            if not np.array_equal(arrays['row_keys'], points[:, :2].astype(np.int64)):
                raise ValueError('Zara image/source row alignment changed')
            if not np.array_equal(arrays['native_xy'], points[:, 2:]):
                raise ValueError('Zara coordinates changed')
            rgb, coverage = np.asarray(arrays['rgb']), np.asarray(arrays['coverage'])
            source_rows[:] = np.arange(len(points)) + offset
            source_status = 'cached_verified_partial_image_arrays'
        elif rid in ('eth_eth', 'eth_hotel'):
            directory = (root / reader.metadata['files'][0]['path']).parent
            h = np.loadtxt(directory / 'H.txt')
            xy = native_to_image_xy(points[:, 2:4], h, projected_axes='row_col')
            rgb = np.zeros((len(points), 3, 32, 32), np.uint8)
            coverage = np.zeros((len(points), 32, 32), np.uint8)
            requests = {}
            for i, point in enumerate(points):
                requests.setdefault(int(point[0]), []).append(i)
            found = set()
            with av.open(str(directory / 'video.avi')) as video:
                for frame_id, frame in enumerate(video.decode(video=0)):
                    if frame_id in requests:
                        image = np.asarray(frame.to_image())
                        found.add(frame_id)
                        for row in requests[frame_id]:
                            rgb[row], coverage[row] = masked_center_patch(image, xy[row], 96, 32)
                    if frame_id % 1000 == 0:
                        heartbeat({'state': 'decode', 'recording': rid, 'frame': frame_id})
                    if frame_id >= max(requests):
                        break
            if found != set(requests):
                raise ValueError('Missing ETH source frames; no silent image removal')
            source_rows[:] = np.arange(len(points)) + offset
            source_status = 'fresh_run_native_index_supplied_H_not_physical_sync'
        else:
            rgb = coverage = None
            source_status = 'not_run_video_unavailable_explicit_zero_mask_rows_retained'
        if rgb is not None:
            images.append(rgb)
            coverages.append(coverage)
            offset += len(points)
        scene = parent.protocol['records'][rid]['physical_scene']
        summaries[rid] = {'rows': len(indices), 'physical_scene': scene,
            'image_source_status': source_status, 'image_source_rows': len(points) if rgb is not None else 0,
            'source_image_coverage_mean': float(coverage.mean() / 9) if coverage is not None else 0.,
            'source_images_empty': int((coverage.sum((1, 2)) == 0).sum()) if coverage is not None else len(points)}
        for item in indices:
            row = reader.index[item]
            begin, current = int(row['history_start']), int(row['current_row'])
            history = np.asarray(points[begin:current + 1])
            if len(history) != 8 or not np.all(history[:, 0] <= history[-1, 0]):
                raise ValueError('Past history support mismatch')
            inputs = reader.get_inputs(int(item))
            transform = causal_coordinate_transform(history[:, 2:4], history[:, 0], int(row['horizon_raw']))
            geometry.append(geometry_features(inputs))
            candidates.append(inputs['baseline_rollouts'])
            image_rows.append(source_rows[begin:current + 1])
            scale.append(transform['scale'])
            meta.append({'recording': rid, 'scene': scene, 'frame': int(history[-1, 0]),
                         'agent': int(history[-1, 1]), 'fold': reg['scene_folds'][scene]})
            # The complete inference payload is constructed before opening labels.
            labels = reader.get_labels(int(item))
            if not np.array_equal(labels['future_frame_ids'], history[-1, 0] + inputs['prediction_frame_offsets']):
                raise ValueError('Label grid mismatch')
            target.append(normalized_targets(history, labels['future_xy_dataset_local'], transform))
        heartbeat({'state': 'feature_build', 'recording': rid, 'rows': len(meta)})
    if len(meta) != reg['expected_rows']:
        raise ValueError('Cohort size differs from registration')
    values = dict(geometry=np.stack(geometry), targets=np.stack(target),
        baselines=np.stack(candidates), scale=np.asarray(scale), image_rows=np.stack(image_rows),
        rgb=np.concatenate(images), coverage=np.concatenate(coverages),
        folds=np.asarray([m['fold'] for m in meta], np.int64))
    output.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        np.save(output / (name + '.npy'), value, allow_pickle=False)
    json_write(output / 'rows.json', meta)
    files = [name + '.npy' for name in values] + ['rows.json']
    report = {'identity': identity, 'result_source': 'fresh_run_features_labels_with_verified_Zara_media_reuse',
        'rows': len(meta), 'records': summaries, 'geometry_dim': values['geometry'].shape[1],
        'baseline_names': BASELINES, 'arrays': {n: file_digest(output / n) for n in files},
        'bytes': sum((output / n).stat().st_size for n in files),
        'new_development_calibration_confirmation_access': False,
        'explicit_future_targets_in_inference': False, 'strict_sensor_as_of_claim': False,
        'offline_interpolated_annotations': True, 'coordinate_claim': 'dataset_local_unverified',
        'source_units_pooled_for_claim': False}
    json_write(manifest, report)
    return report
