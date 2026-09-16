"""Independent raw-CSV recount and bounded duplicate checks, without predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_unification.m3w_causal_recordings import RecordingWindows
from src.evaluation.m3w_recording_lineage import sha256


def count_windows(frames, history=8):
    """Count uniform-gap runs, independently of the conversion/index routine."""
    gaps = np.diff(frames).astype(np.int64)
    counts = {str(h): 0 for h in (10, 25, 50, 100)}
    observed = 0
    if not len(gaps):
        return counts, observed
    if (gaps <= 0).any():
        raise ValueError('Repeated or nonincreasing source frames')
    cuts = np.r_[0, np.flatnonzero(np.diff(gaps)) + 1, len(gaps)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        edges, step = int(right-left), int(gaps[left])
        observed += max(0, edges - (history-1+12) + 1)
        for horizon in counts:
            h = int(horizon)
            if h % step == 0:
                counts[horizon] += max(0, edges - (history-1+h//step) + 1)
    return counts, observed


def geometry_hash(track):
    # Ignore only the frame origin and agent ID; do not make a broad no-alias claim.
    values = np.asarray(track[:, [0, 2, 3]], dtype='<f8').copy()
    values[:, 0] -= values[0, 0]
    return hashlib.sha256(values.tobytes()).hexdigest()


def audit(source_root, cache_root, manifest_path, build_path):
    source_root, cache_root = Path(source_root), Path(cache_root)
    manifest, build = json.loads(Path(manifest_path).read_text()), json.loads(Path(build_path).read_text())
    if manifest['status'] != 'verified' or manifest['upstream_commit'] != build['upstream_commit']:
        raise ValueError('Author source/build identity mismatch')
    if build['run_identity_sha256'] != sha256(cache_root / 'run_identity.json'):
        raise ValueError('Conversion run identity changed')
    source_files = manifest['files']
    pinned_files = {entry['path']: entry for entry in source_files}
    for entry in source_files:
        path = source_root / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Pinned source changed')
    points_checked, lengths, types, strides = 0, [], Counter(), Counter()
    raw_counts, observation_count = Counter(), 0
    tracks_by_hash, clips, track_frames = defaultdict(list), [], {}
    for recording in build['recordings']:
        directory = cache_root / recording['id']
        receipt = json.loads((directory / 'completion.json').read_text())
        if receipt['metadata_sha256'] != sha256(directory / 'metadata.json'):
            raise ValueError('Converted metadata changed')
        reader = RecordingWindows(directory)
        metadata = reader.metadata
        mapping = np.load(directory / 'source_rows.npy', mmap_mode='r', allow_pickle=False)
        clip_counts, clip_steps = Counter(), 0
        for file_id, entry in enumerate(metadata['files']):
            if pinned_files.get(entry['path']) != entry:
                raise ValueError('Converted file identity differs from pinned author file')
            source = source_root / entry['path']
            with source.open() as stream:
                records = list(csv.DictReader(stream))
            points = []
            for row in records:
                vehicle = row['label'] == 'veh'
                points.append([int(row['frame']), 2*int(row['id'])+int(vehicle),
                    float(row['x_c' if vehicle else 'x']), float(row['y_c' if vehicle else 'y'])])
            points = np.array(points, dtype=np.float64)
            selected = np.flatnonzero(mapping[:, 0] == file_id)
            source_row_ids = mapping[selected, 1]-2
            np.testing.assert_array_equal(np.sort(source_row_ids), np.arange(len(points)))
            np.testing.assert_array_equal(reader.points[selected], points[source_row_ids])
            points_checked += len(points)
            for agent in np.unique(points[:, 1]):
                track = points[points[:, 1] == agent]
                kind = 'vehicle' if int(agent)%2 else 'pedestrian'
                types[kind] += 1
                lengths.append(len(track))
                strides.update(str(int(d)) for d in np.diff(track[:, 0]))
                counts, steps = count_windows(track[:, 0])
                clip_counts.update(counts)
                clip_steps += steps
                tracks_by_hash[geometry_hash(track)].append([recording['id'], int(agent)])
                track_frames[(recording['id'], int(agent))] = track[:, 0]
        if dict(clip_counts) != metadata['raw_exact_windows'] or clip_steps != metadata['observation_step_windows']:
            raise ValueError('Independent horizon recount differs from conversion')
        raw_counts.update(clip_counts)
        observation_count += clip_steps
        clips.append({'recording': recording['id'], 'physical_scene': recording['physical_scene'],
                      'points': len(reader.points), 'raw_exact_windows': dict(clip_counts), 'observation_step_windows': clip_steps})
    if points_checked != build['points'] or dict(types) != build['agent_types']:
        raise ValueError('Independent source point/track counts differ')
    references, reference_tracks, matches = [], 0, []
    canonical = json.loads((ROOT / 'configs/m3w_independent_experiment.draft.json').read_text())
    for name, record in canonical['records'].items():
        directory = ROOT / record['cache_path']
        if sha256(directory / 'metadata.json') != record['metadata_sha256']:
            raise ValueError('Canonical reference metadata changed')
        references.append(directory)
    references.extend(sorted((ROOT / 'data/stage_cvpr2027_causal/citr_diagnostic').glob('citr_*')))
    reference_identities = []
    for directory in references:
        reader = RecordingWindows(directory)
        reference_identities.append({'recording': reader.metadata['id'], 'metadata_sha256': sha256(directory / 'metadata.json')})
        for agent in np.unique(reader.points[:, 1]):
            track = reader.points[reader.points[:, 1] == agent]
            reference_tracks += 1
            digest = geometry_hash(track)
            if digest in tracks_by_hash:
                matches.append({'reference': [reader.metadata['id'], int(agent)], 'dut': tracks_by_hash[digest]})
    raw_file_hashes = [e['sha256'] for e in source_files if e['path'].startswith('data/trajectories/')]
    duplicates = [v for v in tracks_by_hash.values() if len(v)>1]
    annotation_flags = []
    for group in duplicates:
        for i, left in enumerate(group):
            for right in group[i+1:]:
                if left[0] == right[0] and np.array_equal(track_frames[tuple(left)], track_frames[tuple(right)]):
                    frames = track_frames[tuple(left)]
                    annotation_flags.append({'recording': left[0], 'agent_ids': [left[1], right[1]],
                        'source_ids': [left[1]//2, right[1]//2], 'points_each': len(frames),
                        'first_frame': int(frames[0]), 'last_frame': int(frames[-1]),
                        'issue': 'different_IDs_identical_positions_at_all_same_frames',
                        'action': 'quarantine_recording_pending_annotation_review_no_silent_merge'})
    quarantined = sorted({f['recording'] for f in annotation_flags})
    ratios = [float((source_root / e['path']).read_text()) for e in source_files if e['path'].startswith('data/ratios/')]
    if not ratios or not np.isfinite(ratios).all() or min(ratios) <= 0:
        raise ValueError('Invalid author scale metadata')
    return {'result_source': 'fresh_run_independent_csv_recount_no_prediction',
        'source_manifest_sha256': sha256(manifest_path), 'build_report_sha256': sha256(build_path),
        'code_sha256': sha256(Path(__file__)), 'source_files_hash_verified': len(source_files),
        'all_raw_rows_verified': points_checked, 'tracks': sum(types.values()), 'agent_types': dict(types),
        'track_length_quantiles': np.quantile(lengths, [0,.25,.5,.75,1]).tolist(),
        'native_frame_stride_counts': dict(strides), 'raw_exact_windows': dict(raw_counts),
        'observation_step_windows': observation_count, 'physical_scene_count': len({r['physical_scene'] for r in clips}),
        'site_recording_counts': dict(Counter(r['physical_scene'] for r in clips)), 'clips': clips,
        'byte_identical_raw_file_duplicates': len(raw_file_hashes)-len(set(raw_file_hashes)),
        'exact_full_track_duplicates_within_dut': duplicates,
        'annotation_flags': annotation_flags, 'quarantine_recordings': quarantined,
        'quarantine_is_not_a_train_test_assignment': True,
        'quarantined_recording_points': sum(r['points'] for r in clips if r['recording'] in quarantined),
        'raw50_windows_outside_flagged_recordings_not_approved': sum(r['raw_exact_windows']['50'] for r in clips if r['recording'] not in quarantined),
        'reference_recordings': reference_identities, 'reference_tracks_compared': reference_tracks,
        'exact_full_track_matches_with_canonical_and_citr': matches,
        'duplicate_check_scope': 'full exact float64 positions plus relative frames; not partial/rounded/transformed or image-level matching',
        'novel_recording_or_independent_confirmation_proven': False,
        'author_scale_files': len(ratios), 'author_scale_value_range': [min(ratios), max(ratios)],
        'author_scale_interpretation': 'raw coordinates divided by value in author filtering code; not independently verified',
        'author_readme_fps': 23.98, 'author_filter_fps': 23.976,
        'metric_or_seconds_claim_allowed': False, 'source_use_approval': False,
        'training_run': False, 'prediction_accuracy_evaluated': False, 'official_protocol_changed': False,
        'stage5c_executed': False, 'smc_enabled': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'external_data/DUT_author_raw')
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'data/stage_cvpr2027_causal/dut_diagnostic')
    report_dir = ROOT / 'outputs/publication_readiness_2026_09/dut_causal_intake'
    parser.add_argument('--manifest', type=Path, default=report_dir / 'source_manifest.json')
    parser.add_argument('--build-report', type=Path, default=report_dir / 'build_report.json')
    parser.add_argument('--report', type=Path, default=report_dir / 'independent_recount.json')
    args = parser.parse_args()
    result = audit(args.source_root, args.cache_root, args.manifest, args.build_report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('clips', 'reference_recordings')}, indent=2))


if __name__ == '__main__':
    main()
