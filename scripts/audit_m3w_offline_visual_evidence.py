"""Replay actual fit inputs and quantify support; never an online-causality certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import verify_registration, json_write
from src.world_model.m3w_offline_visual_forecast import geometry_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg, parent = verify_registration(ROOT, args.registration)
    data, reports = ROOT / reg['output'] / 'inputs', ROOT / reg['reports']
    receipt = json.loads((data / 'data_manifest.json').read_text())
    for name, digest in receipt['arrays'].items():
        if file_digest(data / name) != digest:
            raise ValueError('Input arrays changed')
    a = {k: np.load(data / (k + '.npy'), mmap_mode='r')
         for k in ('geometry', 'image_rows', 'coverage', 'folds')}
    rows = json.loads((data / 'rows.json').read_text())
    counter, source_offset, support = 0, 1, {}
    for rid in reg['recordings']:
        reader, indices = parent.open_recording(rid, purpose='fit')
        agents, intervals, masks, history_stationary = set(), {}, [], 0
        for item in indices:
            row = reader.index[item]
            begin, current = int(row['history_start']), int(row['current_row'])
            history = reader.points[begin:current + 1]
            inputs = reader.get_inputs(int(item))
            if not np.array_equal(a['geometry'][counter], geometry_features(inputs)):
                raise ValueError('Full inference vector replay mismatch')
            expected = (np.zeros(8, dtype=int) if rid == 'ucy_zara03'
                        else np.arange(begin, current + 1) + source_offset)
            if not np.array_equal(a['image_rows'][counter], expected):
                raise ValueError('Image/past row alignment changed')
            agent, frame = int(history[-1, 1]), int(history[-1, 0])
            scene = parent.protocol['records'][rid]['physical_scene']
            if rows[counter] != dict(recording=rid, scene=scene, frame=frame, agent=agent,
                                     fold=reg['scene_folds'][scene]):
                raise ValueError('Identity/fold mismatch')
            agents.add(agent)
            intervals.setdefault(agent, []).append((int(history[0, 0]), frame + int(row['horizon_raw'])))
            masks.append(a['coverage'][expected].sum((1, 2)) / (96 * 96))
            history_stationary += int(np.all(history[:, 2:] == history[-1, 2:]))
            counter += 1
        nonoverlap = 0
        for spans in intervals.values():
            last = -float('inf')
            for start, end in sorted(spans, key=lambda x: x[1]):
                if start > last:
                    nonoverlap += 1
                    last = end
        masks = np.stack(masks)
        support[rid] = {'windows': len(indices), 'unique_source_track_ids_in_windows': len(agents),
            'nonoverlapping_20_step_intervals_within_agent': nonoverlap,
            'exact_stationary_past_windows': history_stationary,
            'full_eight_image_windows': int((masks == 1).all(1).sum()),
            'windows_with_partial_image': int(((masks > 0) & (masks < 1)).any(1).sum()),
            'windows_with_missing_frame_image': int((masks == 0).any(1).sum()),
            'mean_history_pixel_coverage': float(masks.mean()),
            'physical_scene': scene}
        if rid != 'ucy_zara03':
            source_offset += len(reader.points)
    if counter != reg['expected_rows']:
        raise ValueError('Audit did not visit all fit windows')
    result = {'result_source': 'cached_verified_full_input_replay_and_fresh_support_counts',
        'registration_sha256': file_digest(args.registration), 'audit_code_sha256': file_digest(Path(__file__)),
        'all_geometry_vectors_exact': True, 'all_image_history_row_joins_exact': True,
        'all_rows_visited': counter, 'support': support,
        'separate_physical_scene_units': 3, 'independent_agents_or_windows_claim': False,
        'nonoverlap_does_not_imply_independence': True, 'observed_annotation_interpolation_disclosed': True,
        'strict_sensor_as_of_no_leakage_certified': False, 'explicit_future_labels_in_features': False,
        'development_calibration_confirmation_opened': False}
    json_write(reports / 'input_replay_support.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
