"""Re-read raw geometry for every quantized match, without prediction errors."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_european_squares import digest, save
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv

BASE = ROOT/'outputs/publication_readiness_2026_09'
OUT = BASE/'european_squares_overlap_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_squares_overlap_v1'


def main():
    analysis = json.loads((OUT/'analysis.json').read_text())
    path = PRIVATE/'quantized_candidate_groups.json'
    if digest(path) != analysis['modes']['quantized']['candidate_receipt_sha256']:
        raise ValueError('Candidate identity differs')
    groups = json.loads(path.read_text())
    requests = defaultdict(set)
    for group in groups:
        for r in group:
            requests[r['recording']].add(r['start_frame'])
    source = json.loads((BASE/'european_squares_intake_v1/trajectory_manifest.json').read_text())['private_file']
    archive = ROOT/source['path']
    if digest(archive) != source['sha256']:
        raise ValueError('Source archive differs')
    hashes = {r['source_member']: r['rows_sha256'] for r in analysis['recordings']}
    blocks = {}
    with zipfile.ZipFile(archive) as z:
        for i, name in enumerate(sorted(requests)):
            with z.open(name) as f:
                rows, _ = read_raw_csv(f)
            if hashlib.sha256(rows.tobytes()).hexdigest() != hashes[name]:
                raise ValueError('Raw row replay differs')
            for start in sorted(requests[name]):
                raw, quantized, counts = [], [], []
                for frame in range(start, start+8):
                    r = rows[rows['frame'] == frame]
                    value = np.column_stack([r[k] for k in ('class_id', 'x_min', 'y_min', 'x_max', 'y_max')])
                    q = np.rint(value).astype('<i8')
                    order = np.lexsort(tuple(q[:, k] for k in range(4, -1, -1)))
                    raw.append(value[order])
                    quantized.append(q[order])
                    counts.append(len(r))
                blocks[(name, start)] = (raw, quantized, counts)
            print(json.dumps(dict(record=i+1, total=len(requests), blocks=len(requests[name]))), flush=True)
    results = []
    for i, group in enumerate(groups):
        values = [blocks[(r['recording'], r['start_frame'])] for r in group]
        reference = values[0][1]
        if any(any(not np.array_equal(a, b) for a, b in zip(reference, value[1])) for value in values[1:]):
            raise ValueError('Independent raw quantized equality fails')
        single = all(all(n == 1 for n in value[2]) for value in values)
        motion = []
        if single:
            for raw, _, _ in values:
                xy = np.array([[(v[0, 1]+v[0, 3])/2, (v[0, 2]+v[0, 4])/2] for v in raw])
                motion.append(float(np.max(np.linalg.norm(xy-xy[0], axis=1))))
        delta = max(float(np.max(np.abs(a[:, 1:]-b[:, 1:])))
                    for value in values[1:] for a, b in zip(values[0][0], value[0]))
        results.append(dict(candidate_index=i, recording_instances=len(values),
            single_detection_all_frames=single, min_agents=min(min(v[2]) for v in values),
            max_agents=max(max(v[2]) for v in values),
            largest_single_center_excursion_pixels=max(motion) if motion else None,
            largest_unquantized_box_coordinate_difference_pixels=delta,
            direct_quantized_equal=True))
    result = dict(result_source='fresh_run', script_sha256=digest(Path(__file__)),
        analysis_sha256=digest(OUT/'analysis.json'), candidates_sha256=digest(path),
        raw_recordings_reparsed=len(requests), raw_block_instances=len(blocks),
        candidate_groups=len(results), single_detection_groups=sum(r['single_detection_all_frames'] for r in results),
        max_single_center_excursion_pixels=max((r['largest_single_center_excursion_pixels'] for r in results
                                                if r['single_detection_all_frames']), default=None),
        groups=results, exact_duplicate_claim=False,
        approximate_duplicate_exclusion_claim=False, recordings_removed=0,
        consequence='same-locality recordings remain inseparable; matches are not automatically deduplicated',
        prediction_errors_opened=False, metric_claim=False, seconds_claim=False)
    save(OUT/'candidate_review.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'groups'}))


if __name__ == '__main__':
    main()
