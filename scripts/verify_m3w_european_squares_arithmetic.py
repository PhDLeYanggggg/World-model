"""Separate set-based support checks on fixed samples of every raw recording."""
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_m3w_european_squares_v2 import PRIVATE, PUBLIC, heartbeat
from scripts.fetch_m3w_european_squares import PUBLIC as SOURCE_PUBLIC, digest, save
from src.evaluation.m3w_european_squares_intake import frame_support, past_scene
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv


def main():
    start = time.monotonic()
    analysis = json.loads((PUBLIC / 'analysis.json').read_text())
    source = json.loads((SOURCE_PUBLIC / 'trajectory_manifest.json').read_text())
    archive = ROOT / source['private_file']['path']
    if digest(archive) != analysis['identity']['archive_sha256']:
        raise ValueError('Source archive changed')
    receipt = dict(result_source='fresh_run', analysis_sha256=digest(PUBLIC / 'analysis.json'),
                   verifier_sha256=digest(Path(__file__)), records=[], forecast_errors_opened=False)
    with zipfile.ZipFile(archive) as z:
        for i, record in enumerate(analysis['recordings']):
            heartbeat('separate_arithmetic', index=i+1, total=len(analysis['recordings']))
            with z.open(record['source_member']) as f:
                rows, _ = read_raw_csv(f)
            if hashlib.sha256(rows.tobytes()).hexdigest() != record['rows_sha256']:
                raise ValueError('Every-row replay mismatch')
            agents = np.unique(rows['agent'])
            chosen = agents[np.unique(np.linspace(0, len(agents)-1, min(3, len(agents)), dtype=int))]
            comparisons, full_counts, velocity_checks = 0, 0, 0
            for agent in chosen:
                track = rows[rows['agent'] == agent]
                frames = track['frame']
                frame_set = set(map(int, frames))
                selected = frames[np.unique(np.linspace(0, len(frames)-1, min(64, len(frames)), dtype=int))]
                for length in (8, 16, 32, 64):
                    for stride in (1, 12):
                        counts, queries = frame_support(frames, length, stride)
                        query_set = set(map(int, queries))
                        for q in selected:
                            expected = all(int(q)-j*stride in frame_set for j in range(length))
                            if expected != (int(q) in query_set):
                                raise ValueError('Independent past support mismatch')
                            comparisons += 1
                        # Independent future-set arithmetic for every eligible query
                        # of the fixed three tracks, not selected by label difficulty.
                        complete = partial = absent = 0
                        for q in queries:
                            n = sum(int(q)+j*stride in frame_set for j in range(1, 13))
                            complete += n == 12
                            partial += 0 < n < 12
                            absent += n == 0
                        if (complete, partial, absent) != tuple(counts[k] for k in
                                ('complete_future12', 'partial_future12', 'no_future12')):
                            raise ValueError('Independent future support mismatch')
                        full_counts += 1
                q = int(frames[len(frames)//2])
                scene = past_scene(rows, q)
                if set(scene) != set(map(int, rows['agent'][rows['frame'] == q])):
                    raise ValueError('Current agent census mismatch')
                result = scene[int(agent)]
                lookup = {int(row['frame']): row for row in track if row['frame'] <= q}
                for j, frame in enumerate(result['frames']):
                    if frame not in lookup:
                        if result['valid'][j]:
                            raise ValueError('Missing history marked valid')
                        continue
                    r = lookup[int(frame)]
                    center = np.array([(r['x_min']+r['x_max'])/2, (r['y_min']+r['y_max'])/2])
                    np.testing.assert_array_equal(center, result['box_center'][j])
                    previous = int(frame)-12
                    if j and previous in lookup:
                        p = lookup[previous]
                        old = np.array([(p['x_min']+p['x_max'])/2, (p['y_min']+p['y_max'])/2])
                        np.testing.assert_allclose((center-old)/12, result['velocity_causal_fd'][j], rtol=0, atol=0)
                        velocity_checks += 1
            receipt['records'].append(dict(source_member=record['source_member'], rows=len(rows),
                fixed_tracks_checked=len(chosen), past_membership_checks=comparisons,
                complete_future_count_checks=full_counts, direct_velocity_checks=velocity_checks))
    receipt['total_rows_replayed'] = sum(r['rows'] for r in receipt['records'])
    receipt['all_checks_passed'] = True
    save(PUBLIC / 'arithmetic_verification.json', receipt)
    heartbeat('separate_arithmetic_complete', seconds=time.monotonic()-start)


if __name__ == '__main__':
    main()
