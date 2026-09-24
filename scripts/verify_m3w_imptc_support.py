"""Separate set-based support arithmetic and source-to-cache spot-free checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_imptc_sample import RAW, PUBLIC, FILE, digest
from src.evaluation.m3w_imptc_intake import safe_member, strict_json


def counts(frames, length, stride):
    available = set(map(int, frames))
    current = [f for f in sorted(available) if all(f-i*stride in available for i in range(length))]
    future = [sum(f+i*stride in available for i in range(1, 13)) for f in current]
    return dict(past_eligible=len(current), complete_future12=sum(n == 12 for n in future),
        partial_future12=sum(0 < n < 12 for n in future), no_future12=sum(n == 0 for n in future))


def main():
    analysis = json.loads((PUBLIC/'analysis.json').read_text())
    if digest(RAW/FILE) != analysis['archive_sha256']:
        raise ValueError('Raw archive mismatch')
    for entry in analysis['cache_files']:
        if digest(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Private cache changed')
    payloads = {}
    with tarfile.open(RAW/FILE, 'r|gz') as archive:
        for member in archive:
            name = safe_member(member)
            if name.endswith('/track.json') or name.endswith('/master_timestamp_sync.json'):
                payloads[name] = strict_json(archive.extractfile(member).read())
    reports, intervals, checked_rows = [], [], 0
    for source in analysis['sequences']:
        name = source['sequence']
        master = payloads[name+'/context/master_timestamp_sync.json']
        times = sorted(map(int, master))
        intervals.append(dict(sequence=name, first_timestamp=times[0], last_timestamp=times[-1]))
        path = ROOT/'data/stage_cvpr2027_experiments/imptc_intake_v1'/f'{name}_rows.npy'
        rows = np.load(path, allow_pickle=False)
        paths = sorted(n for n in payloads if n.startswith(name+'/') and n.endswith('/track.json'))
        by_class, aggregate = {}, {k: Counter() for k in source['support']}
        for agent, source_path in enumerate(paths):
            observations = payloads[source_path]['track_data'].values()
            cache = rows[rows['agent_id'] == agent]
            expected = sorted((int(o['ts']), *o['coordinates']) for o in observations)
            if len(expected) != len(cache):
                raise ValueError('Direct raw/cache count mismatch')
            for stored, (timestamp, x, y, z) in zip(cache, expected):
                if (int(stored['timestamp']) != timestamp
                    or int(stored['frame_id']) != master[str(timestamp)]['id']
                    or tuple(stored[a] for a in ('x', 'y', 'z')) != (x, y, z)):
                    raise ValueError('Direct row-to-source equality failed')
                checked_rows += 1
            label = payloads[source_path]['overview']['class_name']
            if label not in by_class:
                by_class[label] = dict(tracks=0, rows=0, support={k: Counter() for k in source['support']})
            by_class[label]['tracks'] += 1
            by_class[label]['rows'] += len(cache)
            for length in (8, 16, 32, 64):
                for stride in (1, 10):
                    key = f'K{length}_stride{stride}'
                    c = counts(cache['frame_id'], length, stride)
                    aggregate[key].update(c)
                    by_class[label]['support'][key].update(c)
        for key, sums in aggregate.items():
            for field, value in sums.items():
                if value != source['support'][key][field]:
                    raise ValueError('Independent set arithmetic differs from vectorized audit')
        reports.append(dict(sequence=name, by_class=by_class))
    overlaps = []
    for i, a in enumerate(intervals):
        for b in intervals[i+1:]:
            if max(a['first_timestamp'], b['first_timestamp']) <= min(a['last_timestamp'], b['last_timestamp']):
                overlaps.append([a['sequence'], b['sequence']])
    result = dict(result_source='fresh_run', analysis_sha256=digest(PUBLIC/'analysis.json'),
        every_source_row_matches_cache=True, rows_checked=checked_rows,
        separate_set_arithmetic_matches=True, same_agent_verification_not_external_review=True,
        published_whole_track_classes_for_audit_only=True, per_sequence_class_support=reports,
        source_intervals=intervals, overlapping_source_intervals=overlaps,
        independent_physical_sites=1, independent_calibration_admitted=False,
        predictive_readout=False, script_sha256=digest(Path(__file__)))
    with (PUBLIC/'separate_checks.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(rows_checked=checked_rows, overlapping_intervals=overlaps,
        independent_arithmetic='pass', predictive_readout=False)))


if __name__ == '__main__':
    main()
