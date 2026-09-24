"""Parse every sample annotation, build private rows, audit support without errors."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_imptc_sample import RAW, PUBLIC, FILE, digest
from src.evaluation.m3w_imptc_intake import (parse_master, parse_track, past_scene,
    safe_member, support)

PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/imptc_intake_v1'


def counter(values):
    keys, counts = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(keys, counts)}


def audit_prefix(rows):
    queries = np.unique(rows['frame_id'])
    queries = queries[np.linspace(0, len(queries)-1, min(17, len(queries)), dtype=int)]
    checked = 0
    for q in queries:
        expected = past_scene(rows, int(q))
        changed = rows.copy()
        for axis in ('x', 'y', 'z'):
            changed[axis][changed['frame_id'] > q] = 1e10
        for variant in (changed, rows[rows['frame_id'] <= q]):
            got = past_scene(variant, int(q))
            if expected.keys() != got.keys():
                raise ValueError('Future changes current-agent population')
            for agent in expected:
                for key in expected[agent]:
                    np.testing.assert_equal(expected[agent][key], got[agent][key])
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if any(p.is_symlink() for p in (PRIVATE, *PRIVATE.parents)):
        raise SystemExit('Symlinked cache destination refused')
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE / 'rows.npy')], cwd=ROOT, check=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PUBLIC / 'source_manifest.json').read_text())
    for entry in manifest['private_files']:
        path = ROOT / entry['path']
        if digest(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise SystemExit('Acquired source differs from frozen receipt')
    started = time.monotonic()
    inventory, payloads, names, total = [], {}, set(), 0
    with tarfile.open(RAW / FILE, 'r|gz') as archive:
        for member in archive:
            name = safe_member(member)
            if name in names or len(names) > 5000:
                raise ValueError('Duplicate or excessive archive member count')
            names.add(name); total += member.size
            if total > 1_000_000_000:
                raise ValueError('Archive expanded budget exceeded')
            inventory.append(dict(path=name, bytes=member.size, is_file=member.isreg()))
            if member.isreg() and (name.endswith('/track.json') or name.endswith('/master_timestamp_sync.json')):
                if member.size > 10_000_000:
                    raise ValueError('Annotation member exceeds bound')
                payloads[name] = archive.extractfile(member).read()
    if sum(map(len, payloads.values())) > 200_000_000:
        raise ValueError('Annotation payload budget exceeded')
    sequences = sorted(n.split('/')[0] for n in payloads if n.endswith('/master_timestamp_sync.json'))
    if len(sequences) != 4:
        raise ValueError('Official sample sequence count differs from documentation')
    reports, cache = [], []
    for sequence in sequences:
        master_name = sequence + '/context/master_timestamp_sync.json'
        master = parse_master(payloads[master_name])
        tracks = sorted(n for n in payloads if n.startswith(sequence + '/') and n.endswith('/track.json'))
        arrays, audits, hashes = [], [], []
        for agent_id, name in enumerate(tracks):
            data, audit = parse_track(payloads[name], master, agent_id)
            if audit['overview_length'] != len(data):
                raise ValueError('Overview row count differs from parsed observations')
            arrays.append(data)
            audits.append(dict(source_path=name, **audit))
            hashes.append(dict(path=name, sha256=hashlib.sha256(payloads[name]).hexdigest()))
        rows = np.concatenate(arrays)
        rows.sort(order=['agent_id', 'frame_id'])
        path = PRIVATE / (sequence + '_rows.npy')
        if path.is_symlink():
            raise ValueError('Symlinked cache file refused')
        if args.verify:
            np.testing.assert_array_equal(rows, np.load(path, allow_pickle=False))
        else:
            with path.open('xb') as stream:
                np.save(stream, rows, allow_pickle=False)
        lengths = [a['rows'] for a in audits]
        summary = dict(sequence=sequence, physical_site='imptc_aschaffenburg_intersection',
            rows=len(rows), tracks=len(audits),
            class_track_counts=dict(Counter(str(a['class_name']) for a in audits)),
            class_row_counts=dict(Counter({c: sum(a['rows'] for a in audits if str(a['class_name']) == c)
                for c in sorted({str(a['class_name']) for a in audits})})),
            track_length_quantiles=dict(zip(['min', 'median', 'p95', 'max'], map(float, np.percentile(lengths, [0, 50, 95, 100])))),
            master_rows=len(master), master_index_gaps=int((np.diff(master[:, 1]) != 1).sum()),
            master_timestamp_delta_counts=counter(np.diff(master[:, 0])),
            tracks_with_gaps=sum(a['gap_count'] > 0 for a in audits),
            total_track_gaps=sum(a['gap_count'] for a in audits),
            track_status_counts=dict(sum((Counter(a['status_counts']) for a in audits), Counter())),
            support=support(rows), prefix_invariance_checks=audit_prefix(rows),
            master_sha256=hashlib.sha256(payloads[master_name]).hexdigest())
        row_entry = dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path))
        cache.append(row_entry)
        reports.append(summary)
        # Full track-to-source alignment stays private; public reports are aggregates.
        alignment = PRIVATE / (sequence + '_alignment.json')
        records = [dict(agent_id=i, **entry) for i, entry in enumerate(hashes)]
        if args.verify:
            if records != json.loads(alignment.read_text()):
                raise ValueError('Source-to-row alignment changed')
        else:
            with alignment.open('x') as stream:
                stream.write(json.dumps(records, indent=2) + '\n')
        cache.append(dict(path=str(alignment.relative_to(ROOT)), bytes=alignment.stat().st_size, sha256=digest(alignment)))
        print(json.dumps(dict(pid=os.getpid(), sequence=sequence, rows=len(rows), state='audited',
                             seconds=time.monotonic()-started)), flush=True)
    totals = {}
    for key in reports[0]['support']:
        totals[key] = {field: sum(r['support'][key][field] for r in reports)
            for field in ('past_eligible', 'complete_future12', 'partial_future12', 'no_future12', 'query_frames')}
    analysis = dict(result_source='fresh_run', source_manifest_sha256=digest(PUBLIC / 'source_manifest.json'),
        archive_sha256=manifest['archive_sha256'], sequences=reports,
        total_rows=sum(r['rows'] for r in reports), total_tracks=sum(r['tracks'] for r in reports),
        physical_sites=1, independent_calibration_sites_admitted=0, support=totals,
        cache_files=cache, archive_members=len(inventory), archive_uncompressed_bytes=total,
        inventory_sha256=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        source_unit='publisher_meter_not_independently_recalibrated',
        time_unit='publisher_UTC_timestamp_local_delta_audited_not_sensor_clock_verified',
        velocity_unit='source_coordinate_per_master_index_backward_difference_only',
        role='quarantined_unassigned_source_audit', online_causal_source_verified=False,
        whole_track_class_used_for_inference=False, future_labels_used_to_filter_population=False,
        candidate_goals_built=False, baseline_or_model_error_readout=False, fitting=False,
        dronecrowd_confirmation_opened=False, stage5c_executed=False, smc_enabled=False,
        code_sha256={str(p.relative_to(ROOT)): digest(p) for p in
            (Path(__file__), ROOT/'src/evaluation/m3w_imptc_intake.py', ROOT/'tests/test_m3w_imptc_intake.py')})
    path = PUBLIC / 'analysis.json'
    if args.verify:
        if analysis != json.loads(path.read_text()):
            raise ValueError('Fresh source reparse differs from saved audit')
        receipt = dict(result_source='cached_verified', rows_reparsed=analysis['total_rows'],
            row_arrays_and_alignment_exact=True, aggregate_replay_exact=True,
            analysis_sha256=digest(path), seconds=time.monotonic()-started,
            verified_at_utc=datetime.now(timezone.utc).isoformat())
        with (PUBLIC/'verification.json').open('x') as stream:
            stream.write(json.dumps(receipt, indent=2)+'\n')
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(analysis, indent=2, allow_nan=False)+'\n')
        with (PUBLIC/'execution.json').open('x') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), seconds=time.monotonic()-started,
                completed_at_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
    print(json.dumps(dict(rows=analysis['total_rows'], tracks=analysis['total_tracks'], support=totals), indent=2))


if __name__ == '__main__':
    main()
