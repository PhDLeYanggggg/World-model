"""Bounded ZIP-native structural audit of European Squares raw tracker outputs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_european_squares import RAW, PUBLIC, digest, save
from src.evaluation.m3w_european_squares_intake import read_raw_csv, summarize, safe_zip_member, past_scene

PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/european_squares_intake_v1'


def heartbeat(state, **fields):
    event = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), state=state, **fields)
    tmp = PRIVATE / 'heartbeat.tmp'
    tmp.write_text(json.dumps(event) + '\n')
    tmp.replace(PRIVATE / 'heartbeat.json')
    with (PRIVATE / 'events.jsonl').open('a') as out:
        out.write(json.dumps(event) + '\n')
    print(json.dumps(event), flush=True)


def prefix_checks(rows):
    frames = np.unique(rows['frame'])
    queries = frames[np.linspace(0, len(frames)-1, min(3, len(frames)), dtype=int)]
    count = 0
    for q in queries:
        expected = past_scene(rows, int(q))
        changed = rows.copy()
        for key in ('x_min', 'y_min', 'x_max', 'y_max'):
            changed[key][changed['frame'] > q] = 1e10
        for variant in (changed, rows[rows['frame'] <= q]):
            actual = past_scene(variant, int(q))
            if expected.keys() != actual.keys():
                raise ValueError('Future changes query agent population')
            for agent in actual:
                for field in actual[agent]:
                    np.testing.assert_array_equal(expected[agent][field], actual[agent][field])
            count += 1
    return count


def metadata_sites():
    entries = []
    for kind in ('comparative', 'season'):
        frame = pd.read_csv(RAW / f'stats_{kind}.csv', usecols=['No.', 'City', 'Country', 'Date', 'timeslot', 'lat', 'long'])
        for key, group in frame.groupby('No.'):
            variants = group[['City', 'Country']].drop_duplicates().to_dict('records')
            lat = pd.to_numeric(group['lat'], errors='coerce')
            lon = pd.to_numeric(group['long'], errors='coerce')
            entries.append(dict(dataset=kind, square_id=int(key),
                                city=str(group['City'].iloc[0]) if len(variants) == 1 else None,
                                country=str(group['Country'].iloc[0]) if len(variants) == 1 else None,
                                city_country_variants=variants, site_identity_conflict=len(variants) != 1,
                                recordings=len(group),
                                latitude=float(lat.median()) if len(variants) == 1 and lat.notna().any() else None,
                                longitude=float(lon.median()) if len(variants) == 1 and lon.notna().any() else None,
                                unparsed_latitude_values=sorted(set(map(str, group.loc[lat.isna(), 'lat']))),
                                unparsed_longitude_values=sorted(set(map(str, group.loc[lon.isna(), 'long'])))))
    return entries


def run(args):
    start = time.monotonic()
    meta = json.loads((PUBLIC / 'metadata_manifest.json').read_text())
    for entry in meta['private_files']:
        path = ROOT / entry['path']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Metadata differs from acquisition manifest')
    sites = metadata_sites()
    if args.demo:
        path = RAW / 'demo_raw.csv'
        blob = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode()+path.read_bytes()).hexdigest()
        if blob != 'a5595235d3497e8c11b678740b5c6beab5467ccd':
            raise ValueError('Pinned author raw demo blob differs')
        with path.open('rb') as stream:
            rows, names = read_raw_csv(stream)
        result, _ = summarize(rows, names)
        result.update(result_source='fresh_run', demo_only=True, source_sha256=digest(path),
                      prefix_checks=prefix_checks(rows), code_sha256=digest(ROOT / 'src/evaluation/m3w_european_squares_intake.py'))
        save(PUBLIC / 'demo_probe.json', result)
        heartbeat('demo_complete', rows=len(rows), seconds=time.monotonic()-start)
        return
    source = json.loads((PUBLIC / 'trajectory_manifest.json').read_text())
    entry = source['private_file']
    archive = ROOT / entry['path']
    heartbeat('verifying_archive', bytes=entry['bytes'])
    if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
        raise ValueError('Trajectory archive identity mismatch')
    bound = dict(archive_sha256=entry['sha256'], metadata_sha256=digest(PUBLIC / 'metadata_manifest.json'),
                 module_sha256=digest(ROOT / 'src/evaluation/m3w_european_squares_intake.py'),
                 runner_sha256=digest(Path(__file__)), test_sha256=digest(ROOT / 'tests/test_m3w_european_squares_intake.py'),
                 lengths=[8, 16, 32, 64], diagnostic_strides=[1, 12], future_points=12)
    save(PRIVATE / 'identity.json', bound)
    reports, inventory = [], []
    hashes = defaultdict(list)
    known_sites = {r['square_id'] for r in sites}
    with zipfile.ZipFile(archive) as z:
        names = set()
        for info in z.infolist():
            name = safe_zip_member(info)
            if name in names:
                raise ValueError('Repeated ZIP member')
            names.add(name)
            inventory.append(dict(path=name, bytes=info.file_size, compressed_bytes=info.compress_size, crc=info.CRC))
        if len(inventory) > 10000 or sum(x['bytes'] for x in inventory) > 150_000_000_000:
            raise ValueError('Archive inventory exceeds bounded scope')
        raw_names = sorted(n for n in names if 'trajectories_raw' in n.lower()
                           and n.lower().endswith('.csv') and not n.startswith('__MACOSX/'))
        if not raw_names:
            raise ValueError('No raw tracker CSV members; do not substitute processed data')
        save(PRIVATE / 'inventory.json', inventory)
        selected = raw_names[:1] if args.pilot else raw_names
        heartbeat('inventory_complete', members=len(inventory), raw_recordings=len(raw_names), selected=len(selected))
        for i, name in enumerate(selected):
            key = hashlib.sha256(name.encode()).hexdigest()
            path = PRIVATE / 'records' / (key + '.json')
            track_path = PRIVATE / 'records' / (key + '_track_hashes.json')
            if path.exists() and args.resume and not args.verify:
                result = json.loads(path.read_text())
                if result['identity'] != bound or result['source_member'] != name:
                    raise ValueError('Resume identity mismatch')
                tracks = json.loads(track_path.read_text())
            else:
                before = time.monotonic()
                match = re.match(r'(\d+)[_-]', Path(name).name)
                square = int(match.group(1)) if match else None
                if square not in known_sites:
                    raise ValueError('Unmapped physical-square ID; metadata repair needed: ' + name)
                heartbeat('reading', index=i+1, total=len(selected), member=name, bytes=z.getinfo(name).file_size)
                with z.open(name) as stream:
                    rows, classes = read_raw_csv(stream)
                result, tracks = summarize(rows, classes)
                result.update(identity=bound, source_member=name, square_id=square,
                              prefix_checks=prefix_checks(rows))
                del rows
                if args.verify:
                    if result != json.loads(path.read_text()) or tracks != json.loads(track_path.read_text()):
                        raise ValueError('Exact recording replay differs')
                else:
                    save(path, result)
                    save(track_path, tracks)
                heartbeat('record_complete', index=i+1, rows=result['rows'], tracks=result['tracks'],
                          seconds=time.monotonic()-before, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            reports.append({k: v for k, v in result.items() if k != 'identity'})
            for h in tracks:
                hashes[h].append(name)
        if args.pilot:
            save(PUBLIC / 'pilot.json', dict(result_source='fresh_run', full_audit=False,
                 identity=bound, total_raw_recordings=len(raw_names), recording=reports[0]))
            heartbeat('pilot_complete', seconds=time.monotonic()-start)
            return
    duplicate_groups = [v for v in hashes.values() if len(set(v)) > 1]
    summary = dict(result_source='fresh_run', identity=bound, archive_members=len(inventory),
                   raw_recordings=len(reports), metadata_sites=sites,
                   source_square_ids=sorted({r['square_id'] for r in reports}),
                   rows=sum(r['rows'] for r in reports), tracks=sum(r['tracks'] for r in reports),
                   prefix_checks=sum(r['prefix_checks'] for r in reports),
                   cross_recording_exact_relative_track_duplicate_groups=len(duplicate_groups),
                   recordings=reports, role='quarantined_unassigned_source_audit',
                   forecast_errors_opened=False, fitting=False, future_label_filter=False,
                   source_online_causality_verified=False, metric_claim=False, seconds_claim=False,
                   independent_calibration_sites_admitted=0, stage5c_executed=False, smc_enabled=False)
    save(PRIVATE / 'duplicate_track_members.json', duplicate_groups)
    save(PUBLIC / 'analysis.json', summary)
    if args.verify:
        save(PUBLIC / 'verification.json', dict(exact=True, analysis_sha256=digest(PUBLIC / 'analysis.json')))
    heartbeat('complete', seconds=time.monotonic()-start, rows=summary['rows'], recordings=len(reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for directory in (PRIVATE, PRIVATE / 'records', PUBLIC):
        if any(p.is_symlink() for p in (directory, *directory.parents)):
            raise ValueError('Symlinked destination')
        directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'check-ignore', '--quiet', str(PRIVATE / 'identity.json')], cwd=ROOT, check=True)
    with (PRIVATE / 'audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
