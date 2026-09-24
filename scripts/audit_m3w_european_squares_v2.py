"""Versioned whole-archive raw audit; preserve v1's stricter-schema failure."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_european_squares import PUBLIC as SOURCE_PUBLIC, digest, save
from scripts.audit_m3w_european_squares import metadata_sites, prefix_checks
from src.evaluation.m3w_european_squares_intake import summarize, safe_zip_member
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv

PRIVATE = ROOT / 'data/stage_cvpr2027_experiments/european_squares_intake_v2'
PUBLIC = ROOT / 'outputs/publication_readiness_2026_09/european_squares_intake_v2'


def heartbeat(state, **fields):
    event = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), state=state, **fields)
    tmp = PRIVATE / 'heartbeat.tmp'
    tmp.write_text(json.dumps(event)+'\n')
    tmp.replace(PRIVATE / 'heartbeat.json')
    with (PRIVATE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event), flush=True)


def run(args):
    start = time.monotonic()
    meta = json.loads((SOURCE_PUBLIC / 'metadata_manifest.json').read_text())
    for e in meta['private_files']:
        p = ROOT / e['path']
        if p.stat().st_size != e['bytes'] or digest(p) != e['sha256']:
            raise ValueError('Metadata identity differs')
    entry = json.loads((SOURCE_PUBLIC / 'trajectory_manifest.json').read_text())['private_file']
    archive = ROOT / entry['path']
    heartbeat('verifying_archive', bytes=entry['bytes'])
    if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
        raise ValueError('Archive identity differs')
    files = ['src/evaluation/m3w_european_squares_intake.py',
             'src/evaluation/m3w_european_squares_raw_v2.py',
             'scripts/audit_m3w_european_squares.py', 'scripts/audit_m3w_european_squares_v2.py',
             'scripts/fetch_m3w_european_squares.py', 'tests/test_m3w_european_squares_intake.py',
             'tests/test_m3w_european_squares_raw_v2.py']
    identity = dict(archive_sha256=entry['sha256'], metadata_sha256=digest(SOURCE_PUBLIC/'metadata_manifest.json'),
                    files={p:digest(ROOT/p) for p in files}, lengths=[8,16,32,64],
                    diagnostic_strides=[1,12], future_points=12,
                    optional_source_field='class_name_only_no_inference_of_missing_text')
    save(PRIVATE / 'identity.json', identity)
    sites = metadata_sites()
    known = {r['square_id'] for r in sites}
    reports, hashes = [], defaultdict(list)
    with zipfile.ZipFile(archive) as z:
        inventory, names = [], set()
        for info in z.infolist():
            name = safe_zip_member(info)
            if name in names:
                raise ValueError('Duplicate archive member')
            names.add(name)
            inventory.append(dict(path=name, bytes=info.file_size, compressed_bytes=info.compress_size, crc=info.CRC))
        if len(inventory)>10000 or sum(r['bytes'] for r in inventory)>150_000_000_000:
            raise ValueError('Archive exceeds bounded scope')
        save(PRIVATE / 'inventory.json', inventory)
        raw = sorted(n for n in names if 'trajectories_raw' in n.lower() and n.endswith('.csv') and not n.startswith('__MACOSX/'))
        if not raw:
            raise ValueError('Raw exports missing')
        selected = raw[:1] if args.pilot else raw
        heartbeat('inventory_complete', raw_recordings=len(raw), selected=len(selected))
        for i,name in enumerate(selected):
            key = hashlib.sha256(name.encode()).hexdigest()
            path = PRIVATE/'records'/(key+'.json')
            track_path = PRIVATE/'records'/(key+'_track_hashes.json')
            if args.resume and path.exists() and not args.verify:
                result = json.loads(path.read_text())
                tracks = json.loads(track_path.read_text())
                if result['identity'] != identity or result['source_member'] != name:
                    raise ValueError('Cached recording identity differs')
            else:
                before = time.monotonic()
                match = re.match(r'(\d+)[_-]', Path(name).name)
                square = int(match.group(1)) if match else None
                if square not in known:
                    raise ValueError('Unknown physical square ID: '+name)
                heartbeat('reading', index=i+1, total=len(selected), member=name)
                with z.open(name) as f:
                    rows, classes = read_raw_csv(f)
                result, tracks = summarize(rows, classes)
                result.update(identity=identity, source_member=name, square_id=square, prefix_checks=prefix_checks(rows))
                del rows
                if args.verify:
                    if result != json.loads(path.read_text()) or tracks != json.loads(track_path.read_text()):
                        raise ValueError('Exact replay differs')
                else:
                    save(path,result)
                    save(track_path,tracks)
                heartbeat('record_complete', index=i+1, rows=result['rows'], tracks=result['tracks'],
                          seconds=time.monotonic()-before, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            reports.append({k:v for k,v in result.items() if k!='identity'})
            for h in tracks:
                hashes[h].append(name)
        if args.pilot:
            save(PUBLIC/'pilot.json', dict(result_source='fresh_run', full_audit=False, identity=identity,
                                          total_raw_recordings=len(raw), recording=reports[0]))
            heartbeat('pilot_complete', seconds=time.monotonic()-start)
            return
    aliases = [v for v in hashes.values() if len(set(v))>1]
    summary = dict(result_source='fresh_run',identity=identity,archive_members=len(inventory),
                   raw_recordings=len(reports), metadata_sites=sites,
                   source_square_ids=sorted({r['square_id'] for r in reports}),
                   rows=sum(r['rows'] for r in reports), tracks=sum(r['tracks'] for r in reports),
                   prefix_checks=sum(r['prefix_checks'] for r in reports),
                   cross_recording_exact_relative_track_duplicate_groups=len(aliases),
                   recordings=reports, role='quarantined_unassigned_source_audit',
                   forecast_errors_opened=False, fitting=False, future_label_filter=False,
                   source_online_causality_verified=False, metric_claim=False, seconds_claim=False,
                   independent_calibration_sites_admitted=0, stage5c_executed=False, smc_enabled=False)
    save(PRIVATE/'duplicate_track_members.json',aliases)
    save(PUBLIC/'analysis.json',summary)
    if args.verify:
        save(PUBLIC/'verification.json',dict(exact=True,analysis_sha256=digest(PUBLIC/'analysis.json')))
    heartbeat('complete',seconds=time.monotonic()-start,rows=summary['rows'],recordings=len(reports))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    for directory in (PRIVATE,PRIVATE/'records',PUBLIC):
        if any(p.is_symlink() for p in (directory,*directory.parents)):
            raise ValueError('Symlinked destination')
        directory.mkdir(parents=True,exist_ok=True)
    with (PRIVATE/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__':
    main()
