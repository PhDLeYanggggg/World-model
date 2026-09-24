"""Complete raw-frame partial-overlap audit with resumable per-record signatures."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_m3w_european_squares import digest, PUBLIC as SOURCE_PUBLIC
from scripts.audit_m3w_european_squares_v2 import PUBLIC as INTAKE_PUBLIC
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.evaluation.m3w_european_squares_overlap import frame_blocks,duplicate_groups,BLOCK_DTYPE

PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_squares_overlap_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_squares_overlap_v1'


def save(path,value):
    payload=json.dumps(value,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=payload:
            raise ValueError('Immutable output differs: '+str(path))
        return
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def beat(state,**kwargs):
    event=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),state=state,**kwargs)
    tmp=PRIVATE/'heartbeat.tmp'
    tmp.write_text(json.dumps(event)+'\n')
    tmp.replace(PRIVATE/'heartbeat.json')
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)


def run(args):
    start=time.monotonic()
    source=json.loads((SOURCE_PUBLIC/'trajectory_manifest.json').read_text())['private_file']
    archive=ROOT/source['path']
    beat('verify_archive',bytes=source['bytes'])
    if archive.stat().st_size!=source['bytes'] or digest(archive)!=source['sha256']:
        raise ValueError('Archive identity differs')
    analysis=json.loads((INTAKE_PUBLIC/'analysis.json').read_text())
    verified=json.loads((INTAKE_PUBLIC/'verification.json').read_text())
    if verified['analysis_sha256']!=digest(INTAKE_PUBLIC/'analysis.json') or not verified['exact']:
        raise ValueError('Intake replay missing')
    grouping_path=ROOT/'outputs/publication_readiness_2026_09/european_squares_site_groups_v1/site_groups.json'
    grouping=json.loads(grouping_path.read_text())
    if grouping['analysis_sha256']!=verified['analysis_sha256']:
        raise ValueError('Grouping not based on this raw audit')
    locality={r['source_member']:r['locality_group'] for r in grouping['recordings']}
    files=['scripts/audit_m3w_european_squares_overlap.py',
           'src/evaluation/m3w_european_squares_overlap.py',
           'src/evaluation/m3w_european_squares_raw_v2.py',
           'src/evaluation/m3w_european_squares_intake.py',
           'tests/test_m3w_european_squares_overlap.py']
    identity=dict(archive_sha256=source['sha256'],analysis_sha256=verified['analysis_sha256'],
        grouping_sha256=digest(grouping_path),code={p:digest(ROOT/p) for p in files},
        scope_sha256=digest(PUBLIC/'scope.md'),frame_sequence_length=8,quantization_pixels=1.0,
        data_role='unassigned_source_audit_no_forecast_readout')
    save(PRIVATE/'identity.json',identity)
    summaries=[]
    records=[]
    selected=analysis['recordings'][:1] if args.pilot else analysis['recordings']
    with zipfile.ZipFile(archive) as z:
        for i,entry in enumerate(selected):
            name=entry['source_member']
            key=hashlib.sha256(name.encode()).hexdigest()
            receipt=PRIVATE/'records'/(key+'.json')
            cache=PRIVATE/'records'/(key+'.npy')
            if receipt.exists():
                r=json.loads(receipt.read_text())
                if r['identity']!=identity or r['source_member']!=name or digest(cache)!=r['signature_sha256']:
                    raise ValueError('Cache receipt differs')
                blocks=np.load(cache,allow_pickle=False)
                if blocks.dtype!=BLOCK_DTYPE or blocks.ndim!=1:
                    raise ValueError('Signature cache schema differs')
                stats=r['stats']
            else:
                if args.verify:
                    raise ValueError('Cannot replay absent signature cache')
                before=time.monotonic()
                beat('fingerprinting',index=i+1,total=len(selected),member=name)
                with z.open(name) as f:
                    rows,_=read_raw_csv(f)
                if hashlib.sha256(rows.tobytes()).hexdigest()!=entry['rows_sha256']:
                    raise ValueError('Raw parsed rows differ')
                blocks,stats=frame_blocks(rows)
                del rows
                if shutil.disk_usage(PRIVATE).free<blocks.nbytes+15_000_000_000:
                    raise ValueError('Insufficient cache disk reserve')
                tmp=cache.with_name(cache.name+'.tmp')
                with tmp.open('wb') as f:
                    np.save(f,blocks,allow_pickle=False)
                tmp.replace(cache)
                r=dict(identity=identity,source_member=name,rows_sha256=entry['rows_sha256'],
                       signature_sha256=digest(cache),stats=stats)
                save(receipt,r)
                beat('record_complete',index=i+1,seconds=time.monotonic()-before,
                     peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**stats)
            summaries.append(dict(source_member=name,rows_sha256=entry['rows_sha256'],
                signature_sha256=r['signature_sha256'],locality_group=locality[name],stats=stats))
            records.append((name,blocks))
    if args.pilot:
        save(PUBLIC/'pilot.json',dict(result_source='fresh_run',identity=identity,recordings=summaries,
                                     full=False,predictive_outcomes_opened=False))
        beat('pilot_complete',seconds=time.monotonic()-start)
        return
    modes={}
    for mode in ('exact','quantized'):
        beat('matching',mode=mode,records=len(records))
        groups=duplicate_groups(records,mode)
        save(PRIVATE/(mode+'_candidate_groups.json'),groups)
        pairs=Counter()
        for group in groups:
            names=sorted({r['recording'] for r in group})
            pairs.update(combinations(names,2))
        pair_rows=[dict(recordings=list(pair),matching_block_fingerprints=n,
                        locality_groups=[locality[p] for p in pair],
                        cross_locality=locality[pair[0]]!=locality[pair[1]]) for pair,n in sorted(pairs.items())]
        modes[mode]=dict(candidate_fingerprint_groups=len(groups),recording_pairs=pair_rows,
                        cross_locality_pairs=sum(p['cross_locality'] for p in pair_rows),
                        candidate_receipt_sha256=digest(PRIVATE/(mode+'_candidate_groups.json')))
    summary=dict(result_source='fresh_run',identity=identity,recordings=summaries,
        raw_recordings=len(records),total_rows=sum(r['stats']['rows'] for r in summaries),
        modes=modes,scope='eight_consecutive_raw_frame_geometry_exact_and_integer_pixel_screens',
        totals={k:sum(r['stats'][k] for r in summaries) for k in
                ('nonempty_frames','consecutive_blocks','exact_dynamic_blocks','quantized_dynamic_blocks')},
        role='unassigned_source_audit',predictive_outcomes_opened=False,training=False,
        independence_proven=False,approximate_or_reencoded_duplicates_excluded=False,
        metric_claim=False,seconds_claim=False,stage5c_executed=False,smc_enabled=False)
    save(PUBLIC/'analysis.json',summary)
    if args.verify:
        save(PUBLIC/'verification.json',dict(result_source='cached_verified',exact=True,
            analysis_sha256=digest(PUBLIC/'analysis.json'),raw_rows_reparsed=False,
            signature_bytes_verified=True,matching_recomputed=True))
    beat('complete',seconds=time.monotonic()-start,rows=summary['total_rows'],
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    for directory in (PRIVATE,PRIVATE/'records',PUBLIC):
        if any(p.is_symlink() for p in (directory,*directory.parents)):
            raise ValueError('Symlinked audit destination')
        directory.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','check-ignore','--quiet',str(PRIVATE/'identity.json')],cwd=ROOT,check=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__=='__main__':
    main()
