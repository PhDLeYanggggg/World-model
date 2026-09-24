"""Build the registered source-training cohort, with resumable raw-row verification."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_m3w_european_squares import digest, save
from src.evaluation.m3w_european_squares_raw_v2 import read_raw_csv
from src.evaluation.m3w_european_squares_roles import require_source_training
from src.data_unification.m3w_european_squares_source import (
    choose_query_frames, build_recording, INPUT_FIELDS, LABEL_FIELDS)

BASE = ROOT/'outputs/publication_readiness_2026_09'
OUT = BASE/'european_squares_source_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_squares_source_v1'


def beat(state, **values):
    event = dict(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat(), state=state, **values)
    p = PRIVATE/'heartbeat.tmp'
    p.write_text(json.dumps(event)+'\n')
    p.replace(PRIVATE/'heartbeat.json')
    with (PRIVATE/'events.jsonl').open('a') as f:
        f.write(json.dumps(event)+'\n')
    print(json.dumps(event), flush=True)


def store_array(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        np.save(f, value, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def run(args):
    started = time.monotonic()
    role_path = BASE/'european_squares_roles_v1/roles.json'
    roles = json.loads(role_path.read_text())
    for entry in roles['dependency_bindings'].values():
        if digest(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Role dependency differs')
    source = json.loads((BASE/'european_squares_intake_v1/trajectory_manifest.json').read_text())['private_file']
    archive = ROOT/source['path']
    beat('verify_archive', bytes=source['bytes'])
    if digest(archive) != source['sha256']:
        raise ValueError('Archive differs')
    files = ['scripts/build_m3w_european_squares_source.py',
             'src/data_unification/m3w_european_squares_source.py',
             'tests/test_m3w_european_squares_source.py',
             'src/evaluation/m3w_european_squares_raw_v2.py',
             'src/evaluation/m3w_european_squares_intake.py',
             'src/evaluation/m3w_european_squares_roles.py']
    identity = dict(roles_sha256=digest(role_path), archive_sha256=source['sha256'],
        source_code={p:digest(ROOT/p) for p in files}, per_recording_query_cap=256,
        input_fields=INPUT_FIELDS, label_fields=LABEL_FIELDS, data_role='source_training',
        source_task='eight_observed_twelve_requested_raw_stride12_unsmoothed_box_center')
    # JSON normalization keeps tuple/list comparisons identical after resume.
    identity = json.loads(json.dumps(identity))
    save(PRIVATE/'identity.json', identity)
    selected = [r for r in roles['recordings'] if r['role'] == 'source_training']
    if args.pilot:
        selected = selected[:1]
    receipts = []
    with zipfile.ZipFile(archive) as archive_file:
        for i, row in enumerate(selected):
            name = row['source_member']
            require_source_training(roles, name)
            directory = PRIVATE/'records'/hashlib.sha256(name.encode()).hexdigest()
            directory.mkdir(parents=True, exist_ok=True)
            receipt_path = directory/'receipt.json'
            if receipt_path.exists() and not args.verify:
                receipt = json.loads(receipt_path.read_text())
                if receipt['identity'] != identity or receipt['rows_sha256'] != row['rows_sha256']:
                    raise ValueError('Cached source identity differs')
                for r in receipt['arrays'].values():
                    if digest(directory/r['name']) != r['sha256']:
                        raise ValueError('Cached array differs')
                receipts.append(receipt)
                beat('cached_verified_record', index=i+1, total=len(selected))
                continue
            before = time.monotonic()
            beat('building_record', index=i+1, total=len(selected), member=name)
            with archive_file.open(name) as f:
                rows, _ = read_raw_csv(f)
            if hashlib.sha256(rows.tobytes()).hexdigest() != row['rows_sha256']:
                raise ValueError('Raw row identity differs')
            query, support = choose_query_frames(rows)
            if support['all_past_eligible_targets'] != row['past_eligible_k8_stride12']:
                raise ValueError('Independent past-support count differs')
            inputs, labels = build_recording(rows, query)
            checks = 0
            for qi in sorted(set([0,len(query)//2,len(query)-1])) if len(query) else []:
                q = query[qi]
                prefix, _ = build_recording(rows[rows['frame'] <= q],[q])
                start,end = inputs['query_offsets'][qi:qi+2]
                for key in INPUT_FIELDS:
                    ref = np.array([0,end-start]) if key == 'query_offsets' else inputs[key][start:end]
                    if not np.array_equal(prefix[key],ref):
                        raise ValueError('Future truncation changed input '+key)
                checks += 1
            del rows
            required = sum(v.nbytes for v in [*inputs.values(),*labels.values(),query])
            if shutil.disk_usage(PRIVATE).free < required+15_000_000_000:
                raise ValueError('Insufficient disk reserve')
            arrays = {}
            for kind, collection in [('input',inputs),('label',labels),('index',{'query_frames':query})]:
                for key, value in collection.items():
                    path = directory/(kind+'_'+key+'.npy')
                    if args.verify:
                        if not np.array_equal(np.load(path,allow_pickle=False,mmap_mode='r'),value):
                            raise ValueError('Full raw replay differs: '+path.name)
                    else:
                        store_array(path,value)
                    arrays[kind+'_'+key] = dict(name=path.name,sha256=digest(path),
                                               shape=list(value.shape),dtype=str(value.dtype),bytes=path.stat().st_size)
            eligible = inputs['target_eligible']
            nfuture = labels['future_valid'].sum(axis=1)[eligible]
            receipt = dict(identity=identity,source_member=name,rows_sha256=row['rows_sha256'],
                locality_group=row['locality_group'],source_square_id=row['source_square_id'],
                role='source_training',arrays=arrays,support=support,
                selected_query_frames=len(query),visible_agent_query_rows=len(eligible),
                target_rows=int(eligible.sum()),incomplete_history_context_rows=int((~eligible).sum()),
                target_complete_future=int((nfuture==12).sum()),
                target_partial_future=int(((nfuture>0)&(nfuture<12)).sum()),
                target_no_future=int((nfuture==0).sum()),future_truncation_checks=checks)
            save(receipt_path,receipt)
            receipts.append(receipt)
            beat('record_complete',index=i+1,seconds=time.monotonic()-before,
                 targets=receipt['target_rows'],visible=receipt['visible_agent_query_rows'],
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    fields = ('selected_query_frames','visible_agent_query_rows','target_rows','incomplete_history_context_rows',
              'target_complete_future','target_partial_future','target_no_future','future_truncation_checks')
    summary = dict(result_source='fresh_run',identity=identity,full_registered_source_cohort=not args.pilot,
        raw_recordings=len(receipts),locality_groups=len({r['locality_group'] for r in receipts}),
        totals={k:sum(r[k] for r in receipts) for k in fields},
        cache_bytes=sum(v['bytes'] for r in receipts for v in r['arrays'].values()),
        record_receipts=[dict(source_member=r['source_member'],role=r['role'],locality_group=r['locality_group'],
            directory=str((PRIVATE/'records'/hashlib.sha256(r['source_member'].encode()).hexdigest()).relative_to(ROOT)),
            receipt_sha256=digest(PRIVATE/'records'/hashlib.sha256(r['source_member'].encode()).hexdigest()/'receipt.json'))
            for r in receipts],
        prediction_errors_opened=False,training_executed=False,reserved_roles_read=False,
        no_future_input=True,no_future_label_filter=True,metric_claim=False,seconds_claim=False,
        stage5c_executed=False,smc_enabled=False)
    name = 'pilot.json' if args.pilot else 'analysis.json'
    save(OUT/name,summary)
    if args.verify:
        save(OUT/'verification.json',dict(result_source='fresh_run',raw_reparsed=True,
             analysis_sha256=digest(OUT/'analysis.json'),all_arrays_exact=True))
    beat('complete',seconds=time.monotonic()-started,records=len(receipts),**summary['totals'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    if args.pilot and args.verify:
        parser.error('Verify applies to the complete registered cohort')
    for directory in (OUT,PRIVATE,PRIVATE/'records'):
        if any(p.is_symlink() for p in (directory,*directory.parents)):
            raise ValueError('Symlinked source destination')
        directory.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','check-ignore','--quiet',str(PRIVATE/'identity.json')],cwd=ROOT,check=True)
    with (PRIVATE/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
