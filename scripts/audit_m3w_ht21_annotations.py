"""Audit every official HT21 annotation row without admitting forecasting use."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fetch_m3w_ht21_annotations import ARCHIVE,PUBLIC,inventory,sha
from src.evaluation.m3w_ht21_intake import read_metadata,read_ground_truth,audit_ground_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    receipt = json.loads((PUBLIC/'source_manifest.json').read_text())
    with ARCHIVE.open('rb') as f: archive_hash=hashlib.file_digest(f,'sha256').hexdigest()
    assert archive_hash == receipt['archive_sha256'] and ARCHIVE.stat().st_size == receipt['archive_bytes']
    assert inventory(ARCHIVE) == receipt['members']
    started=time.monotonic(); reports=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        names=set(z.namelist())
        for member in sorted(n for n in names if n.endswith('/seqinfo.ini')):
            prefix=member.rsplit('/',1)[0]
            split,sequence=prefix.split('/')[1:]
            metadata_payload=z.read(member)
            metadata=read_metadata(metadata_payload)
            row=dict(sequence=sequence,supplied_split=split,metadata=metadata,
                metadata_sha256=sha(metadata_payload),ground_truth_present=prefix+'/gt/gt.txt' in names,
                detection_file_present=prefix+'/det/det.txt' in names,
                scientific_role='quarantined_unassigned')
            if row['ground_truth_present']:
                data=z.read(prefix+'/gt/gt.txt')
                gt=read_ground_truth(data,metadata)
                row.update(ground_truth_sha256=sha(data),audit=audit_ground_truth(gt,metadata))
            else:
                row['status']='not_run_no_released_ground_truth_detections_not_substituted'
            reports.append(row)
            print(json.dumps(dict(pid=os.getpid(),sequence=sequence,state='audited',
                ground_truth=row['ground_truth_present'],seconds=time.monotonic()-started)),flush=True)
    gt=[r for r in reports if r['ground_truth_present']]
    totals={}
    for k in ('8','16','32','64'):
        entries=[r['audit']['availability_stride1'][k] for r in gt]
        totals[k]={f:sum(r[f] for r in entries) for f in ('past_eligible','complete_future12','partial_future12','no_future12')}
        totals[k]['endpoint_available']={str(h):sum(r['endpoint_available'][str(h)] for r in entries) for h in (10,25,50,100)}
    result=dict(result_source='fresh_run',source_manifest_sha256=sha((PUBLIC/'source_manifest.json').read_bytes()),
        archive_sha256=archive_hash,recordings=reports,
        aggregate=dict(recordings=len(reports),ground_truth_recordings=len(gt),
            ground_truth_rows=sum(r['audit']['rows'] for r in gt),
            track_ids_with_recording_namespace=sum(r['audit']['track_ids'] for r in gt),
            visible_static_rows=sum(r['audit']['visible_static_rows'] for r in gt),availability_stride1=totals),
        coordinate_unit='head_center_annotation_pixel',metric_status='not_verified',
        time_status='publisher_and_sequence_fps_metadata_only_no_image_clock_verification',
        causal_status='past_indexed_offline_interpolated_annotations_not_verified_online',
        independent_physical_sites_verified=False,source_use_approval=False,scientific_roles_assigned=False,
        official_forecasting_benchmark=False,training=False,baseline_or_model_error_evaluation=False,
        confirmation=False,stage5c_executed=False,smc_enabled=False,
        implementation_sha256={p:sha((ROOT/p).read_bytes()) for p in
            ('scripts/audit_m3w_ht21_annotations.py','src/evaluation/m3w_ht21_intake.py','tests/test_m3w_ht21_intake.py')})
    path=PUBLIC/'analysis.json'
    if args.verify:
        assert result == json.loads(path.read_text()),'Source audit changed'
        verification=dict(all_checks_passed=True,analysis_sha256=sha(path.read_bytes()),
                          result_source='cached_verified',rows_reparsed=result['aggregate']['ground_truth_rows'])
        with (PUBLIC/'verification.json').open('x') as f: f.write(json.dumps(verification,indent=2)+'\n')
    else:
        with path.open('x') as f: f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
        with (PUBLIC/'execution.json').open('x') as f:
            f.write(json.dumps(dict(pid=os.getpid(),completed_at_utc=datetime.now(timezone.utc).isoformat(),
                seconds=time.monotonic()-started,analysis_sha256=sha(path.read_bytes())),indent=2)+'\n')
    print(json.dumps(result['aggregate'],indent=2))


if __name__=='__main__': main()
