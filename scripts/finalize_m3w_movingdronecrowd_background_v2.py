"""Recover preserved complete matching after a duplicate-key summary failure."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import check_m3w_movingdronecrowd_background as check
from src.evaluation.m3w_background_finalization import summarize_roles_once

base,version=check.base,check.version
PUBLIC,PRIVATE=version.PUBLIC,version.PRIVATE
FILES=(Path(__file__),ROOT/'scripts/check_m3w_movingdronecrowd_background.py',
       ROOT/'src/evaluation/m3w_background_finalization.py',
       ROOT/'tests/test_m3w_background_finalization.py',ROOT/'tests/test_m3w_background_readonly_bindings.py',
       PUBLIC/'completion_failure.md')


def pair_inventory(roster):
    if len(list((PRIVATE/'pairs').glob('*.json')))!=len(roster):
        raise ValueError('Preserve exactly the complete original pair checkpoints')
    return {str(base.pair_path(row).relative_to(ROOT)):base.sha(base.pair_path(row)) for row in roster}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','recover','verify'));args=parser.parse_args()
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reg,groups,protected=check.frozen_context();cfg=reg['configuration']
        base.PUBLIC,base.PRIVATE=PUBLIC,PRIVATE
        base.cv2.setNumThreads(1)
        selection=reg['view_selection'];moving=sorted({r['node'] for r in selection})
        roster=base.pair_roster(moving,['drone_'+v for v in protected],cfg['roster_seed'])
        inventory=pair_inventory(roster)
        heartbeat=json.loads((PRIVATE/'heartbeat.json').read_text())
        if heartbeat['pid']!=69529 or heartbeat['complete']!=20580 or heartbeat['new_pairs']!=20530:
            raise ValueError('Require the preserved completed original heartbeat')
        recovery_reg=dict(bindings={str(path.relative_to(ROOT)):base.sha(path) for path in FILES},
            original_registration_sha256=base.sha(PUBLIC/'registration.json'),
            original_matching_log_sha256=base.sha(PRIVATE/'matching.log'),
            original_heartbeat_sha256=base.sha(PRIVATE/'heartbeat.json'),
            original_pair_inventory_sha256=hashlib.sha256(base.intake.encoded(inventory)).hexdigest(),
            original_matching_exit_code=1,original_matching_pid=69529,
            original_pair_loop_seconds=heartbeat['seconds'],original_peak_RSS_bytes=heartbeat['peak_RSS_bytes'],
            original_pair_loop_new_pairs=20530,original_pair_loop_cached_pilot_pairs=50,
            corrected_factor='duplicate_false_role_dictionary_field_only',matching_criteria_changed=False,
            source_roles_assigned=False,predictive_outcomes_read=False,new_model_training=False)
        base.once(PUBLIC/'completion_repair_registration.json',base.stable(recovery_reg))
        if args.phase=='register':return
        for path in (*FILES,PUBLIC/'completion_repair_registration.json'):
            body=subprocess.run(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT,
                                capture_output=True,check=True).stdout
            if hashlib.sha256(body).hexdigest()!=base.sha(path):raise ValueError('Commit the completion adapter first')
        is_verify=args.phase=='verify'
        end=PUBLIC/('verification.json' if is_verify else 'execution.json')
        if end.exists():raise ValueError('Completed recovery phase is immutable')
        if is_verify and not (PUBLIC/'execution.json').exists():raise ValueError('Recover complete summary first')
        if shutil.disk_usage(ROOT).free<cfg['disk_reserve_bytes']:raise OSError('Keep original10GiB reserve')
        started=time.monotonic()
        def beat(**kw):
            event=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),
                seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**kw)
            base.atomic(PRIVATE/('verification_heartbeat.json' if is_verify else 'recovery_heartbeat.json'),event)
            print(json.dumps(event),flush=True)
        beat(state='frozen_pair_finalization',phase=args.phase)
        matcher=dict(json.loads((ROOT/cfg['matcher_config']).read_text()),selected_frames_one_based=[1,2,3])
        features,frames,feature_receipts=base.load_features(selection,protected,matcher,reextract=is_verify)
        results=[]
        for i,row in enumerate(roster):
            value=json.loads(base.pair_path(row).read_text())
            if any(value.get(k)!=v for k,v in dict(registration_sha256=base.sha(PUBLIC/'registration.json'),**row).items()):
                raise ValueError('Changed saved pair identity')
            if is_verify and i<cfg['verification_fresh_pair_prefix']:
                if base.match_pair(row,features,frames,matcher)!={k:v for k,v in value.items() if k!='match_seconds'}:
                    raise ValueError('Fresh fixed-prefix matching differs')
            results.append(value)
        summary=summarize_roles_once(base,results,selection,protected,groups,feature_receipts)
        base.once(PUBLIC/'analysis.json',summary)
        base.once(end,dict(pid=os.getpid(),seconds=time.monotonic()-started,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            result_source='cached_verified_all_receipts_plus_fresh_features_and_fixed50pairs' if is_verify
                          else 'cached_verified_complete_original_pair_work_forward_summary_recovery',
            new_pairs=0,cached_pairs=len(results),analysis_sha256=base.sha(PUBLIC/'analysis.json'),
            completion_adapter_registration_sha256=base.sha(PUBLIC/'completion_repair_registration.json'),
            original_matching_pid=69529,original_matching_exit_code=1,
            original_matching_loop_seconds=heartbeat['seconds'],original_matching_peak_RSS_bytes=heartbeat['peak_RSS_bytes'],
            original_matching_new_pairs=20530,original_cached_pilot_pairs=50,
            all_new_prepared_features_reextracted_exact=is_verify,
            freshly_recomputed_pair_prefix=cfg['verification_fresh_pair_prefix'] if is_verify else 0,
            all_pair_receipts_and_graph_exact=is_verify,predictive_outcomes_read=False))
        beat(state='complete',phase=args.phase,clip_pairs=len(results))


if __name__=='__main__':main()
