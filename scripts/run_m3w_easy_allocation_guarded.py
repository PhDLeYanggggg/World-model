"""Supported replay entrypoint; retain the original experiment source unchanged."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_easy_allocation_repair import load as frozen_load, repair, evaluate, OUTPUT, REPORTS
from scripts.run_m3w_native_forecast import file_digest, assert_current, json_write, immutable_json
from src.evaluation.m3w_frozen_allocation_guard import require_frozen_repair_identity
import torch


def load():
    pack=frozen_load();identity=pack[-1]
    parent_path=ROOT/identity['parent_decision_manifest']
    frozen_parent_path=parent_path.with_name('identity.json')
    if file_digest(frozen_parent_path)!=identity['parent_identity_sha256']:
        raise ValueError('Frozen parent identity hash changed')
    frozen_parent=json.loads(frozen_parent_path.read_text())
    frozen_repair=json.loads((ROOT/OUTPUT/'identity.json').read_text())
    require_frozen_repair_identity(identity,frozen_parent,frozen_repair)
    assert_current(frozen_parent);assert_current(frozen_repair)
    return pack


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['check','repair','evaluate'],default='check')
    p.add_argument('--verify',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();root=ROOT/OUTPUT
    def beat(**v):
        e=dict(pid=os.getpid(),updated_unix=time.time(),**v);json_write(root/'guarded_heartbeat.json',e)
        with (root/'guarded_events.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
        print(json.dumps(e),flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='repair':repair(pack,args.verify,beat)
        elif args.phase=='evaluate':evaluate(pack,beat,args.verify)
        else:beat(state='frozen_parent_and_repair_identity_pass')
    receipt=dict(all_checks_passed=True,experiment_identity_sha256=file_digest(root/'identity.json'),
        source_bindings={p:file_digest(ROOT/p) for p in (
            'scripts/run_m3w_easy_allocation_guarded.py',
            'src/evaluation/m3w_frozen_allocation_guard.py',
            'tests/test_m3w_frozen_allocation_guard.py')},
        complete_parent_binding_preservation=True,complete_repair_identity_equality=True,
        experiment_source_or_choices_modified=False,phase=args.phase,verify=args.verify)
    suffix=args.phase+('_replay' if args.verify else '')
    immutable_json(ROOT/REPORTS/('guarded_'+suffix+'.json'),receipt)


if __name__=='__main__':main()
