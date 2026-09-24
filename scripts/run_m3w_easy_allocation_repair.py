"""Repair only causal solver failures; retain every successful frozen V1 query."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_m3w_easy_allocation import load as load_original, causal_view, evaluate, CONTEXT_KEYS
import numpy as np
import torch
from scripts.run_m3w_native_forecast import file_digest, immutable_json, json_write, assert_current
from scripts.run_m3w_native_nested import write_arrays
from src.world_model.m3w_easy_allocation_repaired import query

OUTPUT = 'data/stage_cvpr2027_experiments/easy_allocation_risk_scaled_v1'
REPORTS = 'outputs/publication_readiness_2026_09/easy_allocation_risk_scaled_v1'
CODE = ('scripts/run_m3w_easy_allocation_repair.py',
        'src/world_model/m3w_easy_allocation_repaired.py',
        'src/world_model/m3w_scaled_risk_controls.py',
        'tests/test_m3w_scaled_risk_controls.py')


def load():
    original = load_original()
    cfg, *rest = original
    old_root = ROOT/cfg['output']
    parent_identity = old_root/'identity.json'
    original_manifest = old_root/'decisions_complete.json'
    manifest = json.loads(original_manifest.read_text())
    assert manifest['identity_sha256'] == file_digest(parent_identity)
    bindings = dict(original[-1]['source_bindings'])
    for path in (parent_identity, original_manifest, ROOT/REPORTS/'registration.md', *(ROOT/p for p in CODE)):
        bindings[str(path.relative_to(ROOT))] = file_digest(path)
    for r in manifest['receipts']:
        assert file_digest(ROOT/r['path']) == r['sha256']
        bindings[r['path']] = r['sha256']
        record = json.loads((ROOT/r['path']).read_text())
        assert file_digest(ROOT/record['path']) == record['sha256']
        bindings[record['path']] = record['sha256']
    cfg = dict(cfg, output=OUTPUT, reports=REPORTS,
        registration=REPORTS+'/registration.md', numerical_repair_only=True)
    identity = dict(original[-1], config=cfg, source_bindings=bindings,
        parent_identity_sha256=file_digest(parent_identity),
        parent_decision_manifest=str(original_manifest.relative_to(ROOT)),
        repair_selection='not matched in frozen causal V1 manifest; no outcome inputs')
    assert_current(identity)
    return cfg, *rest[:-1], identity


def repair(pack, verify, beat):
    cfg,data,context,a,tr,eq,identity=pack
    root=ROOT/cfg['output']; ish=file_digest(root/'identity.json')
    source=json.loads((ROOT/identity['parent_decision_manifest']).read_text())
    receipts=[]; repaired=0; changed=0; current=None; start=time.monotonic()
    for item in source['receipts']:
        old=json.loads((ROOT/item['path']).read_text())
        failed=[i for i,q in enumerate(old['queries']) if not q['matched']]
        if not failed:
            receipts.append(item);continue
        key,action=old['view'],old['action']
        if current!=(key,action):
            values,pred,scale,cut=causal_view(pack,key,action)
            index=np.full(len(data['sites']),-1,np.int64);index[values['ids']]=np.arange(len(values['ids']))
            current=(key,action)
        rec=next(r for r in context['records'] if r['recording']==old['recording'])
        with np.load(ROOT/rec['cache']['path'],allow_pickle=False) as z:c={k:z[k].copy() for k in CONTEXT_KEYS}
        with np.load(ROOT/old['path'],allow_pickle=False) as z:ids=z['ids'].copy();bits=z['choices'].copy()
        new_reports=list(old['queries']);lookup={int(v):i for i,v in enumerate(ids)}
        for qi in failed:
            q=old['queries'][qi]; rows=np.flatnonzero(c['context_frame_ids']==q['frame'])
            tid,b,report=query(rows,c,index,values,pred,data,scale,cut,action,cfg)
            loc=np.array([lookup[int(i)] for i in tid])
            np.testing.assert_array_equal(b[:,[0,1,2,3,4,6]],bits[loc][:,[0,1,2,3,4,6]])
            changed+=int(np.count_nonzero(b!=bits[loc])); bits[loc]=b
            new_reports[qi]=dict(frame=q['frame'],**report);repaired+=1
        path=root/'decisions'/Path(old['path']).relative_to('data/stage_cvpr2027_experiments/easy_allocation_v1/decisions')
        receipt=path.with_suffix('.json')
        if verify:assert path.exists() and receipt.exists()
        write_arrays(path,dict(ids=ids,choices=bits))
        result=dict(old,identity_sha256=ish,path=str(path.relative_to(ROOT)),sha256=file_digest(path),
            queries=new_reports,repair_of=item,repair_frames=[old['queries'][i]['frame'] for i in failed])
        immutable_json(receipt,result)
        receipts.append(dict(path=str(receipt.relative_to(ROOT)),sha256=file_digest(receipt)))
        beat(state='repair_replay' if verify else 'causal_numerical_repair',repaired_queries=repaired,
             changed_agent_arm_bits=changed,seconds=time.monotonic()-start)
    assert repaired==127
    assert_current(identity)
    immutable_json(root/'decisions_complete.json',dict(identity_sha256=ish,receipts=receipts,
        query_action_seed_instances=source['query_action_seed_instances'],future_outcome_arrays_loaded=False,
        repaired_queries=repaired,changed_agent_arm_bits=changed))
    if verify:
        immutable_json(ROOT/REPORTS/'decision_replay.json',dict(all_checks_passed=True,
            decision_manifest_sha256=file_digest(root/'decisions_complete.json'),
            queries_replayed=repaired,other_query_decisions='cached_verified_unchanged'))
    beat(state='repair_replay_complete' if verify else 'repair_complete',repaired_queries=repaired)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['repair','evaluate'],required=True)
    parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    pack=load();root=ROOT/OUTPUT;root.mkdir(parents=True,exist_ok=True)
    immutable_json(root/'identity.json',pack[-1])
    def beat(**v):
        e=dict(pid=os.getpid(),updated_unix=time.time(),**v);json_write(root/'heartbeat.json',e)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
        print(json.dumps(e),flush=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.phase=='repair':repair(pack,args.verify,beat)
        else:evaluate(pack,beat,args.verify)


if __name__=='__main__':main()
