"""Full frozen TRAIN census with cached development comparisons, no fitting."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts import read_m3w_scaled_risk_heads as heads

ROOT, training = heads.ROOT, heads.training
base = training.base
PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_train_census_v1'
PRIVATE = ROOT/'data/stage_cvpr2027_experiments/european_train_census_v1'
REG = PUBLIC/'registration.json'
ARMS = ('quadratic','task_utility','unit_control','budget_scaled')


def registration():
    training.admission()
    files = [Path(__file__).resolve(),ROOT/'src/world_model/m3w_cost_head_census.py',
             ROOT/'tests/test_m3w_cost_head_census.py',PUBLIC/'protocol.md']
    train_reg = json.loads(training.REG.read_text())
    result = dict(bindings={str(p.relative_to(ROOT)):base.sha(p) for p in files},
        parent_freeze_sha256=base.sha(training.PUBLIC/'training_freeze.json'),
        development_complete_sha256=base.sha(training.PUBLIC/'readout/complete.json'),
        input_manifest_sha256=base.sha(base.PRIVATE/'create_train_input_manifest.json'),
        expected=train_reg['expected'], arms=list(ARMS),views=288,
        result_source='fresh_run_frozen_TRAIN_full_census',development_source='cached_verified',
        optimizer_steps=0,new_development_inference=False,independent_roles_read=False)
    base.once(REG,result);return result


def cached_development():
    home = training.PUBLIC/'readout'
    complete = json.loads((home/'complete.json').read_text()); rows = {}
    if base.sha(home/'summary.json') != complete['summary_sha256']: raise ValueError('Changed development summary')
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if base.sha(path) != ref['sha256']: raise ValueError('Changed development aggregate')
        row = json.loads(path.read_text()); key = row['group'],row['source'],row['head_seed']
        if key in rows: raise ValueError('Duplicate development identity')
        rows[key] = row
    return rows


def execute(resume):
    reg = registration()
    for rel in [*reg['bindings'],str(REG.relative_to(ROOT))]:
        p = subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True)
        if p.returncode or p.stdout != (ROOT/rel).read_bytes(): raise ValueError('Commit census before execution')
    if (PUBLIC/'complete.json').exists(): raise ValueError('Complete census immutable')
    from src.world_model import m3w_cost_head_census as api
    api.train.torch.set_num_threads(4);api.train.torch.set_num_interop_threads(1)
    documents,inventory = heads.admission(); dev = cached_development()
    prior_reg,_ = heads.previous.execution.check_registration()
    manifest = json.loads((base.PRIVATE/'create_train_input_manifest.json').read_text())
    PRIVATE.mkdir(parents=True,exist_ok=True); started=time.monotonic(); previous=None; refs=[]; rows=[]
    def beat(**kw):
        row = dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(row)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if time.monotonic()-started > 43200: raise TimeoutError('Resume fixed census after12h boundary')
    with (PRIVATE/'process.lock').open('a') as lock,heads.Reader(inventory) as reader:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for ref in manifest['heads']:
            identity,group = ref['identity'],ref['group']; paths=[PUBLIC/'groups'/(group+'_head'+str(identity['seed'])+'_'+arm+'.json') for arm in ARMS]
            if any(p.exists() for p in paths) and not resume: raise ValueError('Explicit resume required')
            if not all(p.exists() for p in paths) and group != previous:
                beat(state='streaming_TRAIN',group=group)
                args=heads.previous.execution.load_packet(ref,prior_reg);previous=group
            coverage=None
            for mode,path in zip(ARMS,paths):
                key=group,identity['source'],identity['seed'];doc=documents[*key,mode]
                binding=dict(registration_sha256=base.sha(REG),checkpoint_sha256=doc['checkpoint']['sha256'],mode=mode)
                if path.exists():
                    row=json.loads(path.read_text())
                    if any(row[k]!=v for k,v in binding.items()): raise ValueError('Changed census receipt')
                else:
                    state=reader.load(ROOT/doc['checkpoint']['path'])
                    if coverage is None: coverage=api.sampling_coverage(state,args)
                    if coverage['draw_hash']!=state['draw_hash']: raise ValueError('Mismatched sampled controls')
                    row=dict(**binding,identity=identity,census=api.census(state,args),sampling_coverage=coverage,
                        cached_development_policy=dev[key]['result']['policies'][mode],
                        cached_development_diagnostics=dev[key]['result']['diagnostics'][mode],
                        result_source='fresh_run_complete_TRAIN_inference',development_source='cached_verified',
                        optimizer_steps=0,independent_roles_read=False,deployment_changed=False)
                    base.once(path,row)
                rows.append(row);refs.append(dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path)))
                beat(state='head_verified',views=len(rows),total=288)
        expected={(r['group'],r['source'],r['seed'],arm) for r in reg['expected'] for arm in ARMS}
        actual={(r['identity']['group'],r['identity']['source'],r['identity']['seed'],r['mode']) for r in rows}
        if len(rows)!=288 or actual!=expected: raise ValueError('Complete288 identities required')
        base.once(PUBLIC/'summary.json',api.summarize(rows))
        base.once(PUBLIC/'complete.json',dict(groups=refs,summary_sha256=base.sha(PUBLIC/'summary.json'),
            registration_sha256=base.sha(REG),seconds=time.monotonic()-started,
            result_source='fresh_run_full_TRAIN_census',development_source='cached_verified',
            optimizer_steps=0,new_development_inference=False,independent_roles_read=False,deployment_changed=False))
        beat(state='complete',views=288)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','run']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register': registration();print('Registered full TRAIN census')
    else:execute(a.resume)


if __name__=='__main__':main()
