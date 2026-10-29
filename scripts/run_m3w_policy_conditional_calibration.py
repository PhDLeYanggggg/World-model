"""Calibrate frozen rational heads only on source-excluded original TRAIN data."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time

if platform.system()=='Darwin' and platform.machine()!='arm64':
    raise RuntimeError('Native arm64 required before numerical imports')

from scripts import run_m3w_rational_harm as parent
from src.world_model import m3w_policy_conditional_calibration as api
from src.evaluation import m3w_policy_conditional_readout as readout

ROOT=parent.ROOT
PUBLIC=ROOT/'outputs/publication_readiness_2026_09/european_policy_conditional_v1'
PRIVATE=ROOT/'data/stage_cvpr2027_experiments/european_policy_conditional_v1'
REG=PUBLIC/'registration.json'
sha,once=parent.run.base.sha,parent.once
fingerprint=parent.api.previous.old.parent.forest.fingerprint


def register():
    reg,cfg,ancestor=parent.admission()
    files=[Path(__file__).resolve(),PUBLIC/'protocol.md',PUBLIC/'prior_evidence.md',
        ROOT/'src/world_model/m3w_policy_conditional_calibration.py',
        ROOT/'src/evaluation/m3w_policy_conditional_readout.py',
        ROOT/'tests/test_m3w_policy_conditional_calibration.py',
        ROOT/'tests/test_m3w_policy_conditional_readout.py']
    d=dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in files},
        parent_registration_sha256=sha(parent.REG),parent_freeze_sha256=sha(parent.PUBLIC/'training_freeze.json'),
        parent_readout_sha256=sha(parent.PUBLIC/'readout/complete.json'),
        lineage=api.lineage(reg['contexts']),seeds=reg['seeds'],calibrators=72,arms=list(api.ARMS),
        budget=api.BUDGET,gain_edges=list(api.GAIN_EDGES),
        minimum_support=dict(recordings=api.MIN_RECORDINGS,sources=api.MIN_SOURCES,known=api.MIN_KNOWN),
        independent_roles_read=False,outer_development_read=False,deployment_changed=False)
    once(REG,d)
    return reg,cfg,ancestor


def admission():
    reg,cfg,ancestor=register();d=json.loads(REG.read_text())
    for rel in [*d['bindings'],str(REG.relative_to(ROOT))]:
        p=subprocess.run(['git','show','HEAD:'+rel],cwd=ROOT,capture_output=True)
        if p.returncode or p.stdout!=(ROOT/rel).read_bytes():raise ValueError('Commit registered comparison before execution')
    return reg,cfg,ancestor


def runtime():
    parent.api.torch.set_num_threads(4);parent.api.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    def beat(**kw):
        d=dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),**kw)
        tmp=PRIVATE/'heartbeat.tmp';tmp.write_text(json.dumps(d)+'\n');os.replace(tmp,PRIVATE/'heartbeat.json')
        with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        print(json.dumps(d),flush=True)
        if time.monotonic()-began>43200:raise TimeoutError('12h boundary; resume immutable context checkpoints')
    return began,beat


def expected(reg):
    return {(c,i['packet']['identity']['source'],s) for c,items in reg['contexts'].items() for i in items for s in reg['seeds']}


def prepare(resume):
    reg,_,ancestor=admission();_,docs=parent.models(reg)
    if (PUBLIC/'decision_freeze.json').exists():raise ValueError('Completed decision freeze immutable')
    began,beat=runtime();refs=[]
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if shutil.disk_usage(ROOT).free < 10*2**30:raise OSError('Preserve10GiB reserve')
        for context,items in sorted(reg['contexts'].items()):
            cp=PUBLIC/'contexts'/(context+'.json')
            if cp.exists():
                if not resume:raise ValueError('Explicit resume required')
                row=json.loads(cp.read_text())
                if row['registration_sha256']!=sha(REG):raise ValueError('Changed interrupted experiment')
                for rr in row['receipts']:
                    p=ROOT/rr['path'];d=json.loads(p.read_text())
                    if sha(p)!=rr['sha256'] or sha(ROOT/d['cache']['path'])!=d['cache']['sha256']:raise ValueError('Changed saved context')
                refs.extend(row['receipts']);continue
            fit,held=parent.run.parent.prior.packets(items,ancestor,beat)
            del fit
            current=[]
            for seed in reg['seeds']:
                doc=docs[context,seed];state=parent.api.core.read_checkpoint(ROOT/doc['checkpoint']['path'])
                data={}
                for source,args in sorted(held.items()):
                    x,env,y,_,_,recs,frames,_=args
                    pred=parent.api.predict(state,source,x,env)
                    old=parent.PUBLIC/'readout/groups'/('held_'+context+'_'+source+'_seed'+str(seed)+'.json')
                    frozen=json.loads(old.read_text())
                    if fingerprint(pred)!=frozen['prediction_hashes']['rational']:raise ValueError('Frozen raw predictions differ')
                    data[source]=dict(prediction=pred,envelope=env,targets=y,recordings=recs,frames=frames,
                        eligible=parent.api.previous.prior.eligible(args))
                for source,row in sorted(data.items()):
                    # The calibrator API cannot see its own readout source, including its labels.
                    cal=api.fit({s:r for s,r in data.items() if s!=source},source)
                    policies=api.decide(row['prediction'],row['envelope'],row['eligible'],cal)
                    key=context+'_'+source+'_seed'+str(seed)
                    path=PRIVATE/'decisions'/(key+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                    arrays={**row,**{'action_'+a:m for a,m in policies.items()}}
                    hashes={k:fingerprint(v) for k,v in arrays.items()}
                    if path.exists():
                        with parent.api.np.load(path,allow_pickle=False) as saved:
                            if set(saved.files)!=set(arrays) or any(fingerprint(saved[k])!=h for k,h in hashes.items()):
                                raise ValueError('Interrupted decision cache changed')
                    else:
                        temp=path.with_suffix('.tmp.npz');parent.api.np.savez_compressed(temp,**arrays);os.replace(temp,path)
                    out=PUBLIC/'calibrators'/(key+'.json')
                    receipt=dict(context=context,source=source,seed=seed,calibrator=cal,
                        registration_sha256=sha(REG),checkpoint_sha256=doc['checkpoint']['sha256'],
                        cache=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size),
                        hashes=hashes,counts={a:int(m.sum()) for a,m in policies.items()},
                        source_excluded=True,raw_prediction_replayed=True,
                        result_source='fresh_run_calibration_and_frozen_decisions',independent_roles_read=False)
                    once(out,receipt);current.append(dict(path=str(out.relative_to(ROOT)),sha256=sha(out)))
                    beat(state='decision_frozen',views=len(refs)+len(current),total=72)
            once(cp,dict(registration_sha256=sha(REG),receipts=current))
            refs.extend(current)
        if len(refs)!=72:raise ValueError('All72 frozen policies required before readout')
        once(PUBLIC/'decision_freeze.json',dict(registration_sha256=sha(REG),receipts=refs,
            seconds=time.monotonic()-began,pid=os.getpid(),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            calibration_source_exclusion=True,readout_reductions_performed=False,neural_updates=0,
            independent_roles_read=False,outer_development_read=False,deployment_changed=False))
        beat(state='all72_decisions_frozen')


def evaluate(check=False,resume=False):
    reg,_,_=admission();_,docs=parent.models(reg)
    freeze=json.loads((PUBLIC/'decision_freeze.json').read_text())
    if freeze['registration_sha256']!=sha(REG) or len(freeze['receipts'])!=72:raise ValueError('Decision freeze changed')
    began,beat=runtime();rows=[];refs=[]
    if not check and (PUBLIC/'complete.json').exists():raise ValueError('Completed readout immutable')
    with (PRIVATE/'process.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for ref in freeze['receipts']:
            p=ROOT/ref['path'];d=json.loads(p.read_text())
            if sha(p)!=ref['sha256'] or d['registration_sha256']!=sha(REG):raise ValueError('Calibrator receipt changed')
            if d['checkpoint_sha256']!=docs[d['context'],d['seed']]['checkpoint']['sha256']:raise ValueError('Forecast head changed')
            if d['source'] in d['calibrator']['calibration_sources']:raise ValueError('Target source used in calibration')
            path=ROOT/d['cache']['path']
            if path.is_symlink() or not path.resolve().is_relative_to(PRIVATE.resolve()) or sha(path)!=d['cache']['sha256']:
                raise ValueError('Decision cache changed')
            with parent.api.np.load(path,allow_pickle=False) as z:
                if set(z.files)!=set(d['hashes']) or any(fingerprint(z[k])!=v for k,v in d['hashes'].items()):raise ValueError('Array alignment changed')
                actions=api.decide(z['prediction'],z['envelope'],z['eligible'],d['calibrator'])
                for a in api.ARMS:parent.api.np.testing.assert_array_equal(actions[a],z['action_'+a])
                result=readout.evaluate(z['targets'],z['envelope'],z['recordings'],z['frames'],z['prediction'],actions)
            readout.validate(result)
            old=parent.PUBLIC/'readout/groups'/('held_'+d['context']+'_'+d['source']+'_seed'+str(d['seed'])+'.json')
            if result['policies']['raw']!=json.loads(old.read_text())['result']['policies']['rational']:
                raise ValueError('Original raw policy failed exact replay')
            row=dict(context=d['context'],source=d['source'],seed=d['seed'],result=result,
                registration_sha256=sha(REG),decision_freeze_sha256=sha(PUBLIC/'decision_freeze.json'),
                calibrator_sha256=ref['sha256'],result_source='fresh_run_source_excluded_TRAIN_readout',
                independent_roles_read=False,deployment_changed=False)
            out=PUBLIC/'readout'/p.name
            if check:
                if not out.exists() or json.loads(out.read_text())!=row:raise ValueError('Readout no longer replays')
            else:
                if out.exists() and not resume:raise ValueError('Explicit resume required')
                once(out,row)
            rows.append(row);refs.append(dict(path=str(out.relative_to(ROOT)),sha256=sha(out)))
        summary=readout.summarize(rows,expected(reg))
        if check:
            complete=json.loads((PUBLIC/'complete.json').read_text())
            if (complete['readouts']!=refs or complete['summary_sha256']!=sha(PUBLIC/'summary.json')
                    or summary!=json.loads((PUBLIC/'summary.json').read_text())):raise ValueError('Aggregate failed full replay')
        else:
            once(PUBLIC/'summary.json',summary)
            once(PUBLIC/'complete.json',dict(readouts=refs,summary_sha256=sha(PUBLIC/'summary.json'),
                decision_freeze_sha256=sha(PUBLIC/'decision_freeze.json'),registration_sha256=sha(REG),
                seconds=time.monotonic()-began,pid=os.getpid(),independent_roles_read=False,deployment_changed=False))
        beat(state='verified' if check else 'readout_complete',views=72,advance=summary['advance'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','prepare','eval','check'])
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='register':register();print('Registered72 cross-source calibration folds inside original TRAIN')
    elif a.action=='prepare':prepare(a.resume)
    else:evaluate(a.action=='check',a.resume)


if __name__=='__main__':main()
