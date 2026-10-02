"""Centered past-quality leaf learning, with a paired within-recording placebo."""
import argparse
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 runtime required')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_label_support as previous
from scripts.manage_m3w_boundary_diagnostic import RECEIVER
from scripts.m3w_packet_stream import PacketStream
from src.world_model import m3w_past_quality_auxiliary as api

parent,sha,once=previous.parent,previous.sha,previous.once
leaf=previous.leaf
NAME='european_past_quality_auxiliary_v1'
PUBLIC=previous.PUBLIC.parent/NAME;PRIVATE=previous.PRIVATE.parent/NAME
CONFIG=ROOT/'configs'/('m3w_'+NAME+'.json');REMOTE='/users/k24101830/m3w/'+NAME


def registration():
    assert previous.registration()==json.loads((previous.PUBLIC/'registration.json').read_text())
    v=json.loads((previous.PUBLIC/'verification.json').read_text())
    assert sha(previous.PUBLIC/'summary.json')==v['summary_sha256']
    assert sha(previous.PUBLIC/'complete.json')==v['complete_sha256']
    assert sha(previous.PUBLIC/'replay.json')==v['replay_sha256']
    paths=parent.forest.closure(ROOT,['scripts.run_m3w_past_quality_auxiliary'])
    paths += [CONFIG,PUBLIC/'protocol.md',ROOT/'tests/test_m3w_past_quality_auxiliary.py']
    return dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        label_diagnostic_verification_sha256=sha(previous.PUBLIC/'verification.json'),
        features=list(api.FEATURES),source_heads=72,auxiliary_fits=144,new_neural_updates=0,
        independent_roles_read=False)


def past_quality(data):
    obs=previous.observation
    manifest,_,_,_=obs.registration()
    prior=json.loads((obs.PUBLIC/'audit.json').read_text())
    q=np.empty((len(data['sites']),7));seen=np.zeros(len(q),bool);refs=[]
    for ri,ref in enumerate(manifest['record_receipts']):
        r=prior['receipts'][ri];assert sha(ROOT/r['path'])==r['sha256']
        receipt=json.loads((ROOT/r['path']).read_text());assert receipt['source']==ref
        arr=receipt['diagnostics'];assert sha(ROOT/arr['path'])==arr['sha256']
        ids=np.flatnonzero(data['recordings']==ri)
        assert parent.base.inter.array_hash(ids)==receipt['packed_ids_sha256'] and not seen[ids].any()
        with np.load(ROOT/arr['path'],allow_pickle=False) as z: fields={k:z[k] for k in api.FEATURES}
        values=api.quality_matrix(fields);assert values.shape==(len(ids),7)
        q[ids]=values;seen[ids]=True;refs.append(r)
    assert seen.all() and len(q)==318969
    return q,dict(rows=len(q),records=len(refs),past_only=True,features=list(api.FEATURES),
                  quality_hash=parent.base.inter.array_hash(q),receipts=refs)


def remote_storage():
    code=r'''
import hashlib,json,pathlib,shutil,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_past_quality_auxiliary_v1')
assert json.loads((r.parent/'.m3w_owner.json').read_text())['project']=='M3W'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
assert q.returncode==0 and 'm3w_past_quality' not in q.stdout
assert shutil.disk_usage(r.parent).free>p['cap']
r.mkdir(exist_ok=True)
for name,txt in {'.owner.json':json.dumps(dict(experiment=p['name'],project='M3W'))+'\n',
                 'registration.json':p['registration'],'config.json':p['config']}.items():
    f=r/name
    if f.exists():assert f.read_text()==txt
    else:
        with f.open('x') as out:out.write(txt)
existing=[]
for f in sorted((r/'inputs').glob('*.npz')):
    existing.append(dict(group=f.stem,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print(json.dumps(dict(existing=existing,owned_root_verified=True,personal_quota='unknown',
    remote_science_executed=False,m3w_jobs=[l for l in q.stdout.splitlines() if 'm3w' in l.lower()])))
'''
    cfg=json.loads(CONFIG.read_text())
    return leaf.previous.independent.read_remote(code,dict(root=REMOTE,name=NAME,cap=cfg['remote_checkpoint_cap_bytes'],
        registration=(PUBLIC/'registration.json').read_text(),config=CONFIG.read_text()))


class Stream(PacketStream):
    def __init__(self):
        handoff=ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(handoff/'compute_handoff_20260927.json')=='f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh=json.loads((handoff/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.command=ssh+[shlex.join(['/usr/bin/python3','-c',RECEIVER,REMOTE,NAME])];self.process=None


def serialize(fit,identity):
    arrays={k:v for k,v in fit.items() if isinstance(v,np.ndarray)}
    meta=dict(identity=identity,fit={k:v for k,v in fit.items() if k not in arrays})
    stream=io.BytesIO();np.savez_compressed(stream,**arrays,meta_json=np.array(json.dumps(meta,sort_keys=True)))
    return stream.getvalue()


def remote_verify(refs):
    code=r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_past_quality_auxiliary_v1')
assert json.loads((r/'.owner.json').read_text())['experiment']=='european_past_quality_auxiliary_v1'
for v in p['refs']:
    f=r/'inputs'/(v['group']+'.npz')
    assert f.stat().st_size==v['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
print(json.dumps(dict(checkpoints_verified=len(p['refs']),bytes=sum(v['bytes'] for v in p['refs']),remote_science_executed=False)))
'''
    return leaf.previous.independent.read_remote(code,dict(root=REMOTE,refs=refs))


def beat(**kw):
    d=dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    (PRIVATE/'heartbeat.json').write_text(json.dumps(d)+'\n')
    with (PRIVATE/'events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
    print(json.dumps(d),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['register','pilot','run','verify']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    reg=registration();cfg=json.loads(CONFIG.read_text())
    if args.phase=='register':once(PUBLIC/'registration.json',reg);print(json.dumps(dict(registered=True)));return
    assert reg==json.loads((PUBLIC/'registration.json').read_text());parent.base.inter.committed(PUBLIC/'registration.json')
    if args.phase!='verify':assert not (PUBLIC/'complete.json').exists()
    PRIVATE.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    parent.core.torch.set_num_threads(cfg['cpu_threads']);parent.core.torch.set_num_interop_threads(1)
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        storage=remote_storage();beat(state='remote_storage_verified',**{k:v for k,v in storage.items() if k!='existing'})
        if storage['existing'] and not (args.resume or args.phase=='verify'):raise RuntimeError('Existing checkpoints require --resume')
        docs,cal=parent.parent.docs(),parent.docs()
        _,_,data,jobs,oid,_,_,_=parent.inner.old.load();q,provenance=past_quality(data)
        groups=[];refs=[];cprefs=[];fit_seconds=readout_seconds=0.;checks=size=cpbytes=0;pilot=args.phase=='pilot'
        with Stream() as stream:
            for c in parent.parent.contexts(data,jobs,oid):
                for site in parent.inner.sources(c):
                    group=c['name']+'_fit_'+site
                    at,ids,x,env,y,_,upstream=parent.inner.training_arrays(c,data,site)
                    tr,val,partition=parent.forest.parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
                    tid,vid=ids[tr],ids[val];h=parent.base.inter.array_hash
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start<cfg['hard_runtime_limit_seconds']
                        old,ca=docs[group,seed],cal[group,seed];state=joblib.load(ROOT/old['checkpoint']['path'])
                        assert state['identity']['upstream']==upstream and old['partition']==partition
                        pred={};trained={};cps={};diagnostic={};arms=[]
                        for arm in api.ARMS:
                            before=time.monotonic();beat(state='fitting_auxiliary',group=group,head_seed=seed,arm=arm)
                            kw=dict(arm=arm,seed=seed,penalty=cfg['ridge_penalty'])
                            fitted=api.fit(state,x[tr],env[tr],y[tr],q[tid],data['sites'][tid],data['recordings'][tid],data['frames'][tid],**kw)
                            replay=api.fit(state,x[tr],env[tr],y[tr],q[tid],data['sites'][tid],data['recordings'][tid],data['frames'][tid],**kw)
                            for k in fitted:
                                if isinstance(fitted[k],np.ndarray):np.testing.assert_array_equal(fitted[k],replay[k])
                                else:assert fitted[k]==replay[k]
                            fit_seconds+=time.monotonic()-before
                            identity=dict(group=group,source=site,head_seed=seed,arm=arm,partition=partition,
                                checkpoint=old['checkpoint'],training_ids_hash=h(tid),registration_sha256=sha(PUBLIC/'registration.json'))
                            payload=serialize(fitted,identity);assert payload==serialize(replay,identity)
                            cpbytes+=len(payload);assert cpbytes<=cfg['remote_checkpoint_cap_bytes']
                            name=group+'_head'+str(seed)+'_'+arm;cp=stream.send(name,payload);cprefs.append(cp);cps[arm]=cp
                            before=time.monotonic()
                            original,new,support,diag=api.predict(state,fitted,x[val],env[val],q[vid])
                            with np.load(io.BytesIO(payload),allow_pickle=False) as z:loaded={k:z[k] for k in z.files if k!='meta_json'}
                            again=api.predict(state,loaded,x[val],env[val],q[vid])
                            for a,b in zip((original,new,support),again[:3]):np.testing.assert_array_equal(a,b)
                            assert diag==again[3]
                            assert h(original)==old['validation']['prediction_hashes']['forest']==ca['identity']['prediction_hash']
                            if 'original' in pred:np.testing.assert_array_equal(pred['original'],original)
                            pred['original']=original;pred[arm]=new;diagnostic[arm]=diag
                            trained[arm]={k:v for k,v in fitted.items() if not isinstance(v,np.ndarray)}
                            arms.append(fitted);readout_seconds+=time.monotonic()-before
                        np.testing.assert_array_equal(arms[0]['quality_mean'],arms[1]['quality_mean'])
                        np.testing.assert_array_equal(arms[0]['quality_std'],arms[1]['quality_std'])
                        assert h(y[val])==ca['identity']['target_hash'] and h(env[val])==ca['identity']['envelope_hash']
                        assert h(vid)==old['validation']['ids_hash']==ca['identity']['validation_ids_hash']
                        kwargs=dict(state=state,predictions=pred,y=y[val],env=env[val],moving=c['moving'][at][val],support=support,
                            sites=data['sites'][vid],rec=data['recordings'][vid],frames=data['frames'][vid],ids=vid)
                        result,actions=api.evaluate(**kwargs);again,_=api.evaluate(**kwargs);assert result==again
                        assert h(actions['original'])==old['validation']['action_hashes']['forest']
                        for name,a in actions.items():checks+=leaf.previous.check_scalars(result['policies'][name],leaf.previous.independent.scalar_bounds(y[val],a,env[val]))
                        checks+=leaf.previous.check_scalars(result['policies']['original'],old['validation']['completion_screen'])
                        item=dict(group=group,source=site,head_seed=seed,partition=partition,training=trained,checkpoints=cps,
                            result=result,diagnostic=diagnostic,registration_sha256=sha(PUBLIC/'registration.json'),
                            input_source='cached_verified',result_source='fresh_run_train_only_quality_and_placebo',
                            prediction_hashes={k:h(v) for k,v in pred.items()},action_hashes={k:h(a) for k,a in actions.items()},
                            validation_ids_hash=h(vid),training_ids_hash=h(tid),past_quality_hash=h(q[vid]))
                        path=PUBLIC/'groups'/(group+'_head'+str(seed)+'.json');groups.append(item)
                        if not pilot:
                            if path.exists() and not (args.resume or args.phase=='verify'):raise RuntimeError('Existing group requires --resume')
                            once(path,item);size+=path.stat().st_size;assert size<cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                        beat(state='group_complete',groups=len(groups),seconds=time.monotonic()-start,
                            paired_fit_and_replay_seconds=fit_seconds,inference_seconds=readout_seconds,checkpoint_bytes=cpbytes)
                        if pilot:break
                    if pilot:break
                if pilot:break
        verified=remote_verify(cprefs)
        runtime=dict(pid=os.getpid(),seconds=time.monotonic()-start,paired_fit_and_replay_seconds=fit_seconds,
            inference_seconds=readout_seconds,scalar_checks=checks,checkpoint_bytes=cpbytes,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            available_disk_bytes=shutil.disk_usage(ROOT).free,cache_reserve_bytes=cfg['cache_reserve_bytes'],
            native_architecture=platform.machine(),cpu_threads=4,num_workers=0,new_HPC_jobs=0,
            remote_verification=verified,exact_fit_replay=True,exact_inference_replay=True,exact_readout_replay=True,
            local_numeric_cache=False,groups=refs)
        if pilot:
            runtime['first_group_projection_not_runtime_bound_seconds']=(fit_seconds+readout_seconds)*72
            once(PUBLIC/'pilot.json',runtime)
        else:
            assert len(groups)==cfg['source_heads'];summary=api.summarize(groups,cfg)
            once(PUBLIC/'summary.json',summary);once(PUBLIC/'feature_provenance.json',provenance)
            once(PUBLIC/'checkpoint_manifest.json',dict(remote_root=REMOTE,checkpoints=cprefs))
            if args.phase=='verify':once(PRIVATE/'additional_replays'/(str(time.time_ns())+'.json'),dict(runtime,summary_sha256=sha(PUBLIC/'summary.json')))
            else:once(PUBLIC/'complete.json',dict(runtime,summary_sha256=sha(PUBLIC/'summary.json')))
            print(json.dumps(summary,indent=2))
        beat(state='complete',phase=args.phase,seconds=time.monotonic()-start)


if __name__=='__main__':main()
