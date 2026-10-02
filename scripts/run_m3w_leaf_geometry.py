"""Local fixed-routing refits; remote owned storage only for small checkpoints."""
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
    raise RuntimeError('Use native arm64 before numerical imports')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from scripts import run_m3w_forest_projection as previous
from scripts.manage_m3w_boundary_diagnostic import RECEIVER
from scripts.m3w_packet_stream import PacketStream
from src.world_model import m3w_leaf_geometry as api

parent = previous.parent
sha, once = previous.sha, previous.once
NAME = 'european_leaf_geometry_v1'
PUBLIC = previous.PUBLIC.parent/NAME
PRIVATE = previous.PRIVATE.parent/NAME
CONFIG = ROOT/'configs'/('m3w_'+NAME+'.json')
REMOTE = '/users/k24101830/m3w/'+NAME


def registration():
    assert json.loads((previous.PUBLIC/'registration.json').read_text()) == previous.registration()
    prior = json.loads((previous.PUBLIC/'verification.json').read_text())
    assert sha(previous.PUBLIC/'complete.json') == prior['complete_sha256']
    paths = parent.forest.closure(ROOT, ['scripts.run_m3w_leaf_geometry'])
    paths += [CONFIG, PUBLIC/'protocol.md', ROOT/'tests/test_m3w_leaf_geometry.py']
    return dict(bindings={str(p.relative_to(ROOT)):sha(p) for p in paths},
        parent_verification_sha256=sha(previous.PUBLIC/'verification.json'),
        source_heads=72, new_tree_splits=0, new_neural_updates=0, independent_roles_read=False)


def prepare_remote(reg):
    code = r'''
import hashlib,json,pathlib,shutil,subprocess,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_leaf_geometry_v1')
assert json.loads((r.parent/'.m3w_owner.json').read_text())['project']=='M3W'
q=subprocess.run(['squeue','--me','-h','-o','%i|%j|%T'],capture_output=True,text=True,timeout=25)
assert q.returncode==0 and 'm3w_leaf_geometry' not in q.stdout
assert shutil.disk_usage(r.parent).free>p['cap']
r.mkdir(exist_ok=True)
def once(path,text):
    if path.exists():assert path.read_text()==text
    else:
        with path.open('x') as f:f.write(text)
once(r/'.owner.json',json.dumps(dict(experiment=p['name'],project='M3W'))+'\n')
once(r/'registration.json',p['registration']);once(r/'config.json',p['config'])
refs=[]
for f in sorted((r/'inputs').glob('*.npz')):
    refs.append(dict(group=f.stem,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
print(json.dumps(dict(existing=refs,owned_root_verified=True,personal_quota='unknown',
    remote_science_executed=False,m3w_jobs=[l for l in q.stdout.splitlines() if 'm3w' in l.lower()])))
'''
    cfg = json.loads(CONFIG.read_text())
    return previous.independent.read_remote(code, dict(root=REMOTE, name=NAME,
        cap=cfg['remote_checkpoint_cap_bytes'], registration=(PUBLIC/'registration.json').read_text(), config=CONFIG.read_text()))


class CheckpointStream(PacketStream):
    def __init__(self):
        handoff = ROOT/'data/stage_cvpr2027_experiments/create_handoff_20260923'
        assert sha(handoff/'compute_handoff_20260927.json') == 'f78d85acb9d51270747e67b7f3533b1bae023d66874d0fa4ef03fdb6f266d1b9'
        ssh = json.loads((handoff/'observations.json').read_text())['ssh_arguments']
        assert 'BatchMode=yes' in ssh and 'StrictHostKeyChecking=yes' in ssh
        self.command = ssh+[shlex.join(['/usr/bin/python3','-c',RECEIVER,REMOTE,NAME])]
        self.process = None


def serialize(fitted, identity):
    b = io.BytesIO()
    metadata = dict(identity=identity, training={k:v for k,v in fitted.items() if not isinstance(v,np.ndarray)})
    np.savez_compressed(b, values=fitted['values'], offsets=fitted['offsets'], stats=fitted['stats'],
                        meta_json=np.array(json.dumps(metadata,sort_keys=True)))
    return b.getvalue()


def remote_verify(refs):
    code = r'''
import hashlib,json,pathlib,sys
p=json.loads(sys.stdin.read());r=pathlib.Path(p['root'])
assert r==pathlib.Path('/users/k24101830/m3w/european_leaf_geometry_v1')
assert json.loads((r/'.owner.json').read_text())['experiment']=='european_leaf_geometry_v1'
for v in p['refs']:
    f=r/'inputs'/(v['group']+'.npz')
    assert f.stat().st_size==v['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
print(json.dumps(dict(checkpoints_verified=len(p['refs']),bytes=sum(v['bytes'] for v in p['refs']),remote_science_executed=False)))
'''
    return previous.independent.read_remote(code, dict(root=REMOTE, refs=refs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['register','pilot','run'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    reg = registration()
    if args.phase == 'register':
        once(PUBLIC/'registration.json',reg)
        print(json.dumps(dict(registration_sha256=sha(PUBLIC/'registration.json'))))
        return
    assert reg == json.loads((PUBLIC/'registration.json').read_text())
    parent.base.inter.committed(PUBLIC/'registration.json')
    cfg = json.loads(CONFIG.read_text())
    assert cfg['risk_budget'] == .02 and not cfg['independent_roles_read'] and not cfg['threshold_search']
    assert not (PUBLIC/'complete.json').exists(), 'Complete result is immutable'
    parent.core.torch.set_num_threads(cfg['cpu_threads']); parent.core.torch.set_num_interop_threads(1)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    with (PRIVATE/'process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = prepare_remote(reg)
        print(json.dumps(dict(phase='remote_storage_ready',**preflight)),flush=True)
        docs, cal = parent.parent.docs(), parent.docs()
        _,_,data,jobs,oid,_,_,_ = parent.inner.old.load()
        groups, refs, remote_refs = [], [], []
        size = checkpoint_size = checks = 0
        fit_seconds = inference_seconds = 0.
        pilot_done = False
        with CheckpointStream() as stream:
            for c in parent.parent.contexts(data,jobs,oid):
                for site in parent.inner.sources(c):
                    group = c['name']+'_fit_'+site
                    at,ids,x,env,y,_,upstream = parent.inner.training_arrays(c,data,site)
                    tr,val,partition = parent.forest.parent.api.source_partition(data['recordings'][ids],data['frames'][ids],site)
                    for seed in cfg['head_seeds']:
                        assert time.monotonic()-start < cfg['hard_runtime_limit_seconds'], '12h cap; preserve resume state'
                        old, old_cal = docs[group,seed], cal[group,seed]
                        state = joblib.load(ROOT/old['checkpoint']['path'])
                        assert state['identity']['upstream'] == upstream and old['partition'] == partition
                        tid,vid = ids[tr],ids[val]
                        began = time.monotonic()
                        fitted = api.fit(state,x[tr],env[tr],y[tr],data['sites'][tid],data['recordings'][tid],data['frames'][tid])
                        replay = api.fit(state,x[tr],env[tr],y[tr],data['sites'][tid],data['recordings'][tid],data['frames'][tid])
                        for key in fitted:
                            if isinstance(fitted[key],np.ndarray):np.testing.assert_array_equal(fitted[key],replay[key])
                            else:assert fitted[key]==replay[key]
                        fit_seconds += time.monotonic()-began
                        h = parent.base.inter.array_hash
                        identity = dict(group=group,source=site,head_seed=seed,source_checkpoint=old['checkpoint'],
                            partition=partition,registration_sha256=sha(PUBLIC/'registration.json'),training_ids_hash=h(tid))
                        payload = serialize(fitted,identity)
                        assert payload == serialize(replay,identity)
                        checkpoint_size += len(payload)
                        assert checkpoint_size <= cfg['remote_checkpoint_cap_bytes']
                        name = group+'_head'+str(seed)
                        cp = stream.send(name,payload); remote_refs.append(cp)
                        # Decode the checkpoint bytes independently of the fit object's lifetime.
                        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
                            loaded = {key:z[key].copy() for key in ('values','offsets','stats')}
                        began = time.monotonic()
                        p,q,support,diag = api.predict(state,fitted,x[val],env[val])
                        again = api.predict(state,loaded,x[val],env[val])
                        for a,b in zip((p,q,support),again[:3]):np.testing.assert_array_equal(a,b)
                        for key in diag:np.testing.assert_array_equal(diag[key],again[3][key])
                        assert h(p)==old['validation']['prediction_hashes']['forest']==old_cal['identity']['prediction_hash']
                        assert h(y[val])==old_cal['identity']['target_hash']
                        assert h(env[val])==old_cal['identity']['envelope_hash']
                        assert h(vid)==old['validation']['ids_hash']
                        kwargs = dict(state=state,y=y[val],env=env[val],moving=c['moving'][at][val],support=support,
                            sites=data['sites'][vid],rec=data['recordings'][vid],frames=data['frames'][vid],ids=vid,diagnostic=diag)
                        result,actions = api.evaluate(old=p,new=q,**kwargs)
                        repeat,_ = api.evaluate(old=again[0],new=again[1],**kwargs)
                        assert result == repeat
                        assert h(actions['original'])==old['validation']['action_hashes']['forest']
                        for key,action in actions.items():
                            checks += previous.check_scalars(result['policies'][key],previous.independent.scalar_bounds(y[val],action,env[val]))
                        checks += previous.check_scalars(result['policies']['original'],old['validation']['completion_screen'])
                        np.testing.assert_allclose(result['scores']['original'],old['validation']['scores']['forest'],rtol=1e-12)
                        inference_seconds += time.monotonic()-began
                        item = dict(**identity,checkpoint=cp,remote_checkpoint_path=REMOTE+'/inputs/'+name+'.npz',
                            input_source='cached_verified',result_source='fresh_run_train_only_leaf_refit',
                            learned_values_hash=h(fitted['values']),
                            inference_hashes=dict(original=h(p),refit=h(q),ids=h(vid),targets=h(y[val])),
                            action_hashes={k:h(a) for k,a in actions.items()},
                            training={k:v for k,v in fitted.items() if not isinstance(v,np.ndarray)},result=result)
                        path = PUBLIC/'groups'/(name+'.json')
                        if args.phase == 'run':
                            if path.exists() and not args.resume:raise RuntimeError('Use explicit --resume')
                            once(path,item);size+=path.stat().st_size
                            assert size<cfg['aggregate_output_cap_bytes']
                            refs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                        groups.append(item)
                        beat = dict(pid=os.getpid(),phase=args.phase,groups=len(groups),last=name,
                            seconds=time.monotonic()-start,fit_and_refit_seconds=fit_seconds,
                            inference_and_readout_seconds=inference_seconds,scalar_checks=checks,
                            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
                        (PRIVATE/'heartbeat.json').write_text(json.dumps(beat)+'\n')
                        with (PRIVATE/'events.jsonl').open('a') as log:log.write(json.dumps(beat)+'\n')
                        print(json.dumps(beat),flush=True)
                        if args.phase == 'pilot':pilot_done=True;break
                    if pilot_done:break
                if pilot_done:break
        verified = remote_verify(remote_refs)
        receipt = dict(beat,remote_verification=verified,checkpoint_bytes=checkpoint_size,
            exact_refit_replay=True,exact_inference_replay=True,exact_readout_replay=True,
            local_numeric_cache=False,remote_science_executed=False,new_scheduler_job=False,
            cpu_threads=4,num_workers=0,architecture=platform.machine(),torch=parent.core.torch.__version__,
            disk_free_bytes=shutil.disk_usage(ROOT).free,cache_reserve_bytes=cfg['cache_reserve_bytes'])
        if args.phase == 'pilot':
            receipt['single_group_projection_not_an_upper_bound_seconds'] = (fit_seconds+inference_seconds)*72
            receipt['checkpoint_projection_not_an_upper_bound_bytes'] = checkpoint_size*72
            once(PUBLIC/'pilot.json',receipt)
        else:
            assert len(groups)==cfg['source_heads']
            summary = api.summarize(groups,cfg)
            once(PUBLIC/'summary.json',summary)
            once(PUBLIC/'checkpoint_manifest.json',dict(remote_root=REMOTE,checkpoints=remote_refs))
            once(PUBLIC/'complete.json',dict(receipt,groups=refs,summary_sha256=sha(PUBLIC/'summary.json')))
            print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
