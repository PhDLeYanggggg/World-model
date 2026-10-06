"""Allocated matched loss pilot/training over existing immutable TRAIN packets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

NAME = 'european_easy_harm_deviance_v1'
PARENT = 'european_temporal_auxiliary_v1'
PUBLIC = Path('outputs/publication_readiness_2026_09')/NAME
PRIVATE = Path('data/stage_cvpr2027_experiments')/NAME
OLD_PUBLIC = PUBLIC.parent/PARENT
OLD_PRIVATE = PRIVATE.parent/PARENT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('Immutable artifact conflict')
    else:
        temp = path.with_suffix(path.suffix+'.tmp')
        with temp.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        os.replace(temp, path)


def verify(root):
    root = root.resolve(); source = root.parent/PARENT
    if root.parent != Path('/users/k24101830/m3w').resolve() or root.name != NAME:
        raise ValueError('Owned experiment root required')
    reg = json.loads((root/'registration.json').read_text()); cfg = json.loads((root/'config.json').read_text())
    owner = json.loads((root/'.owner.json').read_text())
    if owner != dict(experiment=NAME, registration_sha256=sha(root/'registration.json')):
        raise ValueError('Owner/registration mismatch')
    if sha(root/'config.json') != reg['config_sha256'] or cfg['head_training'] != json.loads((source/'config.json').read_text())['head_training']:
        raise ValueError('Changed fixed training configuration')
    for rel, digest in reg['bindings'].items():
        p = (root/'code'/rel).resolve()
        if not p.is_relative_to(root/'code') or sha(p) != digest: raise ValueError('Changed code/protocol binding')
    if (sha(source/'train_input_manifest.json') != reg['input_manifest_sha256']
            or sha(source/OLD_PUBLIC/'training_freeze.json') != reg['parent_training_freeze_sha256']):
        raise ValueError('Changed immutable parent')
    manifest = json.loads((source/'train_input_manifest.json').read_text())
    if (manifest['input_role'] != 'source_TRAIN_only' or manifest['validation_rows_transferred']
            or manifest['independent_roles_read'] or len(manifest['heads']) != 72):
        raise ValueError('Original TRAIN roles and 72 identities required')
    return root, source, cfg, reg, manifest


def quota(root, source, cfg):
    home = Path('/users/k24101830')
    free = int(os.getxattr(home,'ceph.quota.max_bytes'))-int(os.getxattr(home,'ceph.dir.rbytes'))
    used = sum(p.stat().st_size for directory in (root/PRIVATE, source/OLD_PRIVATE) for p in directory.rglob('*.pt.gz'))
    cap = cfg['combined_parent_child_checkpoint_cap_bytes']; remaining = cap-used
    if remaining < 0 or free < cfg['disk_reserve_bytes']+remaining+cfg['atomic_checkpoint_headroom_bytes']:
        raise OSError('Combined checkpoint cap or storage reserve failed; preserve checkpoints')
    return dict(quota_free=free, combined_checkpoint_bytes=used, cap_bytes=cap)


def key(ref, arm):
    return ref['group']+'_head'+str(ref['identity']['seed'])+'_'+arm


def packet(source, ref):
    from scripts.train_m3w_temporal_auxiliary_portable import unpack
    path = source/'inputs'/(ref['group']+'.npz')
    if sha(path) != ref['sha256'] or path.stat().st_size != ref['bytes']: raise ValueError('Changed TRAIN packet')
    return unpack(path.read_bytes())


def train(root, source, cfg, reg, manifest, phase, shard):
    from src.world_model import m3w_easy_harm_deviance_training as api
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.torch.set_num_threads(4); api.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic checkpoint')))
    home = root/PRIVATE; home.mkdir(parents=True, exist_ok=True); public = root/PUBLIC
    tag = 'pilot' if phase == 'pilot' else 'shard'+str(shard)
    began = time.monotonic(); reg_hash = sha(root/'registration.json')
    expected_submission = 'pilot_submission.json' if phase == 'pilot' else 'train_submission.json'
    submission = json.loads((root/expected_submission).read_text())
    job = os.environ.get('SLURM_ARRAY_JOB_ID', os.environ['SLURM_JOB_ID'])
    assert submission['job_id'] == job and submission['registration_sha256'] == reg_hash
    if phase == 'train':
        pilot = json.loads((public/'pilot.json').read_text())
        if not pilot['engineering_pass']: raise ValueError('Real TRAIN pilot gate required')
        assert int(os.environ['SLURM_ARRAY_TASK_ID']) == shard
    refs = manifest['heads'][:1] if phase == 'pilot' else manifest['heads'][shard::4]
    def heartbeat(**kw):
        if time.monotonic()-began > cfg['hard_runtime_limit_seconds']: raise TimeoutError('Resume saved state')
        value = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            job_id=os.environ['SLURM_JOB_ID'], phase=phase, shard=shard, **kw)
        folder = home/tag; folder.mkdir(exist_ok=True)
        tmp = folder/'heartbeat.tmp'; tmp.write_text(json.dumps(value)+'\n'); os.replace(tmp, folder/'heartbeat.json')
        with (folder/'events.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        print(json.dumps(value), flush=True)
    def guard(): quota(root, source, cfg)
    old_save = api.core.save_checkpoint
    def serialized_save(path, state):
        with (home/'checkpoint_write.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX); guard(); old_save(path, state)
    api.core.save_checkpoint = serialized_save
    fit_refs = []; parent_docs = {}
    for entry in json.loads((source/OLD_PUBLIC/'training_freeze.json').read_text())['fits']:
        path = source/entry['path']; assert sha(path) == entry['sha256']
        doc = json.loads(path.read_text())
        if doc['arm'] == 'none': parent_docs[doc['identity']['group'],doc['identity']['seed']] = doc
    try:
        with (home/(tag+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for ref in refs:
                args = packet(source, ref); identity = ref['identity']; states = []
                for arm in api.ARMS:
                    path = home/'heads'/key(ref, arm)/'checkpoint.pt.gz'
                    kw = dict(arm=arm, settings=cfg['head_training'], seed=identity['seed'], identity=identity,
                        experiment_sha256=reg_hash, path=path, heartbeat=heartbeat, checkpoint_guard=guard)
                    with frozen_preprocess(api.core, args[-1]):
                        if phase == 'pilot' and arm == 'easy_deviance' and not path.exists():
                            api.fit(*args, **kw, stop_at=50)
                        state = api.fit(*args, **kw, resume=path.exists(), stop_at=100 if phase == 'pilot' else None)
                    states.append(state)
                    if phase == 'train':
                        if arm == 'quadratic':
                            doc = parent_docs[ref['group'],identity['seed']]; cp = source/doc['checkpoint']['path']
                            assert sha(cp) == doc['checkpoint']['sha256']
                            api.assert_original_control(state, api.core.read_checkpoint(cp))
                        doc = dict(identity=identity, arm=arm, step=state['step'], trace=state['trace'],
                            input_hashes=state['input_hashes'], draw_hash=state['draw_hash'],
                            checkpoint=dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size),
                            parent_control_exact=arm == 'quadratic', new_forecaster=False,
                            validation_scored=False, independent_roles_read=False)
                        out = public/'fits'/(key(ref,arm)+'.json'); once(out,doc)
                        fit_refs.append(dict(path=str(out.relative_to(root)),sha256=sha(out)))
                api.assert_matched(*states)
                if phase == 'pilot':
                    replay = home/'pilot_replay.pt.gz'; original = home/'original_control_pilot.pt.gz'
                    with frozen_preprocess(api.core, args[-1]):
                        direct = api.fit(*args, **{**kw,'path':replay}, resume=replay.exists(), stop_at=100)
                        control = api.parent.fit(*args, arm='none', settings=cfg['head_training'], seed=identity['seed'],
                            identity=identity,path=original,heartbeat=heartbeat, resume=original.exists(),stop_at=100,checkpoint_guard=guard)
                    for k in direct:
                        if k != 'seconds': api.core.exact(direct[k], states[1][k])
                    api.assert_original_control(states[0],control)
                    seconds = sum(s['seconds'] for s in states)
                    estimate = seconds/100*cfg['head_training']['steps']*72/4*1.5
                    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
                    report = dict(job_id=os.environ['SLURM_JOB_ID'], result_source='fresh_run_real_TRAIN_paired_pilot',
                        identity=identity, steps_per_arm=100, actual_updates_including_two_replays=400,
                        exact_interrupted_resume=True, original_control_exact=True, matched_draws=True,
                        estimate_seconds_per_shard_with_1_5_margin=estimate, peak_RSS_bytes=rss,
                        engineering_pass=estimate < cfg['hard_runtime_limit_seconds'] and rss < 14*2**30,
                        torch=api.torch.__version__, numpy=api.np.__version__, cpu_threads=4, workers=0,
                        gradients={s['arm']:api.gradient_probe(s,args) for s in states},
                        monitor={s['arm']:dict(initial=s['trace'][0]['monitor'],final=s['trace'][-1]['monitor']) for s in states},
                        validation_scored=False, independent_roles_read=False, scientific_lift_established=False,
                        quota=quota(root,source,cfg))
                    once(public/'pilot.json',report); heartbeat(state='pilot_complete',engineering_pass=report['engineering_pass'])
            if phase == 'train':
                assert len(fit_refs) == 36
                once(public/'shards'/(str(shard)+'.json'),dict(fits=fit_refs,registration_sha256=reg_hash,
                    job_id=os.environ['SLURM_JOB_ID'],array_job_id=job,shard=shard,validation_scored=False))
                heartbeat(state='shard_complete',fits=36)
    finally:
        api.core.save_checkpoint = old_save


def join(root, source, cfg, reg, manifest):
    from src.world_model.m3w_temporal_sharding import array_accounting
    sub = json.loads((root/'train_submission.json').read_text()); job = sub['job_id']
    joined = json.loads((root/'join_submission.json').read_text())
    assert joined['job_id'] == os.environ['SLURM_JOB_ID']
    q = subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2','--format=JobID,State,ExitCode'],capture_output=True,text=True,timeout=20)
    assert q.returncode == 0; accounting = array_accounting(q.stdout,job)
    refs = []; found = set()
    expected = {key(r,a) for r in manifest['heads'] for a in cfg['arms']}
    for i in range(4):
        doc = json.loads((root/PUBLIC/'shards'/(str(i)+'.json')).read_text())
        assert doc['array_job_id'] == job and doc['registration_sha256'] == sha(root/'registration.json')
        for ref in doc['fits']:
            path = (root/ref['path']).resolve(); assert path.is_relative_to(root/PUBLIC/'fits') and sha(path) == ref['sha256']
            row = json.loads(path.read_text()); name = path.stem
            assert name in expected and name not in found and row['step'] == 2000 and not row['validation_scored']
            if row['arm'] == 'quadratic': assert row['parent_control_exact']
            cp = (root/row['checkpoint']['path']).resolve()
            assert cp.is_relative_to(root/PRIVATE/'heads') and sha(cp) == row['checkpoint']['sha256']
            assert cp.stat().st_size == row['checkpoint']['bytes']
            refs.append(ref); found.add(name)
    assert found == expected and len(refs) == 144
    once(root/PUBLIC/'training_freeze.json',dict(fits=refs,neural_fits=144,source_heads=72,
        registration_sha256=sha(root/'registration.json'),accounting=accounting,
        original_quadratic_controls_exact=72,validation_scored=False,independent_roles_read=False,
        new_forecasters=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(state='all144_frozen_before_readout',new_forecasters=False)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['pilot','train','join'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--shard',type=int,choices=range(4),default=0)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Allocated compute before numerical imports')
    args=verify(a.root)
    if a.phase=='join': join(*args)
    else: train(*args,a.phase,a.shard)


if __name__=='__main__': main()
