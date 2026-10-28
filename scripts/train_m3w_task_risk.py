"""Allocated TRAIN-only execution for the frozen decision-objective comparison."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time

NAME = 'european_task_risk_v1'
PARENT = 'european_temporal_auxiliary_v1'
PUBLIC = Path('outputs/publication_readiness_2026_09')/NAME
PRIVATE = Path('data/stage_cvpr2027_experiments')/NAME


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text() != raw:
            raise ValueError('Preserve conflicting receipt: '+str(path))
    else:
        with path.open('x') as f:
            f.write(raw)


def verify(root):
    root = root.resolve(); source = root.parent/PARENT
    if root != Path('/users/k24101830/m3w')/NAME:
        raise ValueError('Owned experiment only')
    reg = json.loads((root/'registration.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    if json.loads((root/'.owner.json').read_text()) != dict(experiment=NAME, registration_sha256=sha(root/'registration.json')):
        raise ValueError('Wrong owner')
    for rel, digest in reg['bindings'].items():
        p = (root/'code'/rel).resolve()
        if not p.is_relative_to((root/'code').resolve()) or sha(p) != digest:
            raise ValueError('Changed bound source: '+rel)
    if sha(root/'config.json') != reg['config_sha256']:
        raise ValueError('Changed experiment settings')
    if (sha(source/'train_input_manifest.json') != reg['input_manifest_sha256']
            or json.loads((source/'.owner.json').read_text())['registration_sha256'] != reg['parent_registration_sha256']):
        raise ValueError('Changed source ownership or TRAIN manifest')
    manifest = json.loads((source/'train_input_manifest.json').read_text())
    if (manifest['input_role'] != 'source_TRAIN_only' or manifest['validation_rows_transferred']
            or manifest['independent_roles_read'] or len(manifest['heads']) != 72
            or cfg['head_training'] != json.loads((source/'config.json').read_text())['head_training']):
        raise ValueError('Frozen TRAIN-only scope required')
    return root, source, cfg, reg, manifest


def quota(root, cfg):
    home = Path('/users/k24101830')
    free = int(os.getxattr(home, 'ceph.quota.max_bytes'))-int(os.getxattr(home, 'ceph.dir.rbytes'))
    used = sum(p.stat().st_size for p in (root/PRIVATE).rglob('*.pt.gz'))
    remaining = cfg['checkpoint_cap_bytes']-used
    if remaining < 0 or free < cfg['disk_reserve_bytes']+remaining+cfg['atomic_checkpoint_headroom_bytes']:
        raise OSError('Retain quota reserve and checkpoint cap')
    return dict(quota_free_bytes=free, checkpoint_bytes=used, cap_bytes=cfg['checkpoint_cap_bytes'])


def train(root, source, cfg, reg, manifest, phase, resume):
    from scripts.train_m3w_temporal_auxiliary_portable import unpack
    from src.world_model import m3w_task_risk_training as api
    from src.world_model.m3w_false_safe_diagnostic import moving_from_features
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.torch.set_num_threads(cfg['cpu_threads']); api.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic snapshot')))
    home, public = root/PRIVATE, root/PUBLIC
    home.mkdir(parents=True, exist_ok=True)
    began = time.monotonic(); reg_hash = sha(root/'registration.json')
    submission = json.loads((root/(phase+'_submission.json')).read_text())
    if submission['job_id'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('Only the registered allocation may execute')
    finish = public/('pilot.json' if phase == 'pilot' else 'training_freeze.json')
    if finish.exists():
        raise ValueError('Completed phase is immutable')
    if phase == 'train' and not json.loads((public/'pilot.json').read_text())['engineering_pass']:
        raise ValueError('Real TRAIN pilot required')
    movements = {r['group']:r for r in reg['movement']['groups']}

    def heartbeat(**kw):
        if time.monotonic()-began > cfg['hard_runtime_limit_seconds']:
            raise TimeoutError('Resume saved state; no reduced scientific target')
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   job_id=os.environ['SLURM_JOB_ID'], phase=phase, **kw)
        tmp = home/(phase+'_heartbeat.tmp'); tmp.write_text(json.dumps(row, allow_nan=False)+'\n')
        os.replace(tmp, home/(phase+'_heartbeat.json'))
        with (home/(phase+'_events.jsonl')).open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def guard():
        quota(root, cfg)

    with (home/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard(); refs = []; previous_group = None
        for ref in (manifest['heads'][:1] if phase == 'pilot' else manifest['heads']):
            identity, group = ref['identity'], ref['group']
            if previous_group != group:
                heartbeat(state='loading_frozen_TRAIN_packet', group=group)
                p = source/'inputs'/(group+'.npz')
                raw = p.read_bytes()
                if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
                    raise ValueError('Changed TRAIN packet')
                args = unpack(raw); del raw
                moving = moving_from_features(args[0])
                _, support = api.parent.forest.causal_inputs(args[0], args[1], args[-1])
                eligible = moving & support
                if (api.parent.forest.fingerprint(moving) != movements[group]['moving_sha256']
                        or ref['sha256'] != movements[group]['packet_sha256']):
                    raise ValueError('Causal eligibility lineage changed')
                previous_group = group
            states = []
            for arm in api.ARMS:
                key = group+'_head'+str(identity['seed'])+'_'+arm
                path = home/('pilot_heads' if phase == 'pilot' else 'heads')/key/'checkpoint.pt.gz'
                kw = dict(eligible=eligible, arm=arm, settings=cfg['head_training'],
                    objective_config=cfg['objective'], seed=identity['seed'], identity=identity,
                    experiment_sha256=reg_hash, path=path, heartbeat=heartbeat, checkpoint_guard=guard)
                with frozen_preprocess(api.core, args[-1]):
                    if phase == 'pilot' and arm == 'task_risk' and not path.exists():
                        api.fit(*args, **kw, stop_at=50)
                    state = api.fit(*args, **kw, resume=path.exists() and (resume or phase == 'pilot'),
                                    stop_at=cfg['pilot_updates'] if phase == 'pilot' else None)
                states.append(state)
                if phase == 'train':
                    doc = dict(identity=identity, arm=arm, step=state['step'], trace=state['trace'],
                        dual=state['dual'].tolist(), input_hashes=state['input_hashes'],
                        eligibility_sha256=state['eligibility_sha256'], draw_hash=state['draw_hash'],
                        registration_sha256=reg_hash, checkpoint=dict(path=str(path.relative_to(root)),
                            sha256=sha(path), bytes=path.stat().st_size),
                        validation_scored=False, independent_roles_read=False, new_forecaster=False,
                        risk_calibration_claim=False, result_source='fresh_run_TRAIN_fit')
                    output = public/'fits'/(key+'.json'); once(output, doc)
                    refs.append(dict(path=str(output.relative_to(root)), sha256=sha(output)))
                    heartbeat(state='fit_complete', fits=len(refs), total=216, key=key)
            api.assert_matched(states)
            if phase == 'pilot':
                with frozen_preprocess(api.core, args[-1]):
                    replay = home/'pilot_direct.pt.gz'
                    direct = api.fit(*args, **{**kw,'path':replay}, resume=resume and replay.exists(), stop_at=100)
                    original = home/'pilot_original.pt.gz'
                    control = api.parent.fit(*args, arm='none', settings=cfg['head_training'],
                        seed=identity['seed'], identity=identity, path=original, heartbeat=heartbeat,
                        resume=resume and original.exists(), stop_at=100, checkpoint_guard=guard)
                for k in direct:
                    if k != 'seconds': api.core.exact(direct[k], states[-1][k])
                api.assert_original_control(states[0], control)
                estimate = sum(s['seconds'] for s in states)/100*2000*72*1.5
                rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
                value = dict(result_source='fresh_run_real_TRAIN_pilot', identity=identity,
                    job_id=os.environ['SLURM_JOB_ID'], registration_sha256=reg_hash,
                    updates_per_arm=100, actual_updates_including_replays=500,
                    exact_interrupted_resume=True, original_control_exact=True, matched_draws=True,
                    peak_RSS_bytes=rss, estimated_full_seconds_with_1_5_margin=estimate,
                    engineering_pass=estimate < cfg['hard_runtime_limit_seconds'] and rss < 7*2**30,
                    torch=api.torch.__version__, numpy=api.np.__version__, cpu_threads=4, workers=0,
                    monitor={s['arm']:dict(initial=s['trace'][0], final=s['trace'][-1]) for s in states},
                    seconds=time.monotonic()-began, quota=quota(root,cfg),
                    validation_scored=False, independent_roles_read=False,
                    scientific_lift_established=False, deployment_changed=False)
                once(finish, value); heartbeat(state='pilot_complete', engineering_pass=value['engineering_pass'])
        if phase == 'train':
            if len(refs) != 216 or len({r['path'] for r in refs}) != 216:
                raise ValueError('All 216 fixed finals required')
            once(finish, dict(result_source='fresh_run_TRAIN_fit_grid', fits=refs, neural_fits=216,
                source_heads=72, registration_sha256=reg_hash, job_id=os.environ['SLURM_JOB_ID'],
                seconds=time.monotonic()-began, validation_scored=False, independent_roles_read=False,
                new_forecasters=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False))
            heartbeat(state='all216_frozen_before_readout')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['pilot','train'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    train(*verify(a.root), a.phase, a.resume)


if __name__ == '__main__':
    main()
