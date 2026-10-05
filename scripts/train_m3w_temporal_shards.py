"""Run four registered disjoint TRAIN partitions, or verify their completed join."""
import argparse
from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess

from src.world_model import m3w_temporal_sharding as execution


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value: raise ValueError('Immutable receipt changed')
    else:
        with path.open('x') as f: f.write(json.dumps(value, indent=2)+'\n')


def verify(root):
    root = root.resolve()
    if root.parent != Path('/users/k24101830/m3w').resolve() or root.name != execution.NAME:
        raise ValueError('Owned M3W experiment required')
    owner = json.loads((root/'.owner.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    manifest = json.loads((root/'train_input_manifest.json').read_text())
    regpath = root/'parallel_execution_registration.json'
    reg = json.loads(regpath.read_text())
    if (owner['experiment'] != execution.NAME
            or manifest['registration_sha256'] != owner['registration_sha256']
            or reg['original_registration_sha256'] != owner['registration_sha256']
            or execution.sha(root/'train_input_manifest.json') != reg['manifest_sha256']
            or execution.sha(root/'config.json') != manifest['config_sha256']
            or manifest['config_sha256'] != reg['config_sha256']
            or manifest['input_role'] != 'source_TRAIN_only'
            or manifest['validation_rows_transferred'] is not False
            or manifest['independent_roles_read'] is not False
            or len(manifest['heads']) != cfg['source_heads']):
        raise ValueError('Unchanged frozen TRAIN manifest/config/roles required')
    for rel, digest in {**manifest['code_bindings'], **reg['execution_bindings']}.items():
        execution.checked_file(root/'code', rel, digest, Path('.'))
    parts = execution.partition(manifest)
    if reg['assignment'] != [sorted(execution.fit_keys(p)) for p in parts]:
        raise ValueError('Frozen disjoint assignment required')
    pilot = root/execution.PUBLIC/'pilot.json'
    if execution.sha(pilot) != reg['pilot_sha256']:
        raise ValueError('Original successful pilot must remain unchanged')
    if execution.admission(json.loads(pilot.read_text()), cfg) != reg['admission']:
        raise ValueError('Registered per-shard resource gate failed')
    return root, cfg, manifest, reg, execution.sha(regpath)


def train(root, cfg, manifest, reg_hash, shard, resume):
    # Verify ownership and code first; numerical libraries load only on compute.
    from scripts import train_m3w_temporal_auxiliary_portable as port
    run = port.run
    array_job = os.environ['SLURM_ARRAY_JOB_ID']
    if int(os.environ['SLURM_ARRAY_TASK_ID']) != shard or shard not in range(4):
        raise ValueError('Exact scheduled shard required')
    submission = json.loads((root/'parallel_submission.json').read_text())
    if submission['job_id'] != array_job or submission['execution_registration_sha256'] != reg_hash:
        raise ValueError('Registered submission lineage required')
    private, public = root/execution.PRIVATE, root/execution.PUBLIC
    private.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    run.ROOT, run.PRIVATE, run.PUBLIC = root, private, public
    run.storage = lambda config: port.quota_storage(root, private, config)
    run.api.torch.set_num_threads(cfg['cpu_threads']); run.api.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    part = execution.partition(manifest)[shard]
    local = {**cfg, 'source_heads': len(part['heads']), 'neural_fits': 3*len(part['heads'])}
    with (private/'process.lock').open('a') as common, (private/('shard'+str(shard)+'.lock')).open('a') as lock:
        fcntl.flock(common, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (public/'training_freeze.json').exists() or (public/'training_shards'/('shard'+str(shard)+'.json')).exists():
            raise ValueError('Completed training is immutable; do not refit')
        with execution.scoped_execution(run, private, public, shard, array_job, reg_hash, lambda: run.guard(cfg)):
            run.guard(cfg)
            run.beat(state='shard_started', job_id=os.environ['SLURM_JOB_ID'],
                     identities=len(part['heads']), fits=local['neural_fits'], steps=cfg['head_training']['steps'],
                     optimizer_unchanged=True, validation_labels_scored=False)
            with closing(port.verified_iterator(root, part)) as iterator:
                run.train(local, iterator, pilot=False, resume=resume)
            run.beat(state='shard_complete', fits=local['neural_fits'])


def join(root, cfg, manifest, reg_hash):
    submission = json.loads((root/'parallel_submission.json').read_text())
    canonical = json.loads((root/'train_submission.json').read_text())
    job = submission['job_id']
    if (canonical['job_id'] != os.environ['SLURM_JOB_ID'] or canonical['array_job_id'] != job
            or canonical['execution_registration_sha256'] != reg_hash):
        raise ValueError('Only the registered complete-array join may release readout')
    p = subprocess.run(['sacct', '-j', job, '-X', '--noheader', '--parsable2',
                        '--format=JobID,State,ExitCode'], capture_output=True, text=True, timeout=30)
    if p.returncode: raise RuntimeError(p.stderr)
    accounting = execution.array_accounting(p.stdout, job)
    private, public = root/execution.PRIVATE, root/execution.PUBLIC
    with (private/'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs, envelopes = execution.verify_completed_shards(root, manifest, cfg, job, reg_hash)
        seconds = sum(json.loads((root/r['path']).read_text())['freeze']['seconds'] for r in envelopes)
        once(public/'training_freeze.json', dict(fits=sorted(refs, key=lambda r: r['path']),
            neural_fits=cfg['neural_fits'], seconds=seconds, seconds_scope='sum_of_four_shard_processes',
            validation_labels_scored=False, evaluation_status='not_run_separate_readout_required',
            new_forecasters=False, independent_roles_read=False, deployment_changed=False,
            parallel_execution=dict(array_job_id=job, join_job_id=os.environ['SLURM_JOB_ID'],
                execution_registration_sha256=reg_hash, accounting=accounting, shards=envelopes)))
        print(json.dumps(dict(complete_fits=len(refs), all_four_tasks_verified=True,
                              checkpoint_hashes_verified=True, validation_read=False)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['train', 'join'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated CREATE compute node required before numerical imports')
    root, cfg, manifest, _, reg_hash = verify(a.root)
    if a.phase == 'train': train(root, cfg, manifest, reg_hash, a.shard, a.resume)
    else: join(root, cfg, manifest, reg_hash)


if __name__ == '__main__': main()
