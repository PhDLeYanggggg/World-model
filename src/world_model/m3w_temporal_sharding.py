"""Execution-only partitioning and fail-closed aggregation of frozen TRAIN fits."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

NAME = 'european_temporal_auxiliary_v1'
PUBLIC = Path('outputs/publication_readiness_2026_09')/NAME
PRIVATE = Path('data/stage_cvpr2027_experiments')/NAME
ARMS = ('none', 'rowmean', 'temporal')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_key(identity):
    return identity['group'], identity['source'], identity['seed']


def fit_key(identity, arm):
    return identity['group']+'_head'+str(identity['seed'])+'_'+arm


def fit_keys(manifest):
    return {fit_key(h['identity'], arm) for h in manifest['heads'] for arm in ARMS}


def partition(manifest, count=4):
    heads = manifest['heads']
    if count != 4 or not heads or len(heads) % count:
        raise ValueError('Four equal nonempty execution shards required')
    keys = [identity_key(h['identity']) for h in heads]
    if len(set(keys)) != len(keys) or any(h['group'] != h['identity']['group'] for h in heads):
        raise ValueError('Unique, aligned source/seed identities required')
    if len(fit_keys(manifest)) != 3*len(heads):
        raise ValueError('Fit path collision')
    return [{**manifest, 'heads': heads[i::count]} for i in range(count)]


def admission(pilot, cfg):
    estimate = pilot['estimated_full_seconds']/4
    if (cfg['source_heads'] != 72 or cfg['neural_fits'] != 216
            or not pilot['exact_interrupted_resume'] or not pilot['matched_draws']
            or not math.isfinite(estimate) or not 0 < estimate < cfg['hard_runtime_limit_seconds']
            or not 0 < pilot['peak_RSS_bytes'] < 14*2**30):
        raise ValueError('Registered pilot, memory and per-job time admission required')
    return dict(shards=4, identities_per_shard=18, fits_per_shard=54,
                estimated_seconds_per_shard=estimate, estimate_not_measurement=True,
                single_job_time_feasible=pilot['local_time_feasible'],
                per_job_time_feasible=True, all_216_fits_retained=True)


def array_accounting(text, array_job):
    expected = {array_job+'_'+str(i) for i in range(4)}
    found = {}
    for line in text.strip().splitlines():
        fields = line.strip().split('|')
        if len(fields) != 3 or fields[0] not in expected or fields[0] in found:
            raise ValueError('Exact four-task accounting required')
        if fields[1:] != ['COMPLETED', '0:0']:
            raise ValueError('All four training tasks must complete successfully')
        found[fields[0]] = '|'.join(fields[1:])
    if set(found) != expected: raise ValueError('Partial array cannot authorize readout')
    return found


@contextmanager
def scoped_execution(run, private, public, shard, array_job, registration_hash, guard):
    """Change only destinations and write serialization; never numerical fitting."""
    old_save, old_beat, old_once = run.api.core.save_checkpoint, run.beat, run.once
    home = private/'shards'/str(shard)
    home.mkdir(parents=True, exist_ok=True)
    def save(path, state):
        with (private/'checkpoint_write.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            guard()
            return old_save(path, state)
    def beat(**kw):
        row = dict(pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   array_job_id=array_job, shard=shard, **kw)
        temp = home/'heartbeat.tmp'
        temp.write_text(json.dumps(row)+'\n'); os.replace(temp, home/'heartbeat.json')
        with (home/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    def once(path, value):
        if Path(path) == public/'training_freeze.json':
            return old_once(public/'training_shards'/('shard'+str(shard)+'.json'), dict(
                shard=shard, array_job_id=array_job, task_job_id=os.environ.get('SLURM_JOB_ID'),
                execution_registration_sha256=registration_hash, freeze=value))
        return old_once(path, value)
    run.api.core.save_checkpoint, run.beat, run.once = save, beat, once
    try: yield
    finally: run.api.core.save_checkpoint, run.beat, run.once = old_save, old_beat, old_once


def checked_file(root, relative, expected, prefix):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.is_relative_to(prefix):
        raise ValueError('Path outside registered artifact scope')
    path = (root/p).resolve()
    if not path.is_relative_to((root/prefix).resolve()) or sha(path) != expected:
        raise ValueError('Artifact path/hash mismatch')
    return path


def verify_completed_shards(root, manifest, cfg, array_job, registration_hash):
    refs, envelopes, total_bytes = [], [], 0
    for i, part in enumerate(partition(manifest)):
        path = root/PUBLIC/'training_shards'/('shard'+str(i)+'.json')
        doc = json.loads(path.read_text()); freeze = doc['freeze']
        if (doc['shard'] != i or doc['array_job_id'] != array_job
                or doc['execution_registration_sha256'] != registration_hash
                or freeze['neural_fits'] != 3*len(part['heads'])
                or any(freeze[k] is not False for k in ('validation_labels_scored',
                    'independent_roles_read', 'deployment_changed', 'new_forecasters'))):
            raise ValueError('Unscored complete shard and exact execution lineage required')
        expected = {fit_key(h['identity'], a): (h['identity'], a) for h in part['heads'] for a in ARMS}
        found = set()
        for ref in freeze['fits']:
            p = checked_file(root, ref['path'], ref['sha256'], PUBLIC/'fits')
            row = json.loads(p.read_text()); key = fit_key(row['identity'], row['arm'])
            if (key not in expected or key in found or expected[key] != (row['identity'], row['arm'])
                    or row['step'] != cfg['head_training']['steps']
                    or row['validation_labels_scored'] is not False
                    or ref['path'] != str(PUBLIC/'fits'/(key+'.json'))):
                raise ValueError('Partial, duplicate, changed or unexpected fit')
            cp = row['checkpoint']
            if cp['path'] != str(PRIVATE/'heads'/key/'checkpoint.pt.gz'):
                raise ValueError('Checkpoint identity mismatch')
            p = checked_file(root, cp['path'], cp['sha256'], PRIVATE/'heads')
            if not 0 < p.stat().st_size == cp['bytes'] <= 2*2**20:
                raise ValueError('Checkpoint cap/size mismatch')
            total_bytes += cp['bytes']; found.add(key); refs.append(ref)
        if found != set(expected): raise ValueError('All assigned fits required')
        envelopes.append(dict(path=str(path.relative_to(root)), sha256=sha(path)))
    if len(refs) != cfg['neural_fits'] or total_bytes > cfg['checkpoint_cap_bytes']:
        raise ValueError('Complete grid within original checkpoint cap required')
    return refs, envelopes
