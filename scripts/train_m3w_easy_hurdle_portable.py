"""Run the unchanged registered risk-head API on hash-bound fitting-only packets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import platform

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.world_model import m3w_easy_hurdle as api
from scripts.run_m3w_native_forecast import array_hash
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(path):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    identity = json.loads(str(a['identity_json'])); pr = json.loads(str(a['preprocess_json']))
    for key in ('known', 'weights', 'mean', 'std'):
        pr[key] = a['pr_'+key]
    if not set(a['sites']) == set(identity['roles']['training_sites']):
        raise ValueError('Packet includes a non-fitting source')
    if set(a['sites']) & set(identity['roles']['held_sites']):
        raise ValueError('Held rows in training packet')
    if set(a) != {'x','u','y','sites','recordings','frames','env','ids','identity_json','preprocess_json',
                  'pr_known','pr_weights','pr_mean','pr_std'}:
        raise ValueError('Unregistered packet fields')
    assert array_hash(a['ids']) == identity['fitting_ids_hash']
    for key, field in [('x','x'), ('descriptors','u'), ('envelope','env'), ('targets','y')]:
        assert array_hash(a[field]) == identity['input_hashes'][key]
    return a, pr, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home', required=True); p.add_argument('--pilot', action='store_true')
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute node required')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    home = Path(a.home).resolve(); manifest = json.loads((home/'input_manifest.json').read_text())
    cfg = json.loads((home/'config.json').read_text())
    assert len(manifest['groups']) == 108 and cfg['groups'] == 108 and not cfg['independent_roles_read']
    for relative, expected in manifest['code_bindings'].items():
        assert digest(ROOT/relative) == expected, relative
    assert manifest['registration']['bindings']['src/world_model/m3w_easy_hurdle.py'] == digest(ROOT/'src/world_model/m3w_easy_hurdle.py')
    env = dict(torch=torch.__version__, numpy=np.__version__, python=sys.version, machine=platform.machine(),
               job_id=os.environ['SLURM_JOB_ID'], pid=os.getpid(), cpu_threads=4, interop_threads=1, workers=0)

    def beat(state, **kw):
        row = dict(state=state, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **env, **kw)
        (home/'heartbeat.json').write_text(json.dumps(row)+'\n')
        with (home/'events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    with (home/'train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic(); artifacts = []
        for group in manifest['groups']:
            path = home/'inputs'/(group['group']+'.npz'); assert digest(path) == group['sha256']
            arrays, pr, identity = unpack(path)
            states = {}
            for arm in api.ARMS:
                dest = home/'heads'/group['group']/arm/'checkpoint.pt.gz'
                states[arm] = api.fit(arrays['x'], arrays['u'], arrays['y'], arrays['sites'], arrays['recordings'],
                    arrays['frames'], arrays['env'], pr, arm=arm, seed=identity['seed'],
                    settings=cfg['head_training'], identity=identity, path=dest,
                    heartbeat=lambda **kw: beat(group=group['group'], arm=arm, **kw), resume=a.resume,
                    stop_at=cfg['pilot_updates'] if a.pilot else None)
                artifacts.append(dict(group=group['group'], arm=arm, path=str(dest.relative_to(home)), sha256=digest(dest)))
            api.assert_matched(states['marginal'], states['supervised'])
            beat('paired_fit_complete', group=group['group'])
            if a.pilot:
                break
        report = dict(environment=env, complete=not a.pilot, pilot=a.pilot, artifacts=artifacts,
            groups=len(artifacts)//2, seconds=time.monotonic()-start, manifest_sha256=digest(home/'input_manifest.json'),
            held_outcomes_used=False, independent_roles_read=False)
        out = home/('pilot.json' if a.pilot else 'training_complete.json')
        if out.exists():
            raise ValueError('Completed receipt already exists; verify it rather than overwrite')
        out.write_text(json.dumps(report, indent=2)+'\n'); beat('complete', pilot=a.pilot)


if __name__ == '__main__':
    main()
