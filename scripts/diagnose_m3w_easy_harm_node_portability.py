"""TRAIN-only exact replay on an alternate node; no new reference exceptions."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import time

from scripts import train_m3w_easy_harm_recovery_v2 as recovery
from scripts.diagnose_m3w_easy_harm_control import compare

run = recovery.original
NAME = 'node_portability_v1'


def summarize(rows, expected):
    names = [r['name'] for r in rows]
    if len(names) != 17 or len(set(names)) != 17 or set(names) != set(expected):
        raise ValueError('All17 frozen identities required exactly once')
    exact = sum(r['comparison']['all_control_fields_exact'] for r in rows)
    return dict(identities=17, frozen_reference_exact=exact, alternate_node_admissible=exact == 17,
        verification_updates=34000, new_scientific_fits=0, references_changed=False,
        tolerance_relaxed=False, validation_scored=False, independent_roles_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    original, amendment, _ = recovery.verify(args.root)
    root, source, cfg, _, manifest = original
    binding_path = root/(NAME+'_registration.json')
    binding = json.loads(binding_path.read_text())
    if (binding['training_registration_sha256'] != run.sha(root/'registration.json')
            or binding['amendment_sha256'] != run.sha(root/recovery.AMENDMENT)
            or platform.node().split('.')[0] != binding['node']):
        raise ValueError('Frozen experiment, references and allocated node required')
    for rel, digest in binding['code_bindings'].items():
        if run.sha(root/'code'/rel) != digest:
            raise ValueError('Diagnostic code drift')
    if json.loads((root/(NAME+'_submission.json')).read_text())['job_id'] != os.environ['SLURM_JOB_ID']:
        raise ValueError('Owned diagnostic submission required')
    refs = [r for r in manifest['heads'] if run.key(r, 'quadratic') in binding['identities']]
    if len(refs) != 17 or any(amendment['references'][run.key(r, 'quadratic')]['preserve_v1'] for r in refs):
        raise ValueError('Only the17 remaining reference identities are in scope')
    from src.world_model import m3w_easy_harm_deviance_training as api
    from src.world_model.m3w_preprocess_portability import frozen_preprocess
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume atomic diagnostic')))
    home, public = root/run.PRIVATE/NAME, root/run.PUBLIC/NAME
    home.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()

    def beat(**value):
        row = dict(pid=os.getpid(), job_id=os.environ['SLURM_JOB_ID'], node=platform.node(),
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **value)
        temp = home/'heartbeat.tmp'
        temp.write_text(json.dumps(row)+'\n')
        os.replace(temp, home/'heartbeat.json')
        print(json.dumps(row), flush=True)
        if time.monotonic()-began > 3600:
            raise TimeoutError('One-hour diagnostic bound; preserve checkpoints')

    saved_save = api.core.save_checkpoint

    def guarded_save(path, state):
        with (root/run.PRIVATE/'checkpoint_write.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run.quota(root, source, cfg)
            saved_save(path, state)

    api.core.save_checkpoint = guarded_save
    rows = []
    try:
        with (home/'process.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for ref in refs:
                name = run.key(ref, 'quadratic')
                output = public/(name+'.json')
                cp = amendment['references'][name]['reference_checkpoint']
                reference_path = recovery.checked_path(root, cp, run.PRIVATE/'control_diagnostic_v2')
                if output.exists():
                    row = json.loads(output.read_text())
                    assert row['registration_sha256'] == run.sha(binding_path)
                    assert row['reference_checkpoint'] == cp
                    assert run.sha(root/row['checkpoint']['path']) == row['checkpoint']['sha256']
                    rows.append(row)
                    continue
                data = run.packet(source, ref)
                path = home/name/'checkpoint.pt.gz'
                with frozen_preprocess(api.core, data[-1]):
                    state = api.fit(*data, arm='quadratic', settings=cfg['head_training'],
                        seed=ref['identity']['seed'], identity=ref['identity'],
                        experiment_sha256=run.sha(root/'registration.json'), path=path,
                        heartbeat=lambda **kw: beat(identity=name, **kw), resume=path.exists(),
                        checkpoint_guard=lambda: run.quota(root, source, cfg))
                old = api.core.read_checkpoint(reference_path)
                api.core.exact(state['identity'], old['identity'])
                result = compare(state, old, api.core.exact)
                row = dict(name=name, identity=ref['identity'], job_id=os.environ['SLURM_JOB_ID'],
                    registration_sha256=run.sha(binding_path), reference_checkpoint=cp,
                    checkpoint=dict(path=str(path.relative_to(root)),sha256=run.sha(path),bytes=path.stat().st_size),
                    comparison=result, verification_updates=2000,
                    result_source='fresh_run_TRAIN_only_cross_node_verification',
                    references_changed=False, new_scientific_fits=0,
                    validation_scored=False, independent_roles_read=False)
                run.once(output, row)
                rows.append(row)
                beat(state='reference_checked', completed=len(rows), exact=result['all_control_fields_exact'])
            result = dict(summary=summarize(rows, binding['identities']),
                results=[dict(path=str((public/(r['name']+'.json')).relative_to(root)),
                              sha256=run.sha(public/(r['name']+'.json'))) for r in rows],
                registration_sha256=run.sha(binding_path), job_id=os.environ['SLURM_JOB_ID'],
                node=platform.node(), torch=api.torch.__version__, cpu_threads=4, workers=0)
            run.once(public/'complete.json', result)
            beat(state='diagnostic_complete', **result['summary'])
    finally:
        api.core.save_checkpoint = saved_save


if __name__ == '__main__':
    main()
