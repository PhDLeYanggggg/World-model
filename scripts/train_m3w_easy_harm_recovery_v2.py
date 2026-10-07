"""Explicit 18-reference recovery; preserve accepted fits and numerical training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from scripts import train_m3w_easy_harm_verified as previous

original = previous.original
AMENDMENT = 'control_execution_amendment_v2.json'


def verify(root):
    args, old = previous.verify(root)
    root = args[0]
    reg = json.loads((root/AMENDMENT).read_text())
    if (reg['training_registration_sha256'] != original.sha(root/'registration.json')
            or reg['previous_amendment_sha256'] != original.sha(root/'control_execution_amendment.json')):
        raise ValueError('Training and previous amendment must remain frozen')
    for rel, digest in reg['execution_bindings'].items():
        path = (root/'code'/rel).resolve()
        if not path.is_relative_to(root/'code') or original.sha(path) != digest:
            raise ValueError('Changed recovery implementation')
    for ref in reg['proofs'] + reg['preserved_fit_refs']:
        path = (root/ref['path']).resolve()
        if not path.is_relative_to(root/original.PUBLIC) or original.sha(path) != ref['sha256']:
            raise ValueError('Changed diagnostic or preserved receipt')
    expected = {original.key(r, 'quadratic'):r['identity'] for r in args[4]['heads'][0::4]}
    if len(expected) != 18 or set(reg['references']) != set(expected):
        raise ValueError('Only the original 18 shard-zero identities may use replay')
    for name, ref in reg['references'].items():
        if ref['identity'] != expected[name]:
            raise ValueError('Reference identity drift')
        checked_path(root, ref['reference_checkpoint'], original.PRIVATE/'control_diagnostic_v1'
                     if ref['preserve_v1'] else original.PRIVATE/'control_diagnostic_v2')
    return args, reg, old


def checked_path(root, ref, directory):
    path = (root/ref['path']).resolve()
    if (not path.is_relative_to(root/directory) or original.sha(path) != ref['sha256']
            or path.stat().st_size != ref['bytes']):
        raise ValueError('Owned hash/size-verified checkpoint required')
    return path


def compare_reference(new, historical, reference, replay, api):
    if reference is None:
        if new['identity'] != historical['identity']:
            raise ValueError('Historical identity mismatch')
        api.assert_original_control(new, historical)
        return dict(historical_control_exact=True, original_implementation_control_exact=True,
                    reference='historical_fixed_final', floating_tolerance_relaxed=False)
    if reference['identity'] != new['identity']:
        raise ValueError('Unregistered identity')
    result = previous.check_reference(new, historical, replay, reference['identity'], api)
    if result['historical_control_exact']:
        raise ValueError('Registered historical mismatch unexpectedly disappeared; inspect')
    return result


def train(args, reg):
    from src.world_model import m3w_easy_harm_deviance_training as api
    root, source, cfg, _, _ = args
    adopted = []
    # Reuse verified TRAIN-only quadratic fits. Never repeat their optimizer work.
    for name, ref in reg['references'].items():
        if ref['preserve_v1']:
            continue
        source_path = checked_path(root, ref['adopt_checkpoint'], original.PRIVATE/'control_diagnostic_v2')
        state = api.core.read_checkpoint(source_path)
        replay = api.core.read_checkpoint(checked_path(root, ref['reference_checkpoint'], original.PRIVATE/'control_diagnostic_v2'))
        if state['identity'] != ref['identity'] or state['step'] != 2000:
            raise ValueError('Cannot adopt wrong identity or partial state')
        if state['experiment_sha256'] != original.sha(root/'registration.json'):
            raise ValueError('Cannot adopt state from a different experiment')
        api.assert_original_control(state, replay)
        target = root/original.PRIVATE/'heads'/name/'checkpoint.pt.gz'
        target.parent.mkdir(parents=True, exist_ok=True)
        present = target.exists()
        if present:
            saved = api.core.read_checkpoint(target)
            for key in state:
                if key != 'seconds':
                    api.core.exact(saved[key], state[key])
        else:
            original.quota(root, source, cfg)
            temp = target.with_suffix('.adopting')
            with source_path.open('rb') as src, temp.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if original.sha(temp) != ref['adopt_checkpoint']['sha256']:
                raise ValueError('Atomic checkpoint adoption failed')
            os.replace(temp, target)
        adopted.append(dict(name=name, already_present=present, source=ref['adopt_checkpoint'],
                            accepted_checkpoint_sha256=original.sha(target)))
    adoption = root/original.PUBLIC/'recovery_v2_adoption.json'
    if not adoption.exists():
        original.once(adoption, dict(fits=adopted, new_quadratic_optimizer_updates=0,
                                    diagnostic_updates_previously_recorded=68000))
    saved_assert, saved_once = api.assert_original_control, original.once
    receipt = {}

    class ReferenceAPI:
        core = api.core
        assert_original_control = staticmethod(saved_assert)

    def compare(new, historical):
        identity = new['identity']
        name = identity['group']+'_head'+str(identity['seed'])+'_quadratic'
        ref = reg['references'].get(name)
        replay = None if ref is None else api.core.read_checkpoint(root/ref['reference_checkpoint']['path'])
        receipt.clear()
        receipt.update(compare_reference(new, historical, ref, replay, ReferenceAPI))
        if ref is not None:
            receipt['reference_checkpoint'] = ref['reference_checkpoint']
            receipt['amendment_sha256'] = original.sha(root/('control_execution_amendment.json' if ref['preserve_v1'] else AMENDMENT))

    def write(path, value):
        if path.parent == root/original.PUBLIC/'fits' and value.get('arm') == 'quadratic':
            value = {**value, **receipt, 'parent_control_exact':receipt['historical_control_exact']}
        return saved_once(path, value)

    api.assert_original_control, original.once = compare, write
    try:
        original.train(*args, 'train', 0)
    finally:
        api.assert_original_control, original.once = saved_assert, saved_once


def validate_fit(row, name, reg, old_sha, new_sha):
    if row['step'] != 2000 or any(row[k] is not False for k in ('validation_scored','independent_roles_read','new_forecaster')):
        raise ValueError('Complete unscored final fit required')
    if row['arm'] != 'quadratic':
        return 'candidate'
    ref = reg['references'].get(name)
    if ref is None:
        if row['parent_control_exact'] is not True:
            raise ValueError('Unregistered historical exception')
        return 'historical'
    if (row['identity'] != ref['identity'] or row['parent_control_exact'] is not False
            or row.get('historical_control_exact') is not False
            or row.get('original_implementation_control_exact') is not True
            or row.get('floating_tolerance_relaxed') is not False
            or row.get('reference_checkpoint') != ref['reference_checkpoint']
            or row.get('amendment_sha256') != (old_sha if ref['preserve_v1'] else new_sha)):
        raise ValueError('Exact registered replay proof required')
    return 'replay'


def join(args, reg):
    root, _, cfg, _, manifest = args
    job = json.loads((root/'train_submission.json').read_text())['job_id']
    own = json.loads((root/'join_submission.json').read_text())['job_id']
    if own != os.environ['SLURM_JOB_ID']:
        raise ValueError('Owned verification join required')
    refs, found, accounting = [], set(), {}
    counts = dict(historical=0, replay=0, candidate=0)
    expected = {original.key(r,a):(r['identity'],a) for r in manifest['heads'] for a in cfg['arms']}
    assignment = {'0':dict(task=job+'_0',path=str(original.PUBLIC/'shards/0.json')),
                  **reg['preserved_shards']}
    for shard, assignment_ref in assignment.items():
        task = assignment_ref['task']
        p = subprocess.run(['sacct','-j',task,'-X','--noheader','--parsable2','--format=State,ExitCode'],capture_output=True,text=True,timeout=30)
        if p.returncode or p.stdout.strip() != 'COMPLETED|0:0':
            raise ValueError('Every task must complete: '+task)
        accounting[task] = p.stdout.strip()
        path = root/assignment_ref['path']
        if shard != '0' and original.sha(path) != assignment_ref['sha256']:
            raise ValueError('Preserved shard changed')
        doc = json.loads(path.read_text())
        if doc['array_job_id']+'_'+str(doc['shard']) != task or doc['registration_sha256'] != original.sha(root/'registration.json'):
            raise ValueError('Task provenance mismatch')
        for ref in doc['fits']:
            path = (root/ref['path']).resolve()
            if not path.is_relative_to(root/original.PUBLIC/'fits') or original.sha(path) != ref['sha256']:
                raise ValueError('Invalid fit receipt')
            row = json.loads(path.read_text()); name = path.stem
            if name not in expected or name in found or (row['identity'],row['arm']) != expected[name]:
                raise ValueError('Registered unique identity required')
            checked_path(root, row['checkpoint'], original.PRIVATE/'heads'/name)
            counts[validate_fit(row,name,reg,original.sha(root/'control_execution_amendment.json'),original.sha(root/AMENDMENT))] += 1
            refs.append(ref); found.add(name)
    if found != set(expected) or counts != dict(historical=54,replay=18,candidate=72):
        raise ValueError('Complete54 historical+18 replay controls+72 candidates required')
    original.once(root/original.PUBLIC/'training_freeze.json', dict(fits=refs,neural_fits=144,source_heads=72,
        registration_sha256=original.sha(root/'registration.json'), accounting=accounting,
        control_execution_amendment_sha256=original.sha(root/AMENDMENT),
        previous_amendment_sha256=original.sha(root/'control_execution_amendment.json'),
        original_implementation_controls_exact=72,historical_quadratic_controls_exact=54,
        historical_nonexact_replay_verified=18,historical_bitwise_reproduction_complete=False,
        preserved_accepted_fits=110,adopted_diagnostic_quadratic_fits=17,new_candidate_fits=17,
        validation_scored=False,independent_roles_read=False,new_forecasters=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(state='all144_frozen',counts=counts)),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','join']); p.add_argument('--root',type=Path,required=True)
    a = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute required before numerical imports')
    args, reg, _ = verify(a.root)
    if a.phase == 'train':
        if os.environ.get('SLURM_ARRAY_TASK_ID') != '0':
            raise ValueError('Only unfinished shard zero is authorized')
        train(args,reg)
    else:
        join(args,reg)


if __name__ == '__main__':
    main()
