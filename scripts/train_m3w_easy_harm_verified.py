"""Resume paired training with one explicit, exact reference-replay exception."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts import train_m3w_easy_harm_deviance as original


def verify(root):
    args = original.verify(root)
    root = args[0]
    amendment = json.loads((root/'control_execution_amendment.json').read_text())
    if amendment['training_registration_sha256'] != original.sha(root/'registration.json'):
        raise ValueError('Original training registration changed')
    for rel,digest in amendment['execution_bindings'].items():
        path = (root/'code'/rel).resolve()
        if not path.is_relative_to(root/'code') or original.sha(path) != digest:
            raise ValueError('Changed execution amendment')
    diag = root/original.PUBLIC/'control_diagnostic_v1.json'
    if original.sha(diag) != amendment['diagnostic_sha256']:
        raise ValueError('Changed diagnostic proof')
    return args, amendment


def check_reference(new, historical, replay, identity, api):
    """No floating tolerance: only this registered identity may use replay."""
    try:
        api.assert_original_control(new, historical)
        return dict(historical_control_exact=True, original_implementation_control_exact=True,
                    reference='historical_fixed_final', floating_tolerance_relaxed=False)
    except AssertionError:
        if new['identity'] != identity or historical['identity'] != identity or new['step'] != 2000:
            raise
        # A model/optimizer portability discrepancy is the sole permitted case.
        for key in ('initial_model','input_hashes','preprocess','settings','seed','step',
                    'sampler_rng','torch_rng','draw_hash','row_draws','queries'):
            api.core.exact(new[key],historical[key])
        if replay is None or replay['identity'] != identity:
            raise ValueError('Exact original-trainer replay required')
        api.assert_original_control(new,replay)
        return dict(historical_control_exact=False, original_implementation_control_exact=True,
                    reference='registered_same_node_original_trainer_replay', floating_tolerance_relaxed=False)


def train(args, amendment, shard):
    from src.world_model import m3w_easy_harm_deviance_training as api
    root = args[0]
    cp = amendment['reference_checkpoint']
    path = (root/cp['path']).resolve()
    if not path.is_relative_to(root/original.PRIVATE/'control_diagnostic_v1') or original.sha(path) != cp['sha256']:
        raise ValueError('Owned hash-verified reference required')
    replay = api.core.read_checkpoint(path)
    identity = amendment['exception_identity']
    saved_assert, saved_once = api.assert_original_control, original.once
    receipt = {}
    # Keep a stable reference to the unmodified numerical verifier while the
    # runner calls its scoped wrapper. No optimizer or data helper is replaced.
    class ReferenceAPI:
        core = api.core
        assert_original_control = staticmethod(saved_assert)

    def compare(new, old):
        receipt.clear()
        receipt.update(check_reference(new,old,replay,identity,ReferenceAPI))
        receipt['amendment_sha256'] = original.sha(root/'control_execution_amendment.json')
        if not receipt['historical_control_exact']:
            receipt['reference_checkpoint'] = cp

    def write(path, value):
        if path.parent == root/original.PUBLIC/'fits' and value.get('arm') == 'quadratic':
            value = {**value, **receipt,
                     'parent_control_exact':receipt['historical_control_exact']}
        return saved_once(path,value)

    api.assert_original_control, original.once = compare, write
    try:
        original.train(*args,'train',shard)
    finally:
        api.assert_original_control, original.once = saved_assert, saved_once


def join(args, amendment):
    root, source, cfg, reg, manifest = args
    new = json.loads((root/'train_submission.json').read_text())['job_id']
    own = json.loads((root/'join_submission.json').read_text())['job_id']
    if own != os.environ['SLURM_JOB_ID']:
        raise ValueError('Owned join required')
    assignment = {str(i):new for i in (0,2,3)}
    assignment['1'] = amendment['preserved_array_job_id']
    accounting, refs, found, historical, replayed = {}, [], set(), 0, 0
    expected = {original.key(r,a) for r in manifest['heads'] for a in cfg['arms']}
    for shard,job in assignment.items():
        task = job+'_'+shard
        p = subprocess.run(['sacct','-j',task,'-X','--noheader','--parsable2','--format=State,ExitCode'],
                           capture_output=True,text=True,timeout=30)
        if p.returncode or p.stdout.strip() != 'COMPLETED|0:0':
            raise ValueError('Every preserved/resumed task must complete: '+task)
        accounting[task] = p.stdout.strip()
        path = root/original.PUBLIC/'shards'/(shard+'.json')
        if shard == '1' and original.sha(path) != amendment['preserved_shard_sha256']:
            raise ValueError('Preserved completed shard changed')
        doc = json.loads(path.read_text())
        assert doc['array_job_id'] == job and doc['registration_sha256'] == original.sha(root/'registration.json')
        for ref in doc['fits']:
            path = (root/ref['path']).resolve()
            assert path.is_relative_to(root/original.PUBLIC/'fits') and original.sha(path) == ref['sha256']
            fit = json.loads(path.read_text()); name = path.stem
            assert name in expected and name not in found and fit['step'] == 2000 and not fit['validation_scored']
            cp = (root/fit['checkpoint']['path']).resolve()
            assert cp.is_relative_to(root/original.PRIVATE/'heads') and original.sha(cp) == fit['checkpoint']['sha256']
            assert cp.stat().st_size == fit['checkpoint']['bytes']
            if fit['arm'] == 'quadratic':
                if fit['parent_control_exact']:
                    historical += 1
                else:
                    assert fit['identity'] == amendment['exception_identity']
                    assert fit['original_implementation_control_exact'] and not fit['floating_tolerance_relaxed']
                    assert fit['reference_checkpoint'] == amendment['reference_checkpoint']
                    assert fit['amendment_sha256'] == original.sha(root/'control_execution_amendment.json')
                    replayed += 1
            refs.append(ref); found.add(name)
    assert found == expected and len(refs) == 144 and historical == 71 and replayed == 1
    original.once(root/original.PUBLIC/'training_freeze.json',dict(fits=refs,neural_fits=144,source_heads=72,
        registration_sha256=original.sha(root/'registration.json'),accounting=accounting,
        control_execution_amendment_sha256=original.sha(root/'control_execution_amendment.json'),
        original_implementation_controls_exact=72,historical_quadratic_controls_exact=71,
        historical_nonexact_replay_verified=1,historical_bitwise_reproduction_complete=False,
        validation_scored=False,independent_roles_read=False,new_forecasters=False,
        deployment_changed=False,stage5c_executed=False,smc_enabled=False))
    print(json.dumps(dict(state='all144_frozen',historical_controls_exact=71,replay_controls_exact=1)),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','join']); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shard',type=int,choices=[0,2,3]); args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Allocated compute before numerical imports')
    original_args, amendment = verify(args.root)
    if args.phase == 'join': join(original_args,amendment)
    else: train(original_args,amendment,args.shard)


if __name__ == '__main__':
    main()
