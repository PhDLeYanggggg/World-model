"""Fail closed on a null solver certificate without editing sealed algorithms."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import OptimizeResult
from scripts import run_m3w_fixed_occurrence_policy as run
from src.world_model import m3w_query_utility as legacy


def isolate(function, **dependencies):
    # Copy the dependency namespace, not the algorithm or any live module globals.
    out = FunctionType(function.__code__, {**function.__globals__, **dependencies},
                       function.__name__, function.__defaults__, function.__closure__)
    out.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return out


def compatible_solver(solver, record):
    def solve(**kwargs):
        result = solver(**kwargs)
        if getattr(result, 'mip_dual_bound', np.nan) is not None: return result
        bound, constraint = kwargs['bounds'], kwargs['constraints']
        arrays = [kwargs['c'], kwargs['integrality'], bound.lb, bound.ub,
                  constraint.A, constraint.lb, constraint.ub]
        signature = [dict(shape=list(np.shape(a)), sha256=hashlib.sha256(
            np.asarray(a, dtype='<f8').tobytes()).hexdigest()) for a in arrays]
        signature.append(kwargs['options'])
        record(dict(changed_field='mip_dual_bound', original_value=None,
            replacement='NaN_missing_certificate_not_zero', success=bool(result.success),
            status=int(result.status), message=str(getattr(result, 'message', '')),
            input_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()))
        copied = OptimizeResult(result); copied.mip_dual_bound = np.nan
        return copied
    return solve


def decision_runner(record):
    alloc = isolate(legacy.allocate, milp=compatible_solver(legacy.milp, record))
    grouped = isolate(legacy.grouped, allocate=alloc)
    arms = isolate(run.allocator.decisions, api=SimpleNamespace(grouped=grouped))
    decisions = isolate(run.decisions, allocator=SimpleNamespace(decisions=arms))
    action_group = isolate(run.action_group, decisions=decisions)
    return isolate(run.decide, action_group=action_group)


def register():
    assert not (run.PUBLIC/'decision_freeze.json').exists()
    assert not (run.PUBLIC/'summary.json').exists()
    cfg, ident = run.identity()
    assert ident == json.loads((run.PUBLIC/'decision_registration.json').read_text())
    existing = sorted((run.PRIVATE/'decisions').glob('*.json'))
    assert len(existing) == 95
    refs = []
    for path in existing:
        doc = json.loads(path.read_text()); assert doc['identity'] == ident and not doc['held_outcomes_used']
        assert run.base.artifact(run.ROOT/doc['arrays']['path']) == doc['arrays']
        refs.append(run.base.artifact(path))
    paths = run.train.closure(run.ROOT, ['scripts.recover_m3w_fixed_occurrence_solver'])
    paths += [run.ROOT/'tests/test_m3w_fixed_occurrence_solver_compat.py', run.PUBLIC/'solver_recovery_protocol.md']
    doc = dict(original_decision_registration=run.base.artifact(run.PUBLIC/'decision_registration.json'),
        training_freeze=run.base.artifact(run.PUBLIC/'training_freeze.json'),
        retained_prefix=refs, source_bindings={str(p.relative_to(run.ROOT)): run.train.digest(p) for p in paths},
        failure='nullable_mip_dual_bound_float_conversion_after95_complete_groups',
        sole_change='None_dual_certificate_to_NaN_existing_checked_anchor_fallback',
        registered_algorithm_code_objects_unchanged=True, global_modules_mutated=False,
        objective_changed=False, budget_changed=False, new_training=False, held_outcomes_used=False)
    run.train.immutable(run.PUBLIC/'solver_recovery_registration.json', doc)
    print(json.dumps(dict(registered=True, preserved_groups=95, files=len(doc['source_bindings']))))


def verify_registration():
    path = run.PUBLIC/'solver_recovery_registration.json'; run.base.inter.committed(path)
    doc = json.loads(path.read_text())
    for k in ('original_decision_registration', 'training_freeze'):
        assert run.base.artifact(run.ROOT/doc[k]['path']) == doc[k]
    for p, sha in doc['source_bindings'].items(): assert run.train.digest(run.ROOT/p) == sha, p
    assert len(doc['retained_prefix']) == 95
    for ref in doc['retained_prefix']: run.checked(ref)
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['register', 'decide', 'replay_decide']); a = p.parse_args()
    if a.phase == 'register': register(); return
    verify_registration(); run.api.torch.set_num_threads(4); run.api.torch.set_num_interop_threads(1)
    records = []
    log = run.PRIVATE/('solver_compatibility_'+a.phase+'.jsonl')
    def record(row):
        records.append(row)
        with log.open('a') as f: f.write(json.dumps(row)+'\n')
    with (run.PRIVATE/'policy.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); run.train.guard()
        decision_runner(record)(*run.load(), resume=a.phase == 'decide', replay=a.phase == 'replay_decide')
        doc = dict(registration=run.base.artifact(run.PUBLIC/'solver_recovery_registration.json'),
            phase=a.phase, null_certificates=records, completed_groups=108, held_outcomes_used=False,
            log=run.base.artifact(log) if log.exists() else None)
        run.train.immutable(run.PUBLIC/('solver_compatibility_'+a.phase+'.json'), doc)
        run.train.beat('solver_compatibility_phase_complete', phase=a.phase, null_certificates=len(records))


if __name__ == '__main__': main()
