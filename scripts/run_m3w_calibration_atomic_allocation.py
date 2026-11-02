"""Prospective storage-only repair before full calibration allocation/outcomes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import run_m3w_independent_risk_calibration as base
from src.world_model import m3w_calibration_atomic_progress as atomic

np = base.np
PUBLIC, PRIVATE = base.PUBLIC, base.PRIVATE
REG = PUBLIC/'allocation_storage_registration_v2.json'
FILES = (Path(__file__), ROOT/'src/world_model/m3w_calibration_atomic_progress.py',
         ROOT/'tests/test_m3w_calibration_atomic_progress.py')


def admission(register=False):
    values = base.admission(); cfg = values[0]
    registration = dict(parent=base.artifact(base.REG), bindings={str(p.relative_to(ROOT)): base.sha(p) for p in FILES},
        numerical_rule_unchanged=True, model_and_threshold_changes=False,
        reason='one_atomic_outer_grid_prefix_instead_of_nonatomic_choices_and_json_pair',
        lambda_grid=cfg['multipliers'], original_risk_budgets_unchanged=True,
        calibration_outcomes_opened_before_repair=False, final_confirmation_access=False)
    if register:
        if (PUBLIC/'policy.json').exists() or (PUBLIC/'labels.json').exists():
            raise ValueError('Register storage repair before full allocation or outcomes')
        base.once(REG, registration)
    if json.loads(REG.read_text()) != registration:
        raise ValueError('Storage-only contract changed')
    if not register:
        for p in (*FILES, REG):
            old = subprocess.run(['git', 'show', 'HEAD:'+str(p.relative_to(ROOT))],
                                 check=True, capture_output=True, cwd=ROOT).stdout
            if hashlib.sha256(old).hexdigest() != base.sha(p):
                raise ValueError('Commit storage repair before execution')
    return values


def one(r, row, data, cfg, beat, guard, *, family=None, stop_after=None):
    origin = base.home(r)/row['trial']
    directory = origin if family is None else PRIVATE/family
    directory.mkdir(parents=True, exist_ok=True)
    final = directory/'policy.json'; inference = origin/'inference.json'
    ref = json.loads(inference.read_text())
    if final.exists():
        result = json.loads(final.read_text()); base.checked(result['choices'])
        if result['inference'] != base.artifact(inference):
            raise ValueError('Frozen policy inputs changed')
        return base.artifact(final)
    p = np.load(base.checked(ref['moments']), allow_pickle=False)
    ok = np.load(base.checked(ref['allowed']), allow_pickle=False)
    identity = dict(parent_registration_sha256=base.sha(base.REG), storage_registration_sha256=base.sha(REG),
                    inference=base.artifact(inference))
    outer = directory/'grid_prefix_v2.gz'
    if outer.exists():
        saved = atomic.load(outer, identity, len(p)); choices = saved['choices']; info = saved['allocation']
    else:
        choices = np.full((13, 4, len(p)), -1, np.int8); info = []
    for j, multiplier in enumerate(base.api.MULTIPLIERS):
        if j < len(info):
            continue
        guard(); inner = directory/('query_v2_'+str(multiplier)+'.gz')
        q = base.api.coefficient_view(p, ok, multiplier)
        out, _, metadata = base.prior.allocator.allocate(q, ok, data['recordings'], data['frames'], data['ids'],
            seconds=cfg['solver_seconds_per_query'], checkpoint=lambda v: base.prior.api.save_checkpoint(inner, v),
            heartbeat=lambda **v: beat(state='atomic_grid_allocation', trial=row['trial'], multiplier=multiplier, **v),
            resume_state=base.prior.api.read_checkpoint(inner) if inner.exists() else None)
        choices[j] = np.stack([out[a] for a in base.prior.allocator.ARMS]).astype(np.int8)
        info.append(metadata); atomic.commit(outer, identity, choices, info)
        if inner.exists():
            inner.unlink()
        if stop_after is not None and len(info) == stop_after:
            return dict(interrupted_after_committed_multipliers=stop_after)
    matrix = directory/'choices.npy'; base.old_inputs.store_array(matrix, choices)
    base.once(final, dict(registration_sha256=base.sha(base.REG), storage_registration=base.artifact(REG),
        inference=base.artifact(inference), choices=base.artifact(matrix),
        policy_names=list(base.prior.allocator.ARMS), multipliers=list(base.api.MULTIPLIERS),
        allocation=info, labels_read=False))
    return base.artifact(final)


def main(phase):
    cfg, roles, rows, producers, _, _ = admission(phase == 'register')
    if phase == 'register':
        return
    beat, guard, receipt = base.runtime(cfg, 'atomic_'+phase)
    if phase == 'pilot':
        final = PUBLIC/'allocation_storage_pilot_v2.json'
        if final.exists():
            raise ValueError('Complete recovery pilot immutable')
        r = max(rows, key=lambda r: r['raw_rows']); row = producers[0]
        _, data, _ = base.record_data(r, roles, rows.index(r))
        one(r, row, data, cfg, beat, guard, family='storage_resume_pilot', stop_after=1)
        resumed = one(r, row, data, cfg, beat, guard, family='storage_resume_pilot')
        direct = one(r, row, data, cfg, beat, guard, family='storage_direct_pilot')
        a, b = [json.loads(base.checked(ref).read_text()) for ref in (resumed, direct)]
        legacy = json.loads((base.home(r)/row['trial']/'policy.json').read_text())
        for other in (b, legacy):
            np.testing.assert_array_equal(np.load(base.checked(a['choices']), allow_pickle=False),
                                          np.load(base.checked(other['choices']), allow_pickle=False))
            if a['allocation'] != other['allocation']:
                raise ValueError('Numerical allocation metadata changed')
        base.once(final, dict(**receipt(), storage_registration=base.artifact(REG),
            real_largest_record_resume_and_direct_grid_exact=True, original_pilot_grid_exact=True,
            labels_read=False, engineering_pass=True))
    else:
        if (PUBLIC/'policy.json').exists():
            raise ValueError('Complete policy freeze immutable')
        if not json.loads((PUBLIC/'allocation_storage_pilot_v2.json').read_text())['engineering_pass']:
            raise ValueError('Real exact interrupted/direct recovery pilot required')
        if not (PUBLIC/'verify_inference.json').exists():
            raise ValueError('Complete fresh forecast/cost replay required')
        refs = []
        for i, r in enumerate(rows):
            _, data, _ = base.record_data(r, roles, i)
            for row in producers:
                refs.append(one(r, row, data, cfg, beat, guard))
            beat(state='full_policy_record_complete', recording=i+1, total=102)
        base.once(PUBLIC/'policy.json', dict(**receipt(), model_record_receipts=refs,
            storage_registration=base.artifact(REG), future_labels_read=False, full_cohort=102,
            all_nine_models=True, new_optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('register', 'pilot', 'run')); args = parser.parse_args()
    if args.phase == 'register':
        main(args.phase)
    else:
        with (PRIVATE/'process.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); main(args.phase)
