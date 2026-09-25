"""Targeted past-only numerical replay; preserve original scientific readout."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.world_model.m3w_event_risk_feasibility import bounded_risk_coefficients
from src.world_model.m3w_joint_intervention import InterventionProblem
from src.world_model.m3w_scaled_risk_controls import solve_scaled_risk_control

PUBLIC = ROOT/'outputs/publication_readiness_2026_09/european_conditional_risk_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((PUBLIC/'analysis.json').read_text())
    analysis_sha = sha(PUBLIC/'analysis.json')
    audit = json.loads((PUBLIC/'accounting_audit.json').read_text())
    if audit['analysis_sha256'] != analysis_sha or not audit['all_passed']:
        raise ValueError('Verified completed accounting required')
    private = ROOT/'data/stage_cvpr2027_experiments'
    packed = private/'european_source_forecast_v1/packed'
    pr = json.loads((packed/'receipt.json').read_text())
    data = {}
    for name in ('recordings', 'frames', 'history'):
        path = packed/(name+'.npy')
        if sha(path) != pr['arrays'][name]:
            raise ValueError('Past array changed')
        data[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    checks = []
    for item in result['controls']:
        rp = ROOT/item['path']
        if sha(rp) != item['sha256']:
            raise ValueError('Receipt changed')
        receipt = json.loads(rp.read_text())
        failures = [q for q in receipt['queries'] if not q['arms']['independent']['solver_optimal']]
        if not failures:
            continue
        lineage = receipt['identity']['head_identity']['lineage']
        name = lineage['final_producer']+'_'+lineage['event']+'_'+receipt['identity']['head_identity']['arm']
        r = result['training'][name]
        path = ROOT/r['artifacts']['scores']['path']
        if sha(path) != r['artifacts']['scores']['sha256']:
            raise ValueError('Moment scores changed')
        with np.load(path, allow_pickle=False) as z:
            ids, m = z['ids'].copy(), z['moments'].copy()
        udir = private/'european_cv_reference_v1/heads'/(lineage['final_producer']+'_neural_underharm4')
        ur = json.loads((udir/'complete.json').read_text())
        upath = ROOT/ur['artifacts']['predicted_costs']['path']
        if sha(upath) != ur['artifacts']['predicted_costs']['sha256']:
            raise ValueError('Frozen utility changed')
        with np.load(upath, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], ids)
            utility = (z['costs'][:, 0]-z['costs'][:, 1])/lineage['cost_scale']
        if sha(ROOT/receipt['path']) != receipt['sha256']:
            raise ValueError('Old choices changed')
        with np.load(ROOT/receipt['path'], allow_pickle=False) as z:
            oldids, oldbits = z['ids'].copy(), z['independent'].copy()
        for q in failures:
            use = (data['recordings'][ids] == q['recording']) & (data['frames'][ids] == q['frame'])
            if use.sum() != 1 or q['agents'] != 1:
                raise ValueError('This targeted diagnostic only covers observed singleton failures')
            selected_ids = ids[use]
            moving = np.linalg.norm(np.diff(data['history'][selected_ids], axis=1), axis=2).sum(1) > 0
            bounded = bounded_risk_coefficients(utility[use], m[use], moving, budget=.02,
                support_available=receipt['source_support_available'])
            p = InterventionProblem(bounded['expected_gain'], bounded['expected_harm'], bounded['supported'],
                np.empty((0, 2), int), np.empty((0, 2, 2)), .1, .02, 1)
            fresh = solve_scaled_risk_control(p, objective_kind='independent', time_limit_seconds=2.)
            old = oldbits[np.isin(oldids, selected_ids)]
            if not fresh['solver_optimal'] or not np.array_equal(fresh['switch'], old):
                raise ValueError('Targeted repair failed or changed the forecast choice')
            checks.append(dict(head=name, guard=receipt['identity']['support_guard'],
                site=q['site'], recording=q['recording'], frame=q['frame'],
                old_reason=q['arms']['independent']['reason'], predicted_moments=m[use][0].tolist(),
                native_budget=bounded['native_budget'], pruned=int((~bounded['supported']).sum()),
                new_solver_optimal=True, same_choice=True))
    if not checks:
        raise ValueError('No failure cases to verify')
    output = dict(result_source='fresh_run_targeted_past_only_numerical_replay', analysis_sha256=analysis_sha,
        checks=checks, scientific_readout_changed=False, new_training=False, deployment_changed=False,
        replay_scope='all_observed_independent_singleton_solver_failures_not_full_solver_reexecution',
        future_targets_read=False, tolerance_relaxed=False,
        bindings={p: sha(ROOT/p) for p in ('src/world_model/m3w_event_risk_feasibility.py',
            'scripts/audit_m3w_event_risk_numerics.py', 'tests/test_m3w_event_risk_feasibility.py')})
    (PUBLIC/'numerical_audit.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    if sha(PUBLIC/'analysis.json') != analysis_sha:
        raise ValueError('Original scientific result changed')
    print(json.dumps(dict(targeted_cases=len(checks), all_repaired=True, choices_changed=0)))


if __name__ == '__main__':
    main()
