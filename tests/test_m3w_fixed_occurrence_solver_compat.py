import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from src.world_model import m3w_query_utility as legacy
from scripts.recover_m3w_fixed_occurrence_solver import isolate, compatible_solver, decision_runner


def inputs():
    return (np.array([1., 3., 2.]), np.array([[-.2, -.2], [-.1, .1], [-.1, -.1]]),
            np.ones(3, bool), np.array([True, False, False]), np.arange(3))


def test_original_failure_is_reproduced_and_compatibility_abstains():
    result = OptimizeResult(success=False, x=None, fun=None, mip_dual_bound=None,
                            status=4, message='numerical failure')
    solve = lambda **kw: result
    original = isolate(legacy.allocate, milp=solve)
    with pytest.raises(TypeError): original(*inputs())
    notes = []
    repaired = isolate(legacy.allocate, milp=compatible_solver(solve, notes.append))
    out, info = repaired(*inputs())
    np.testing.assert_array_equal(out['joint_utility'], inputs()[3])
    assert info['fallback'] and not info['optimal']
    assert notes[0]['status'] == 4 and notes[0]['changed_field'] == 'mip_dual_bound'
    assert result.mip_dual_bound is None


@pytest.mark.parametrize('success', [False, True])
def test_missing_certificate_cannot_admit_an_unverified_solution(success):
    result = OptimizeResult(success=success, x=np.array([0., 0., 1.]), fun=-2/3,
                            mip_dual_bound=None, status=0 if success else 4)
    f = isolate(legacy.allocate, milp=compatible_solver(lambda **kw: result, lambda _: None))
    out, info = f(*inputs())
    np.testing.assert_array_equal(out['joint_utility'], inputs()[3])
    assert info['fallback']


def test_valid_result_is_identical_and_legacy_globals_untouched():
    result = OptimizeResult(success=True, x=np.array([0., 0., 1.]), fun=-2/3,
                            mip_dual_bound=-2/3, mip_node_count=1, status=0)
    solve = lambda **kw: result
    old = isolate(legacy.allocate, milp=solve)
    notes = []
    new = isolate(legacy.allocate, milp=compatible_solver(solve, notes.append))
    a, x = old(*inputs()); b, y = new(*inputs())
    assert x == y and not notes
    for k in a: np.testing.assert_array_equal(a[k], b[k])
    assert legacy.allocate.__globals__['milp'] is legacy.milp
    assert new.__code__ is old.__code__ is legacy.allocate.__code__
    assert new.__kwdefaults__ == legacy.allocate.__kwdefaults__


def test_runner_only_replaces_local_dependencies():
    from scripts import run_m3w_fixed_occurrence_policy as run
    f = decision_runner(lambda _: None)
    assert f.__code__ is run.decide.__code__
    assert run.decide.__globals__['action_group'] is run.action_group
    assert f.__globals__['action_group'] is not run.action_group
    assert run.decisions.__globals__['allocator'] is run.allocator
