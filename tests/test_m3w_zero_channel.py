import numpy as np
import pytest
from src.evaluation.m3w_zero_channel import attribute


def test_zero_nonzero_and_both_vetoes_are_partitioned():
    y = np.array([[[-1., 0.]], [[-2., 0.]], [[-3., 0.]], [[100., 0.]]])
    r = np.array([[[-1., 1.]], [[1., -1.]], [[1., 1.]], [[-1., -1.]]])
    d = np.full_like(r, -1.)
    a = attribute(r, d, y, np.ones((4, 1), bool), np.array([True, True, True, False]), np.ones(4))
    assert a['false_reject_count'] == 3 and a['false_safe_count'] == 0
    for key, value in zip(('zero_only', 'nonzero_only', 'both'), (1., 2., 3.)):
        assert a['false_reject_partition'][key] == dict(rows=1, cost=value)
    assert a['blocking_outputs']['risk_easy']['zero_target_rows'] == 2
    assert a['blocking_outputs']['risk_easy']['zero_target_cost'] == pytest.approx(4.)
    assert a['unknown_proposals'] == 1


def test_diagnostic_runner_resolves_real_router():
    from scripts import diagnose_m3w_decision_semantics as run
    assert callable(run.direct.api.parent.assert_excluded)
    assert callable(run.direct.api.parent.router.choose)
