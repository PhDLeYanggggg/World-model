import numpy as np
import pytest

from scripts.report_m3w_completion_cost import support_ceiling
from src.world_model.m3w_multiaction_cost import select_actions


def test_empty_support_ceiling_cannot_be_repaired_by_nested_veto():
    value = support_ceiling(dict(required_record_views=444, undefined_easy_record_views=62))
    assert value['maximum_defined_easy_views_for_nested_policy'] == 382
    assert value['fixed_head_veto_cannot_pass_complete_support']
    assert not value['population_or_arbitrary_model_impossibility_claim']
    with pytest.raises(ValueError):
        support_ceiling(dict(required_record_views=2, undefined_easy_record_views=3))


def test_fixed_head_availability_factorization_cannot_expand_gate_rows():
    rng = np.random.default_rng(11); p = rng.uniform(.001, 2., (200, 7, 5))
    p[:30] = [1., .001, 10., 3., .001]
    envelope = np.full((200, 7), 3.); allowed = np.ones((200, 7), bool)
    old, _ = select_actions(p, allowed, arm='physical_neural')
    for _ in range(20):
        q = rng.uniform(0, 1, (200, 1)); q[:15] = .999; q[0] = 0.; q[1] = 1.
        transported = p*q[..., None]
        transported[..., 1] += (1-q)*envelope
        transported[..., 4] += (1-q)*envelope
        new, _ = select_actions(transported, allowed, arm='physical_neural')
        assert not np.any((new >= 0) & (old < 0))
        assert new[0] == -1 and new[1] >= 0


def test_same_row_reference_is_action_independent_in_the_nested_class():
    reference = np.array([0., 2., 0., 1., 0.])
    old = np.array([True, False, True, False, True])
    for bits in range(32):
        nested = old & np.array([bool(bits & (1 << i)) for i in range(5)])
        assert reference[nested].sum() <= reference[old].sum()
        assert reference[nested].sum() == 0.
