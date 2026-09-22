from argparse import Namespace
import numpy as np
import pytest
from scripts.run_m3w_eqmotion_cost_refit import validate_args
from scripts.run_m3w_bounded_cost import selections


def args(**kw):
    defaults = dict(audit_only=False, resume=False, evaluate=False, verify=False,
                    view=None, arm=None, stop_at=None)
    return Namespace(**(defaults | kw))


@pytest.mark.parametrize('changes', [dict(evaluate=True, resume=True), dict(audit_only=True, arm='bounded_fraction'),
    dict(verify=True, view='coupa_seed17'), dict(evaluate=True, verify=True), dict(stop_at=100),
    dict(view='coupa_seed17', arm='direct_native', stop_at=0)])
def test_invalid_phase_arguments_rejected(changes):
    with pytest.raises(ValueError): validate_args(args(**changes))


def test_allowed_pilot_and_resume():
    validate_args(args(view='coupa_seed17', arm='bounded_fraction', stop_at=100))
    validate_args(args(resume=True))
    validate_args(args(verify=True))


def test_all_new_arms_match_frozen_count_not_own_strict_count():
    past = np.zeros((6, 8, 2)); past[:, -1, 0] = 1
    ids = np.array([5, 6, 7, 8, 9, 10]); d = np.ones(6)
    anchor = np.array([False, True, False, True, False, False])
    for score in (np.tile([10., 0.], (6, 1)), np.tile([0., 10.], (6, 1))):
        selected = selections(score, past, d, anchor, ids)
        assert selected['matched_count'].sum() == anchor.sum() == 2
        np.testing.assert_array_equal(selected['matched_count'], [True, True, False, False, False, False])
    assert selections(np.tile([10., 0.], (6, 1)), past, d, anchor, ids)['strict_stop'].sum() == 6


def test_future_label_poisoning_cannot_change_policy():
    past = np.zeros((4, 8, 2)); past[:, -1, 0] = 1
    score = np.array([[2., .1], [0., 1.], [1., .5], [0., 0.]])
    kw = dict(score=score, past=past, distance=np.ones(4), legacy=np.array([True, False, False, False]), ids=np.arange(4))
    original = selections(**kw)
    # The interface has no argument for labels, outcomes or future support.
    with pytest.raises(TypeError): selections(**kw, future_endpoint=np.ones((4, 2)))
    for name, bits in original.items(): np.testing.assert_array_equal(bits, selections(**kw)[name])
