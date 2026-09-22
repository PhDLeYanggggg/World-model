import numpy as np
import pytest
from src.evaluation.m3w_prefix_veto_audit import diagnose


def example():
    score = np.tile([1., .01], (5, 12, 1)); score[0, 0] = [1., .5]; score[1, 3] = [1., .5]
    costs = np.tile([1., 0.], (5, 12, 1)); costs[0, 0] = [0., .2]
    available = np.ones((5, 12), bool); available[3, 2:] = False; available[4] = False
    costs[~available] = np.nan
    return score, costs, available, np.ones(5, bool), np.array([0, 0, 1, 1, 1], bool)


def test_beneficial_terminal_can_have_harmed_prefix():
    r = diagnose(*example()); v = r['groups']['vetoed']
    assert r['vetoed'] == 2 and r['guarded'] == 3
    assert v['complete_terminal_beneficial'] == 2 and v['complete_terminal_beneficial_but_prefix_harmed'] == 1
    assert r['first_veto_prefix_counts']['1'] == r['first_veto_prefix_counts']['4'] == 1


def test_missing_labels_not_safe_zeros():
    r = diagnose(*example())['groups']['retained']
    assert r['complete'] == 1 and r['incomplete'] == 2 and r['no_valid_prefix'] == 1


def test_wrong_guard_or_future_gap_rejected():
    args = list(example()); args[-1][0] = True
    with pytest.raises(ValueError):
        diagnose(*args)
    args = list(example()); args[2][3, 4] = True; args[1][3, 4] = [0, 0]
    with pytest.raises(ValueError):
        diagnose(*args)


def test_no_input_mutation():
    args = example(); saved = [x.copy() for x in args]
    diagnose(*args)
    for a, b in zip(args, saved):
        np.testing.assert_array_equal(a, b)
