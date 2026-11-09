import numpy as np
import pytest

from src.evaluation.m3w_opendd_pair_disagreement import decompose, gate_transitions


def example():
    control = np.zeros((8, 2, 2))
    treatment = control.copy()
    treatment[[0, 2, 4], :, 0] = 1
    y = np.ones_like(control)
    y[..., 1] = 0.
    mask = np.ones((8, 2), bool)
    mask[2, 1] = mask[6, 1] = False
    y[~mask] = np.nan
    a = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    b = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    return treatment, control, mask, y, np.ones(8), 100., a, b


def test_only_unobserved_disagreements_are_charged_and_bounds_reconcile():
    d = decompose(*example())
    assert d['known_pp'] == 5
    assert d['missing_debit_pp'] == 1
    assert d['lower_pp'] == 4
    assert d['upper_pp'] == 6
    assert d['categories']['all_disagreements_observed']['recordings'] == 2
    assert d['categories']['some_disagreements_missing']['recordings'] == 1
    assert d['categories']['identical']['recordings'] == 1
    assert d['choice']['same_recipients_different_paths'] == 3


def test_outside_edits_are_rejected():
    t, c, m, y, w, mass, a, b = example()
    t = np.append(t, np.ones((1, 2, 2)), axis=0)
    c = np.append(c, np.zeros((1, 2, 2)), axis=0)
    y = np.append(y, np.ones((1, 2, 2)), axis=0)
    with pytest.raises(ValueError, match='outside'):
        decompose(t, c, np.append(m, [[True, True]], axis=0), y, np.r_[w, 1], mass, a, b)


def test_replacing_a_recipient_is_distinct_from_changing_the_same_path():
    t, c, m, y, w, mass, a, b = example()
    a[-1, 1] = 8
    t = np.concatenate((t, [[[1., 0.], [1., 0.]]]))
    c = np.concatenate((c, np.zeros((1, 2, 2))))
    y = np.concatenate((y, [[[1., 0.], [1., 0.]]]))
    r = decompose(t, c, np.append(m, [[True, True]], axis=0), y, np.r_[w, 1.], mass, a, b)
    assert r['choice']['different_recipients'] == 1
    assert r['choice']['same_recipients_different_paths'] == 3


def test_unsupported_easy_is_not_a_supported_risk_failure():
    def make(*status):
        x = np.array(status)
        return dict(missing=x == 0, risk=x >= 2, dual=x == 3)

    d = gate_transitions(make(1, 0, 3, 2, 0), make(0, 3, 2, 2, 0))
    assert d['no_easy_support']['risk_failed'] == 1
    assert d['dual_pass']['no_easy_support'] == 1
    assert d['reference_gain_failed']['dual_pass'] == 1
    assert sum(sum(x.values()) for x in d.values()) == 5
