import numpy as np

from src.evaluation import m3w_support_value_attribution as m


def test_support_swap_changes_only_support_bonus_and_preserves_fallback():
    pred = np.zeros((1, 3, 10), dtype=np.float32)
    truth = pred.copy()
    truth[0, 1, [0, 1, 9]] = 1.
    truth[0, 1, [2, 3, 5]] = .1
    s = m.oracle_scores(pred, truth)
    assert s["gain_oracle"][0, 1] > s["support_oracle"][0, 1] > s["motion"][0, 1]
    assert s["risk_oracle"][0, 1] < s["motion"][0, 1]
    assert all(v[0, 0] == 0. for v in s.values())


def test_fixed_count_witness_cannot_use_the_unfunded_best_action():
    legal = np.tile([True, True, False], (5, 1))
    gain = np.zeros((5, 3, 2))
    gain[0, 1], gain[4, 1] = 100., 1.
    easy = np.zeros((5, 3), dtype=bool)
    easy[:, 1] = True
    risk = np.zeros((5, 3, 3))
    assert m.fixed_count_witness(gain, risk, easy, legal, 0., 1., 100., 0)[1] == "zero_edits"
    n, status = m.fixed_count_witness(gain, risk, easy, legal, 0., 1., 100., 1)
    assert status == "positive_incumbent"
    assert n.tolist() == [0, 0, 0, 0, 1]
    assert m.fixed_count_witness(gain, risk, easy, legal, 0., 1., 100., 2)[1] == "risk_infeasible"


def test_prefix_risk_score_uses_only_explicit_prefix_errors():
    cv = np.array([1., 2.])
    err = np.array([[1.5, .5, .5], [2., 2., 2.]])
    s = m.prefix_scores([(cv, err)], [1.], np.array([[0, 1]]), np.array([[0, 0]]), np.ones(1))
    assert s[0, 0] == 0.
    assert (s[0, 1:] < 0.).all()
