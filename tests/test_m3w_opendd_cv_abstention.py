import numpy as np

from src.evaluation.m3w_opendd_cv_abstention import edit, select_threshold
from scripts.report_m3w_opendd_cv_abstention import known_easy_oracle


def test_threshold_is_strict_and_matched_controls_cannot_edit_other_queries():
    d = dict(offsets=np.array([0, 2, 4]), cv=np.zeros((4, 12, 2)),
             bank=np.ones((4, 1, 12, 2)), legal=np.ones((4, 1), dtype=bool))
    score = np.array([[0.2], [0.1], [0.3], [0.4]])
    out, query = edit(d, score, threshold=.2)
    assert query.tolist() == [False, True]
    assert out[:, 0, 0].tolist() == [0., 0., 0., 1.]
    _, matched = edit(d, -score, allowed_queries=query)
    np.testing.assert_array_equal(query, matched)


def test_inner_selection_requires_support_safety_and_gain():
    def row(t, harm, gain, switches=1):
        return dict(threshold=t, sites=[dict(signed_easy_pct=harm, lower_pp=gain, switches=switches)] * 6)

    invalid = [row(.001, None, 6), row(.002, 2.01, 6), row(.003, 0, -.1), row(.004, 0, 0, switches=0)]
    assert select_threshold(invalid) is None
    assert select_threshold([*invalid, row(.02, 1.1, .4), row(.01, 1.0, .4)]) == .01


def test_hindsight_lower_bound_rejects_gain_that_depends_on_missing_future():
    cv = np.zeros((3, 12, 2))
    bank = np.zeros((3, 1, 12, 2))
    bank[:, 0, :, 0] = [np.full(12, 2.), np.full(12, 3.), np.full(12, 2.)]
    y = np.zeros_like(cv)
    y[:, :, 0] = [np.full(12, .1), np.full(12, 3.), np.full(12, 2.)]
    valid = np.ones((3,12),dtype=bool)
    valid[2,1:] = False
    y[~valid] = np.nan
    out = known_easy_oracle(dict(cv=cv,bank=bank,y=y,valid=valid,offsets=[0,2,3],
        legal=np.ones((3,1),dtype=bool),dist=np.linalg.norm(bank-cv[:,None],axis=-1)), .5)
    assert out['switches'] == 1 and out['positive_easy_pct'] == 0.
