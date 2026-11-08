import numpy as np

from src.evaluation.m3w_opendd_cv_abstention import edit, select_threshold


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
