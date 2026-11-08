import numpy as np

from src.evaluation.m3w_opendd_location import choose, cohorts, geometry, paired


def test_queries_keep_past_complete_targets_even_without_any_future_labels():
    visible = np.ones((3, 41), dtype=bool)
    visible[1, 8:20] = False
    first, second = list(cohorts(visible, stride=1))
    assert first[0] == 0 and first[1].tolist() == [0, 1, 2]
    assert second[0] == 20 and second[1].tolist() == [0, 1, 2]


def test_matched_one_edit_and_direct_missing_step_debit():
    score = np.array([[1., 2.], [3., 5.], [8., 9.], [7., 8.]])
    legal = np.array([[1, 1], [1, 0], [0, 0], [0, 0]], dtype=bool)
    assert choose(score, legal, [0, 2, 4]).tolist() == [-1, 0, -1, -1]
    a = np.zeros((1, 2, 2))
    b = np.ones_like(a)
    label = np.array([[[1., 1.], [np.nan, np.nan]]])
    obs, low = paired(a, b, label, np.array([[True, False]]))
    np.testing.assert_allclose(obs, [[np.sqrt(2), 0.]])
    np.testing.assert_allclose(low, [[np.sqrt(2), -np.sqrt(2)]])


def test_local_geometry_translation_invariant():
    p = np.arange(48.).reshape(3, 8, 2)
    scale = np.ones(3)
    np.testing.assert_array_equal(geometry(p, [0, 3], scale), geometry(p + 50., [0, 3], scale))
