import numpy as np

from src.world_model import m3w_postsearch_mass as m


def test_ledger_can_spend_only_past_expected_credit_and_resets_by_recording():
    p = np.array([[.3, .3, 10., 10.], [0., 0., 10., 10.],
                  [.3, .3, 10., 10.], [.3, .3, 10., 10.]])
    frame = np.array([4, 2, 1, 5])
    rec = np.array(['a', 'a', 'a', 'b'])
    np.testing.assert_array_equal(m.ledger(p, rec, frame, 1.), [True, True, False, False])


def test_mass_keeps_missing_rows_in_both_harm_numerators():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[:, :, 0] = np.array([1., 4., 10.])[:, None]
    valid = np.ones((3, 12), bool)
    valid[-1] = False
    path = cv.copy()
    path[0, :, 0] = -1.
    path[2, :, 0] = 2.
    np.testing.assert_allclose(m.targets(cv, path, y, valid, 2.), [3., 3., 5., 1.])
