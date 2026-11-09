import numpy as np

from scripts.run_m3w_opendd_optional_edits import options, oracle, paint, release


def test_options_can_drop_either_slot_and_leave_a_cv_recipient():
    cv = np.zeros((3, 12, 2))
    bank = cv[:, None].copy()
    bank[..., 0] = 4.
    paths, row, arm, opts, family = options(cv, bank, np.array([1., 2.]))
    for o, f in zip(opts, family, strict=True):
        p = paint(cv, paths, row, arm, o)
        radius = np.linalg.norm(p-cv, axis=-1).max(-1)
        expected = ([1., 2.] if f == 2 else [float(f+1)])
        np.testing.assert_array_equal(sorted(radius[radius > 0.]), expected)
        assert (radius == 0.).any()
    assert set(family) == {0, 1, 2}


def test_abstention_when_risky_or_reference_is_missing():
    gain = np.full((3, 2), .1)
    z = np.array([[.03, .03, .03, 1., 1., 0.],
                  [0., 0., 0., 1., 0., 0.],
                  [.01, .01, .01, 1., 1., 0.]])
    np.testing.assert_array_equal(release(gain, z, 1., 0., 0.), [2])
    np.testing.assert_array_equal(release(gain[:0], z[:0], 1., 0., 0.), [])


def test_subset_oracle_retains_helpful_edit_only_and_verifies_risk():
    cv = np.zeros((3, 12, 2))
    y = cv.copy()
    y[..., 0] = 1.
    p = cv.copy()
    p[:2, :, 0] = np.array([.5, -.5])[:, None]
    row = oracle(cv, cv, p, y, np.ones((3, 12), bool), np.zeros(3, int), [1.5, 2.])
    assert row["verified"] and row["positive"] and row["edits"] == 1
    np.testing.assert_allclose(row["value_pp"], 100./6)
