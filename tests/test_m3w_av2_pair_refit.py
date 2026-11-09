import numpy as np

from scripts import run_m3w_av2_pair_refit as run


def test_roles_are_city_disjoint_and_cycle_once():
    cities = ["PIT", "ATX", "MIA", "DTW", "PAO", "WDC"]
    used = set()
    for held in cities:
        cal, donors = run.roles(cities, held)
        assert len(donors) == 4
        assert set(donors).isdisjoint({held, cal})
        assert held != cal
        used.add(cal)
    assert used == set(cities)


def test_calibration_must_have_positive_nonempty_supported_prefix():
    rows = [dict(score=.8, strict=True, add_ok=True),
            dict(score=.6, strict=True, add_ok=False),
            dict(score=.6, strict=True, add_ok=True),
            dict(score=-1., strict=True, add_ok=True)]
    assert run.calibrate(rows) == (.8, 2, 1)
    rows[0]["strict"] = False
    assert run.calibrate(rows) == (None, 2, 0)
    assert run.calibrate([]) == (None, 0, 0)


def test_pair_target_rejects_missing_easy_reference():
    y = np.zeros((3, 12, 2))
    cv = y.copy()
    cv[:, :, 0] = np.array([.1, 2., 1.])[:, None]
    data = dict(cv=cv, ols4=cv+.1)
    v = np.ones((3, 12), dtype=bool)
    a, b = cv.copy(), cv.copy()
    a[:2] = y[:2]
    b[1:] = y[1:]
    paths = np.stack([a, b])
    gain, strict, support = run.label_terms(data, paths, cv, y, v, (.2, 1.))
    np.testing.assert_array_equal(strict, [True, False])
    np.testing.assert_array_equal(support, [True, False])
    assert gain[0] > 0.
    assert run.metrics(a, data, y, v, (.2, 1.))["risk"]["supported"]
    v[0] = False
    _, strict, support = run.label_terms(data, paths, cv, y, v, (.2, 1.))
    assert not strict.any() and not support.any()
