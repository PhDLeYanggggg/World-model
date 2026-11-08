import numpy as np

from src.evaluation import m3w_tumdot_three_reference as m
from src.evaluation.m3w_opendd_location import paired


def test_mask_attribution_uses_one_full_reference_denominator():
    cv = np.zeros((3, 4, 2))
    valid = np.array([[True]*4, [True, True, False, False], [False]*4])
    y = np.where(valid[..., None], np.broadcast_to([2., 0.], cv.shape), np.nan)
    z = m.mask_parts(cv, cv+[1., 0.], cv, y, valid)
    assert all(v["rows"] == 1 for v in z.values())
    np.testing.assert_allclose([v["lower"] for v in z.values()], [100/3, 0., -100/3])


def test_solver_keeps_roster_and_finds_three_reference_gain():
    cv = np.zeros((3, 4, 2))
    bank = np.stack((cv+[.5, 0.], cv-[.5, 0.]), axis=1)
    bank[-1] = cv[-1]
    y = cv+[1., 0.]
    valid = np.ones((3, 4), bool)
    active = np.array([True, True, False])
    c = m.coefficients(cv, cv, bank, 1, active, y, valid, 2.)
    a, r = m.solve(c)
    assert r["status"] == "optimal"
    np.testing.assert_array_equal(a, [0, 0])
    p = cv.copy()
    p[active] = bank[active, a]
    gain = min(paired(ref, p, y, valid)[1].sum() for ref in (cv, bank[:, 1]))
    np.testing.assert_allclose(r["objective_pp"], 100*gain/c["mass"])
