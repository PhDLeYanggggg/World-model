import numpy as np
import pytest

from scripts import run_m3w_av2_approach_menu as evaluation
from src.world_model import m3w_approach_candidate_paths as model


def crossing():
    t = np.arange(-7, 1)
    a = np.column_stack((t, 0.*t))
    b = np.column_stack((4.+0.*t, 4.-t))
    c = np.column_stack((t+100., 30.+0.*t))
    return np.stack((a, b, c))


def test_crossing_produces_directed_brake_and_lateral_paths():
    h = crossing()
    cv, own, social, who = model.candidates(h, 12)
    np.testing.assert_array_equal(who, [1, 0, -1])
    np.testing.assert_array_equal(own, np.broadcast_to(cv[:, None], own.shape))
    assert social[0, 0, -1, 0] < cv[0, -1, 0]
    assert social[0, 1, -1, 1] < cv[0, -1, 1]
    np.testing.assert_array_equal(social[2], np.repeat(cv[2:3], 2, axis=0))


def test_causal_paths_equivariant_to_origin_and_units():
    h = crossing().astype(float)
    h[0, -2, 0] -= .125
    r = np.array([[0., 1.], [-1., 0.]])
    a = model.candidates(h, 12)
    b = model.candidates(3.*h@r+[11., -2.], 12)
    for x, y in zip(a[:3], b[:3], strict=True):
        np.testing.assert_allclose(3.*x@r, y)
    np.testing.assert_array_equal(a[3], b[3])
    with pytest.raises(ValueError, match="eight"):
        model.candidates(np.zeros((1, 20, 2)), 12)


def test_vectorized_checks_match_original_risk_including_empty_easy():
    y = np.zeros((3, 12, 2))
    cv = y.copy()
    cv[..., 0] = np.array([1., 2., 3.])[:, None]
    bank = np.stack([cv, cv.copy(), cv.copy(), cv.copy()])
    bank[1, :2] *= .5
    bank[2, :2] *= 1.1
    bank[3, 1:] *= .5
    v = np.ones(y.shape[:2], bool)
    _, *checks = evaluation.constraints(bank, cv, cv, cv, y, 1.5)
    expected = [evaluation.study.policy(p, cv, cv, y, v, ["r"]*3, easy_cut=1.5, hard_cut=2.5)
                ["recordings"]["r"]["supported"] for p in bank]
    np.testing.assert_array_equal(np.logical_and.reduce(checks), expected)
    assert expected == [False, True, False, False]
