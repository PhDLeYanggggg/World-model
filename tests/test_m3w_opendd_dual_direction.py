import numpy as np

from src.world_model import m3w_opendd_dual_direction as m


def test_dual_and_cv_choose_different_shapes_without_relaxing_legality():
    z = np.array([[0., 0., 0., 0., 0.], [5., -1., 0., 0., 0.],
                  [2., 3., 1., 0., 0.], [6., 6., -1., 0., 0.]])
    legal = np.array([True, True, True, False])
    assert m.choose(z, legal, "cv") == 1
    assert m.choose(z, legal, "dual") == 2
    assert m.choose(z, legal, "guard") == 0


def test_joint_combines_complementary_candidates_in_same_units():
    z = np.array([[0., 0., 0., 0., 0.], [3., -1., 1., 0., 0.]])
    w = np.array([[0., 0., 0., 0., 0.], [-1., 3., -1., 0., 0.]])
    mask = np.ones(2, dtype=bool)
    assert m.choose(z, mask, "guard") == m.choose(w, mask, "guard") == 0
    assert m.joint(z, w, mask, mask) == (1, 1)
    assert m.joint(z, w, mask, np.array([True, False])) == (0, 0)
