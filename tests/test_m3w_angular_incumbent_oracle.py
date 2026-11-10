import numpy as np

from src.evaluation.m3w_angular_incumbent_oracle import choose_rowwise, rotate, stepwise_oracle


def test_oracle_projects_to_bounded_arc_and_does_not_alter_missing_steps():
    cv = np.zeros((1, 3, 2))
    anchor = cv.copy()
    anchor[0, :, 0] = 1.
    target = np.array([[[-1., .1], [0., 2.], [0., -2.]]])
    valid = np.array([[True, False, True]])
    p = stepwise_oracle(cv, anchor, target, valid, np.pi/2)
    np.testing.assert_allclose(p, [[[0., 1.], [1., 0.], [0., -1.]]], atol=1e-12)


def test_finite_rowwise_oracle_charges_disagreement_and_can_keep_original():
    cv = np.zeros((2, 2, 2))
    anchor = cv.copy()
    anchor[..., 0] = 1.
    alt = rotate(cv, anchor, np.pi/2)
    valid = np.array([[True, False], [True, True]])
    p = choose_rowwise(anchor, [anchor, alt], alt, valid)
    np.testing.assert_array_equal(p[0], anchor[0])
    np.testing.assert_array_equal(p[1], alt[1])
