import numpy as np

from scripts.audit_m3w_exact_count_joint_capacity import dominant_paths


def test_dominant_path_uses_observed_error_and_unknown_cv_disagreement():
    cv = np.zeros((3, 2, 2))
    y = cv+4.
    bank = np.stack((cv, cv+1., cv+4.), axis=1)
    bank[2] = 0.
    valid = np.array([[1, 0], [0, 0], [1, 1]], dtype=bool)
    np.testing.assert_array_equal(dominant_paths(cv, bank, y, valid, 9.),
                                  np.stack((cv[0]+4., cv[1]+1., cv[2])))
