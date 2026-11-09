import numpy as np
import pytest

from src.world_model.m3w_av2_direction_menu import menus, rotate


def test_direction_preserves_exact_recipients_and_each_step_radius():
    cv = np.arange(3*12*2, dtype=float).reshape(3, 12, 2)/9.
    incumbent = cv.copy()
    incumbent[0, :, 0] += np.linspace(.1, 1., 12)
    incumbent[2, :, 1] += np.linspace(1., .2, 12)
    p, slots, groups = menus(cv, incumbent)
    assert [len(v) for v in groups.values()] == [9, 9, 17]
    np.testing.assert_array_equal(p[0], incumbent)
    np.testing.assert_array_equal(slots, np.broadcast_to([0, 2], slots.shape))
    d = p[groups["direction"]]-cv
    np.testing.assert_allclose(np.linalg.norm(d, axis=-1), np.broadcast_to(np.linalg.norm(incumbent-cv, axis=-1), d.shape[:-1]))
    assert np.any(d[1:, 0, :, 1] != 0.)


def test_reject_any_other_intervention_count():
    cv = np.zeros((3, 12, 2))
    with pytest.raises(ValueError, match="two edits"):
        rotate(cv, cv)
