import numpy as np

from scripts import run_m3w_av2_eqmotion as run
from scripts import audit_m3w_av2_eqmotion_choices as audit
from scripts import run_m3w_av2_eqmotion_uniform as uniform


def test_city_training_pools_leave_non_donors_out():
    t = np.arange(8, dtype=float)
    h = np.array([np.column_stack((t, t*0)), np.column_stack((t, t*0+1))])
    y = np.ones((2, 12, 2))
    v = np.ones((2, 12), dtype=bool)
    data = {"a": dict(h=h, city="donor"), "b": dict(h=h*5, city="held")}
    (pool,) = run.city_batches(data, {"a": (y, v), "b": (y*9, v)}, ["donor"])
    assert len(pool[0]) == 2
    np.testing.assert_array_equal(pool[4].numpy(), y)
    np.testing.assert_array_equal(pool[5].numpy(), v)


def test_public_shape_extends_exact_peak_menu_without_removing_old_options():
    cv = np.zeros((3, 12, 2))
    bank = np.repeat(cv[:, None], 2, axis=1)
    bank[:, 0, :, 0] = 2.
    bank[:, 1, :, 1] = 2.
    d = dict(cv=cv, bank=bank)
    eq = np.full_like(cv, 1.5)
    radii = np.array([1., 2.])
    before = run.slates(d, radii, bank)
    after = run.slates(d, radii, run.expanded(d, eq))
    assert len(after) > len(before)
    np.testing.assert_allclose(np.sort(run.av.motion.peak(after-cv), axis=1),
                               np.broadcast_to([0., 1., 2.], (len(after), 3)))
    assert all(np.any(np.all(after == p, axis=(1, 2, 3))) for p in before)


def test_shape_usage_does_not_count_an_alias_of_the_old_bank():
    cv = np.zeros((3, 12, 2))
    old = np.repeat(cv[:, None], 1, axis=1)
    old[:, 0, :, 0] = 2.
    eq = old[:, 0].copy()
    eq[1, :, 1] = 2.
    d = dict(cv=cv, bank=old)
    prediction = np.stack((old[0, 0]*.5, eq[1]*.5, cv[2]))
    np.testing.assert_array_equal(audit.new_shapes(prediction, d, eq), [False, True, False])


def test_scene_uniform_uses_one_family_without_changing_slots():
    opts = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    np.testing.assert_array_equal(uniform.uniform_opts(opts, np.array([0, 1, 1, 0]), 2),
                                  [[0, 3], [1, 2]])
    single = np.array([[0, -1], [1, -1]])
    np.testing.assert_array_equal(uniform.uniform_opts(single, np.array([0, 1]), 1), single)
