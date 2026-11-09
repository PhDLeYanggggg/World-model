from itertools import product

import numpy as np

from scripts.report_m3w_opendd_interaction_effects import interval, site_means
from src.evaluation import m3w_interaction_frontier as model
from src.evaluation.m3w_exact_slot_headroom import POSITIVE_PP, make_design
from src.evaluation.m3w_selection_forecast_readout import policy


def test_full_grid_debit_lower_bounds_every_completion():
    control = np.zeros((2, 2, 2))
    edit = control.copy()
    edit[0, :, 0] = 1.
    target = edit.copy()
    valid = np.array([[True, False], [True, True]])
    lower = model.pair_reduction(edit[None], control, target, valid)[0]
    np.testing.assert_allclose(lower, [1., 0.])
    for unseen in [-1000., -.1, .25, 1., 1000.]:
        target[0, 1, 0] = unseen
        actual = model.pair_reduction(edit[None], control, target, np.ones_like(valid))[0, 0]
        assert actual >= lower[1]


def test_exact_interaction_optimizer_matches_enumeration_under_original_gate():
    cv = np.zeros((6, 2, 2))
    bank = np.repeat(cv[:, None], 2, axis=1)
    bank[:, 0, :, 0], bank[:, 1, :, 0] = .5, -.5
    target = cv.copy()
    target[:, :, 0] = np.array([.6, -.7, 2., -.8, .7, 1.4])[:, None]
    valid = np.ones((6, 2), bool)
    valid[2, 1] = False
    q = np.repeat(np.arange(2), 3)
    control = cv.copy()
    control[0], control[1], control[3] = bank[0, 1], bank[1, 0], bank[3, 1]
    design, reason = make_design(cv, cv, bank, control, control, target, valid, q, 1., double_only=True)
    assert reason is None
    den = model.denominators(q, valid)
    menu = model.query_options(design, cv, control, target, valid, q, den)
    contenders = []
    for indices in product(*(np.flatnonzero(menu["group"] == j) for j in (0, 1))):
        ix = list(indices)
        gain = design["offset"] + menu["gain"][ix].sum(0)
        actions = np.concatenate([menu["actions"][i] for i in ix])
        path = cv.copy()
        path[design["row"][actions]] = design["path"][actions]
        np.testing.assert_array_equal(path[3:], control[3:])
        gate = policy(path, cv, cv, target, valid, ["rec"] * len(cv), easy_cut=1., hard_cut=1.5)
        if min(gain[:2]) > POSITIVE_PP and gain[2] >= 0. and gate["recordings"]["rec"]["supported"]:
            contenders.append(model.aggregate_pair(path, control, target, valid, q, den))
    assert contenders
    for objective in (0, 1):
        taken, report = model.solve(menu, design, objective, 2.)
        assert report["status"] == "optimal"
        expected = np.asarray(contenders)[:, objective].max()
        np.testing.assert_allclose(menu["pair"][taken, objective].sum(), expected, atol=1e-10)
        np.testing.assert_allclose(report["upper_native"], expected, atol=1e-10)


def test_no_coobserved_steps_do_not_become_positive():
    ref = np.zeros((2, 4, 2))
    edit = ref.copy()
    edit[0, :, 0] = 1.
    observed = np.array([[True, True, False, False], [False, False, True, True]])
    np.testing.assert_array_equal(model.denominators([0, 0], observed), [0, 1])
    np.testing.assert_allclose(model.pair_reduction(edit[None], ref, ref, observed), [[0., -1.]])


def test_resampling_weights_recordings_and_locations_not_seed_counts():
    a = site_means([("r0", [0., 2.]), ("r0", [2., 4.]), ("r1", [3., 1.])])
    assert a == [2., 2.]
    b = site_means([("r2", [4., 8.])])
    receipt = interval([a, None, b])
    assert receipt["sites"] == 2 and receipt["site_mean"] == [3., 5.]
