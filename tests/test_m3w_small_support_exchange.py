from itertools import product

import numpy as np

from src.evaluation import m3w_small_support_exchange as m
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import policy


def test_exchange_risk_and_exact_paired_costs_with_unknown_removed():
    cv = np.zeros((4, 2, 2))
    bank = cv[:, None].copy()
    bank[:, 0, :, 0] = np.array([.5, .5, -.5, .5])[:, None]
    target = cv.copy()
    target[:, :, 0] = np.array([1., 1., 5., 1.])[:, None]
    valid = np.ones((4, 2), bool)
    valid[0] = False
    initial = np.array([0, -1, 0, -1])
    groups = [np.array([0, 1]), np.array([2, 3])]
    p = m.menu(cv, bank, cv, initial, groups, target, valid, 2.)
    np.testing.assert_array_equal(p["choice"][:, :3], [[0, 0, 1], [1, 2, 3]])
    assert p["terms"][0, 4] == 0.
    assert m.solve(p, single=True, joint=True)[0] is None
    plan, status = m.solve(p, single=False, joint=True)
    assert status == "optimal"
    out, choices = m.paths(p, plan, initial, cv, bank)
    np.testing.assert_array_equal(choices, [-1, 0, -1, 0])
    assert policy(out, cv, cv, target, valid, ["r"]*4, easy_cut=2., hard_cut=4.)["recordings"]["r"]["supported"]
    z = comparison_masses(p["original"], out, cv, target, valid, p["weights"])
    np.testing.assert_allclose(z["known_delta"]-z["unknown_radius"], p["terms"][plan, 4].sum())


def test_radius_and_empty_support_are_not_silently_relaxed():
    cv = np.zeros((2, 2, 2))
    bank = cv[:, None].copy()
    bank[:, 0, :, 0] = np.array([.4, .5])[:, None]
    target = cv.copy()
    target[..., 0] = 1.
    valid = np.ones((2, 2), bool)
    groups = [np.arange(2)]
    a = m.menu(cv, bank, cv, np.array([0, -1]), groups, target, valid, .9)
    assert m.solve(a, single=False, joint=False) == (None, "structural_no_easy")
    b = m.menu(cv, bank, cv, np.array([0, -1]), groups, target, valid, 2.)
    assert len(b["choice"]) == 0
    assert m.solve(b, single=False, joint=True) == (None, "no_legal_exchange")


def test_sparse_program_matches_full_path_enumeration():
    rng = np.random.default_rng(121)
    for _ in range(12):
        cv = np.zeros((6, 3, 2))
        bank = rng.normal(0, .25, (6, 3, 3, 2))
        # Known old paths turn away from the observed target; smaller candidates
        # include both beneficial and harmful directions.
        bank[::2, 0, :, 0] = -.65
        target = cv.copy()
        target[:, :, 0] = .5
        valid = np.ones((6, 3), bool)
        valid[rng.choice(6, 2, replace=False)] = False
        ols = rng.normal(0, .15, cv.shape)
        initial = np.array([0, -1, 0, -1, 0, -1])
        groups = np.arange(6).reshape(3, 2)
        p = m.menu(cv, bank, ols, initial, groups, target, valid, 1.)
        radius = np.linalg.norm(bank-cv[:, None], axis=-1).max(2)
        raw = [[-1]+[k for k in range(3) if valid[b].any() and 0 < radius[b, k] <= radius[a, 0]]
               for a, b in groups]
        feasible = []
        for arm in product(*raw):
            out = p["original"].copy()
            for (a, b), k in zip(groups, arm, strict=True):
                if k >= 0:
                    out[a], out[b] = cv[a], bank[b, k]
            risk = policy(out, cv, cv, target, valid, ["r"]*6, easy_cut=1., hard_cut=2.)["recordings"]["r"]
            if risk["supported"]:
                z = [comparison_masses(ref, out, cv, target, valid, np.full(6, .5))
                     for ref in (p["original"], cv, ols)]
                feasible.append([a["known_delta"]-a["unknown_radius"] for a in z])
        for joint in (False, True):
            options = [a[0] for a in feasible if not joint or min(a) >= max(1., p["reference"])*m.POS]
            plan, _ = m.solve(p, single=False, joint=joint)
            if options:
                assert plan is not None
                np.testing.assert_allclose(p["terms"][plan, 4].sum(), max(options), atol=1e-9)
            else:
                assert plan is None
