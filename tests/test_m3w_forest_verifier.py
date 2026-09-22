import numpy as np

from scripts.verify_m3w_forest_cost import manual_choices
from src.world_model.m3w_forest_cost_head import selections


def test_separate_choice_formula_matches_fixed_gate_and_budget():
    rng = np.random.default_rng(7)
    f, n = rng.uniform(0, 10, (2, 100, 2))
    h = rng.normal(size=(100, 8, 2)); h[:7, -1] = h[:7, -2]
    d = rng.uniform(0, 4, 100); d[7:12] = 0
    ids = rng.permutation(100)
    f[12:20, 1] = 0
    a, b = manual_choices(f, n, h, d, ids), selections(f, n, h, d, ids)
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert a['forest'].sum() == a['neural_matched'].sum()
    assert not a['forest'][:12].any()


def test_separate_matching_zero_budget_never_forces_switch():
    h = np.ones((4, 8, 2)); h[:, -1] = 2
    f = np.array([[0., 1.]]*4); n = np.array([[4., 1.]]*4)
    a = manual_choices(f, n, h, np.ones(4), np.arange(4))
    assert not a['forest'].any() and not a['neural_matched'].any()
