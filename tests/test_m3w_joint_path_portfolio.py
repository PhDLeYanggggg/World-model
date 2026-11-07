import numpy as np

from src.world_model import m3w_joint_path_portfolio as model
from src.world_model.m3w_record_joint_actions import PAIR_WEIGHT, local_edges


def test_pair_score_equals_direct_graph_energy_change():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(5, 8, 2))
    cv, ols = rng.normal(size=(2, 5, 12, 2))
    bank = rng.normal(size=(5, 3, 12, 2))
    scale = np.arange(1., 6.)
    u = rng.normal(size=(5, 3))
    z, ones = np.zeros_like(u), np.ones_like(u)
    rows, arms, _, values, _ = model.options(h, cv, ols, bank, scale, u, z, z, ones, ones > 0, 2)
    i, j = local_edges(h)
    degree = np.bincount(np.r_[i, j])
    weight = 1./(np.sqrt(degree[i]*degree[j])*scale[i]*scale[j])
    target = ols[i]-ols[j]
    energy = lambda p: np.sum(weight*((p[i]-p[j]-target)**2).sum(-1).mean(1))
    expected = []
    for r, a in zip(rows, arms):
        pred = cv.copy()
        pred[r] = bank[r, a]
        expected.append((u[r, a]/scale[r]).sum()-PAIR_WEIGHT*(energy(pred)-energy(cv)))
    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_late_gate_does_not_change_the_matched_activation_mask():
    cv = np.zeros((3, 12, 2))
    h = np.zeros((3, 8, 2))
    bank = np.ones((3, 2, 12, 2))
    u = np.ones((3, 2))
    risk = np.array([[1., -1.], [1., -1.], [1., -1.]])
    old = np.array([0, 0, -1])
    raw, add, joint, late, _, _ = model.choose(h, cv, cv, bank, np.ones(3), u,
        risk, risk, u, u > 0., 2, old, False)
    assert (raw >= 0).sum() == (late >= 0).sum() == 2
    assert (add == -1).all() and (joint == -1).all()
