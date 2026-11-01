import itertools
import numpy as np
import pytest

from src.world_model import m3w_pair_action_allocation as api


def fixture(seed=0, n=3):
    rng = np.random.default_rng(seed); p = np.zeros((n, 7, 5)); p[..., 0] = rng.uniform(.05, 1, (n, 7))
    p[..., 2:4] = 1.; p[..., 1] = rng.uniform(0, .025, (n, 7)); p[..., 4] = p[..., 1]/2
    ok = np.zeros((n, 7), bool); ok[:, :2] = True
    origin = rng.normal(size=(n, 2)); floor = rng.normal(size=(n, 12, 2))
    paths = rng.normal(size=(n, 7, 12, 2)); scales = rng.uniform(.5, 2, n); ids = np.arange(n)
    edges, table = api.geometry(paths, floor, origin, scales, ids, neighbors=2)
    parent = np.zeros(n, np.int64)
    if not api.base.feasible(p, ok, parent): p[..., 1] *= .1; p[..., 4] *= .1
    return p, ok, paths, floor, origin, scales, ids, edges, table, parent


@pytest.mark.parametrize('seed', range(20))
def test_exact_pair_solver_matches_exhaustive(seed):
    p, ok, _, _, _, s, _, edges, table, parent = fixture(seed)
    c, info = api.solve(p, ok, parent, edges, table, s, weight=1., seconds=5.)
    objective = lambda v: -api.base.terms(p, v)[0]/s.sum()+api.pair_value(edges, table, v)
    possible = [np.array(v) for v in itertools.product((-1, 0, 1), repeat=len(p))]
    legal = [v for v in possible if (v >= 0).sum() == (parent >= 0).sum() and api.base.feasible(p, ok, v)]
    assert objective(c) == pytest.approx(min(map(objective, legal)), abs=1e-9)
    assert info['optimal']


def test_common_frame_rigid_and_scale_invariance():
    _, _, paths, floor, origin, s, ids, edges, table, _ = fixture()
    rot = np.array([[0., -1.], [1., 0.]])
    ee, tt = api.geometry(paths@rot*3, floor@rot*3, origin@rot*3+100, s*3, ids, neighbors=2)
    np.testing.assert_array_equal(edges, ee); np.testing.assert_allclose(table, tt, atol=1e-13)
    assert (table[:, 0, 0] == 0).all()


def test_zero_weight_exact_control_and_nonoptimal_parent_fallback(monkeypatch):
    p, ok, _, _, _, s, _, edges, table, parent = fixture()
    c, info = api.solve(p, ok, parent, edges, table, s, weight=0., seconds=1.)
    np.testing.assert_array_equal(c, parent)
    monkeypatch.setattr(api, 'milp', lambda **_: type('Result', (), dict(status=1, x=None))())
    c, info = api.solve(p, ok, parent, edges, table, s, weight=1., seconds=1.)
    np.testing.assert_array_equal(c, parent); assert info['reason'] == 'solver_status_1_parent_fallback'


def test_resume_and_geometry_binding():
    p, ok, r, f, o, s, ids, _, _, parent = fixture()
    rec = np.array(['a']*3); frame = np.array([1]*3); state = []
    args = (p, ok, rec, frame, ids, r, f, o, s, parent)
    direct = api.allocate(*args, weight=1., neighbors=2, seconds=3., checkpoint=state.append)
    resumed = api.allocate(*args, weight=1., neighbors=2, seconds=3., resume_state=state[-1])
    for name in api.NAMES: np.testing.assert_array_equal(direct[0][name], resumed[0][name])
    assert direct[2] == resumed[2]
    with pytest.raises(ValueError, match='changed'):
        api.allocate(*args, weight=2., neighbors=2, seconds=3., resume_state=state[-1])
    with pytest.raises(TypeError): api.allocate(*args, weight=1., neighbors=2, seconds=3., future_endpoint=np.zeros((3, 2)))


def test_invalid_graph_and_undefined_scale_rejected():
    p, ok, _, _, _, s, _, edges, table, parent = fixture()
    with pytest.raises(ValueError): api.solve(p, ok, parent, edges[:, ::-1], table, s, weight=1., seconds=1.)
    with pytest.raises(ValueError): api.solve(p, ok, parent, edges, table, s*0, weight=1., seconds=1.)
