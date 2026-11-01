import itertools
import numpy as np
import pytest

from src.world_model import m3w_pair_action_allocation as api
from tests.test_m3w_pair_action_allocation import fixture


@pytest.mark.parametrize('seed', range(12))
def test_transport_with_floor_and_changing_selected_agents(seed):
    p, ok, _, _, _, scales, _, edges, table, parent = fixture(seed, n=4)
    parent[2:] = -1
    if not api.base.feasible(p, ok, parent):
        p[..., 1] *= .1; p[..., 4] *= .1
    choice, info = api.solve(p, ok, parent, edges, table, scales, weight=1., seconds=5.)
    objective = lambda c: -api.base.terms(p, c)[0]/scales.sum()+api.pair_value(edges, table, c)
    candidates = [np.array(c) for c in itertools.product((-1, 0, 1), repeat=4)]
    candidates = [c for c in candidates if (c >= 0).sum() == 2 and api.base.feasible(p, ok, c)]
    assert info['optimal']
    assert objective(choice) == pytest.approx(min(map(objective, candidates)), abs=1e-9)
    assert int((choice >= 0).sum()) == 2


def test_graph_cost_matches_existing_complete_pair_formula():
    _, _, paths, floor, origin, scales, ids, _, _, parent = fixture(n=4)
    edges, table = api.geometry(paths, floor, origin, scales, ids, neighbors=3)
    choice = parent.copy(); choice[1] = -1; choice[2] = 1
    selected = floor.copy(); at = choice >= 0; selected[at] = paths[np.flatnonzero(at), choice[at]]
    selected = selected+origin[:, None]; baseline = floor+origin[:, None]
    i, j = edges.T
    reference = np.linalg.norm(baseline[i]-baseline[j], axis=-1)
    actual = np.linalg.norm(selected[i]-selected[j], axis=-1)
    expected = (np.maximum(reference-actual, 0)/np.sqrt(scales[i]*scales[j])[:, None]).mean()
    assert api.pair_value(edges, table, choice) == pytest.approx(expected, abs=1e-13)


def test_numerically_invalid_optimum_returns_parent(monkeypatch):
    p, ok, _, _, _, scales, _, edges, table, parent = fixture()
    monkeypatch.setattr(api, 'milp', lambda **kw: type('R', (), dict(status=0,
        x=np.full(len(kw['c']), .5), fun=-1e10))())
    choice, info = api.solve(p, ok, parent, edges, table, scales, weight=1., seconds=1.)
    np.testing.assert_array_equal(choice, parent)
    assert info['reason'] == 'numerical_postcheck_parent_fallback'
