import numpy as np

from scripts import build_m3w_songdo_candidate_bank as bank
from src.world_model import m3w_equal_radius_slates as search
from src.world_model import m3w_slate_shared_menu as shared
from src.world_model.m3w_slate_imitation import Slate


def test_combinatorial_counts_match_explicit_ordered_candidates():
    peak = np.array([[0., 2., 3.], [3., 4., 5.], [3., 2., 2.], [2., .001, 1.]])
    n, pairs, uniform, complete, radius = bank.slot_counts(peak)
    choices = np.argwhere(peak >= radius)
    assert n == len(choices)
    assert pairs == sum(a != b for a, _ in choices for b, _ in choices)
    assert uniform == sum(a != b and c == d for a, c in choices for b, d in choices)
    assert complete == int((peak >= radius).all(axis=1).sum())
    assert bank.slot_counts(peak[:2]) == (0, 0, 0, 0, 0.)


def test_symmetric_search_matches_existing_exhaustive_search_and_counts_pairs_once():
    h = np.cumsum(np.random.default_rng(781).normal(size=(6, 8, 2)), axis=1)
    cv, ols = bank.analytic(h)
    paths = np.stack([ols, cv - 1.1, cv + .5], axis=1)
    radius = .3
    with bank.torch.random.fork_rng():
        bank.torch.manual_seed(79)
        net = Slate().eval()
    additive, joint, total = shared.choose(net, h, cv, ols, paths, [radius, radius])
    a, count = search.choose(net, h, cv, ols, paths, radius, block=3)
    assert a['uniform'] is not None
    assert count * 2 == total
    np.testing.assert_allclose(additive, a['additive']['path'], atol=1e-10)
    np.testing.assert_allclose(joint, a['joint']['path'], atol=1e-10)


def test_empty_session_preserves_valid_output_shapes():
    paths = bank.forecasts(np.empty((0, 8, 2)), {17: []})
    assert all(a.shape == (0, 12, 2) for a in paths.values())
    info = bank.aggregate(bank.count_queries(paths, np.array([0]), 17))
    assert info['queries'] == info['joint_feasible'] == 0
