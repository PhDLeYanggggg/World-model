import itertools

import numpy as np

from src.evaluation.m3w_quarter_swap_capacity import construct, passes, prepare, solve
from src.evaluation.m3w_recording_forecast_comparison import score_recording


def test_new_easy_support_and_actual_score_agree():
    floor = np.zeros((3, 12, 2))
    bank = np.zeros((3, 1, 12, 2))
    bank[:, 0, :, 0] = [np.full(12, 2.), np.full(12, 4.), np.ones(12)]
    target = floor.copy()
    target[:, :, 0] = [np.full(12, .5), np.ones(12), np.zeros(12)]
    valid = np.ones((3, 12), bool)
    valid[-1] = False
    p = prepare(floor, bank, np.array([-1, -1, 0]), [np.arange(3)], target, valid, .75)
    assert not passes(p['base'], p['limits'])
    take, state, value = solve(p, 1)
    assert state == 'optimal'
    pred = construct(p, take)
    s = score_recording({'constant_velocity_causal_fd': floor, 'before': p['before'], 'after': pred},
        {'pair': ('before', 'after', 'constant_velocity_causal_fd')}, target, valid,
        np.zeros(3, int), 'rec', easy_cut=.75, hard_cut=1.)
    assert s['metrics']['after']['risk']['supported']
    effect = s['comparisons']['pair']
    np.testing.assert_allclose(value, effect['known_delta']-effect['unknown_radius'])


def test_milp_matches_exhaustive_small_swaps_and_keeps_query_counts():
    p = dict(gains=np.array([1., 4., 1., 2., 10.]), query=np.array([0, 0, 1, 1, 2]),
        changes=np.array([[-3., -2., -2., 1.], [1., -1., -1., 0.], [-3., -2., -1., 1.],
                          [-2., -1., -1., 1.], [7., 3., 1., 0.]]),
        base=np.array([1., 1., 0., 0.]), limits=np.array([0., 0., 0., 1.]), queries=3)
    good = []
    for count in range(4):
        for picks in itertools.combinations(range(5), count):
            i = np.array(picks, int)
            if len(np.unique(p['query'][i])) == count and passes(p['base']+p['changes'][i].sum(0), p['limits']):
                good.append(p['gains'][i].sum())
    take, status, bound = solve(p, 4)
    assert status == 'optimal'
    assert p['gains'][take].sum() == max(good)
    np.testing.assert_allclose(bound, max(good))
