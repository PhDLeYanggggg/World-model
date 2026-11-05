import inspect

import numpy as np

from src.world_model import m3w_donor_augmentation as m
from src.data_unification.m3w_moving_forecast_data import encode_scene
from src.world_model.m3w_european_source_forecast import baseline_numpy


def test_causal_features_and_label_support_are_separate():
    h = np.stack([np.column_stack((np.arange(8)+i*10, np.arange(8)*.5)) for i in range(3)])
    g, local, context = encode_scene(h, np.ones((3, 8), bool), np.arange(3))
    origin = h[:, -1]
    physical = np.stack([baseline_numpy(h, k)-origin[:, None] for k in range(6)], 1)
    proposal = physical[:, 1]+np.arange(1, 13)[None, :, None]*np.array([.2, .1])
    args = (g, context, physical, proposal, np.zeros(3, int), np.ones(3, int), np.arange(3),
            np.array(['source']*3), 1)
    x, scale, allowed, paths = m.causal_features(*args)
    y = physical[:, 1].copy()
    valid = np.ones((3, 12), bool)
    valid[1] = False
    first, known = m.supervision(paths, y, valid, scale, 1, 10.)
    y[0] += 100
    second, _ = m.supervision(paths, y, valid, scale, 1, 10.)
    np.testing.assert_array_equal(x, m.causal_features(*args)[0])
    assert not np.array_equal(first[0], second[0])
    assert known.tolist() == [True, False, True]
    assert np.isnan(first[1]).all()
    assert not {'target', 'valid', 'future_endpoint'} & set(inspect.signature(m.causal_features).parameters)
