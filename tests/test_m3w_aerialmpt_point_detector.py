import numpy as np

from scripts import audit_m3w_aerialmpt_track_guard as guard
from scripts import run_m3w_aerialmpt_point_detector as point


def test_held_locations_excluded_from_detector_and_threshold_fits():
    folds = point.plans(['bauma1', 'bauma2', 'marienplatz', 'oac', 'pasing1L'],
                        ['bauma2', 'marienplatz', 'pasing1L'])
    assert folds['bauma']['evaluate'] == ['bauma2']
    for k, v in folds.items():
        a = {point.group(x) for x in v['fit']}
        b = {point.group(x) for x in v['calibrate']}
        assert a and len(b) == 1 and not a & b and k not in a | b


def test_point_detector_grid_and_causal_track_guard():
    imgs = [(np.full((80, 80, 3), 100, np.uint8), np.array([[20., 24.]]))]
    x, y = point.patches(imgs, np.random.default_rng(17))
    assert point.model()(x).shape == y.shape == (6, 1, 40, 40)
    assert int(y.sum()) == 6
    s = {.2: {'stable': 20, 'online': 30}, .35: {'stable': 10, 'online': 30},
         .5: {'stable': 8, 'online': 10}, .8: {'stable': 1, 'online': 1}}
    assert guard.choose(s, .35) == .5
    assert guard.choose(s, .65) is None
