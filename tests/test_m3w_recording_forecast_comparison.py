import numpy as np

from src.evaluation.m3w_recording_forecast_comparison import merge_comparisons, score_recording


def test_unknown_path_disagreement_counts_without_imputing_error_or_easy_support():
    b = np.zeros((3, 12, 2)); p = b.copy(); p[:, :, 0] = 2
    y = b.copy(); y[0, :, 0] = 3
    mask = np.zeros((3,12), bool); mask[0, :4] = True
    paths = {'constant_velocity_causal_fd':b, 'candidate':p}
    r = score_recording(paths, {'main':('constant_velocity_causal_fd','candidate','constant_velocity_causal_fd')},
        y, mask, np.arange(3), 'a', easy_cut=1., hard_cut=2.)
    assert r['metrics']['candidate']['risk']['known_easy_risk'] is None
    assert not r['metrics']['candidate']['risk']['supported']
    assert r['metrics']['candidate']['known_ADE_pixel'] == 1.
    m = merge_comparisons([r['comparisons']['main']])
    assert m['reference'] == 3 and m['known_delta'] == 2 and m['unknown_radius'] == 4
    assert m['completion_lower_pp'] < 0 < m['known_pp']


def test_merge_sums_mass_before_dividing_and_keeps_empty_undefined():
    a = dict(reference=5., known_delta=2., unknown_radius=1.)
    b = dict(reference=95., known_delta=0., unknown_radius=5.)
    assert merge_comparisons([a,b])['completion_lower_pp'] == -4.
    assert merge_comparisons([])['completion_lower_pp'] is None
