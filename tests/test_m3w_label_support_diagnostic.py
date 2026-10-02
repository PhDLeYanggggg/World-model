import numpy as np
import pytest
from src.world_model import m3w_label_support_diagnostic as api


def test_unknown_and_invalid_garbage_are_not_outcomes():
    r = np.zeros((2, 12, 2)); c = r+1; t = r.copy(); m = np.zeros((2, 12), bool)
    m[0, :2] = True; t[~m] = np.nan
    d = api.temporal_errors(r, c, t, m)
    assert np.isnan(d['signed_error'][1]) and not d['leave_one_out_defined'][1]
    assert d['valid_steps'].tolist() == [2, 0]
    np.testing.assert_allclose(d['signed_error'][0], np.sqrt(2))


def test_temporal_fragility_and_early_late_reversal():
    r = np.zeros((1, 12, 2)); c = r.copy(); c[..., 0] = 2
    t = r.copy(); t[:, 0, 0] = 0; t[:, 11, 0] = 1.5
    m = np.zeros((1, 12), bool); m[:, [0, 11]] = True
    d = api.temporal_errors(r, c, t, m)
    assert d['signed_error'][0] == .5
    assert d['leave_one_out_sign_flip'][0] and d['early_late_opposite_sign'][0]


def test_single_label_fragility_undefined_not_stable():
    r = np.zeros((1, 12, 2)); m = np.zeros((1, 12), bool); m[0, 0] = True
    d = api.temporal_errors(r, r+1, r, m)
    assert not d['leave_one_out_defined'][0] and not d['early_late_defined'][0]


def test_raw_future_quality_requires_exact_agent_frame_labels():
    dtype = [(k, 'f8') for k in ('frame', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id')]
    t = np.zeros(3, dtype=dtype); t['frame'] = [0, 12, 36]
    t['x_max'] = 2; t['y_max'] = 4; t['confidence'] = [.9, .5, .7]; t['class_id'][2] = 1
    v = np.zeros((1, 12), bool); v[0, [0, 2]] = True
    xy = np.zeros((1, 12, 2)); xy[v] = [1, 2]
    out = api.raw_future_quality(t, [0], xy, v)
    assert out['future_class_id_changed'][0]
    assert out['future_mean_detector_confidence'][0] == .6
    xy[0, 0, 0] += 1
    with pytest.raises(AssertionError): api.raw_future_quality(t, [0], xy, v)


def test_unknown_cohort_remains_separate_and_empty_risk_undefined():
    y = np.array([[0, 1, 2, 2, 1], [np.nan]*5]); m = np.array([True, True])
    temp = {k:np.array([False, False]) for k in ('leave_one_out_defined','leave_one_out_sign_flip',
                                                'early_late_defined','early_late_opposite_sign')}
    d = api.cohort(y, m, [1, 2], [1, 2], [0, 0], [5, 5], np.array([1, 0]), temp)
    assert d['unique_tracks'] == 1
    assert d['strata']['unknown']['unknown_envelope'] == 2
    assert d['strata']['unknown']['easy_harm_ratio'] is None
    assert d['strata']['one_to_three']['easy_harm_ratio'] == .5
    y[1] = 0
    with pytest.raises(ValueError): api.cohort(y, m, [1, 2], [1, 2], [0, 0], [5, 5], np.array([1, 0]), temp)


def test_query_matching_does_not_compare_unmatched_recordings():
    d = api.contrasts([9., 3., 100., 0.], np.array([1,0,1,0], bool), np.array([0,1,0,1], bool),
                      [0, 0, 1, 2], [1, 1, 1, 1])
    assert d['matched_queries'] == 1 and d['query_matched_mean_difference'] == 6


def test_empty_contrast_not_zero():
    d = api.contrasts([1.], np.array([True]), np.array([False]), [0], [0])
    assert d['raw_mean_difference'] is None and d['query_matched_mean_difference'] is None


def test_bootstrap_clusters_localities_before_repeated_heads():
    a = [dict(source='A', value=1), dict(source='B', value=3)]
    b = a+[dict(source='A', value=1)]*20
    assert api.locality_interval(a, lambda g:g['value']) == api.locality_interval(b, lambda g:g['value'])


def test_temporal_mask_must_be_boolean():
    x = np.zeros((1, 12, 2))
    with pytest.raises(ValueError): api.temporal_errors(x, x, x, np.zeros((1, 12)))
