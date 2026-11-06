import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from src.data_unification import m3w_past_background_motion as method

CFG = json.loads((Path(__file__).parents[1]/'configs/m3w_moving_past_background_v1.json').read_text())['measurement']


def textured():
    rng = np.random.default_rng(17)
    image = cv2.GaussianBlur(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), (3, 3), .6)
    return image


def test_true_image_motion_recovered_in_original_coordinates_and_repeats():
    cv2.setNumThreads(4); cv2.ocl.setUseOpenCL(False)
    image = textured(); transform = np.array([[1., 0., 4.], [0., 1., -3.]])
    current = cv2.warpAffine(image, transform, (640, 360))
    box = np.array([[100., 80., 150., 140.]])
    a = method.prepare(image, box, CFG); b = method.prepare(current, box+np.array([4, -3, 4, -3]), CFG)
    x = method.estimate(a, b, CFG); y = method.estimate(a, b, CFG)
    assert x == y and x['supported']
    np.testing.assert_allclose(np.array(x['native_affine'])[:2], transform, atol=.15)
    assert x['spatial_coverage'] > .5


def test_native_resizing_is_explicit_and_boxes_excluded():
    image = np.zeros((720, 1280, 3), np.uint8)
    p = method.prepare(image, np.array([[100., 80., 200., 160.]]), CFG)
    assert p['gray'].shape == (360, 640) and p['centers_in_bounds'] == 1
    assert p['mask'][60, 75] == 0
    np.testing.assert_array_equal(p['resize_scale'], [.5, .5])


def test_rotation_and_native_scale_conversion_match_known_digital_transform():
    image = cv2.resize(textured(), (1280, 720))
    transform = cv2.getRotationMatrix2D((640, 360), 1., 1.005)
    transform[:, 2] += [3, -2]
    current = cv2.warpAffine(image, transform, (1280, 720))
    a = method.prepare(image, np.empty((0, 4)), CFG)
    b = method.prepare(current, np.empty((0, 4)), CFG)
    result = method.estimate(a, b, CFG)
    assert result['supported']
    grid = np.array([[200., 200.], [1000., 200.], [1000., 500.], [200., 500.]])
    got = np.asarray(result['native_affine'])
    expected = grid@transform[:, :2].T+transform[:, 2]
    actual = grid@got[:2, :2].T+got[:2, 2]
    assert np.linalg.norm(expected-actual, axis=1).max() < .5


def test_textureless_measurement_is_unknown_not_verified_zero_motion():
    p = method.prepare(np.zeros((360, 640, 3), np.uint8), np.empty((0, 4)), CFG)
    result = method.estimate(p, p, CFG)
    assert result['supported'] is False and result['reason'] == 'insufficient_corners'
    assert result['residual_median'] is None and result['fb_median'] is None


def flow(T, supported=True):
    return dict(supported=supported, native_affine=np.asarray(T).tolist(), inlier_fraction=.9,
        matches=100, residual_median=.1, fb_median=.1, spatial_coverage=.7,
        previous_background_fraction=.8, current_background_fraction=.8,
        log_scale=0., rotation_cos=1., rotation_sin=0.)


def test_context_has48past_only_translation_invariant_features_and_correct_stabilization():
    h = np.stack((np.c_[np.arange(8)*2+100, np.ones(8)*200], np.c_[np.arange(8)*3+120, np.ones(8)*220]))
    p = np.tile(np.c_[np.arange(1, 13)*2, np.zeros(12)], (2, 1, 1))
    T = np.eye(3); T[0, 2] = 2.
    flows = {f: flow(T) for f in range(1, 8)}
    a = method.contextual_features(h, p, 7, flows)
    assert a.shape == (2, 48) and len(method.NAMES) == 48
    # Stationary background point transported2pixels/step has zero stabilized speed.
    assert np.linalg.norm(a[0, 28:34]) < 1e-6
    offset = np.array([40., -25.]); C = np.eye(3); C[:2, 2] = offset
    moved = {f: flow(C@T@np.linalg.inv(C)) for f in flows}
    np.testing.assert_allclose(a, method.contextual_features(h+offset, p, 7, moved), atol=1e-6)
    with pytest.raises(ValueError):
        method.contextual_features(h, p, 7, {**flows, 8: flow(T)})
    assert not {'target', 'future', 'valid'}&set(inspect.signature(method.contextual_features).parameters)


def test_box_indexing_does_not_mix_future_boxes_into_past_mask():
    rows = b'0,1,10,20,5,6,1,1,1,1\n1,1,100,200,10,10,1,1,1,1\n'
    a = method.boxes_by_frame(rows)
    b = method.boxes_by_frame(rows.replace(b'100,200,10,10', b'999,888,10,10'))
    np.testing.assert_array_equal(a[0], b[0])
    assert not np.array_equal(a[1], b[1])


def test_past_transform_composition_order_is_correct_for_noncommuting_motion():
    history = [np.array([100., 150.])]; flows = {}
    for f in range(1, 8):
        theta = .005*f
        T = np.array([[np.cos(theta), -np.sin(theta), f], [np.sin(theta), np.cos(theta), -f/2], [0, 0, 1.]])
        flows[f] = flow(T); history.append(T[:2, :2]@history[-1]+T[:2, 2])
    x = method.contextual_features(np.asarray(history)[None], np.zeros((1, 12, 2)), 7, flows)
    assert np.linalg.norm(x[0, 28:34]) < 1e-5
