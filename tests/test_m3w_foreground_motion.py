import cv2
import numpy as np

from src.world_model.m3w_foreground_motion import features, flow


def test_shifted_pixels_have_independent_flow():
    a = np.random.default_rng(4).integers(0, 255, (100, 100), dtype=np.uint8)
    a = cv2.GaussianBlur(a, (5, 5), 1.1)
    b = cv2.warpAffine(a, np.array([[1., 0., 3.], [0., 1., -2.]]), (100, 100))
    payload = lambda g: dict(gray=g, resize_scale=np.ones(2))
    x = flow(payload(a), payload(b), np.array([[30., 30., 20., 20.]]))
    assert x[0, 2] == 1.
    np.testing.assert_allclose(x[0, :2], [3., -2.], atol=.18)


def test_unavailable_local_flow_is_explicit_and_zero():
    x = features(np.zeros((3, 8, 2)), np.zeros((3, 7, 5)))
    assert x.shape == (3, 12)
    np.testing.assert_array_equal(x, 0.)
